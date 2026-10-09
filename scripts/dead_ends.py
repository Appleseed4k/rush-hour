"""Where a pass's lapse exposure goes, and how much recurring dead-end patterns
("negative chunks") could prune.

Each pass is a full lapse-free (gamma 0) and_or search from a board the
gamma-0.07 solver actually visits. Every OrNode visit that reaches its lapse
check is one check, classified as:

- dead end: inside a maximal subgoal that returned no resolution,
- other: on the returned plan, or in subgoals that resolved but whose
  resolution the search discarded (a blocker order that failed, say).

A pass whose search takes C checks gets through without lapsing with
probability (1 - gamma)**C, so C is the quantity chunks of any kind act on.

Each subgoal is described by sig(): its car's length and how it sits
relative to the move it unblocks, and for every candidate move, the cars
blocking it - "P" if protected (an ancestor in the plan, an instant fail),
"V" if that subgoal is already open above (a loop, also an instant fail),
else the blocker's own description one level shallower ("*" at depth 0).
Like chunks.relation(), it never mentions absolute position or direction.
A signature that fails in nearly every training occurrence is a negative
chunk: recognizing it skips the subgoal's search for a single check."""
import argparse
import collections
import random
from multiprocessing import Pool

import and_or as ao
from chunks import sample_entries

STACK, ROOTS, COUNT = [], [], [0]
LIMIT = [3_000]
RECORDING = [False]
_or_solve = ao.OrNode.solve


class _Exhausted(Exception):
    pass


def _recording_solve(self):
    if not RECORDING[0]:
        return _or_solve(self)
    rec = {"node": self, "children": [], "free": (self.car_name, self.collisions) in self.visited, "ok": False}
    (STACK[-1]["children"] if STACK else ROOTS).append(rec)
    if rec["free"]:  # returns before its lapse check
        return None
    COUNT[0] += 1
    if COUNT[0] > LIMIT[0]:
        raise _Exhausted
    STACK.append(rec)
    try:
        result = _or_solve(self)
    finally:
        STACK.pop()
    rec["ok"] = result is not None
    return result


ao.OrNode.solve = _recording_solve

# At gamma 0, _solve_pass only calls legal_moves for its random fallback move,
# i.e. when no plan was found.
FALLBACK = [False]
_legal_moves = ao.legal_moves


def _flagging_legal_moves(state):
    FALLBACK[0] = True
    return _legal_moves(state)


ao.legal_moves = _flagging_legal_moves


def sig(state, car_name, collisions, parent_direction, visited, protected, k):
    """Depth-k description of the subgoal "car_name must vacate collisions"."""
    if k == 0:
        return "*"
    node = ao.OrNode(state, car_name, collisions)
    horizontal = node.car[0][0] == node.car[1][0]
    parallel = (parent_direction in ("l", "r")) == horizontal
    inner_visited = visited | {(car_name, frozenset(collisions))}
    by_direction = collections.defaultdict(list)
    for direction, steps in node.directions():
        parts = []
        for blocker, cells in ao.AndNode(state, car_name, direction, steps).blockers().items():
            if blocker in protected:
                parts.append("P")
            elif (blocker, frozenset(cells)) in inner_visited:
                parts.append("V")
            else:
                parts.append(sig(state, blocker, cells, direction, inner_visited, ao.protect(protected, car_name),
                                 k - 1))
        by_direction[direction].append("[" + ",".join(sorted(parts)) + "]")
    sides = []
    for direction in (("l", "r") if horizontal else ("u", "d")):
        label = ("f" if direction == parent_direction else "b") if parallel else ""
        sides.append(label + "".join(by_direction[direction]))
    return f"{'p' if parallel else 'x'}{len(node.car)}{{{'|'.join(sorted(sides))}}}"


def _summarize(rec, depths):
    """Compact, picklable copy of a record tree, with signatures."""
    n = rec["node"]
    sigs = () if rec["free"] else tuple(
        sig(n.state, n.car_name, n.collisions, n.parent_direction, n.visited, n.protected, k) for k in depths)
    return {"ok": rec["ok"], "free": rec["free"], "sigs": sigs,
            "children": [_summarize(c, depths) for c in rec["children"]]}


def _checks(rec):
    return 0 if rec["free"] else 1 + sum(_checks(c) for c in rec["children"])


def _dead_end_checks(rec):
    if rec["free"]:
        return 0
    if not rec["ok"]:
        return _checks(rec)
    return sum(_dead_end_checks(c) for c in rec["children"])


def measure_pass(state, depths):
    """One gamma-0 pass from `state`: dict of checks by kind, plus record trees."""
    STACK.clear()
    ROOTS.clear()
    COUNT[0] = 0
    FALLBACK[0] = False
    RECORDING[0] = True
    try:
        result = ao._solve_pass(state, None, 0.0)
    except _Exhausted:
        return None
    finally:
        RECORDING[0] = False
    found = not FALLBACK[0]
    moves = result if type(result) is list else result[1] or []
    trees = [_summarize(r, depths) for r in ROOTS]
    dead = sum(_dead_end_checks(t) for t in trees)
    return {"checks": COUNT[0], "dead": dead, "found": found, "plan": len(moves) if found else 0,
            "trees": trees}


def measure_puzzle(args):
    """Up to `per_puzzle` distinct pass-start boards from one gamma-0.07 run."""
    idx, distance, state, per_puzzle, seed, depths = args
    rng = random.Random(seed)
    random.seed(seed)
    boards, moves, s = [state], 0, state
    while ao._red_steps(s) > 0 and moves < 200:
        attempt = ao.solve(s, gamma=0.07)
        step = attempt if type(attempt) is list else attempt[1]
        if not step:
            break
        for move in step:
            s = ao.AndNode(s, *move).apply(s)
        moves += len(step)
        if ao._red_steps(s) > 0:
            boards.append(s)
    boards = list(dict.fromkeys(boards))
    picked = [boards[0]] + rng.sample(boards[1:], min(per_puzzle - 1, len(boards) - 1))
    out = []
    for board in picked:
        random.seed(seed)
        m = measure_pass(board, depths)
        if m is not None:
            m["puzzle"], m["distance"] = idx, distance
        out.append(m)
    return out


def _walk(t):
    yield t
    for c in t["children"]:
        yield from _walk(c)


def pruned_checks(t, negative, k, cost=1):
    """(checks, wrong prunes) if every subgoal whose depth-k signature is in
    `negative` were recognized and skipped for `cost` checks. A recognized
    subgoal that actually resolves is a wrong prune; it's counted and left
    unpruned."""
    if t["free"]:
        return 0, 0
    if t["sigs"][k] in negative:
        if not t["ok"]:
            return cost, 0
        wrong = 1
    else:
        wrong = 0
    total = 1
    for c in t["children"]:
        checks, w = pruned_checks(c, negative, k, cost)
        total += checks
        wrong += w
    return total, wrong


def _maximal_dead_ends(t):
    """Maximal failing subgoals in a record tree (no failing ancestor)."""
    if t["free"]:
        return
    if not t["ok"]:
        yield t
        return
    for c in t["children"]:
        yield from _maximal_dead_ends(c)


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="../data/rush_nw.txt")
    ap.add_argument("--count", type=int, default=800, help="puzzles sampled")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--per-puzzle", type=int, default=4, help="boards measured per puzzle")
    ap.add_argument("--limit", type=int, default=3_000, help="checks per pass before it counts as exhausted")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2], help="signature depths compared")
    ap.add_argument("--min-support", type=int, default=3, help="training puzzles a negative chunk must occur in")
    ap.add_argument("--precision", type=float, default=0.98,
                    help="share of training occurrences that must fail for a signature to be a negative chunk")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.07, 0.2])
    args = ap.parse_args()
    LIMIT[0] = args.limit

    entries = sample_entries(args.pool, args.count, args.seed)
    jobs = [(i, d, s, args.per_puzzle, hash((args.seed, i)) & 0xFFFFFFFF, tuple(args.depths))
            for i, (d, s, _line) in enumerate(entries)]
    with Pool(initializer=LIMIT.__setitem__, initargs=(0, args.limit)) as pool:
        results = [m for out in pool.imap_unordered(measure_puzzle, jobs, chunksize=2) for m in out]
    exhausted = sum(m is None for m in results)
    passes = [m for m in results if m is not None]
    test_ids = set(random.Random(args.seed).sample(range(len(entries)), int(len(entries) * args.test_frac)))
    train = [m for m in passes if m["puzzle"] not in test_ids]
    test = [m for m in passes if m["puzzle"] in test_ids]
    print(f"{len(passes)} passes measured ({len(train)} train / {len(test)} test), "
          f"{exhausted} exhausted the {args.limit}-check limit and were dropped")

    print("\nwhere a pass's lapse checks go (all passes):")
    print(f"  {'dist':>5}  {'passes':>6}  {'plan found':>10}  {'plan moves':>10}  {'checks':>7}  "
          f"{'median':>6}  {'dead end':>8}")
    bins = [(1, 8), (9, 12), (13, 16), (17, 99)]
    for lo, hi in bins + [(1, 99)]:
        ms = [m for m in passes if lo <= m["distance"] <= hi]
        total = sum(m["checks"] for m in ms) or 1
        label = "all" if (lo, hi) == (1, 99) else (f"{lo}-{hi}" if hi < 99 else f"{lo}+")
        found = [m for m in ms if m["found"]]
        print(f"  {label:>5}  {len(ms):>6}  {_mean(m['found'] for m in ms):>10.0%}  "
              f"{_mean(m['plan'] for m in found):>10.1f}  {_mean(m['checks'] for m in ms):>7.1f}  "
              f"{sorted(m['checks'] for m in ms)[len(ms) // 2]:>6}  {sum(m['dead'] for m in ms) / total:>8.0%}")

    print("\ndead-end checks by size of the maximal failing subgoal they sit in (all passes):")
    sizes = collections.Counter()
    for m in passes:
        for t in m["trees"]:
            for d in _maximal_dead_ends(t):
                n = _checks(d)
                sizes[next(label for label, hi in (("1", 1), ("2-10", 10), ("11-100", 100), (">100", 10**9))
                           if n <= hi)] += n
    total_dead = sum(sizes.values()) or 1
    print("  " + "  ".join(f"{label}: {sizes[label] / total_dead:.0%}" for label in ("1", "2-10", "11-100", ">100")))
    for ki, k in enumerate(args.depths):
        stats = collections.defaultdict(lambda: [0, 0, set()])
        for m in train:
            for t in m["trees"]:
                for node in _walk(t):
                    if node["free"]:
                        continue
                    entry = stats[node["sigs"][ki]]
                    entry[0 if node["ok"] else 1] += 1
                    entry[2].add(m["puzzle"])
        negative = {s for s, (ok, fail, sup) in stats.items()
                    if len(sup) >= args.min_support and fail / (ok + fail) >= args.precision}
        dead_nodes = [n for m in test for t in m["trees"] for n in _walk(t) if not n["free"] and not n["ok"]]
        ok_nodes = [n for m in test for t in m["trees"] for n in _walk(t) if not n["free"] and n["ok"]]
        print(f"\ndepth-{k} signatures: {len(stats)} distinct in train, {len(negative)} negative chunks "
              f"(support >= {args.min_support}, >= {args.precision:.0%} fail)")
        print(f"  held-out failing subgoals recognized: "
              f"{_mean(n['sigs'][ki] in negative for n in dead_nodes):.0%} of {len(dead_nodes)}; "
              f"resolving subgoals wrongly matched: {_mean(n['sigs'][ki] in negative for n in ok_nodes):.1%} "
              f"of {len(ok_nodes)}")
        print("  per held-out pass with a plan: before -> recognized dead end costs 1 check / 0 checks")
        print(f"  {'dist':>5}  {'checks':>24}  " + "  ".join(f"{'P(no lapse) g=' + str(g):>24}" for g in args.gammas)
              + f"  {'wrong prune':>11}")
        for lo, hi in bins:
            ms = [m for m in test if lo <= m["distance"] <= hi and m["found"]]
            if not ms:
                continue
            after1, after0, wrong = [], [], 0
            for m in ms:
                pruned = [pruned_checks(t, negative, ki, 1) for t in m["trees"]]
                after1.append(sum(c for c, _ in pruned))
                after0.append(sum(pruned_checks(t, negative, ki, 0)[0] for t in m["trees"]))
                wrong += any(w for _, w in pruned)
            before = [m["checks"] for m in ms]
            label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
            cols = "  ".join(f"{_mean((1 - g) ** c for c in before):>6.0%} -> "
                             f"{_mean((1 - g) ** c for c in after1):>5.0%} / {_mean((1 - g) ** c for c in after0):>5.0%}"
                             for g in args.gammas)
            print(f"  {label:>5}  {_mean(before):>6.1f} -> {_mean(after1):>6.1f} / {_mean(after0):>6.1f}  {cols}  "
                  f"{wrong / len(ms):>11.0%}")


if __name__ == "__main__":
    main()
