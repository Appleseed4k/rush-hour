import argparse
import itertools
import json
import os
import pickle
import random
from collections import Counter, defaultdict

from tqdm import tqdm

from and_or import PROTECT_MOVERS, AndNode, OrNode, legal_moves, or_budget, protect, red_candidates, seed_protected
from lib import DELTAS, parse_puzzle_line, visualize

BOARD_SIZE = 6
OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}


# Chunk mining: every OrNode subtree of a lapse-free AND/OR solution is one
# candidate chunk - a subgoal ("car X must clear these cells") plus the whole
# sub-plan that resolved it. Each move is described only relative to the move
# it unblocks (see relation()), so patterns carry no absolute orientation,
# direction or board position: a pattern and any rotation, reflection or
# translation of it are the same pattern.


class _Exhausted(Exception):
    """Raised when capture_forest's expansion budget runs out."""


def capture_forest(state, rng, max_attempts=100, max_expansions=1_000, protect_movers=PROTECT_MOVERS):
    """Tree-capturing, lapse-free AND-OR solve (like fingerprint.solve_forest,
    but trying every blocker order as AndNode.solve does, and each subgoal's
    candidates fewest-steps-first so mirrored boards resolve alike). Skips any
    resolution that would return to a board already seen, so repositioning
    red can't cycle. When no pass from the current board reaches a new one
    within max_expansions OrNode expansions, plays a random legal move to an
    unseen board instead, as and_or.solve does (drawn from `rng`, not
    recorded as a tree). `protect_movers` is and_or.solve's, with the same
    default (and_or.PROTECT_MOVERS).

    Returns (trees, fallbacks): one tree per successful top-level red pass,
    and how many random moves were needed. None if red doesn't reach the
    exit within max_attempts passes and moves combined."""
    attempts = []
    fallbacks = 0
    seen = {state}
    for _ in range(max_attempts):
        red = dict(state)["red"]
        steps = (BOARD_SIZE - 1) - max(j for _, j in red)
        if steps <= 0:
            return attempts, fallbacks
        budget = [max_expansions]
        red_moves = [("r", steps)] + red_candidates(state, steps)
        results = (result for direction, red_steps in red_moves
                   for result in _capture_and(state, "red", direction, red_steps, None, frozenset(),
                                              seed_protected(protect_movers), budget))
        try:
            result = next((r for r in results if r[0] not in seen), None)
        except _Exhausted:
            result = None
        if result is not None:
            state, tree = result
            attempts.append(tree)
        else:
            moves = [AndNode(state, *move).apply(state) for move in legal_moves(state)]
            moves = [s for s in moves if s not in seen]
            if not moves:
                return None
            state = rng.choice(moves)
            fallbacks += 1
        seen.add(state)
    return None


def _capture_and(state, car_name, direction, steps, collisions, visited, protected, budget):
    """Lazily yields (new_state, tree) for every way AndNode(...).solve() could
    resolve this move without lapsing, in the order its search finds them.

    A tree's "action" is the move as declared when its subgoal was posed
    (what patterns are described by, see relation()); "played" is the slide
    actually made once its blockers were cleared - the same unless clearing
    them moved this car (possible only with protection off, see AndNode), and
    None if they left it on its destination already."""
    node = AndNode(state, car_name, direction, steps)
    blockers = node.blockers()
    if protected & blockers.keys():
        return
    next_protected = protect(protected, car_name)
    for order in itertools.permutations(blockers.items()):
        for working_state, children in _capture_order(state, order, visited, next_protected, budget):
            played = node.move_from(working_state)
            if played is not None and AndNode(working_state, car_name, *played).blockers():
                continue
            tree = {
                "car": car_name,
                "state": state,
                "action": (direction, steps),
                "played": played,
                "collisions": collisions,
                "children": children,
            }
            final_state = working_state if played is None else \
                AndNode(working_state, car_name, *played).apply(working_state)
            yield final_state, tree


def _capture_order(state, order, visited, protected, budget):
    """Resolves `order`'s blockers in sequence, yielding (state, child trees)."""
    if not order:
        yield state, []
        return
    (blocker_name, collisions), rest = order[0], order[1:]
    for new_state, child in _capture_or(state, blocker_name, collisions, visited, protected, budget):
        for final_state, children in _capture_order(new_state, rest, visited, protected, budget):
            yield final_state, [child] + children


def _capture_or(state, car_name, collisions, visited, protected, budget):
    key = (car_name, frozenset(collisions))
    if key in visited:
        return
    budget[0] -= 1
    if budget[0] < 0:
        raise _Exhausted
    next_visited = visited | {key}
    candidates = sorted(OrNode(state, car_name, collisions).directions(), key=lambda c: (c[1], c[0]))
    for direction, steps in candidates:
        yield from _capture_and(state, car_name, direction, steps, collisions, next_visited, protected, budget)


def relation(parent_direction, node):
    """Describes node's move relative only to the move it unblocks (parent's
    direction), so the description survives any rotation or reflection:

    - Crossing parent's lane (perpendicular): "x{need}:{other}" - the steps
      needed to clear the lane the way it went, then the way it didn't
      (together they give its length and which of its cells sat in the lane).
    - In parent's lane (parallel): "p{length}{f|b}{need}" - its length,
      whether it moved forward (same way as parent) or back, and the steps
      needed to clear.

    Either gets "+{n}" when the move overshot what was needed by n steps.
    Sides relative to siblings or grandparents aren't recorded: on a real
    board, the local geometry above already determines them."""
    car = dict(node["state"])[node["car"]]
    direction, steps = node["action"]
    horizontal = car[0][0] == car[1][0]
    axis = 1 if horizontal else 0
    along = [cell[axis] for cell in car]
    hits = [cell[axis] for cell in node["collisions"]]
    lo, hi = min(along), max(along)

    def need(d):
        return max(hits) - lo + 1 if DELTAS[d][axis] > 0 else hi - min(hits) + 1

    if (parent_direction in ("l", "r")) == horizontal:
        rel = "f" if direction == parent_direction else "b"
        desc = f"p{len(car)}{rel}{need(direction)}"
    else:
        desc = f"x{need(direction)}:{need(OPPOSITE[direction])}"
    extra = steps - need(direction)
    return desc + (f"+{extra}" if extra else "")


def annotate(tree):
    """Fills in, for every OrNode below a top-level red tree: "rel" (see
    relation()), "shape" (its unlabeled canonical pattern, used to order
    siblings), "size" (moves in its subtree) and "depth" (OrNodes on its
    longest chain)."""
    def visit(node, parent_direction):
        node["rel"] = relation(parent_direction, node)
        for child in node["children"]:
            visit(child, node["action"][0])
        kids = sorted(c["shape"] for c in node["children"])
        node["shape"] = node["rel"] + (f"({','.join(kids)})" if kids else "")
        node["size"] = 1 + sum(c["size"] for c in node["children"])
        node["depth"] = 1 + max((c["depth"] for c in node["children"]), default=0)

    for child in tree["children"]:
        visit(child, tree["action"][0])


def pattern(node):
    """Canonical pattern of node's subtree: its shape, with siblings in sorted
    order (the order independent blockers were cleared in doesn't matter),
    plus "#k" tags on any car that moves more than once within it."""
    counts = Counter(n["car"] for n in _walk(node))
    labels = {}

    def render(n):
        s = n["rel"]
        if counts[n["car"]] > 1:
            s += f"#{labels.setdefault(n['car'], len(labels))}"
        kids = sorted(n["children"], key=lambda c: c["shape"])
        return s + (f"({','.join(render(c) for c in kids)})" if kids else "")

    return render(node)


def _walk(node):
    yield node
    for child in node["children"]:
        yield from _walk(child)


def chunk_moves(node):
    """node's subtree as the moves actually played, in execution order
    (blockers before the move they unblock)."""
    moves = []
    for child in node["children"]:
        moves.extend(chunk_moves(child))
    if node["played"] is not None:
        moves.append((node["car"], *node["played"]))
    return moves


def state_to_bitboard(state):
    """Inverse of lib.parse_puzzle_line's bitboard decoding."""
    cells = ["o"] * (BOARD_SIZE * BOARD_SIZE)
    for name, positions in state:
        for i, j in positions:
            cells[i * BOARD_SIZE + j] = "A" if name == "red" else name
    return "".join(cells)


def mine(forests, min_size=1, max_size=6):
    """Every OrNode subtree with min_size..max_size moves across `forests`
    (a list of (puzzle_index, forest)). Returns pattern -> {"size", "count",
    "support" (set of puzzle indices), "example" (first occurrence)}."""
    patterns = {}
    for puzzle_idx, forest in forests:
        for tree in forest:
            for top in tree["children"]:
                for node in _walk(top):
                    if not min_size <= node["size"] <= max_size:
                        continue
                    key = pattern(node)
                    entry = patterns.setdefault(key, {"size": node["size"], "count": 0, "support": set()})
                    entry["count"] += 1
                    entry["support"].add(puzzle_idx)
                    entry.setdefault("example", (puzzle_idx, node))
    return patterns


def compress(tree, library):
    """(ornodes, units, depth, unit_depth) for one top-level attempt: its
    OrNode count and longest OrNode chain, before and after replacing every
    maximal subtree whose pattern is in `library` with a single unit. An
    upper bound on what chunking could save - it assumes every library
    pattern present gets recognized."""
    def visit(node):
        if node["size"] >= 2 and pattern(node) in library:
            return 1, 1
        kids = [visit(c) for c in node["children"]]
        return 1 + sum(u for u, _ in kids), 1 + max((d for _, d in kids), default=0)

    kids = [visit(c) for c in tree["children"]]
    ornodes = sum(c["size"] for c in tree["children"])
    depth = max((c["depth"] for c in tree["children"]), default=0)
    units = sum(u for u, _ in kids)
    unit_depth = max((d for _, d in kids), default=0)
    return ornodes, units, depth, unit_depth


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def sample_entries(pool_path, count, seed=None):
    """`count` random (distance, state, line) puzzles from a rush_nw.txt-format pool."""
    rng = random.Random(seed)
    with open(pool_path) as f:
        lines = f.readlines()
    sampled = rng.sample(lines, min(count, len(lines)))
    return [(*parse_puzzle_line(line), line.rstrip("\n")) for line in sampled]


def solve_and_split(pool_path, count, seed, max_expansions, test_frac, cache=None, protect_movers=PROTECT_MOVERS):
    """Samples `count` puzzles from pool_path, solves each with capture_forest
    (annotated), and splits the solved ones into (entries, train, test), each
    split a list of (puzzle_index, forest) with indices into `entries`.
    Prints a one-line summary. If `cache` is given, the result is pickled
    there and reused on later calls with the same arguments. `protect_movers` is
    as in and_or.solve (same default). The cache key holds it, and also names
    the plan semantics, so a cache mined under fixed step counts, or under
    target cells with red alone unprotected, is never reused."""
    key = (pool_path, count, seed, max_expansions, test_frac, protect_movers, "target cells, protect_movers")
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            cached_key, result, summary = pickle.load(f)
        if cached_key == key:
            print(summary + " (cached)\n")
            return result

    entries = sample_entries(pool_path, count, seed)
    rng = random.Random(seed)
    forests, failures, fallbacks = [], 0, 0
    for idx, (_dist, state, _line) in enumerate(tqdm(entries, desc="solving", unit="puzzle")):
        solved = capture_forest(state, rng, max_expansions=max_expansions, protect_movers=protect_movers)
        if solved is None:
            failures += 1
            continue
        forest, n_fallbacks = solved
        fallbacks += n_fallbacks
        for tree in forest:
            annotate(tree)
        forests.append((idx, forest))

    random.Random(seed).shuffle(forests)
    n_test = int(len(forests) * test_frac)
    result = entries, forests[n_test:], forests[:n_test]
    summary = (f"{len(entries)} puzzles sampled from {pool_path}; {failures} did not converge; "
               f"{fallbacks} random fallback moves played\n{len(result[1])} train / {n_test} test")
    print(summary + "\n")
    if cache:
        with open(cache, "wb") as f:
            pickle.dump((key, result, summary), f)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Mine recurring sub-plans (chunks) from lapse-free AND/OR solutions of sampled Rush Hour "
                    "puzzles, described invariantly to rotation, reflection and translation, and measure how well "
                    "a library learned on one split covers and compresses held-out puzzles.")
    parser.add_argument("--pool", default="data/rush_nw.txt", help="rush_nw.txt-format puzzle pool")
    parser.add_argument("--count", type=int, default=2000, help="number of puzzles to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2, help="fraction of solved puzzles held out")
    parser.add_argument("--min-size", type=int, default=2, help="smallest chunk (in moves) kept in the library")
    parser.add_argument("--max-size", type=int, default=6, help="largest chunk (in moves) mined")
    parser.add_argument("--min-support", type=int, default=3,
                        help="library keeps patterns seen in at least this many training puzzles")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="lapse probability for the reasoning-budget comparison (see and_or.or_budget)")
    parser.add_argument("--max-expansions", type=int, default=1_000,
                        help="OrNode expansions allowed per pass before falling back to a random move")
    parser.add_argument("--protect-movers", action=argparse.BooleanOptionalAction, default=PROTECT_MOVERS,
                        help="forbid subgoals from moving a car whose action is being resolved "
                             "(and_or.solve's protect_movers; default: and_or.PROTECT_MOVERS)")
    parser.add_argument("--top", type=int, default=25, help="how many of the most-supported patterns to print")
    parser.add_argument("--output", help="write the library (with one example per pattern) to this JSON file")
    parser.add_argument("--visualize-dir", help="render one example per printed pattern into this folder")
    parser.add_argument("--cache", help="pickle solved puzzles here, reused when the sampling arguments match")
    args = parser.parse_args()

    entries, train, test = solve_and_split(args.pool, args.count, args.seed, args.max_expansions,
                                           args.test_frac, args.cache, args.protect_movers)

    all_train = mine(train, 1, args.max_size)
    by_size = defaultdict(lambda: [0, 0])
    for p in all_train.values():
        by_size[p["size"]][0] += 1
        by_size[p["size"]][1] += p["count"]
    print("train subtrees by size:  size  distinct  occurrences  occurrences/distinct")
    for size in sorted(by_size):
        distinct, occ = by_size[size]
        print(f"                         {size:>4}  {distinct:>8}  {occ:>11}  {occ / distinct:>20.1f}")

    library = {k: v for k, v in all_train.items()
               if v["size"] >= args.min_size and len(v["support"]) >= args.min_support}
    candidates = {k: v for k, v in all_train.items() if v["size"] >= args.min_size}
    lib_occ = sum(v["count"] for v in library.values())
    cand_occ = sum(v["count"] for v in candidates.values())
    print(f"\nlibrary: {len(library)} of {len(candidates)} distinct size>={args.min_size} patterns have support "
          f">={args.min_support}, covering {lib_occ}/{cand_occ} ({lib_occ / max(cand_occ, 1):.0%}) of train occurrences")

    test_patterns = mine(test, args.min_size, args.max_size)
    test_occ = sum(v["count"] for v in test_patterns.values())
    hit_occ = sum(v["count"] for k, v in test_patterns.items() if k in library)
    print(f"held-out coverage: {hit_occ}/{test_occ} ({hit_occ / max(test_occ, 1):.0%}) of test size "
          f"{args.min_size}-{args.max_size} subtrees match a library pattern")

    budget = or_budget(args.gamma)
    stats = [compress(tree, library) for _, forest in test for tree in forest if tree["children"]]
    if stats:
        ornodes, units, depth, unit_depth = zip(*stats)
        fit_before = _mean([o <= budget for o in ornodes])
        fit_after = _mean([u <= budget for u in units])
        print(f"\nheld-out passes with any subgoal ({len(stats)}), before -> after chunking (optimistic):")
        print(f"  mean OrNodes per pass   {_mean(ornodes):5.2f} -> {_mean(units):5.2f}")
        print(f"  mean longest chain      {_mean(depth):5.2f} -> {_mean(unit_depth):5.2f}")
        print(f"  passes within budget {budget} (gamma={args.gamma}):  {fit_before:.0%} -> {fit_after:.0%}")

    ranked = sorted(library.items(), key=lambda kv: (-len(kv[1]["support"]), -kv[1]["count"], kv[0]))
    print(f"\ntop {min(args.top, len(ranked))} library patterns by support:")
    print(f"  {'support':>7}  {'count':>6}  {'size':>4}  pattern")
    for key, v in ranked[:args.top]:
        print(f"  {len(v['support']):>7}  {v['count']:>6}  {v['size']:>4}  {key}")

    if args.output:
        out = []
        for key, v in ranked:
            puzzle_idx, node = v["example"]
            out.append({
                "pattern": key,
                "size": v["size"],
                "count": v["count"],
                "support": len(v["support"]),
                "example": {
                    "puzzle": entries[puzzle_idx][2],
                    "state": state_to_bitboard(node["state"]),
                    "moves": chunk_moves(node),
                },
            })
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {len(out)} patterns to {args.output}")

    if args.visualize_dir:
        for rank, (key, v) in enumerate(ranked[:args.top]):
            _, node = v["example"]
            visualize(os.path.join(args.visualize_dir, f"pattern_{rank:03d}"), node["state"],
                      chunk_moves(node), highlight=node["car"])
        print(f"rendered {min(args.top, len(ranked))} examples to {args.visualize_dir}")


if __name__ == "__main__":
    main()
