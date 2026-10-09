"""How often does each way of ranking a subgoal's candidate moves pick the
move an optimal solver would? A cheap stand-in for end-to-end solve rate.

Probe: oracle-guided, lapse-free and_or searches on held-out boards (see
search_control.oracle_heuristic). At a sample of the OrNodes they visit, every
scorer ranks the candidate moves; it agrees if its top choice is among the
oracle's best (how early the car reaches that destination on an optimal
solution). Only OrNodes where the oracle prefers some candidates over others
count. "covers" is how often the scorer itself prefers some over others
(otherwise its pick is random).

Scorers (all learned from plan trees of oracle-guided passes on training
puzzles - chunk_ordering's "opt" source - unless noted):
- pref: choice rate (chosen / available in training plans) of the move's
  relational description plus the kinds of cars blocking it;
- pref+ctx: the same plus two consequence cues - does the move land in red's
  exit lane, does it land on cells an action higher up the plan still has to
  sweep through;
- ctx-rule: those two cues alone, no learning (avoid landing on needed cells);
- full: whole-template recognition (ChunkLibrary.rank); full-self: the same
  with templates from the solver's own random-order plans;
- partial: templates also cut down to their top 1, 2 and 3 levels (deeper
  levels become holes), deepest match preferred;
- lookahead: no learning - prefer unblocked moves, then moves whose every
  blocker has an unblocked way out."""
import argparse
import collections
import random
from multiprocessing import Pool

import and_or as ao
import chunk_library as cl
import chunk_ordering as co
import chunks
from lib import multi_bfs
from search_control import load_puzzles, oracle_heuristic

SCORERS = ["pref", "pref+ctx", "ctx-rule", "full", "full-self", "partial", "lookahead"]


# --- move descriptions -------------------------------------------------------

def needed_cells(ancestors):
    return set().union(*ancestors) if ancestors else set()


def describe(state, car, collisions, parent_direction, direction, steps, needed):
    """(relational description incl. blocker kinds, consequence cues)."""
    rel = chunks.relation(parent_direction, {"state": state, "car": car, "collisions": collisions,
                                             "action": (direction, steps)})
    cars = dict(state)
    kinds = []
    for b in ao.AndNode(state, car, direction, steps).blockers():
        horizontal = cars[b][0][0] == cars[b][1][0]
        kinds.append(("p" if horizontal == (direction in "lr") else "x") + str(len(cars[b])))
    moved = set(dict(ao.AndNode(state, car, direction, steps).apply(state))[car])
    red = cars["red"]
    exit_lane = {(red[0][0], j) for j in range(max(c for _, c in red) + 1, 6)}
    cues = f"R{int(bool(moved & exit_lane))}A{int(bool(moved & needed))}"
    return rel + "|" + ",".join(sorted(kinds)), cues


def count_choices(forests):
    """chosen / available counts per description, with and without cues."""
    avail, chosen = collections.Counter(), collections.Counter()

    def visit(node, parent_direction, ancestors):
        state, car, collisions = node["state"], node["car"], node["collisions"]
        for direction, steps in ao.OrNode(state, car, collisions).directions():
            # ancestors[-1] is the parent's own lane, which any candidate clears by construction
            base, cues = describe(state, car, collisions, parent_direction, direction, steps,
                                  needed_cells(ancestors[:-1]))
            for key in (base, base + "|" + cues):
                avail[key] += 1
                chosen[key] += (direction, steps) == tuple(node["action"])
        swept = set(ao.AndNode(state, car, *node["action"]).swept_cells())
        for child in node["children"]:
            visit(child, node["action"][0], ancestors + [swept])

    for forest in forests:
        for tree in forest:
            root = set(ao.AndNode(tree["state"], "red", *tree["action"]).swept_cells())
            for child in tree["children"]:
                visit(child, tree["action"][0], [root])
    return avail, chosen


def truncate(t, depth):
    if t is cl.HOLE:
        return t
    rel, kids = t
    if depth <= 1:
        return rel, tuple(cl.HOLE for _ in kids)
    return rel, tuple(sorted((truncate(k, depth - 1) for k in kids), key=cl.render))


# --- probing -------------------------------------------------------------------

MODELS = {}
ANCESTORS = []
ORACLE = [None]
ROWS = []
SAMPLE = [0.2]


def _probe_and_solve(self):
    ANCESTORS.append(set(self.swept_cells()))
    try:
        return co._and_solve(self)
    finally:
        ANCESTORS.pop()


def _scores(name, node, candidates, visited):
    """candidate -> score (higher first) for one scorer."""
    m = MODELS
    if name in ("pref", "pref+ctx", "ctx-rule"):
        needed = needed_cells(ANCESTORS[:-1])  # the parent's own lane is cleared by construction
        out = {}
        for c in candidates:
            base, cues = describe(node.state, node.car_name, node.collisions, node.parent_direction, *c, needed)
            if name == "ctx-rule":
                out[c] = -cues.count("1")
                continue
            key = base + "|" + cues if name == "pref+ctx" and m["avail"][base + "|" + cues] >= 5 else base
            out[c] = (m["chosen"][key] + 1) / (m["avail"][key] + 2)
        return out
    if name == "lookahead":
        out = {}
        for c in candidates:
            blockers = ao.AndNode(node.state, node.car_name, *c).blockers()
            if not blockers:
                out[c] = 2
            elif not (node.protected & blockers.keys()) and all(
                    any(not ao.AndNode(node.state, b, *bc).blockers()
                        for bc in ao.OrNode(node.state, b, cells).directions())
                    for b, cells in blockers.items()):
                out[c] = 1
            else:
                out[c] = 0
        return out
    library = {"full": m["full"], "full-self": m["full_self"], "partial": m["partial"]}[name]
    ranked = library.rank(node, list(candidates), visited)
    recognized = set()
    for c in candidates:
        rel = cl._rel(node.state, node.car_name, node.collisions, node.parent_direction, *c)
        if any(next(cl._and(node.state, node.car_name, *c, t[1], visited, node.protected, None, []), None)
               for t in library.by_root.get(rel, ())):
            recognized.add(c)
    # The library's own order among recognized moves, then everything else tied.
    return {c: (len(ranked) - ranked.index(c) if c in recognized else 0) for c in candidates}


def _probe_or_solve(self):
    key = (self.car_name, self.collisions)
    if self.parent_direction is not None and key not in self.visited and random.random() < SAMPLE[0]:
        candidates = self.directions()
        if len(candidates) > 1:
            oracle = ORACLE[0](self.state, [(self.car_name, *c) for c in candidates])
            depth = {c: int(round(oracle[(self.car_name, *c)] - 0.5)) for c in candidates}
            best = {c for c, v in depth.items() if v == max(depth.values())}
            if len(best) < len(candidates):
                visited = self.visited | {key}
                row = {"n": len(candidates), "best": len(best)}
                for name in SCORERS:
                    scores = _scores(name, self, candidates, visited)
                    top = max(scores.values())
                    tops = [c for c, v in scores.items() if v == top]
                    row[name] = (random.choice(tops) in best, len(tops) < len(candidates))
                ROWS.append(row)
    return co._or_solve(self)


def _init_probe(models, sample):
    MODELS.update(models)
    SAMPLE[0] = sample
    ao.OrNode.solve = _probe_or_solve
    ao.AndNode.solve = _probe_and_solve


def probe(state):
    ROWS.clear()
    ORACLE[0] = oracle_heuristic(multi_bfs(state))
    random.seed(0)
    moves = 0
    while ao._red_steps(state) > 0 and moves < 60:
        ANCESTORS.clear()
        result = ao._solve_pass(state, ORACLE[0], 0.0)
        step = result if type(result) is list else result[1]
        if not step:
            break
        for move in step:
            state = ao.AndNode(state, *move).apply(state)
        moves += len(step)
    return list(ROWS)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", default="data/rush_nw.txt")
    ap.add_argument("--train-per-bin", type=int, default=60)
    ap.add_argument("--test-per-bin", type=int, default=8)
    ap.add_argument("--sample", type=float, default=0.2, help="fraction of OrNodes probed")
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bins = [(5, 8), (9, 12), (13, 16), (17, 20), (21, 30)]
    test = load_puzzles(args.pool, bins, 40, args.seed)
    exclude = {s for b in bins for s in test[b]}
    train = co.sample_train(args.pool, bins, args.train_per_bin, exclude, args.seed)
    with Pool() as pool:
        forests = {}
        for source in ("opt", "self"):
            jobs = [(i, s, source, hash((args.seed, source, i)) & 0xFFFFFFFF) for i, s in enumerate(train)]
            forests[source] = [f for _, f in sorted(pool.map(co.mine_one, jobs, chunksize=2))]
    indexed = {k: list(enumerate(v)) for k, v in forests.items()}
    full = cl.flat_templates(indexed["opt"], 1, 6, args.min_support)
    partial = full + [(truncate(t, d), w) for t, w in full for d in (1, 2, 3)]
    avail, chosen = count_choices(forests["opt"])
    models = {"avail": avail, "chosen": chosen,
              "full": cl.ChunkLibrary(full, min_moves=1),
              "full_self": cl.ChunkLibrary(cl.flat_templates(indexed["self"], 1, 6, args.min_support), min_moves=1),
              "partial": cl.ChunkLibrary(partial, min_moves=1)}
    print(f"{len(train)} training puzzles; templates: full {models['full'].size}, "
          f"full-self {models['full_self'].size}, partial {models['partial'].size}", flush=True)

    held = [s for b in bins[1:] for s in test[b][:args.test_per_bin]]
    with Pool(initializer=_init_probe, initargs=(models, args.sample)) as pool:
        rows = [r for out in pool.map(probe, held) for r in out]
    n = len(rows)
    print(f"{n} probed OrNodes where the oracle prefers some candidates "
          f"(mean {sum(r['n'] for r in rows) / n:.1f} candidates, {sum(r['best'] for r in rows) / n:.1f} best)")
    print(f"  {'scorer':<10} {'agrees':>7} {'covers':>7} {'agrees when covering':>21}")
    print(f"  {'random':<10} {sum(r['best'] / r['n'] for r in rows) / n:>7.0%}")
    for name in SCORERS:
        covered = [r for r in rows if r[name][1]]
        print(f"  {name:<10} {sum(r[name][0] for r in rows) / n:>7.0%} {len(covered) / n:>7.0%} "
              f"{sum(r[name][0] for r in covered) / max(len(covered), 1):>21.0%}")


if __name__ == "__main__":
    main()
