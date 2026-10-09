import argparse
import json
import os
from collections import Counter, defaultdict

from tqdm import tqdm

import chunks
from and_or import or_budget
from lib import visualize


# Nested chunking, byte-pair-encoding style. Every OrNode subtree starts as a
# tree of units, one per move, labelled by chunks.relation(). Each round finds
# the fragment that would save the most units across the training trees,
# names it as a new chunk, and rewrites every occurrence into a single unit.
#
# A fragment is a unit plus some of its descendants; any child left out is a
# hole - a blocker the chunk leaves to be cleared by separate reasoning (its
# own units, or other chunks), whatever that takes. Holes are what let a
# chunk recur even when the plan beneath it varies, and later rounds can
# merge chunk units with each other, so big plans emerge as chunks of chunks.
#
# Rendering: "x1:2(p2f1,?)" is a fragment whose root crosses its parent's lane
# with one blocker resolved inside the chunk and one left as a hole. Children
# of a move are unordered (sorted); children of a chunk unit are its holes,
# ordered by which hole they fill. Unlike chunks.pattern(), there are no "#k"
# co-reference tags - which car fills which role is left to the board.

HOLE = "?"


def to_units(node):
    """Unit tree for one annotated OrNode subtree (see chunks.annotate)."""
    return {
        "label": node["rel"],
        "children": [to_units(c) for c in node["children"]],
        "slots": None,
        "nodes": [node],
    }


def ordered_children(unit):
    """(slot, child) pairs in canonical order: a move's children sorted by
    shape, a chunk's children by which of its holes they fill."""
    if unit["slots"] is None:
        return sorted((("", c) for c in unit["children"]), key=lambda sc: shape(sc[1]))
    return sorted(zip(unit["slots"], unit["children"]), key=lambda sc: (sc[0], shape(sc[1])))


def shape(unit):
    """Full canonical rendering of unit's subtree (memoized on the unit)."""
    if "shape" not in unit:
        kids = [shape(c) for _, c in ordered_children(unit)]
        unit["shape"] = unit["label"] + (f"({','.join(kids)})" if kids else "")
    return unit["shape"]


def fragments(unit, max_units):
    """Every fragment rooted at `unit` of at most max_units units: unit itself
    plus, for each child, either a hole or a fragment rooted at that child."""
    if max_units < 1:
        return []
    partial = [([], 1)]
    for slot, child in ordered_children(unit):
        extended = []
        for kids, n in partial:
            extended.append((kids + [(slot, child, None)], n))
            for sub in fragments(child, max_units - n):
                extended.append((kids + [(slot, child, sub)], n + sub["units"]))
        partial = extended
    return [{"unit": unit, "kids": kids, "units": n} for kids, n in partial]


def render(fragment, mark=None):
    """Canonical string for a fragment. With `mark` (one of its hole units),
    that hole renders as "!" instead of "?" - giving each hole a signature
    that identifies its position, shared only by interchangeable holes."""
    parts = []
    for slot, child, sub in fragment["kids"]:
        if sub is None:
            parts.append((slot, "!" if child is mark else HOLE))
        else:
            parts.append((slot, render(sub, mark)))
    parts.sort()
    label = fragment["unit"]["label"]
    return label + (f"({','.join(s for _, s in parts)})" if parts else "")


def holes(fragment):
    for _slot, child, sub in fragment["kids"]:
        if sub is None:
            yield child
        else:
            yield from holes(sub)


def covered_nodes(fragment):
    """The original tree nodes (moves) a fragment covers, excluding its holes."""
    nodes = list(fragment["unit"]["nodes"])
    for _slot, _child, sub in fragment["kids"]:
        if sub is not None:
            nodes.extend(covered_nodes(sub))
    return nodes


def _walk(unit):
    yield unit
    for child in unit["children"]:
        yield from _walk(child)


def rewrite(unit, chunk):
    """Copy of unit's tree with every occurrence of `chunk` collapsed into one
    unit, matched top-down (an occurrence is taken before any overlapping
    one beneath it)."""
    for fragment in fragments(unit, chunk["units"]):
        if fragment["units"] == chunk["units"] and render(fragment) == chunk["pattern"]:
            open_holes = list(holes(fragment))
            return {
                "label": chunk["name"],
                "children": [rewrite(h, chunk) for h in open_holes],
                "slots": [render(fragment, mark=h) for h in open_holes],
                "nodes": covered_nodes(fragment),
            }
    return {**{k: v for k, v in unit.items() if k != "shape"},
            "children": [rewrite(c, chunk) for c in unit["children"]]}


def count_fragments(passes, max_units):
    """pattern -> (units, occurrences, set of puzzle indices) over every
    fragment of 2..max_units units in `passes` ((puzzle_index, roots) pairs)."""
    counts = Counter()
    support = defaultdict(set)
    units_of = {}
    for puzzle_idx, roots in passes:
        for root in roots:
            for unit in _walk(root):
                # Identical siblings yield the same fragment more than once; count it once per unit.
                keys = {(render(f), f["units"]) for f in fragments(unit, max_units) if f["units"] >= 2}
                for key, units in keys:
                    units_of[key] = units
                    counts[key] += 1
                    support[key].add(puzzle_idx)
    return {k: (units_of[k], counts[k], support[k]) for k in counts}


def learn(train, max_units=2, min_support=3, min_savings=3, max_chunks=500):
    """Greedily learns chunks from `train` passes, best savings first, rewriting
    the passes after each. Savings = (units - 1) * occurrences, counted only
    for patterns seen in at least min_support puzzles. Stops when no pattern
    saves at least min_savings or after max_chunks. Returns (chunks, train
    rewritten under all of them)."""
    learned = []
    levels = {}
    with tqdm(total=max_chunks, desc="learning chunks", unit="chunk") as bar:
        while len(learned) < max_chunks:
            stats = count_fragments(train, max_units)
            scored = [((units - 1) * count, key, units, count, len(sup))
                      for key, (units, count, sup) in stats.items() if len(sup) >= min_support]
            if not scored:
                break
            savings, key, units, count, support = max(scored, key=lambda s: (s[0], s[1]))
            if savings < min_savings:
                break
            name = f"C{len(learned)}"
            used = [tok for tok in levels if tok in _labels(key)]
            levels[name] = 1 + max((levels[tok] for tok in used), default=0)
            chunk = {"name": name, "pattern": key, "units": units, "count": count, "support": support,
                     "savings": savings, "level": levels[name]}
            train = [(idx, [rewrite(r, chunk) for r in roots]) for idx, roots in train]
            example = _find_example(train, name)
            chunk["moves"] = len(example["nodes"]) if example else None
            chunk["example"] = example
            learned.append(chunk)
            bar.update(1)
    return learned, train


def _labels(pattern):
    """Unit labels appearing in a rendered pattern."""
    return set(pattern.replace("(", ",").replace(")", ",").split(",")) - {"", HOLE}


def _find_example(passes, name):
    for _idx, roots in passes:
        for root in roots:
            for unit in _walk(root):
                if unit["label"] == name:
                    return unit
    return None


def pass_stats(roots):
    """(units, depth) for one pass: units in all its OrNode trees, and the
    longest unit chain."""
    def depth(unit):
        return 1 + max((depth(c) for c in unit["children"]), default=0)
    return sum(1 for r in roots for _ in _walk(r)), max((depth(r) for r in roots), default=0)


def summarize(passes, budget):
    """(mean units per pass, mean longest chain, fraction within budget) over
    every pass in `passes` ((puzzle_index, roots) pairs) with at least one
    subgoal."""
    stats = [pass_stats(roots) for _, roots in passes if roots]
    if not stats:
        return 0.0, 0.0, 0.0
    units, depths = zip(*stats)
    return chunks._mean(units), chunks._mean(depths), chunks._mean([u <= budget for u in units])


def as_passes(split):
    """(puzzle_index, unit roots) for every top-level red attempt in a
    chunks.solve_and_split() split."""
    return [(idx, [to_units(c) for c in tree["children"]]) for idx, forest in split for tree in forest]


def _execution_order(node, covered):
    """Moves of `node`'s original subtree in execution order, restricted to `covered`."""
    moves = []
    for child in node["children"]:
        moves.extend(_execution_order(child, covered))
    if id(node) in covered and node["played"] is not None:
        moves.append((node["car"], *node["played"]))
    return moves


def main():
    parser = argparse.ArgumentParser(
        description="Learn nested chunks (chunks of chunks, with holes for separately-resolved blockers) from "
                    "lapse-free AND/OR solutions of sampled Rush Hour puzzles, byte-pair-encoding style, and "
                    "measure how much a library learned on one split compresses held-out puzzles.")
    parser.add_argument("--pool", default="data/rush_nw.txt", help="rush_nw.txt-format puzzle pool")
    parser.add_argument("--count", type=int, default=2000, help="number of puzzles to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-frac", type=float, default=0.2, help="fraction of solved puzzles held out")
    parser.add_argument("--max-expansions", type=int, default=1_000,
                        help="OrNode expansions allowed per pass before falling back to a random move")
    parser.add_argument("--cache", help="pickle solved puzzles here, reused when the sampling arguments match")
    parser.add_argument("--max-units", type=int, default=2,
                        help="most units merged into one new chunk per round (2 = pairwise, like BPE)")
    parser.add_argument("--min-support", type=int, default=3,
                        help="a chunk must occur in at least this many training puzzles")
    parser.add_argument("--min-savings", type=int, default=3, help="stop once no chunk saves this many units")
    parser.add_argument("--max-chunks", type=int, default=500, help="stop after learning this many chunks")
    parser.add_argument("--gamma", type=float, default=0.2,
                        help="lapse probability for the reasoning-budget comparison (see and_or.or_budget)")
    parser.add_argument("--flat-max-size", type=int, default=6,
                        help="largest pattern in the flat (chunks.py) library this is compared against")
    parser.add_argument("--top", type=int, default=30, help="how many chunks to print")
    parser.add_argument("--output", help="write the learned chunks (with one example each) to this JSON file")
    parser.add_argument("--visualize-dir", help="render one example per printed chunk into this folder")
    args = parser.parse_args()

    entries, train_split, test_split = chunks.solve_and_split(args.pool, args.count, args.seed,
                                                              args.max_expansions, args.test_frac, args.cache)
    budget = or_budget(args.gamma)

    flat_library = {k for k, v in chunks.mine(train_split, 2, args.flat_max_size).items()
                    if len(v["support"]) >= args.min_support}
    flat = [chunks.compress(tree, flat_library) for _, forest in test_split for tree in forest if tree["children"]]

    learned, _ = learn(as_passes(train_split), args.max_units, args.min_support, args.min_savings,
                       args.max_chunks)

    checkpoints = sorted({n for n in (0, 10, 25, 50, 100, 200, 300, 400, 500) if n <= len(learned)} | {len(learned)})
    test = as_passes(test_split)
    print(f"held-out passes, by library size (budget {budget} at gamma={args.gamma}):")
    print(f"  {'chunks':>6}  {'units/pass':>10}  {'longest chain':>13}  {'within budget':>13}")
    for i, chunk in enumerate([None] + learned):
        if chunk is not None:
            test = [(idx, [rewrite(r, chunk) for r in roots]) for idx, roots in test]
        if i in checkpoints:
            units, depth, fit = summarize(test, budget)
            print(f"  {i:>6}  {units:>10.2f}  {depth:>13.2f}  {fit:>13.0%}")
    if flat:
        ornodes, units, _depth, unit_depth = zip(*flat)
        print(f"  {'flat':>6}  {chunks._mean(units):>10.2f}  {chunks._mean(unit_depth):>13.2f}  "
              f"{chunks._mean([u <= budget for u in units]):>13.0%}   "
              f"({len(flat_library)}-pattern chunks.py library, for comparison)")

    by_level = Counter(c["level"] for c in learned)
    print(f"\n{len(learned)} chunks learned; by nesting level: "
          + ", ".join(f"level {lvl}: {n}" for lvl, n in sorted(by_level.items())))

    print(f"\nfirst {min(args.top, len(learned))} chunks learned:")
    print(f"  {'name':>5}  {'level':>5}  {'moves':>5}  {'savings':>7}  {'support':>7}  pattern")
    for chunk in learned[:args.top]:
        print(f"  {chunk['name']:>5}  {chunk['level']:>5}  {chunk['moves'] or '-':>5}  {chunk['savings']:>7}  "
              f"{chunk['support']:>7}  {chunk['pattern']}")

    deepest = sorted(learned, key=lambda c: (-c["level"], -c["savings"]))[:10]
    print("\ndeepest-nested chunks:")
    for chunk in deepest:
        print(f"  {chunk['name']:>5}  {chunk['level']:>5}  {chunk['moves'] or '-':>5}  {chunk['savings']:>7}  "
              f"{chunk['support']:>7}  {chunk['pattern']}")

    def example_of(chunk):
        unit = chunk["example"]
        root = unit["nodes"][0]
        return root, {id(n) for n in unit["nodes"]}

    if args.output:
        out = []
        for chunk in learned:
            entry = {k: chunk[k] for k in ("name", "pattern", "level", "units", "moves", "count", "support",
                                           "savings")}
            if chunk["example"] is not None:
                root, covered = example_of(chunk)
                entry["example"] = {
                    "state": chunks.state_to_bitboard(root["state"]),
                    "moves": chunks.chunk_moves(root),
                    "chunk_moves": _execution_order(root, covered),
                }
            out.append(entry)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {len(out)} chunks to {args.output}")

    if args.visualize_dir:
        for chunk in learned[:args.top]:
            if chunk["example"] is None:
                continue
            root, _ = example_of(chunk)
            visualize(os.path.join(args.visualize_dir, chunk["name"]), root["state"], chunks.chunk_moves(root),
                      highlight=root["car"])
        print(f"rendered examples to {args.visualize_dir}")


if __name__ == "__main__":
    main()
