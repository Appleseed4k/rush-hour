import argparse
import string

from rush_hour_and_or import AndNode
from rush_hour_lib import multi_bfs, parse_puzzle_line
from rush_hour_fingerprint import forest_distance, solve_forest

BOARD_SIZE = 6


def _replay(state, tree, swept_accum):
    """Executes one AND-node's move (after first executing its children, same
    order resolve_and used at fingerprinting time), recording every cell the
    moving car's slide passes through - not just where it starts or ends,
    since a filler car placed on any swept-but-not-final cell would still
    obstruct the slide even though it was never the move's destination."""
    for child in tree["children"]:
        state = _replay(state, child, swept_accum)
    node = AndNode(state, tree["car"], tree["action"][0], tree["action"][1])
    swept_accum.update(node.swept_cells())
    return node.apply(state)


def skeleton_cells(state, forest):
    """Cells that matter to `forest`'s AND/OR tree: every cell any skeleton
    car (one that's ever moved or ever blocks a move) occupies at any point -
    its start, and everything its slide(s) sweep through/into. Everything
    else is genuinely free to repack: the solver only ever looks at
    occupancy along a slide's swept cells (AndNode.blockers()/swept_cells()
    in rush_hour_and_or.py), so a cell no move ever touches cannot affect the
    tree no matter what's placed there. Returns (skeleton_names, cells)."""
    names = {"red"}

    def collect_names(tree):
        names.add(tree["car"])
        for c in tree["children"]:
            collect_names(c)

    for attempt in forest:
        collect_names(attempt)

    cells = set()
    state_dict = dict(state)
    for name in names:
        cells.update(state_dict[name])

    working_state = state
    for attempt in forest:
        working_state = _replay(working_state, attempt, cells)

    return names, cells


def enumerate_tilings(free_cells):
    """Every way to place disjoint straight 2- or 3-cell filler cars on a
    subset of free_cells, leaving the rest empty - exhaustive backtracking in
    a fixed cell order so each tiling is produced exactly once. Free-cell
    counts here are small (a handful to a couple dozen, since these are the
    leftover cells around an already-dense solve skeleton), so exhaustive
    search is cheap and gives the true yield instead of a random sample's
    lower-bound estimate."""
    cells = sorted(free_cells)
    free_set = set(free_cells)
    n = len(cells)
    results = []

    def backtrack(idx, remaining, placed):
        if idx == n:
            results.append(list(placed))
            return
        cell = cells[idx]
        if cell not in remaining:
            backtrack(idx + 1, remaining, placed)
            return
        backtrack(idx + 1, remaining, placed)  # leave this cell empty
        r, c = cell
        for length in (2, 3):
            for dr, dc in ((0, 1), (1, 0)):
                span = [(r + dr * k, c + dc * k) for k in range(length)]
                if all(0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE and (rr, cc) in remaining for rr, cc in span):
                    placed.append(span)
                    backtrack(idx + 1, remaining - set(span), placed)
                    placed.pop()

    backtrack(0, free_set, [])
    return results


def mirror_state(state):
    """Row-mirror (row -> BOARD_SIZE-1-row): preserves every move's
    (horizontal, steps, edge) fingerprint descriptor exactly, since direction
    ('u' vs 'd') isn't part of it - a free structural variant on top of
    whatever filler tiling produced `state`."""
    return tuple(
        (name, tuple((BOARD_SIZE - 1 - r, c) for r, c in positions))
        for name, positions in state
    )


def apply_tiling(state, skeleton_names, cars):
    """Rebuilds a full state: the skeleton cars unchanged (in their original
    positions), plus one new lettered car per `cars` entry (a list of cell
    tuples from enumerate_tilings)."""
    state_dict = dict(state)
    used_letters = (set(skeleton_names) - {"red"}) | {"A"}  # "A" is red's reserved bitboard letter
    letters = [ch for ch in string.ascii_uppercase if ch not in used_letters]
    new_state = [("red", state_dict["red"])]
    for name in sorted(skeleton_names - {"red"}):
        new_state.append((name, state_dict[name]))
    for letter, cells in zip(letters, cars):
        new_state.append((letter, tuple(cells)))
    return tuple(new_state)


def state_to_bitboard(state):
    grid = ["o"] * (BOARD_SIZE * BOARD_SIZE)
    for name, positions in state:
        letter = "A" if name == "red" else name
        for r, c in positions:
            grid[r * BOARD_SIZE + c] = letter
    return "".join(grid)


def generate_variants(line, mirror=True):
    """Expands one rush_nw.txt-format exemplar line into every filler-tiling
    (and, if `mirror`, row-mirrored) variant that shares its exact AND/OR
    solve shape - verified via forest_distance()==0 against real
    solve_forest() output, not just assumed from the construction. Returns
    (exemplar_dist, list of new "<dist> <bitboard> 0" lines, sorted by
    piece-count then bitboard, excluding the exemplar's own bitboard)."""
    dist, state = parse_puzzle_line(line)
    forest = solve_forest(state)
    names, skel_cells = skeleton_cells(state, forest)
    free_cells = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if (r, c) not in skel_cells]

    tilings = enumerate_tilings(free_cells)
    candidates = [apply_tiling(state, names, cars) for cars in tilings]
    if mirror:
        candidates += [mirror_state(s) for s in candidates]

    seen = {state_to_bitboard(state)}
    new_lines = []
    for cand in candidates:
        bb = state_to_bitboard(cand)
        if bb in seen:
            continue
        cand_forest = solve_forest(cand)
        if cand_forest is None or forest_distance(forest, cand_forest) != 0.0:
            continue
        seen.add(bb)
        real_dist = multi_bfs(cand)[cand]
        new_lines.append(f"{real_dist} {bb} 0")

    return dist, new_lines


def main():
    parser = argparse.ArgumentParser(
        description="Expands each puzzle in a rush_nw.txt-format exemplar file (e.g. one cluster written by "
                     "rush_hour_fingerprint.py) into every same-shape variant reachable by repacking the cells its "
                     "AND/OR solve never touches, plus a row-mirror of each. Every variant is verified to have "
                     "forest_distance()==0 to its own exemplar via real solve_forest() execution, so puzzles "
                     "generated from the same exemplar are exact solution-shape matches, while puzzles from "
                     "different exemplars keep whatever variation the input file already had between them - useful "
                     "for growing a training pool without collapsing it onto a single memorizable shape.")
    parser.add_argument("input", help="rush_nw.txt-format file of exemplar puzzles to expand")
    parser.add_argument("output", help="where to write the exemplars plus their generated variants")
    parser.add_argument("--no-mirror", action="store_true", help="skip the row-mirrored variants")
    args = parser.parse_args()

    with open(args.input) as f:
        lines = [l.strip() for l in f if l.strip()]

    out_lines = []
    total_new = 0
    for line in lines:
        dist, new_lines = generate_variants(line, mirror=not args.no_mirror)
        out_lines.append(line)
        out_lines.extend(new_lines)
        total_new += len(new_lines)
        print(f"dist={dist:2d}  {len(new_lines):4d} variants  {line.split()[1]}")

    with open(args.output, "w") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"\n{len(lines)} exemplars -> {len(lines) + total_new} puzzles written to {args.output}")


if __name__ == "__main__":
    main()
