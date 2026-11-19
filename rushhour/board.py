"""Boards and moves: states, legal moves, breadth-first search over a puzzle's states, and the puzzle pool
(rush_nw.txt lines -> (distance, state); sampling by distance)."""
import random
from collections import defaultdict, deque

from . import paths

MOVES = {"h": "lr", "v": "ud"}
DELTAS = {"l": (0, -1), "r": (0, 1), "u": (-1, 0), "d": (1, 0)}


def car_orientations(state):
    orientations = {}
    for name, positions in state:
        if positions[0][0] == positions[1][0]:
            orientations[name] = "h"
        else:
            orientations[name] = "v"
    return orientations


def neighbors(state, orientations, n_rows=6, n_cols=6):
    """Yields ((car_name, direction, steps), new_state) for every state reachable
    from `state` in one move, where a move is any legal multi-cell slide."""
    occupied = {pos: name for name, positions in state for pos in positions}
    order = [name for name, _ in state]
    state_dict = dict(state)

    for name, positions in state:
        horizontal = orientations[name] == "h"
        bound = n_cols if horizontal else n_rows
        axis_pos = [j for _, j in positions] if horizontal else [i for i, _ in positions]
        cross = positions[0][0] if horizontal else positions[0][1]
        lo, hi = min(axis_pos), max(axis_pos)

        for direction in MOVES[orientations[name]]:
            di, dj = DELTAS[direction]
            delta = dj if horizontal else di
            steps = 1
            while True:
                edge = hi + steps if delta > 0 else lo - steps
                if not (0 <= edge < bound):
                    break
                cell = (cross, edge) if horizontal else (edge, cross)
                occupant = occupied.get(cell)
                if occupant is not None and occupant != name:
                    break

                new_positions = tuple((i + di * steps, j + dj * steps) for i, j in positions)
                new_state = tuple(
                    (n, new_positions if n == name else state_dict[n])
                    for n in order
                )
                yield (name, direction, steps), new_state
                steps += 1


def multi_bfs(state, n_rows=6, n_cols=6):
    """Distance-to-goal for every state reachable from `state`: the same dict, in the same order, as
    bfs(find_goal_states(bfs([state], ...)), ...) - a forward search listing the states, then one backward from
    the goal states - but several times faster. Each car is reduced to its position along its axis (the lowest
    cell it covers), a board to the tuple of those plus a bitmask of the occupied cells, and moves are generated
    once, in the forward search, in neighbors()' order, then reused backward."""
    names = [name for name, _ in state]
    orientations = car_orientations(state)
    cars = []  # per car: (index, horizontal, length, bound, {anchor: cell mask}, {cell: bit} along the axis)
    positions_at = []  # per car: {anchor: positions tuple}, as neighbors() would build it
    for k, (name, positions) in enumerate(state):
        horizontal = orientations[name] == "h"
        axis = [j for _, j in positions] if horizontal else [i for i, _ in positions]
        lo, length = min(axis), max(axis) - min(axis) + 1
        bound = n_cols if horizontal else n_rows
        di, dj = (0, 1) if horizontal else (1, 0)
        at, masks = {}, {}
        for anchor in range(bound - length + 1):
            shift = anchor - lo
            cells = tuple((i + di * shift, j + dj * shift) for i, j in positions)
            at[anchor] = cells
            masks[anchor] = sum(1 << (i * n_cols + j) for i, j in cells)
        cross = positions[0][0] if horizontal else positions[0][1]
        bits = {e: 1 << ((cross * n_cols + e) if horizontal else (e * n_cols + cross)) for e in range(bound)}
        cars.append((k, horizontal, length, bound, masks, bits))
        positions_at.append(at)

    def anchors_of(st):
        out = []
        for (name, positions), (_, horizontal, _, _, _, _) in zip(st, cars):
            out.append(min(j for _, j in positions) if horizontal else min(i for i, _ in positions))
        return tuple(out)

    start = anchors_of(state)
    occupied = 0
    for (k, _, _, _, masks, _), a in zip(cars, start):
        occupied |= masks[a]
    # forward: every reachable board, with its moves in neighbors()' order (each direction's slides by 1, 2, ...;
    # "l" before "r", "u" before "d" - the lower end first, as MOVES lists them)
    adjacency = {}
    queue = deque([(start, occupied)])
    adjacency[start] = None
    while queue:
        board, occ = queue.popleft()
        nxt = []
        for k, horizontal, length, bound, masks, bits in cars:
            a = board[k]
            free = occ & ~masks[a]
            for delta in (-1, 1):
                steps = 1
                while True:
                    edge = a - steps if delta < 0 else a + length - 1 + steps
                    if not 0 <= edge < bound or free & bits[edge]:
                        break
                    b = a + delta * steps
                    new = board[:k] + (b,) + board[k + 1:]
                    nxt.append(new)
                    if new not in adjacency:
                        adjacency[new] = None
                        queue.append((new, free | masks[b]))
                    steps += 1
        adjacency[board] = nxt

    def full(board):
        return tuple((names[k], positions_at[k][a]) for k, a in enumerate(board))

    # goal states in the forward search's order (find_goal_states: red's second cell in the last column)
    goals = [board for board in adjacency if positions_at[0][board[0]][1][1] == n_cols - 1]
    distance = {board: 0 for board in goals}
    queue = deque(goals)
    while queue:
        board = queue.popleft()
        d = distance[board] + 1
        for new in adjacency[board]:
            if new not in distance:
                distance[new] = d
                queue.append(new)
    return {full(board): d for board, d in distance.items()}


def parse_puzzle_line(line):
    """Parses one "<distance> <bitboard> <count>" line (rush_nw.txt format) into
    (distance, state), state in Environment's (name, positions) state-tuple
    format (red first, then every other car sorted by name)."""
    dist_str, bitboard, _count = line.split()
    cars = defaultdict(list)
    for idx, cell in enumerate(bitboard):
        if cell != "o":
            row, col = divmod(idx, 6)
            cars[cell].append((row, col))

    other_names = sorted(name for name in cars if name != "A")
    state = (("red", tuple(cars["A"])),) + tuple((name, tuple(cars[name])) for name in other_names)
    return int(dist_str), state


def load_puzzle_pool(path=None):
    """Every (distance, state) in a rush_nw.txt-format file (default: the analysis pool, paths.POOL)."""
    with open(path or paths.POOL) as f:
        return [parse_puzzle_line(line) for line in f]


def sample_by_distance(pool, bins, per_bin, seed=0, exclude=()):
    """`per_bin` random (distance, state) puzzles from each (lo, hi) distance
    bin of `pool`, skipping states in `exclude`."""
    rng = random.Random(seed)
    exclude = set(exclude)
    out = []
    for lo, hi in bins:
        candidates = [p for p in pool if lo <= p[0] <= hi and p[1] not in exclude]
        out += rng.sample(candidates, min(per_bin, len(candidates)))
    return out
