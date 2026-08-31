import os
import random
import shutil
from collections import defaultdict, deque

import matplotlib.pyplot as plt

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
    """Yield ((car_name, direction, steps), new_state) pairs reachable from `state`
    in one move, where a move slides a car by any number of clear cells in one
    direction - matching rush_hour_and_or.py's move semantics, so distances from
    bfs() count a multi-cell slide as a single move rather than one per cell."""
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


def find_goal_states(states):
    goal_states = []
    for state in states:
        if state[0][1][1][1] == 5:
            goal_states.append(state)
    return goal_states


def bfs(initial, orientations):
    n_rows, n_cols = 6, 6

    distance = {state: 0 for state in initial}
    queue = deque(initial)

    while queue:
        state = queue.popleft()
        for _, new_state in neighbors(state, orientations, n_rows, n_cols):
            if new_state not in distance:
                distance[new_state] = distance[state] + 1
                queue.append(new_state)

    return distance


def multi_bfs(state):
    orientations = car_orientations(state)
    distances = bfs([state], orientations)
    return bfs(find_goal_states(distances), orientations)


def hint(state, distances):
    """Return the (car, direction, steps) macro-move from `state` that is one
    move closer to the goal, taken from the AND-OR-optimal slide that achieves
    the distance reduction."""
    orientations = car_orientations(state)
    current = distances[state]
    for move, new_state in neighbors(state, orientations):
        if distances[new_state] < current:
            return move
    return None


def sample_puzzle(path="data/rush_nw.txt"):
    """Samples a random puzzle from a `rush_nw.txt`-format file (see conv.ipynb's
    Data Preparation section): each line is "<distance> <bitboard> <count>",
    where `bitboard` is a 36-character, row-major layout of a 6x6 grid ("o" for
    empty, "A" for the target car, and one other letter per remaining car).
    Returns a list of (name, positions) pairs in the format `Environment`
    expects: the target car (renamed "red", matching sample_unique() and
    Environment.get_state()) first, followed by every other car under its
    original letter name, each a list of (row, col) tuples.
    """
    with open(path) as f:
        bitboard = random.choice(f.readlines()).split()[1]

    cars = defaultdict(list)
    for idx, cell in enumerate(bitboard):
        if cell != "o":
            row, col = divmod(idx, 6)
            cars[cell].append((row, col))

    other_names = sorted(name for name in cars if name != "A")
    return [("red", cars["A"])] + [(name, cars[name]) for name in other_names]


def sample_unique(path="data/rush_nw_unique.txt"):
    """Reads every puzzle out of a `rush_nw_unique.txt`-format file - same
    "<distance> <bitboard> <count>" line format as sample_puzzle, but one line per
    distance-to-goal - and returns them as a dict keyed by that distance. Each value
    is in the (name, positions) state-tuple format rush_hour_and_or.py and
    visualize() expect: red first, then every other car sorted by name, each a
    tuple of (row, col) tuples.
    """
    puzzles = {}
    with open(path) as f:
        for line in f:
            dist_str, bitboard, _count = line.split()
            cars = defaultdict(list)
            for idx, cell in enumerate(bitboard):
                if cell != "o":
                    row, col = divmod(idx, 6)
                    cars[cell].append((row, col))

            other_names = sorted(name for name in cars if name != "A")
            state = (("red", tuple(cars["A"])),) + tuple((name, tuple(cars[name])) for name in other_names)
            puzzles[int(dist_str)] = state
    return puzzles


def _draw_state(state, save_path, step_number, n=6):
    """Renders one (name, positions) state tuple to save_path, in the same style as
    Environment.visualize."""
    fig, ax = plt.subplots()
    ax.add_patch(plt.Rectangle((0, 0), n, n, facecolor="white", edgecolor="none"))

    for name, positions in state:
        rows, cols = [i for i, j in positions], [j for i, j in positions]
        i0, i1, j0, j1 = min(rows), max(rows), min(cols), max(cols)
        ax.add_patch(plt.Rectangle(
            (j0, i0), j1 - j0 + 1, i1 - i0 + 1,
            facecolor="red" if name == "red" else "gray",
            edgecolor="black", linewidth=1.5,
        ))
        if name != "red":
            ax.text((j0 + j1 + 1) / 2, (i0 + i1 + 1) / 2, name,
                     ha="center", va="center", color="white", fontsize=12)

    ax.add_patch(plt.Rectangle((0, 0), n, n, facecolor="none", edgecolor="black", linewidth=1.5))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Step {step_number}")

    fig.savefig(save_path)
    plt.close(fig)


def visualize(file_path, state, moves=None):
    """Saves one PNG per state - the initial `state` plus the state after each move
    in `moves` (a (car_name, direction, steps) sequence, e.g. from
    rush_hour_and_or.solve) - into the folder at file_path, one frame per file as
    move_000.png, move_001.png, ..., plus a trajectory.png in that same folder
    plotting bfs-calculated distance-to-goal at every one of those states against
    move number - the trajectory plot from Olieslagers et al., tracking how a solve
    attempt's distance to the goal rises and falls before (if ever) reaching it,
    rather than assuming it falls monotonically."""
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)

    distances = multi_bfs(state)
    trace = [distances[state]]

    _draw_state(state, os.path.join(file_path, "move_000.png"), step_number=0)
    for i, (car_name, direction, steps) in enumerate(moves or [], start=1):
        di, dj = DELTAS[direction]
        car = dict(state)[car_name]
        new_car = tuple((r + di * steps, c + dj * steps) for r, c in car)
        state = tuple((n, new_car if n == car_name else pos) for n, pos in state)
        _draw_state(state, os.path.join(file_path, f"move_{i:03d}.png"), step_number=i)
        trace.append(distances[state])

    fig, ax = plt.subplots()
    ax.plot(range(len(trace)), trace, color="gray")
    ax.set_xlabel("Move #")
    ax.set_ylabel("Distance to goal")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Length {trace[0]}")
    plt.tight_layout()
    fig.savefig(os.path.join(file_path, "trajectory.png"))
    plt.close(fig)
