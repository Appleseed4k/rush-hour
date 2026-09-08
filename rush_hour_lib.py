import os
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
    """Distance-to-goal for every state reachable from `state`."""
    orientations = car_orientations(state)
    distances = bfs([state], orientations)
    return bfs(find_goal_states(distances), orientations)


def sample_unique(path="data/rush_nw_unique.txt"):
    """Reads a `rush_nw_unique.txt`-format file ("<distance> <bitboard> <count>"
    per line, one line per distance-to-goal) and returns a dict keyed by
    distance, each value in Environment's (name, positions) state-tuple format
    (red first, then every other car sorted by name)."""
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
    """Renders one (name, positions) state tuple to save_path."""
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
    """Saves one PNG per state (initial state plus after each move in `moves`)
    into file_path, plus a trajectory.png plotting distance-to-goal per move."""
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
