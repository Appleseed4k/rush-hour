import math
import os
import random
import shutil
from collections import defaultdict, deque

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


def sample_puzzles(output_path, min_distance=1, max_distance=51, num=1, source_path="data/rush_nw.txt"):
    """Samples up to `num` random puzzles for each distance in [min_distance, max_distance]
    from a `rush_nw.txt`-format pool file and writes them to output_path in the same
    "<distance> <bitboard> <count>" line format, ordered by decreasing distance (if a
    distance has fewer than `num` puzzles in the pool, every puzzle at that distance is
    written). Read back with read_puzzles()."""
    pool = defaultdict(list)
    with open(source_path) as f:
        for line in f:
            dist = int(line.split()[0])
            pool[dist].append(line.rstrip("\n"))

    lines = []
    for dist in range(max_distance, min_distance - 1, -1):
        candidates = pool.get(dist, [])
        lines.extend(random.sample(candidates, min(num, len(candidates))))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


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


def read_puzzles(path="data/rush_nw_unique.txt"):
    """Reads a puzzle file written by sample_puzzles() and returns a dict keyed
    by distance-to-goal, each value the list of states sampled at that distance
    (one per matching line, in file order)."""
    puzzles = defaultdict(list)
    with open(path) as f:
        for line in f:
            dist, state = parse_puzzle_line(line)
            puzzles[dist].append(state)
    return dict(puzzles)


def _draw_state(ax, state, n=6, highlight=None, shading=None):
    """Draws one (name, positions) state tuple onto `ax`. `shading`, if given,
    maps car names to weights in [0, 1]: each such car is filled from gray
    (0) to blue (1) and labeled with its weight as a percentage. Car
    `highlight`, if given, gets a thick blue outline."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    gray, blue = to_rgb("gray"), to_rgb("royalblue")
    shading = shading or {}
    ax.add_patch(plt.Rectangle((0, 0), n, n, facecolor="white", edgecolor="none"))

    outline = None
    for name, positions in state:
        rows, cols = [i for i, j in positions], [j for i, j in positions]
        i0, i1, j0, j1 = min(rows), max(rows), min(cols), max(cols)
        weight = shading.get(name, 0)
        fill = "red" if name == "red" else tuple((1 - weight) * g + weight * b for g, b in zip(gray, blue))
        ax.add_patch(plt.Rectangle(
            (j0, i0), j1 - j0 + 1, i1 - i0 + 1,
            facecolor=fill, edgecolor="black", linewidth=1.5,
        ))
        if name != "red":
            label = f"{name}\n{weight:.0%}" if weight else name
            ax.text((j0 + j1 + 1) / 2, (i0 + i1 + 1) / 2, label,
                     ha="center", va="center", color="white", fontsize=12)
        if name == highlight:
            outline = (j0, i0, j1 - j0 + 1, i1 - i0 + 1)
    if outline is not None:
        # Drawn after every car so neighbouring cars' edges can't cover it.
        j0, i0, width, height = outline
        ax.add_patch(plt.Rectangle((j0, i0), width, height, facecolor="none", edgecolor="royalblue", linewidth=5))

    ax.add_patch(plt.Rectangle((0, 0), n, n, facecolor="none", edgecolor="black", linewidth=1.5))
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def visualize(file_path, state, moves=None, boundary=None, highlight=None, shading=None):
    """Saves one PNG per state (initial state plus after each move in `moves`)
    into file_path, plus a trajectory.png plotting distance-to-goal per move.
    `boundary`, if given, is a move index (e.g. from
    and_or.final_pass_start) marked on trajectory.png with a vertical red line.
    `highlight`, if given, is a car name outlined in blue in every move PNG,
    or a `highlight(state) -> car name or None` callable (e.g. wrapping
    and_or.horizon_car) picking one per PNG. `shading`, if given, maps car
    names to weights in [0, 1] (e.g. how often critical_car picked each car
    across attempts), filling each from gray to blue in every move PNG."""
    import matplotlib.pyplot as plt

    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)

    distances = multi_bfs(state)
    trace = [distances[state]]

    def save(state, save_path, step_number):
        fig, ax = plt.subplots()
        _draw_state(ax, state, highlight=highlight(state) if callable(highlight) else highlight, shading=shading)
        ax.set_title(f"Step {step_number}")
        fig.savefig(save_path)
        plt.close(fig)

    save(state, os.path.join(file_path, "move_000.png"), step_number=0)
    for i, (car_name, direction, steps) in enumerate(moves or [], start=1):
        di, dj = DELTAS[direction]
        car = dict(state)[car_name]
        new_car = tuple((r + di * steps, c + dj * steps) for r, c in car)
        state = tuple((n, new_car if n == car_name else pos) for n, pos in state)
        save(state, os.path.join(file_path, f"move_{i:03d}.png"), step_number=i)
        trace.append(distances[state])

    fig, ax = plt.subplots()
    ax.plot(range(len(trace)), trace, color="gray")
    if boundary is not None:
        ax.axvline(boundary, color="red", label="final pass start")
        ax.legend()
    ax.set_xlabel("Move #")
    ax.set_ylabel("Distance to goal")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Length {trace[0]}")
    plt.tight_layout()
    fig.savefig(os.path.join(file_path, "trajectory.png"))
    plt.close(fig)


def critical_car(state, moves, budget):
    """The last car this solve attempt had to reason past before its final
    pass, judged from the moves alone: and_or.horizon_car at the latest state
    up to the final pass's start (and_or.final_pass_start) that has one - the
    car sitting just outside a `budget`-OrNode reasoning horizon (see
    and_or.or_budget) before the rest of the plan fell within reach. None if
    `moves` is None or no such state has a car outside the horizon.

    A property of one attempt, not of the puzzle: different attempts wander
    through different states, and so can hinge on different cars."""
    import and_or

    if moves is None:
        return None
    boundary = and_or.final_pass_start(state, moves)
    if boundary is None:
        return None
    states = [state]
    for car_name, direction, steps in moves[:boundary]:
        states.append(and_or.AndNode(states[-1], car_name, direction, steps).apply(states[-1]))
    return next((car for s in reversed(states) if (car := and_or.horizon_car(s, budget)) is not None), None)


def hint_move(state, distances):
    """An all-knowing observer's one-move hint from `state`: the move that
    starts the reasoning chain a pass must get down (and_or.chain_first_moves)
    - the step beyond the model's horizon that it can't find on its own.
    When several plans start differently, picks the move reaching the lowest
    distance-to-goal in `distances` (e.g. multi_bfs of the puzzle), ties at
    random. None if there's no chain to start."""
    import and_or

    options = [(move, distances[and_or.AndNode(state, *move).apply(state)])
               for move in and_or.chain_first_moves(state)]
    if not options:
        return None
    best = min(distance for _, distance in options)
    return random.choice(sorted(move for move, distance in options if distance == best))


def show_puzzles(states, titles=None, n_cols=5):
    """Displays each state in `states` as a subplot of one inline matplotlib figure,
    arranged in a grid with up to `n_cols` columns per row. `titles`, if given, must
    have one entry per state; otherwise subplots are titled by their index."""
    import matplotlib.pyplot as plt

    n_cols = min(n_cols, len(states))
    n_rows = math.ceil(len(states) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)

    for idx, state in enumerate(states):
        ax = axes[idx // n_cols][idx % n_cols]
        _draw_state(ax, state)
        ax.set_title(titles[idx] if titles is not None else f"Puzzle {idx}")
    for idx in range(len(states), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    return fig


def save_puzzle_images(states, output_dir, titles=None):
    """Saves one puzzle_NNN.png per state into output_dir (created if missing;
    existing contents are cleared first). `titles`, if given, must have one
    entry per state and is used as that image's title; otherwise states are
    titled by their index."""
    import matplotlib.pyplot as plt

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for idx, state in enumerate(states):
        fig, ax = plt.subplots()
        _draw_state(ax, state)
        ax.set_title(titles[idx] if titles is not None else f"Puzzle {idx}")
        fig.savefig(os.path.join(output_dir, f"puzzle_{idx:03d}.png"))
        plt.close(fig)


def visualize_puzzle_file(input_path, output_dir):
    """Renders every puzzle in a rush_nw.txt-format file (e.g. one written by
    fingerprint.write_cluster() or generate_variants.generate_variants()) to a
    puzzle_NNN.png in output_dir, titled by its distance-to-goal."""
    entries = [(dist, state) for dist, states in sorted(read_puzzles(input_path).items()) for state in states]
    states = [state for _, state in entries]
    titles = [f"distance {dist}" for dist, _ in entries]
    save_puzzle_images(states, output_dir, titles)
