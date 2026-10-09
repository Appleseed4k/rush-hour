import io
import random
from collections import deque
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from and_or import AndNode, OrNode, _red_steps, solve as and_or_solve
from lib import MOVES, DELTAS, car_orientations, multi_bfs, neighbors, parse_puzzle_line

OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}
DIRECTIONS = "lrud"
# Target for a destination first reached k moves into an optimal solution:
# DEST_DECAY ** (k - 1), so the destinations of optimal next moves score 1.
DEST_DECAY = 0.8


class Car():
    def __init__(self, positions, name):
        self.name = name
        self.positions = positions
        self.orientation = "h" if positions[0][0] == positions[1][0] else "v"


class Environment():
    """Puzzle grid containing car objects that can be moved until the red car
    reaches its goal cell.
    """
    def __init__(self, size, cars):
        """`cars` is a sequence of (name, positions) pairs, red first - the same
        shape get_state() returns, so a captured state can be fed straight back
        in to rebuild an equivalent grid."""
        self.puzzle = [[0 for j in range(size[1])] for i in range(size[0])]
        self.goal = (2, 5)
        self.cars = {}
        for name, positions in cars:
            self.cars[name] = self.add_car(positions, name)

    def add_car(self, positions, name):
        car = Car(positions, name)
        for i, j in positions:
            self.puzzle[i][j] = car
        return car

    def move(self, car_name, direction, steps=1):
        """Slides car_name `steps` cells in `direction`. Every swept cell must be
        in bounds and clear of other cars, not just the final resting cells."""
        car = self.cars[car_name]
        if direction not in MOVES[car.orientation]:
            return False
        old_pos = car.positions
        di, dj = DELTAS[direction]
        n_rows, n_cols = len(self.puzzle), len(self.puzzle[0])
        for s in range(1, steps + 1):
            for i, j in old_pos:
                new_i, new_j = i + di * s, j + dj * s
                if not (0 <= new_i < n_rows and 0 <= new_j < n_cols):
                    return False
                if self.puzzle[new_i][new_j] not in [0, car]:
                    return False
        new_pos = [(i + di * steps, j + dj * steps) for i, j in old_pos]
        for i, j in old_pos:
            self.puzzle[i][j] = 0
        for new_i, new_j in new_pos:
            self.puzzle[new_i][new_j] = car
        car.positions = new_pos
        return self.get_state()

    def get_state(self):
        return tuple((name, tuple(car.positions)) for name, car in self.cars.items())

    def car_at(self, cell):
        """Name of the car occupying `cell`, or None if it's empty."""
        i, j = cell
        occupant = self.puzzle[i][j]
        return occupant.name if occupant != 0 else None

    def check_win(self):
        target = self.puzzle[self.goal[0]][self.goal[1]]
        if target != 0:
            if target.name == "red":
                return True
        return False


def legal_actions(environment, action_space):
    """Subset of action_space currently executable in `environment`, without
    leaving it mutated."""
    legal = []
    for cell, direction, steps in action_space:
        car_name = environment.car_at(cell)
        if car_name is None:
            continue
        if environment.move(car_name, direction, steps):
            environment.move(car_name, OPPOSITE[direction], steps)
            legal.append((cell, direction, steps))
    return legal


def car_anchor(state, car_name):
    """(row, col) of car_name's topmost/leftmost occupied cell in `state` -
    used as a puzzle-independent stand-in for car identity, since a car's
    letter carries no information a policy trained across puzzles could use."""
    positions = dict(state)[car_name]
    return min(i for i, j in positions), min(j for i, j in positions)


class StateEncoder():
    """Encodes a state as a fixed-size (channels, n_rows, n_cols) grid of
    binary channels: each cell's occupant (empty / red / horizontal car /
    vertical car), then which cells start and end a car and which cars are
    trucks. Without the start/end channels, two cars parked end to end in one
    lane look the same as a single longer car. Both the encoding and its
    shape depend only on board size, never on a puzzle's specific cars, so one
    agent trains across many puzzles. (PolicyNetwork adds row/column
    coordinates itself.)"""
    CHANNELS = ("empty", "red", "h", "v", "start", "end", "truck")

    def __init__(self, size=(6, 6)):
        self.n_rows, self.n_cols = size

    @property
    def shape(self):
        return len(self.CHANNELS), self.n_rows, self.n_cols

    def encode_numpy(self, state):
        """The encoding as a (channels, n_rows, n_cols) uint8 array."""
        x = np.zeros(self.shape, dtype=np.uint8)
        x[0] = 1
        for name, positions in state:
            horizontal = positions[0][0] == positions[1][0]
            category = 1 if name == "red" else (2 if horizontal else 3)
            ordered = sorted(positions)
            for i, j in positions:
                x[0, i, j] = 0
                x[category, i, j] = 1
                x[6, i, j] = len(positions) == 3
            x[(4, *ordered[0])] = 1
            x[(5, *ordered[-1])] = 1
        return x

    def encode(self, state):
        return torch.from_numpy(self.encode_numpy(state)).float()


def valid_action_mask(environment, action_space):
    """Boolean mask over action_space of the moves currently legal in `environment`."""
    legal = set(legal_actions(environment, action_space))
    return torch.tensor([action in legal for action in action_space], dtype=torch.bool)


class LaneBlock(nn.Module):
    """Residual block whose full-row and full-column convolutions let every
    cell see its whole lane - what blocks a car, or red's path to the exit,
    is often several cells away."""
    def __init__(self, hidden, n_rows, n_cols):
        super().__init__()
        self.row = nn.Conv2d(hidden, hidden, (1, 2 * n_cols - 1), padding=(0, n_cols - 1))
        self.col = nn.Conv2d(hidden, hidden, (2 * n_rows - 1, 1), padding=(n_rows - 1, 0))
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm = nn.BatchNorm2d(hidden)

    def forward(self, x):
        return F.relu(x + self.norm(self.local(F.relu(self.row(x) + self.col(x)))))


class PolicyNetwork(nn.Module):
    """Lane-aware residual trunk with three heads. Move heads are shared
    across board positions: each cell predicts its car's moves (4 directions
    x max_slide steps), so a pattern learned in one place applies everywhere.

    - policy: which legal move to play now (masked softmax);
    - dest: for every car and destination, legal now or not, how soon an
      optimal solution brings that car there - what orders AND-OR's
      candidate resolutions (PolicyAgent.heuristic);
    - value: distance to the goal, in tens of moves.

    forward() returns just the policy logits."""
    def __init__(self, in_channels, n_rows, n_cols, moves_per_cell, hidden=96, blocks=4):
        super().__init__()
        rows = torch.linspace(-1, 1, n_rows)[:, None].expand(n_rows, n_cols)
        cols = torch.linspace(-1, 1, n_cols)[None, :].expand(n_rows, n_cols)
        self.register_buffer("coords", torch.stack([rows, cols]))
        self.stem = nn.Conv2d(in_channels + 2, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[LaneBlock(hidden, n_rows, n_cols) for _ in range(blocks)])
        self.policy_head = nn.Conv2d(hidden, moves_per_cell, 1)
        self.dest_head = nn.Conv2d(hidden, moves_per_cell, 1)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def heads(self, x):
        """(policy logits, dest logits, value) for a batch of encoded states."""
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        h = self.blocks(F.relu(self.stem(x)))

        def per_action(head):
            return head(h).permute(0, 2, 3, 1).reshape(x.shape[0], -1)

        return per_action(self.policy_head), per_action(self.dest_head), self.value_head(h.mean((2, 3))).squeeze(1)

    def forward(self, x):
        return self.heads(x)[0]


class PolicyAgent():
    """Policy network plus its action space. Actions are keyed by board cell
    rather than car name (see car_anchor) - (cell, direction, steps), in
    cell-major order to match the network's per-cell heads - so the same
    agent applies to any puzzle of this board size regardless of how many
    cars it has or what they're named.

    Trained on the solver's own solved attempts by expert iteration (see
    expert_iteration and train_policy). Used two ways, which
    solve_arbitrated combines: playing the policy head's top legal move
    (intuition; policy_avoiding_cycles), and as the `heuristic` that orders
    and_or.solve's candidate moves (deliberation: AND-OR's lapsing,
    subgoal-by-subgoal reasoning with learned judgment about which option
    to try first).

    `temperature` (default 0) adds Gumbel noise of that scale to every
    heuristic score, so AND-OR tries options in an order sampled in
    proportion to the network's preference instead of always the same one -
    otherwise the same board always gets the same choice, and a pair of
    plans that undo each other repeats until a lapse breaks it.
    """
    def __init__(self, size=(6, 6), max_slide=None, lr=1e-3, hidden=96, blocks=4, device="cpu", temperature=0.0):
        self.size = size
        self.temperature = temperature
        self.encoder = StateEncoder(size)
        self.max_slide = max_slide if max_slide is not None else max(size) - 1
        self.hidden = hidden
        self.blocks = blocks
        cells = [(i, j) for i in range(size[0]) for j in range(size[1])]
        self.action_space = [(cell, direction, steps)
                             for cell in cells for direction in DIRECTIONS
                             for steps in range(1, self.max_slide + 1)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        in_channels, n_rows, n_cols = self.encoder.shape
        self.net = PolicyNetwork(in_channels, n_rows, n_cols, 4 * self.max_slide, hidden, blocks)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.device = "cpu"
        self.to(device)
        self._scores = {}

    def to(self, device):
        """Moves the network to `device` (e.g. "mps" for training, "cpu" for
        the many single-board evaluations AND-OR makes). Returns self."""
        self.device = device
        self.net.to(device)
        self._scores = {}
        return self

    def save(self, path):
        """Saves this agent's weights plus the hyperparameters load() needs to
        rebuild a network and action space of the same shape. Optimizer state
        is not saved."""
        torch.save({
            "version": 2,
            "size": self.size,
            "max_slide": self.max_slide,
            "hidden": self.hidden,
            "blocks": self.blocks,
            "state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, lr=1e-3, device="cpu"):
        """Rebuilds a PolicyAgent from a save() checkpoint, with a fresh Adam
        optimizer."""
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        if checkpoint.get("version") != 2:
            raise ValueError(f"{path} was saved by the previous network (no car-boundary channels, no "
                             f"dest/value heads); retrain and save a new agent")
        agent = cls(size=checkpoint["size"], max_slide=checkpoint["max_slide"], hidden=checkpoint["hidden"],
                    blocks=checkpoint["blocks"], lr=lr)
        agent.net.load_state_dict(checkpoint["state_dict"])
        return agent.to(device)

    def action_id(self, state, car_name, direction, steps):
        """Index of a (car_name, direction, steps) move from `state`, or None
        if it's outside the action space."""
        return self.action_index.get((car_anchor(state, car_name), direction, steps))

    def _evaluate(self, state):
        """(policy logits, dest logits) for one state, on the CPU. Cached,
        since AND-OR asks about the same boards over and over."""
        if state not in self._scores:
            if len(self._scores) > 50_000:
                self._scores.clear()
            self.net.eval()
            with torch.no_grad():
                policy, dest, _ = self.net.heads(self.encoder.encode(state).unsqueeze(0).to(self.device))
            self._scores[state] = (policy[0].cpu(), dest[0].cpu())
        return self._scores[state]

    def policy_avoiding_cycles(self, environment, mask, visited=frozenset()):
        """Ranks every currently legal action by the policy head's logit (highest
        first) and returns the first one whose resulting state isn't already in
        `visited`, falling back to the plain top-ranked action only when every
        candidate leads somewhere already visited. Returns a (car_name,
        direction, steps) move, resolving each candidate cell to whichever car
        currently occupies it.

        Catches the failure mode where a single wrong greedy decision lands the
        rollout on an undertrained state and, with no escape hatch, walks
        straight into a 2-cycle (undoing its own last move) until max_steps
        runs out - turning one bad decision into a total non-solve.
        """
        state = environment.get_state()
        logits = self._evaluate(state)[0]

        scored = []
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            cell, direction, steps = self.action_space[i]
            car_name = environment.car_at(cell)
            environment.move(car_name, direction, steps)
            next_state = environment.get_state()
            environment.move(car_name, OPPOSITE[direction], steps)
            scored.append(((car_name, direction, steps), logits[i].item(), next_state))

        scored.sort(key=lambda entry: entry[1], reverse=True)
        for action, _score, next_state in scored:
            if next_state not in visited:
                return action
        return scored[0][0]

    def heuristic(self, state, actions):
        """Dest-head scores for `actions` (full (car_name, direction, steps)
        tuples, legal now or not) from `state`: how soon an optimal solution
        brings that car to that destination, as the network estimates it. This
        is and_or.solve's `heuristic` - it orders AND-OR's candidate moves
        best-first, and picks the move a lapse plays (and_or.lapse_move).
        Returns a dict mapping each action to its score (-inf if outside this
        agent's action space)."""
        dest = self._evaluate(state)[1]
        scores = {}
        for action in actions:
            idx = self.action_id(state, *action)
            scores[action] = dest[idx].item() if idx is not None else float("-inf")
            if self.temperature > 0:
                scores[action] -= self.temperature * np.log(-np.log(random.random() or 1e-12))
        return scores


# --- labels ----------------------------------------------------------------------------------
# The network learns only what a solver itself could know: its own solved
# AND-OR attempts, with loops cut out, and which option resolved each
# subgoal in the plans that worked (see expert_iteration). Exact labels from
# a breadth-first search of each puzzle's state space (label_puzzles,
# label_subgoals) are for measuring the network, not for training it.

class Example():
    """One labelled board: its encoding, distance to the goal, legal and
    target move indices, and dest-head targets (action indices and values).
    From attempts (attempt_examples), the distance is the moves the best
    attempt still took and the target move is the one it played."""
    __slots__ = ("state", "x", "distance", "legal", "optimal", "dest_idx", "dest_target")

    def __init__(self, state, x, distance, legal, optimal, dest_idx, dest_target):
        self.state, self.x, self.distance = state, x, distance
        self.legal, self.optimal = legal, optimal
        self.dest_idx, self.dest_target = dest_idx, dest_target


class SubgoalExample():
    """One subgoal from an AND-OR search: the board, the action indices of
    the options AND-OR compares there (the subgoal car's candidate moves),
    and which of them are targets - the option that resolved it in a plan
    that worked, or the options an optimal solution prefers."""
    __slots__ = ("state", "x", "candidates", "targets")

    def __init__(self, state, x, candidates, targets):
        self.state, self.x, self.candidates, self.targets = state, x, candidates, targets


def _move_index(state_dict, car, direction, steps, n_cols, max_slide):
    """Action index of a move, keyed by the car's top/left cell (the order of
    PolicyAgent.action_space)."""
    r = min(i for i, _ in state_dict[car])
    c = min(j for _, j in state_dict[car])
    return ((r * n_cols + c) * 4 + DIRECTIONS.index(direction)) * max_slide + steps - 1


def _dest_labels(board, reached, size, max_slide):
    """Dest-head (action indices, targets) for `board`: every on-board
    destination of every car, with target DEST_DECAY ** (k - 1) if
    reached[(car, positions)] = k, else 0."""
    n_rows, n_cols = size
    sd = dict(board)
    dest_idx, dest_target = [], []
    for car, cells in board:
        horizontal = cells[0][0] == cells[1][0]
        for direction in ("lr" if horizontal else "ud"):
            di, dj = DELTAS[direction]
            for steps in range(1, max_slide + 1):
                moved = tuple((i + di * steps, j + dj * steps) for i, j in cells)
                if not all(0 <= i < n_rows and 0 <= j < n_cols for i, j in moved):
                    break
                k = reached.get((car, moved))
                dest_idx.append(_move_index(sd, car, direction, steps, n_cols, max_slide))
                dest_target.append(DEST_DECAY ** (k - 1) if k is not None else 0.0)
    return np.array(dest_idx, dtype=np.int16), np.array(dest_target, dtype=np.float32)


def optimal_destinations(state, distances, orientations, max_states=20_000):
    """{(car_name, positions): k} - the fewest moves after which some optimal
    solution from `state` has that car at those positions (k >= 1; the
    car's current positions are left out). Walks the shortest-path DAG layer
    by layer, stopping after `max_states` boards."""
    first = {}
    layer, depth, seen = {state}, 0, 0
    while layer and seen < max_states:
        nxt = set()
        for s in layer:
            if depth > 0:
                for car in s:
                    first.setdefault(car, depth)
            for _, t in neighbors(s, orientations):
                if distances.get(t) == distances[s] - 1:
                    nxt.add(t)
        seen += len(layer)
        layer, depth = nxt, depth + 1
    for car in state:
        first.pop(car, None)
    return first


def _label_state(agent_shape, state, distances, orientations):
    size, max_slide = agent_shape
    n_cols = size[1]
    d = distances[state]
    sd = dict(state)
    legal, optimal = [], []
    for (car, direction, steps), nxt in neighbors(state, orientations):
        idx = _move_index(sd, car, direction, steps, n_cols, max_slide)
        legal.append(idx)
        if distances.get(nxt) == d - 1:
            optimal.append(idx)
    dest_idx, dest_target = _dest_labels(state, optimal_destinations(state, distances, orientations), size,
                                         max_slide)
    return Example(state, StateEncoder(size).encode_numpy(state), d, legal, optimal, dest_idx, dest_target)


def label_puzzle(job):
    """Labelled examples for one puzzle: its starting state plus up to
    `n_states` other unsolved states of its state space (all of them if
    None), sampled with `seed`."""
    state, n_states, seed, agent_shape = job
    distances = multi_bfs(state)
    orientations = car_orientations(state)
    others = [s for s, d in distances.items() if d > 0 and s != state]
    if n_states is not None:
        others = random.Random(seed).sample(others, min(n_states, len(others)))
    return [_label_state(agent_shape, s, distances, orientations) for s in [state] + others]


def label_puzzles(puzzles, n_states=50, seed=0, size=(6, 6), max_slide=5, processes=None):
    """Breadth-first-search labels for every puzzle (in parallel): a list,
    one entry per puzzle, of its Examples - see label_puzzle. Labels are
    exact: every optimal move from each state, its distance to the goal, and
    how soon an optimal solution brings each car to each destination."""
    jobs = [(state, n_states, hash((seed, i)) & 0xFFFFFFFF, (size, max_slide)) for i, state in enumerate(puzzles)]
    with Pool(processes) as pool:
        return list(tqdm(pool.imap(label_puzzle, jobs, chunksize=4), total=len(jobs), desc="labelling puzzles"))


def oracle_heuristic(distances, orientations):
    """An and_or.solve heuristic that knows every board's distance to the
    goal: a candidate move scores -k when an optimal solution first has
    that car at that destination k moves from now (much lower if never),
    plus a random tie-break below 1."""
    cache = {}

    def heuristic(state, actions):
        if state not in cache:
            cache[state] = optimal_destinations(state, distances, orientations) if state in distances else {}
        first = cache[state]
        scores = {}
        for action in actions:
            moved = dict(AndNode(state, *action).apply(state))[action[0]]
            scores[action] = -first.get((action[0], moved), 10 ** 6) + random.random()
        return scores
    return heuristic


class SearchRecorder():
    """While active (a `with` block), records the search tree of every
    and_or.solve pass: one frame per OrNode.solve call (its attempted
    actions) and per AndNode.solve call (its subgoals, and whether it
    succeeded). Call reset() before each pass."""

    def __enter__(self):
        self._or, self._and = OrNode.solve, AndNode.solve
        recorder = self

        def or_solve(node):
            frame = {"node": node, "attempts": [], "ok": False,
                     "free": (node.car_name, node.collisions) in node.visited}
            if recorder.stack:
                recorder.stack[-1]["children"].append(frame)
            recorder.subgoals.append(frame)
            recorder.stack.append(frame)
            try:
                result = recorder._or(node)
            finally:
                recorder.stack.pop()
            frame["ok"] = result is not None
            return result

        def and_solve(node):
            frame = {"node": node, "children": [], "ok": False}
            (recorder.stack[-1]["attempts"] if recorder.stack else recorder.roots).append(frame)
            recorder.stack.append(frame)
            try:
                result = recorder._and(node)
            finally:
                recorder.stack.pop()
            frame["ok"] = result is not None
            return result

        OrNode.solve, AndNode.solve = or_solve, and_solve
        self.reset()
        return self

    def __exit__(self, *exc):
        OrNode.solve, AndNode.solve = self._or, self._and

    def reset(self):
        self.roots, self.stack, self.subgoals = [], [], []

    def plan_choices(self):
        """[(OrNode, (direction, steps))] - every subgoal on the plan the last
        pass returned and the option that resolved it; empty if the pass
        lapsed or found no plan. A successful AndNode's subgoals are its last
        k OrNode calls (k = its blocker count): it returns right after the
        blocker order that worked."""
        root = next((r for r in reversed(self.roots) if r["ok"]), None)
        choices = []

        def walk(and_frame):
            k = len(and_frame["node"].blockers())
            for or_frame in (and_frame["children"][-k:] if k else []):
                chosen = next(a for a in reversed(or_frame["attempts"]) if a["ok"])
                choices.append((or_frame["node"], (chosen["node"].direction, chosen["node"].steps)))
                walk(chosen)

        if root is not None:
            walk(root)
        return choices


def _subgoal_example(node, targets, size, max_slide):
    """SubgoalExample for an OrNode; `targets` is the set of its (direction,
    steps) options to rank first. None if it has fewer than two options."""
    options = node.directions()
    if len(options) < 2:
        return None
    sd = dict(node.state)
    candidates = [_move_index(sd, node.car_name, d, s, size[1], max_slide) for d, s in options]
    return SubgoalExample(node.state, StateEncoder(size).encode_numpy(node.state),
                          np.array(candidates, dtype=np.int16), np.array([o in targets for o in options]))


def label_subgoals_puzzle(job):
    """Oracle subgoal labels for one puzzle: lapse-free AND-OR passes from its
    start, ordered by oracle_heuristic, until red exits; every subgoal they
    visit, with the options an optimal solution prefers as targets. At most
    `limit` subgoals, sampled with `seed`."""
    state, limit, seed, (size, max_slide) = job
    random.seed(seed)
    distances = multi_bfs(state)
    orientations = car_orientations(state)
    heuristic = oracle_heuristic(distances, orientations)
    out, moves = [], 0
    with SearchRecorder() as recorder:
        while _red_steps(state) > 0 and moves < 100:
            recorder.reset()
            att = and_or_solve(state, heuristic=heuristic)
            step = att if type(att) is list else att[1]
            for frame in recorder.subgoals:
                node = frame["node"]
                if frame["free"]:
                    continue
                options = node.directions()
                scores = heuristic(node.state, [(node.car_name, d, s) for d, s in options])
                best = max(int(np.floor(v)) for v in scores.values()) if scores else None
                targets = {(d, s) for d, s in options if int(np.floor(scores[(node.car_name, d, s)])) == best}
                example = _subgoal_example(node, targets, size, max_slide)
                if example is not None:
                    out.append(example)
            if not step:
                break
            for move in step:
                state = AndNode(state, *move).apply(state)
            moves += len(step)
    return random.Random(seed).sample(out, min(limit, len(out))) if limit is not None else out


def label_subgoals(puzzles, limit=200, seed=0, size=(6, 6), max_slide=5, processes=None):
    """Oracle subgoal labels for every puzzle (in parallel): a list, one
    entry per puzzle, of SubgoalExamples - see label_subgoals_puzzle."""
    jobs = [(state, limit, hash((seed, i)) & 0xFFFFFFFF, (size, max_slide)) for i, state in enumerate(puzzles)]
    with Pool(processes) as pool:
        return list(tqdm(pool.imap(label_subgoals_puzzle, jobs, chunksize=2), total=len(jobs),
                         desc="labelling subgoals"))


def erase_loops(boards, moves):
    """An attempt's boards (start first) and moves with every loop cut out:
    a board seen again rewinds the path to its first visit."""
    path, played, position = [boards[0]], [], {boards[0]: 0}
    for move, board in zip(moves, boards[1:]):
        if board in position:
            cut = position[board]
            for b in path[cut + 1:]:
                del position[b]
            path, played = path[:cut + 1], played[:cut]
        else:
            position[board] = len(path)
            path.append(board)
            played.append(move)
    return path, played


def attempt_examples(attempts, size=(6, 6), max_slide=5):
    """Examples from solved attempts ((boards, moves) pairs): per board, the
    attempt with the fewest moves left from it supplies the labels - the
    move it played (moves tied for fewest all count), the moves it still
    took as the distance, and how many moves later it had each car on each
    destination as dest targets."""
    best = {}  # board -> (moves left, set of move indices, {(car, positions): moves until reached})
    for boards, moves in attempts:
        n = len(moves)
        for t in range(n):
            board, left = boards[t], n - t
            idx = _move_index(dict(board), *moves[t], size[1], max_slide)
            if board in best and best[board][0] < left:
                continue
            if board in best and best[board][0] == left:
                best[board][1].add(idx)
                continue
            reached = {}
            for k in range(t + 1, n + 1):
                for entry in boards[k]:
                    reached.setdefault(entry, k - t)
            for entry in board:
                reached.pop(entry, None)
            best[board] = (left, {idx}, reached)

    encoder = StateEncoder(size)
    examples = []
    for board, (left, targets, reached) in best.items():
        sd = dict(board)
        legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
        dest_idx, dest_target = _dest_labels(board, reached, size, max_slide)
        examples.append(Example(board, encoder.encode_numpy(board), left, legal, sorted(targets), dest_idx,
                                dest_target))
    return examples


_WORKER = {}


def _init_attempt_worker(weights, config):
    """Pool initializer: a CPU copy of the guiding agent (None = plain AND-OR)."""
    torch.set_num_threads(1)
    _WORKER.clear()
    if weights is not None:
        size, max_slide, hidden, blocks, temperature = config
        agent = PolicyAgent(size, max_slide, hidden=hidden, blocks=blocks, temperature=temperature)
        agent.net.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
        _WORKER["agent"] = agent


def attempt_puzzle(job):
    """(Examples, SubgoalExamples, solved attempts) for one puzzle from
    `n_attempts` AND-OR attempts at lapse probability `gamma`, guided by the
    worker's agent if it has one. Only solved attempts teach: their boards,
    loops cut out (attempt_examples), and every subgoal of every plan they
    completed, with the option that resolved it as the target."""
    state0, n_attempts, gamma, seed, max_steps, (size, max_slide) = job
    random.seed(seed)
    agent = _WORKER.get("agent")
    heuristic = agent.heuristic if agent is not None else None
    attempts, subgoals = [], []
    with SearchRecorder() as recorder:
        for _ in range(n_attempts):
            state, boards, moves, choices = state0, [state0], [], []
            while len(moves) < max_steps and _red_steps(state) > 0:
                recorder.reset()
                att = and_or_solve(state, heuristic=heuristic, gamma=gamma)
                step = att if type(att) is list else att[1]
                if not step:
                    break
                choices += recorder.plan_choices()
                for move in step:
                    state = AndNode(state, *move).apply(state)
                    boards.append(state)
                    moves.append(move)
            if _red_steps(state) <= 0:
                attempts.append(erase_loops(boards, moves))
                for node, chosen in choices:
                    example = _subgoal_example(node, {chosen}, size, max_slide)
                    if example is not None:
                        subgoals.append(example)
    return attempt_examples(attempts, size, max_slide), subgoals, len(attempts)


def label_attempts(puzzles, n_attempts=4, gamma=0.07, agent=None, seed=0, max_steps=200, processes=None,
                   size=(6, 6), max_slide=5):
    """Labels from the solver's own attempts on every puzzle (in parallel):
    (Examples, SubgoalExamples, solved attempts) - see attempt_puzzle. With
    `agent`, attempts are made by the human-like solver it guides (its
    heuristic, at its temperature); without, by plain AND-OR. `size` and
    `max_slide` (the agent's, when one is given) set the action indexing."""
    if agent is not None:
        size, max_slide = agent.size, agent.max_slide
    weights, config = None, None
    if agent is not None:
        buffer = io.BytesIO()
        torch.save({k: v.cpu() for k, v in agent.net.state_dict().items()}, buffer)
        weights = buffer.getvalue()
        config = (agent.size, agent.max_slide, agent.hidden, agent.blocks, agent.temperature)
    jobs = [(state, n_attempts, gamma, hash((seed, i)) & 0xFFFFFFFF, max_steps, (size, max_slide))
            for i, state in enumerate(puzzles)]
    examples, subgoals, solved = [], [], 0
    with Pool(processes, initializer=_init_attempt_worker, initargs=(weights, config)) as pool:
        for ex, sg, n in tqdm(pool.imap(attempt_puzzle, jobs, chunksize=2), total=len(jobs), desc="attempts"):
            examples += ex
            subgoals += sg
            solved += n
    return examples, subgoals, solved


def load_puzzle_pool(path="../data/rush_nw.txt"):
    """Every (distance, state) in a rush_nw.txt-format file."""
    with open(path) as f:
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


# --- training -------------------------------------------------------------------------------

def _batch(agent, examples):
    device = agent.device
    n = len(examples)
    x = torch.from_numpy(np.stack([e.x for e in examples])).float().to(device)
    distance = torch.tensor([e.distance for e in examples], dtype=torch.float32, device=device)
    n_actions = len(agent.action_space)
    legal = torch.zeros(n, n_actions, dtype=torch.bool)
    optimal = torch.zeros(n, n_actions, dtype=torch.bool)
    dest_mask = torch.zeros(n, n_actions, dtype=torch.bool)
    dest_target = torch.zeros(n, n_actions)
    for r, e in enumerate(examples):
        legal[r, e.legal] = True
        optimal[r, e.optimal] = True
        idx = torch.from_numpy(e.dest_idx.astype(np.int64))
        dest_mask[r, idx] = True
        dest_target[r, idx] = torch.from_numpy(e.dest_target)
    return (x, distance, legal.to(device), optimal.to(device), dest_mask.to(device), dest_target.to(device))


def _loss(agent, examples):
    x, distance, legal, optimal, dest_mask, dest_target = _batch(agent, examples)
    policy, dest, value = agent.net.heads(x)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    policy_loss = -logp.masked_fill(~optimal, float("-inf")).logsumexp(1).mean()
    dest_loss = F.binary_cross_entropy_with_logits(dest[dest_mask], dest_target[dest_mask])
    value_loss = F.smooth_l1_loss(value, distance / 10)
    return policy_loss + dest_loss + 0.5 * value_loss


def _subgoal_batch(agent, subgoals):
    device = agent.device
    width = max(len(s.candidates) for s in subgoals)
    x = torch.from_numpy(np.stack([s.x for s in subgoals])).float().to(device)
    idx = torch.zeros(len(subgoals), width, dtype=torch.long)
    valid = torch.zeros(len(subgoals), width, dtype=torch.bool)
    target = torch.zeros(len(subgoals), width, dtype=torch.bool)
    for r, s in enumerate(subgoals):
        n = len(s.candidates)
        idx[r, :n] = torch.from_numpy(s.candidates.astype(np.int64))
        valid[r, :n] = True
        target[r, :n] = torch.from_numpy(s.targets)
    return x, idx.to(device), valid.to(device), target.to(device)


def _subgoal_logits(agent, subgoals):
    """Dest-head logits of each subgoal's options (-inf past its last), and
    the valid/target masks."""
    x, idx, valid, target = _subgoal_batch(agent, subgoals)
    _, dest, _ = agent.net.heads(x)
    return dest.gather(1, idx).masked_fill(~valid, float("-inf")), valid, target


def _subgoal_loss(agent, subgoals):
    """Ranking loss over each subgoal's options: probability (softmax over the
    options' dest scores) on the target options."""
    logits, _, target = _subgoal_logits(agent, subgoals)
    logp = torch.log_softmax(logits, dim=1)
    return -logp.masked_fill(~target, float("-inf")).logsumexp(1).mean()


@torch.no_grad()
def evaluate_agreement(agent, examples, batch_size=2048):
    """How well the agent's heads match the labels on `examples`:
    - agreement: how often the policy head's top legal move is a target;
    - chance: the same for a uniformly random legal move;
    - distance_error: mean absolute error of the value head, in moves.
    """
    if not examples:
        return {"agreement": float("nan"), "chance": float("nan"), "distance_error": float("nan")}
    agent.net.eval()
    hits = chance = err = 0.0
    for i in range(0, len(examples), batch_size):
        x, distance, legal, optimal, _, _ = _batch(agent, examples[i:i + batch_size])
        policy, _, value = agent.net.heads(x)
        top = policy.masked_fill(~legal, float("-inf")).argmax(1)
        hits += optimal.gather(1, top[:, None]).float().sum().item()
        chance += (optimal.sum(1).float() / legal.sum(1).float()).sum().item()
        err += (value * 10 - distance).abs().sum().item()
    agent.net.train()
    n = len(examples)
    return {"agreement": hits / n, "chance": chance / n, "distance_error": err / n}


@torch.no_grad()
def evaluate_subgoals(agent, subgoals, batch_size=2048):
    """How well the agent orders AND-OR's options on `subgoals`:
    - agreement: how often its top-scoring option is a target;
    - chance: the same for a uniformly random option."""
    if not subgoals:
        return {"agreement": float("nan"), "chance": float("nan")}
    agent.net.eval()
    hits = chance = 0.0
    for i in range(0, len(subgoals), batch_size):
        logits, valid, target = _subgoal_logits(agent, subgoals[i:i + batch_size])
        hits += target.gather(1, logits.argmax(1)[:, None]).float().sum().item()
        chance += (target.sum(1).float() / valid.sum(1).float()).sum().item()
    agent.net.train()
    n = len(subgoals)
    return {"agreement": hits / n, "chance": chance / n}


def _validation_score(scores):
    """Mean of a train_policy history entry's scores, ignoring missing (nan) ones."""
    values = [v for v in scores.values() if v == v]
    return sum(values) / len(values) if values else 0.0


def train_policy(agent, examples, val_examples, subgoals=(), val_subgoals=(), epochs=40, patience=6,
                 batch_size=256):
    """Supervised training: on board Examples, the policy head puts
    probability on any target move, the dest head predicts each
    destination's target and the value head the distance; on
    SubgoalExamples (`subgoals`), the dest head ranks each subgoal's target
    options first - the ordering AND-OR uses. Board and subgoal batches are
    interleaved at random.

    Keeps the weights with the best validation score - the policy head's
    agreement on `val_examples`, averaged with the ordering agreement on
    `val_subgoals` when given (both from puzzles held out of training) - and
    stops after `patience` epochs without improvement. Returns per-epoch
    validation scores as dicts."""
    examples, subgoals = list(examples), list(subgoals)
    agent._scores = {}
    best, best_state, stale, history = -1.0, None, 0, []
    bar = tqdm(range(epochs), desc="training epochs")
    for _ in bar:
        agent.net.train()
        random.shuffle(examples)
        random.shuffle(subgoals)
        batches = [(False, examples[i:i + batch_size]) for i in range(0, len(examples), batch_size)]
        batches += [(True, subgoals[i:i + batch_size]) for i in range(0, len(subgoals), batch_size)]
        random.shuffle(batches)
        for is_subgoal, batch in batches:
            loss = _subgoal_loss(agent, batch) if is_subgoal else _loss(agent, batch)
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        scores = {"agreement": evaluate_agreement(agent, val_examples)["agreement"]}
        if val_subgoals:
            scores["subgoal_agreement"] = evaluate_subgoals(agent, list(val_subgoals))["agreement"]
        history.append(scores)
        score = _validation_score(scores)
        bar.set_postfix({k: f"{v:.1%}" for k, v in scores.items()})
        if score > best:
            best, best_state, stale = score, {k: v.clone() for k, v in agent.net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    agent.net.load_state_dict(best_state)
    agent._scores = {}
    return history


def expert_iteration(agent, puzzles, val_puzzles, rounds=3, n_attempts=4, gamma=0.07, max_steps=200, seed=0,
                     **train_kwargs):
    """Trains `agent` on the solver's own attempts, by expert iteration. Round
    1's attempts are plain AND-OR; each later round's come from the
    human-like solver the agent has become (its heuristic, at its
    temperature), so the examples improve as the agent does. Each round
    labels attempts on `puzzles` (training) and `val_puzzles` (validation)
    with label_attempts and keeps training the same weights on them. No
    breadth-first search is involved.

    Returns one dict per round: attempts made and solved on the training
    puzzles, boards and subgoals learned from, and the best validation
    scores (agreement with the round's own attempts)."""
    history = []
    for r in range(rounds):
        guide = agent if r > 0 else None
        shape = {"size": agent.size, "max_slide": agent.max_slide}
        examples, subgoals, solved = label_attempts(puzzles, n_attempts, gamma, guide, seed + 2 * r, max_steps,
                                                    **shape)
        val_examples, val_subgoals, _ = label_attempts(val_puzzles, n_attempts, gamma, guide, seed + 2 * r + 1,
                                                       max_steps, **shape)
        scores = train_policy(agent, examples, val_examples, subgoals, val_subgoals, **train_kwargs)
        best = max(scores, key=_validation_score)
        history.append({"round": r + 1, "attempts": len(puzzles) * n_attempts, "solved": solved,
                        "boards": len(examples), "subgoals": len(subgoals), **best})
    return history


# --- solving ---------------------------------------------------------------------------------

def solve_and_or(state, max_steps=200, heuristic=None, gamma=0.0):
    """Repeated and_or.solve passes from `state` until red exits or
    `max_steps` moves are played. Returns the moves if solved, else None.

    `heuristic`, if given (a trained PolicyAgent's `.heuristic`), orders
    AND-OR's candidate moves and picks what a lapse plays - the human-like
    solver. `gamma` is the lapse probability - see and_or.solve."""
    moves = []
    while len(moves) < max_steps:
        att = and_or_solve(state, heuristic=heuristic, gamma=gamma)
        if type(att) is list:
            moves.extend(att)
            return moves
        state, step_moves = att
        if not step_moves:
            return None
        moves.extend(step_moves)
    return None


def solve_arbitrated(agent, state, max_steps=200, gamma=0.0, threshold=0.5, memory=10):
    """Solves from `state` by arbitrating, every turn, between intuition and
    deliberation - the human-like solver. Returns (moves if solved else
    None, stats).

    - Intuition: the policy head's top legal move, played when its
      probability (softmax over legal moves) is at least `threshold` and it
      doesn't lead back to one of the last `memory` boards (None = every
      board this attempt). One move.
    - Deliberation: otherwise, one and_or.solve pass ordered by the agent's
      heuristic, with lapse probability `gamma`; its moves are played. A
      move that would revisit a recent board counts as low confidence -
      going in circles is a reason to stop and think.

    threshold 0 is pure intuition (apart from revisits); above 1, pure
    deliberation. stats counts "intuitive" moves, "deliberations" (passes)
    and "deliberate" moves."""
    orientations = car_orientations(state)
    recent = deque([state], maxlen=memory)
    moves = []
    stats = {"intuitive": 0, "deliberations": 0, "deliberate": 0}
    while len(moves) < max_steps and _red_steps(state) > 0:
        options = list(neighbors(state, orientations))
        if threshold <= 1:
            logits = agent._evaluate(state)[0][[agent.action_id(state, *move) for move, _ in options]]
            probs = torch.softmax(logits, dim=0)
            best = int(probs.argmax())
            move, nxt = options[best]
            if probs[best].item() >= threshold and nxt not in recent:
                state = nxt
                moves.append(move)
                recent.append(state)
                stats["intuitive"] += 1
                continue
        att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
        step = att if type(att) is list else att[1]
        stats["deliberations"] += 1
        if not step:
            break
        for move in step:
            state = AndNode(state, *move).apply(state)
            moves.append(move)
            recent.append(state)
        stats["deliberate"] += len(step)
    return (moves if _red_steps(state) <= 0 else None), stats


def _arbitrated_job(job):
    """One solve_arbitrated attempt by the worker's agent (see label_attempts' Pool setup)."""
    state, distance, gamma, threshold, memory, max_steps, seed = job
    random.seed(seed)
    moves, stats = solve_arbitrated(_WORKER["agent"], state, max_steps, gamma, threshold, memory)
    return {"solved": moves is not None, "moves": len(moves) if moves is not None else None,
            "distance": distance, **stats}


def evaluate_arbitrated(agent, puzzles, n_runs=3, gamma=0.0, threshold=0.5, memory=10, max_steps=200, seed=0,
                        processes=None):
    """solve_arbitrated `n_runs` times on every (distance, state) in `puzzles`,
    in parallel (each worker gets a CPU copy of the agent, at its
    temperature). Returns one dict per attempt, puzzle by puzzle: solved,
    moves (None if unsolved), distance, and the stats of solve_arbitrated."""
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in agent.net.state_dict().items()}, buffer)
    config = (agent.size, agent.max_slide, agent.hidden, agent.blocks, agent.temperature)
    jobs = [(state, d, gamma, threshold, memory, max_steps, hash((seed, i, r)) & 0xFFFFFFFF)
            for i, (d, state) in enumerate(puzzles) for r in range(n_runs)]
    with Pool(processes, initializer=_init_attempt_worker, initargs=(buffer.getvalue(), config)) as pool:
        return pool.map(_arbitrated_job, jobs, chunksize=2)


def evaluate_and_or(initial_cars, size, n_trials, max_steps=200, heuristic=None, gamma=0.0):
    """Solve rate and average move count over `n_trials` independent AND-OR
    attempts from initial_cars. Pass a trained agent's `.heuristic` for the
    human-like solver, None for plain (randomly ordered) AND-OR. Returns
    (solve_rate, avg_moves) - avg_moves is nan when nothing solved.
    """
    state0 = tuple((name, tuple(positions)) for name, positions in initial_cars)
    lengths = [
        len(moves)
        for moves in (solve_and_or(state0, max_steps, heuristic=heuristic, gamma=gamma) for _ in range(n_trials))
        if moves
    ]
    solve_rate = len(lengths) / n_trials
    avg_moves = float(np.mean(lengths)) if lengths else float("nan")
    return solve_rate, avg_moves


def evaluate_policy(agent, initial_cars, size, max_steps=200):
    """Rolls the agent forward from initial_cars via policy_avoiding_cycles,
    purely over the legal-move action space (no AND-OR involved). Tracks every
    state seen this rollout as `visited` so a shallow 2-cycle gets routed
    around via the next-best logit instead of looping until max_steps.
    Returns (solved, steps, moves).
    """
    environment = Environment(size, initial_cars)
    moves = []
    visited = set()
    for step in range(max_steps):
        visited.add(environment.get_state())
        mask = valid_action_mask(environment, agent.action_space)
        if not mask.any():
            break
        move = agent.policy_avoiding_cycles(environment, mask, visited=visited)
        environment.move(*move)
        moves.append(move)
        if environment.check_win():
            return True, step + 1, moves
    return False, None, moves
