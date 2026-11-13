import copy
import io
import json
import math
import os
import random
from collections import Counter, deque
from types import SimpleNamespace
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
    solve_arbitrated combines: playing the policy head's moves (intuition;
    policy_avoiding_cycles, solve_intuition), and as the `heuristic` that orders
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
        self.train_device = None  # see train_on
        self._trainer = None

    def train_on(self, device):
        """Has online_update train a copy of the network on `device` (e.g. "mps"), with its own Adam, while
        play stays on this agent's device: one board at a time is faster on the CPU, a batch of 256 on the GPU.
        After each update the copy's weights are copied back. None: train the network itself. Returns self."""
        self.train_device, self._trainer = device, None
        return self

    def _learner(self):
        """What online_update trains: this agent, or (with train_on) the copy on the training device - made
        from the current weights the first time, then kept with its optimizer's state."""
        if self.train_device is None:
            return self
        if self._trainer is None:
            net = copy.deepcopy(self.net).to(self.train_device)
            self._trainer = SimpleNamespace(
                net=net, device=self.train_device, action_space=self.action_space,
                optimizer=optim.Adam(net.parameters(), lr=self.optimizer.param_groups[0]["lr"]))
        return self._trainer

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
    sd = dict(board)
    dest_idx, dest_target = [], []
    for car, direction, steps, moved in _destinations(board, size, max_slide):
        k = reached.get((car, moved))
        dest_idx.append(_move_index(sd, car, direction, steps, size[1], max_slide))
        dest_target.append(DEST_DECAY ** (k - 1) if k is not None else 0.0)
    return np.array(dest_idx, dtype=np.int16), np.array(dest_target, dtype=np.float32)


def _destinations(board, size, max_slide):
    """(car, direction, steps, positions moved to) for every on-board destination of every car, in the order of
    _dest_labels's targets."""
    n_rows, n_cols = size
    for car, cells in board:
        horizontal = cells[0][0] == cells[1][0]
        for direction in ("lr" if horizontal else "ud"):
            di, dj = DELTAS[direction]
            for steps in range(1, max_slide + 1):
                moved = tuple((i + di * steps, j + dj * steps) for i, j in cells)
                if not all(0 <= i < n_rows and 0 <= j < n_cols for i, j in moved):
                    break
                yield car, direction, steps, moved


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


def erase_loops(boards, moves, window=None):
    """An attempt's boards (start first) and moves with loops cut out: a
    board seen again rewinds the path to its earlier visit. With `window`,
    only a board last seen at most `window` moves earlier is recognized - a
    longer loop goes unnoticed and stays in the path (hindsight limited to
    recent boards); None recognizes every board."""
    kept = _kept_moves(boards, window)
    return [boards[0]] + [boards[t + 1] for t in kept], [moves[t] for t in kept]


def _kept_moves(boards, window=None):
    """The indices of the moves erase_loops keeps, in order."""
    kept, position, seen = [], {boards[0]: 0}, {boards[0]: 0}
    for t, board in enumerate(boards[1:], 1):
        if board in position and (window is None or t - seen[board] <= window):
            kept = kept[:position[board]]
            position = {boards[0]: 0} | {boards[k + 1]: i + 1 for i, k in enumerate(kept)}
        else:
            position[board] = len(kept) + 1
            kept.append(t - 1)
        seen[board] = t
    return kept


def shortcut_routes(boards, moves, taught=None, everywhere=False, window=None):
    """Shortcut labels for a solved attempt (`boards`, start first, ending solved; `moves`; `taught`, per move,
    whether a teacher showed it): hindsight over everything the student saw. The visited boards form a graph - two
    are joined when one legal move turns one into the other (moves are reversible) - and the shortest route from a
    board to the solved board through it is often far shorter than the path played, even with loops cut out.
    `window` limits the hindsight: two boards are joined only if the student was at them within `window` moves of
    each other (None: any), so a connection between boards far apart in the attempt goes unnoticed; the moves
    played are always joined, so every visited board still gets a label.
    Returns (boards, moves, taught) routes for attempt_examples: from each labelled board, one shortest route per
    first move that starts one (so moves tied for shortest are all targets). Labelled boards: the ones on shortest
    routes from the start, or with `everywhere` every visited board that reaches the solved board. A route's move is
    taught when the teacher showed that move on that board in the attempt."""
    taught = taught or [False] * len(moves)
    shown = {(b, m) for b, m, t in zip(boards, moves, taught) if t}
    visited = set(boards)
    goal = boards[-1]
    orient = car_orientations(boards[0])
    if window is None:
        adj = {b: [(m, n) for m, n in neighbors(b, orient) if n in visited] for b in visited}
    else:
        times = {}
        for t, b in enumerate(boards):
            times.setdefault(b, []).append(t)

        def near(a, b):
            return any(abs(x - y) <= window for x in times[a] for y in times[b])

        adj = {b: [(m, n) for m, n in neighbors(b, orient) if n in visited and near(b, n)] for b in visited}
    dist, queue = {goal: 0}, [goal]
    for b in queue:
        for _, n in adj[b]:
            if n not in dist:
                dist[n] = dist[b] + 1
                queue.append(n)

    def route(b):
        bs, ms = [b], []
        while dist[b] > 0:
            m, b = next((m, n) for m, n in adj[b] if dist.get(n) == dist[b] - 1)
            ms.append(m)
            bs.append(b)
        return bs, ms

    if everywhere:
        starts = [b for b in visited if dist.get(b, 0) > 0]
    else:
        starts, frontier = set(), {boards[0]} if dist[boards[0]] > 0 else set()
        while frontier:
            starts |= frontier
            frontier = {n for b in frontier for _, n in adj[b] if dist.get(n) == dist[b] - 1 and dist[n] > 0}
    routes = []
    for b in starts:
        for m, n in adj[b]:
            if dist.get(n) == dist[b] - 1:
                bs, ms = route(n)
                bs, ms = [b] + bs, [m] + ms
                routes.append((bs, ms, [(x, y) in shown for x, y in zip(bs, ms)]))
    return routes


LABELS = ("own", "path", "all")


def solved_routes(result, labels="own", window=None):
    """What a solved attempt (a solve_intuition result) labels, as (boards, moves, taught) routes for
    attempt_examples. `labels` "own": the path it played with loops cut out (erase_loops, recognizing boards within
    `window` moves; None = all); "path" / "all": shortcut labels (shortcut_routes - the boards on shortest routes from
    the start, or every visited board; connections noticed only between boards visited within `window` moves of
    each other, None = any)."""
    boards, moves, taught = result["boards"], result["moves"], result["taught"]
    if labels == "own":
        kept = _kept_moves(boards, window)
        return [([boards[0]] + [boards[t + 1] for t in kept], [moves[t] for t in kept], [taught[t] for t in kept])]
    if labels not in LABELS:
        raise ValueError(f"labels: one of {LABELS}, not {labels!r}")
    return shortcut_routes(boards, moves, taught, everywhere=labels == "all", window=window)


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


def _cap(max_steps, state):
    """The move cap for an attempt from `state`: `max_steps` itself, or
    max_steps(state) when it's a callable (e.g. a cap scaled by the puzzle's
    optimal length)."""
    return max_steps(state) if callable(max_steps) else max_steps


def _shown_example(board, moves, size, max_slide):
    """Example from a teacher showing `moves` on `board`: the moves are the
    targets; there is no distance or destination target."""
    sd = dict(board)
    legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
    optimal = sorted(_move_index(sd, *m, size[1], max_slide) for m in moves)
    return Example(board, StateEncoder(size).encode_numpy(board), None, legal, optimal,
                   np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.float32))


def intuition_examples(results, size=(6, 6), max_slide=5, window=None, encode=1.0, rng=None, labels="own"):
    """Examples from one puzzle's solve_intuition attempts (`results`). Solved
    attempts teach their boards along solved_routes(..., `labels`, `window`):
    by default the path played with loops cut out (erase_loops, recognizing
    boards within `window` moves; None = all), or shortcut labels ("path",
    "all") - see attempt_examples. Every
    move the teacher showed teaches too, solved or not: its board with that
    move as the target, unless a solved attempt already labels the board.

    `encode` < 1: encoding lapses - each board label from a solved attempt
    and each move shown is learned only with probability `encode` (drawn
    from `rng`); at 1 no random numbers are drawn."""
    rng = rng or random

    def kept():
        return encode >= 1 or rng.random() < encode

    attempts, shown = [], {}
    for result in results:
        for board, move, was_taught in zip(result["boards"], result["moves"], result["taught"]):
            if was_taught and kept():
                shown.setdefault(board, set()).add(move)
        if result["solved"]:
            attempts += [(bs, ms) for bs, ms, _ in solved_routes(result, labels, window)]
    examples = [e for e in attempt_examples(attempts, size, max_slide) if kept()]
    covered = {e.state for e in examples}
    examples += [_shown_example(board, moves, size, max_slide) for board, moves in shown.items()
                 if board not in covered]
    return examples


def _intuition_attempts(state0, n_attempts, seed, max_steps, size, max_slide, agent, mode):
    """attempt_puzzle for the pure-intuition student (see solve_intuition),
    with the mode's teacher, if any, showing optimal moves along the way."""
    rng = random.Random(seed)
    teacher = mode.get("teacher")
    if teacher is not None:
        teacher.counts = Counter()  # this puzzle's own tally (jobs can share a teacher object)
    results = []
    for _ in range(n_attempts):
        if teacher is not None:
            teacher.start(state0, rng)
        results.append(solve_intuition(agent, state0, max_steps, mode["memory"], mode["explore"], teacher, rng))
        if teacher is not None:
            teacher.end_attempt(results[-1])
    examples = intuition_examples(results, size, max_slide)
    return examples, [], {"solved": sum(r["solved"] for r in results),
                          "taught": sum(sum(r["taught"]) for r in results),
                          "moves": sum(len(r["moves"]) for r in results),
                          "interventions": sum(sum(r["taught"]) for r in results),
                          "teacher_counts": dict(teacher.counts) if teacher is not None else {}}


def attempt_puzzle(job):
    """(Examples, SubgoalExamples, info) for one puzzle from `n_attempts`
    attempts by the student `mode` describes (see label_attempts). info
    counts solved attempts, moves the teacher showed and moves played.

    - "and_or" student: AND-OR attempts at lapse probability `gamma`, guided
      by the worker's agent if it has one. Only solved attempts teach: their
      boards, loops cut out (attempt_examples), and every subgoal of every
      plan they completed, with the option that resolved it as the target.
    - "intuition" student: the worker's agent playing by intuition alone
      (solve_intuition), sampling at the mode's "explore" temperature, with
        the mode's "teacher" (see teach.py) free to show optimal moves.
      Solved attempts teach as above (no subgoals: there is no search); every
      move the teacher showed teaches too, solved or not - its board with
      that move as the target, unless a solved attempt already labels the
      board."""
    state0, n_attempts, gamma, seed, max_steps, (size, max_slide), mode = job
    random.seed(seed)
    agent = _WORKER.get("agent")
    if mode["student"] == "intuition":
        return _intuition_attempts(state0, n_attempts, seed, max_steps, size, max_slide, agent, mode)
    heuristic = agent.heuristic if agent is not None else None
    attempts, subgoals, played = [], [], 0
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
            played += len(moves)
            if _red_steps(state) <= 0:
                attempts.append(erase_loops(boards, moves))
                for node, chosen in choices:
                    example = _subgoal_example(node, {chosen}, size, max_slide)
                    if example is not None:
                        subgoals.append(example)
    return attempt_examples(attempts, size, max_slide), subgoals, {"solved": len(attempts), "taught": 0,
                                                                   "moves": played}


def _agent_payload(agent):
    """(weights as bytes, config) for rebuilding a CPU copy of `agent` in a
    worker (see _init_attempt_worker)."""
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in agent.net.state_dict().items()}, buffer)
    return buffer.getvalue(), (agent.size, agent.max_slide, agent.hidden, agent.blocks, agent.temperature)


def label_attempts(puzzles, n_attempts=4, gamma=0.07, agent=None, seed=0, max_steps=200, processes=None,
                   size=(6, 6), max_slide=5, student="and_or", teacher=None, explore=1.0, memory=10):
    """Labels from the student's own attempts on every puzzle (in parallel):
    (Examples, SubgoalExamples, info) - see attempt_puzzle. info sums
    attempts made, attempts solved, moves the teacher showed ("taught"; also
    "interventions", the teaching budget spent), moves played, and the
    teacher's own counts ("teacher_counts", see teach.py).

    `student` "and_or": with `agent`, attempts are made by the human-like
    solver it guides (its heuristic, at its temperature); without, by plain
    AND-OR. `student` "intuition": `agent` (required) plays by intuition
    alone, sampling at temperature `explore` with recent-board memory
    `memory`, and `teacher` (see teach.py), if given, may show optimal moves.
    `size` and `max_slide` (the agent's, when one is given) set the action
    indexing. `max_steps` caps each attempt's moves: a number, or a callable
    giving each puzzle's cap from its starting state."""
    if student == "intuition" and agent is None:
        raise ValueError("the intuition student needs an agent")
    if teacher is not None and student != "intuition":
        raise ValueError("teachers are only wired to the intuition student")
    weights, config = None, None
    if agent is not None:
        size, max_slide = agent.size, agent.max_slide
        weights, config = _agent_payload(agent)
    mode = {"student": student, "teacher": teacher, "explore": explore, "memory": memory}
    jobs = [(state, n_attempts, gamma, hash((seed, i)) & 0xFFFFFFFF, _cap(max_steps, state), (size, max_slide), mode)
            for i, state in enumerate(puzzles)]
    examples, subgoals = [], []
    info = {"attempts": len(puzzles) * n_attempts, "solved": 0, "taught": 0, "moves": 0, "interventions": 0,
            "teacher_counts": Counter()}
    with Pool(processes, initializer=_init_attempt_worker, initargs=(weights, config)) as pool:
        for ex, sg, counts in tqdm(pool.imap(attempt_puzzle, jobs, chunksize=2), total=len(jobs),
                                   desc="attempts"):
            examples += ex
            subgoals += sg
            for k, v in counts.items():
                if k == "teacher_counts":
                    info[k].update(v)
                else:
                    info[k] += v
    info["teacher_counts"] = dict(info["teacher_counts"])
    return examples, subgoals, info


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
    distance = torch.tensor([e.distance if e.distance is not None else float("nan") for e in examples],
                            dtype=torch.float32, device=device)  # nan: no distance target
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
    return x, distance, legal.to(device), optimal.to(device), dest_mask.to(device), dest_target.to(device)


def _loss(agent, examples):
    """Policy loss (probability on the target moves), plus dest and value
    losses where there are targets."""
    x, distance, legal, optimal, dest_mask, dest_target = _batch(agent, examples)
    policy, dest, value = agent.net.heads(x)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    loss = -logp.masked_fill(~optimal, float("-inf")).logsumexp(1).mean()
    if dest_mask.any():
        loss = loss + F.binary_cross_entropy_with_logits(dest[dest_mask], dest_target[dest_mask])
    has_distance = ~torch.isnan(distance)
    if has_distance.any():
        loss = loss + 0.5 * F.smooth_l1_loss(value[has_distance], distance[has_distance] / 10)
    return loss


def _weighted_loss(agent, examples, weights):
    """_loss with a weight per example: a number scales its move, dest and distance targets; a pair (number, array)
    gives its dest targets their own weights (one per dest_idx entry, in order). Each term is still a mean over the
    batch's targets, so weights below 1 mean less learning, not the same learning shared out differently."""
    x, distance, legal, optimal, dest_mask, dest_target = _batch(agent, examples)
    policy, dest, value = agent.net.heads(x)
    w = torch.tensor([a[0] if isinstance(a, tuple) else a for a in weights], dtype=torch.float32, device=x.device)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    loss = (-logp.masked_fill(~optimal, float("-inf")).logsumexp(1) * w).mean()
    if dest_mask.any():
        w_dest = torch.zeros(dest.shape)
        for r, (e, a) in enumerate(zip(examples, weights)):
            w_dest[r, torch.from_numpy(e.dest_idx.astype(np.int64))] = \
                torch.from_numpy(a[1]) if isinstance(a, tuple) else float(w[r])
        terms = F.binary_cross_entropy_with_logits(dest[dest_mask], dest_target[dest_mask], reduction="none")
        loss = loss + (terms * w_dest.to(x.device)[dest_mask]).mean()
    has_distance = ~torch.isnan(distance)
    if has_distance.any():
        terms = F.smooth_l1_loss(value[has_distance], distance[has_distance] / 10, reduction="none")
        loss = loss + 0.5 * (terms * w[has_distance]).mean()
    return loss


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
    - agreement: how often the policy head's top legal move is a target
      (over examples that have one);
    - chance: the same for a uniformly random legal move;
    - distance_error: mean absolute error of the value head, in moves.
    """
    if not examples:
        return {"agreement": float("nan"), "chance": float("nan"), "distance_error": float("nan")}
    agent.net.eval()
    hits = chance = err = 0.0
    with_distance = with_target = 0
    for i in range(0, len(examples), batch_size):
        x, distance, legal, optimal, _, _ = _batch(agent, examples[i:i + batch_size])
        policy, _, value = agent.net.heads(x)
        top = policy.masked_fill(~legal, float("-inf")).argmax(1)
        has_target = optimal.any(1)
        hits += optimal.gather(1, top[:, None])[has_target].float().sum().item()
        chance += (optimal.sum(1).float() / legal.sum(1).float())[has_target].sum().item()
        with_target += int(has_target.sum())
        has_distance = ~torch.isnan(distance)
        err += (value * 10 - distance)[has_distance].abs().sum().item()
        with_distance += int(has_distance.sum())
    agent.net.train()
    return {"agreement": hits / with_target if with_target else float("nan"),
            "chance": chance / with_target if with_target else float("nan"),
            "distance_error": err / with_distance if with_distance else float("nan")}


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
    """Mean of a train_policy history entry's scores, ignoring missing (nan)
    ones; None if none are left (nothing to validate on)."""
    values = [v for v in scores.values() if v == v]
    return sum(values) / len(values) if values else None


def _kept_epoch(history):
    """The epoch (from 1) whose weights train_policy keeps: the first with the
    best validation score - or, for epochs with nothing to validate on,
    simply the latest."""
    best, kept = -1.0, 1
    for epoch, scores in enumerate(history, 1):
        score = _validation_score(scores)
        if score is None or score > best:
            best = best if score is None else score
            kept = epoch
    return kept


def train_policy(agent, examples, val_examples, subgoals=(), val_subgoals=(), epochs=40, patience=6,
                 batch_size=256, on_epoch=None):
    """Supervised training: on board Examples, the policy head puts
    probability on any target move, the dest head predicts each
    destination's target and the value head the distance; on
    SubgoalExamples (`subgoals`), the dest head ranks each subgoal's target
    options first - the ordering AND-OR uses. Board and subgoal batches are
    interleaved at random.

    Keeps the weights with the best validation score - the policy head's
    agreement on `val_examples`, averaged with the ordering agreement on
    `val_subgoals` when given (both from puzzles held out of training) - and
    stops after `patience` epochs without improvement. With nothing to
    validate on (e.g. no validation attempt solved), it keeps each epoch's
    weights as it goes and doesn't stop early. Returns per-epoch
    validation scores as dicts.

    `on_epoch`, if given, is called as on_epoch(epoch, scores) after each
    epoch (from 1), on that epoch's own weights - e.g. to save checkpoints."""
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
        if on_epoch is not None:
            on_epoch(len(history), scores)
        score = _validation_score(scores)
        bar.set_postfix({k: f"{v:.1%}" for k, v in scores.items()})
        if score is None or score > best:
            best = best if score is None else score
            best_state, stale = {k: v.clone() for k, v in agent.net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    agent.net.load_state_dict(best_state)
    agent._scores = {}
    return history


def expert_iteration(agent, puzzles, val_puzzles, rounds=3, n_attempts=4, gamma=0.07, max_steps=200, seed=0,
                     checkpoint_dir=None, checkpoint_epochs=True, student="and_or", teacher=None, explore=1.0,
                     memory=10, resume=False, **train_kwargs):
    """Trains `agent` on the student's own attempts, by expert iteration. Each
    round labels attempts on its training puzzles and on `val_puzzles`
    (validation) with label_attempts and keeps training the same weights on
    them, so the examples improve as the agent does. `puzzles` is either one
    list of starting states, used every round, or a list of such lists, one
    per round (e.g. fresh puzzles each round). `max_steps` caps each
    attempt's moves - a number, or a callable giving a puzzle's cap from its
    starting state.

    `student` "and_or": round 1's attempts are plain AND-OR; each later
    round's come from the human-like solver the agent has become (its
    heuristic, at its temperature). No breadth-first search is involved.
    `student` "intuition": every round's attempts are the agent's own
    intuition (solve_intuition, sampling at temperature `explore`), with
    `teacher` (see teach.py) - which knows the optimal moves - free to show
    them, on validation attempts too.

    Returns one dict per round: attempts made and solved on the training
    puzzles, the share and number of their moves the teacher showed
    ("taught", "taught_moves") out of all moves played ("moves"), its
    interventions - the teaching budget ("interventions") - the teacher's own counts
    ("teacher_counts", see teach.py), boards and
    subgoals learned from, the best validation scores (agreement with the
    round's own attempts) and the epoch they came from.

    `checkpoint_dir`, if given, gets the learning trajectory: the agent
    before training (round0_epoch00.pt) and after every epoch of every round
    (round{r}_epoch{e}.pt), plus manifest.json listing them in training
    order - round, epoch, validation scores, and "kept": whether these are
    the weights the round ended on (its best epoch), which the next round
    starts from. The manifest is rewritten after every round, along with
    history.json (this function's return value so far), so a run can be
    re-evaluated later without retraining. With `checkpoint_epochs` False,
    only the untrained agent and each round's kept weights are saved.

    `resume`: continue the run saved in `checkpoint_dir` - load its history,
    manifest and last kept weights into `agent` and train from the next
    round up to `rounds` in all (the optimizer starts fresh). Without a
    saved run there, it starts from scratch."""
    per_round = bool(puzzles) and isinstance(puzzles[0], list)
    if per_round and len(puzzles) < rounds:
        raise ValueError(f"{len(puzzles)} rounds of puzzles for {rounds} rounds")
    history, manifest = [], []

    def checkpoint(round_, epoch, scores):
        name = f"round{round_}_epoch{epoch:02d}.pt"
        agent.save(os.path.join(checkpoint_dir, name))
        manifest.append({"file": name, "round": round_, "epoch": epoch, "kept": round_ == 0, **scores})

    saved = checkpoint_dir is not None and os.path.exists(os.path.join(checkpoint_dir, "history.json"))
    if resume and saved:
        with open(os.path.join(checkpoint_dir, "history.json")) as f:
            history = json.load(f)
        with open(os.path.join(checkpoint_dir, "manifest.json")) as f:
            manifest = json.load(f)
        last = max((m for m in manifest if m["kept"]), key=lambda m: m["round"])
        agent.net.load_state_dict(PolicyAgent.load(os.path.join(checkpoint_dir, last["file"])).net.state_dict())
        agent._scores = {}
    elif checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint(0, 0, {})
    for r in range(len(history), rounds):
        round_puzzles = puzzles[r] if per_round else puzzles
        guide = agent if r > 0 or student == "intuition" else None
        settings = {"size": agent.size, "max_slide": agent.max_slide, "student": student, "teacher": teacher,
                    "explore": explore, "memory": memory}
        examples, subgoals, info = label_attempts(round_puzzles, n_attempts, gamma, guide, seed + 2 * r, max_steps,
                                                  **settings)
        val_examples, val_subgoals, _ = label_attempts(val_puzzles, n_attempts, gamma, guide, seed + 2 * r + 1,
                                                       max_steps, **settings)
        on_epoch = ((lambda epoch, s, round_=r + 1: checkpoint(round_, epoch, s))
                    if checkpoint_dir is not None and checkpoint_epochs else None)
        scores = train_policy(agent, examples, val_examples, subgoals, val_subgoals, on_epoch=on_epoch,
                              **train_kwargs)
        best_epoch = _kept_epoch(scores)
        best = scores[best_epoch - 1]
        history.append({"round": r + 1, "attempts": info["attempts"], "solved": info["solved"],
                        "taught": info["taught"] / info["moves"] if info["moves"] else float("nan"),
                        "taught_moves": info["taught"], "moves": info["moves"],
                        "interventions": info["interventions"], "teacher_counts": info["teacher_counts"],
                        "boards": len(examples), "subgoals": len(subgoals), "best_epoch": best_epoch, **best})
        if checkpoint_dir is not None:
            if not checkpoint_epochs:
                checkpoint(r + 1, best_epoch, best)  # train_policy has restored the kept weights
            for entry in manifest:
                if entry["round"] == r + 1 and entry["epoch"] == best_epoch:
                    entry["kept"] = True
            with open(os.path.join(checkpoint_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=1)
            with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
    return history


def agency_weights(examples, results, shown_weight, size=(6, 6), max_slide=5, window=None, labels="own"):
    """Loss weights (see _weighted_loss) for intuition_examples(results, ...): what a teacher's shown move taught
    counts `shown_weight`, the student's own play 1 - with `shown_weight` < 1 it learns less from being shown a move
    than from finding one itself (the generation effect), at 0 nothing at all. An example on a board where a move
    was shown weighs `shown_weight` throughout; on a solved path's other boards, so do the dest targets of
    destinations a shown move first brought a car to (the rest of the example weighs 1; with several routes from a
    board, as shortcut labels have, a destination any of them got from a shown move). `size`, `max_slide`, `window`
    and `labels` as given to intuition_examples."""
    shown = {b for r in results for b, t in zip(r["boards"], r["taught"]) if t}
    given = {}  # path board -> destinations a shown move first brought a car to, later on the path
    for r in results:
        if r["solved"]:
            for path, moves, taught in solved_routes(r, labels, window):
                for i, board in enumerate(path[:-1]):
                    first = {}
                    for j in range(i, len(moves)):
                        car = moves[j][0]
                        first.setdefault((car, dict(path[j + 1])[car]), taught[j])
                    given.setdefault(board, set()).update(entry for entry, t in first.items() if t)
    weights = []
    for e in examples:
        if e.state in shown:
            weights.append(shown_weight)
        elif given.get(e.state):
            dest = [shown_weight if (car, moved) in given[e.state] else 1.0
                    for car, _, _, moved in _destinations(e.state, size, max_slide)]
            weights.append((1.0, np.array(dest, dtype=np.float32)))
        else:
            weights.append(1.0)
    return weights


def progress_examples(results, judge, recent, tail, size=(6, 6), max_slide=5, window=None, labels="own"):
    """Learning from its own moves, solved or not, by its sense of progress: `judge`'s value head (moves to go; a
    frozen copy of the student's starting network) rates each move the student made itself by the moves to go
    before it minus after. Among its last moves (`recent`, a deque this adds to), the clearest `tail` share each way
    are labelled: a step back teaches "not that move here" (the targets are every other legal move - the policy loss
    then pushes that move's probability down), a step forward "that move here" (it is the target). Moves a solved
    path already labels (solved_routes with `labels` and `window`) and moves a teacher showed get no label; a step
    back needs another legal move. Returns (examples, {"forward": n, "back": n})."""
    covered = set()
    for r in results:
        if r["solved"]:
            covered |= {(b, m) for bs, ms, _ in solved_routes(r, labels, window) for b, m in zip(bs, ms)}
    encoder = StateEncoder(size)
    judge.net.eval()
    moves = []
    for r in results:
        with torch.no_grad():
            v = judge.net.heads(torch.from_numpy(np.stack([encoder.encode_numpy(b) for b in r["boards"]])).float()
                                .to(judge.device))[2].cpu().numpy() * 10
        for t, (move, taught) in enumerate(zip(r["moves"], r["taught"])):
            if not taught:
                recent.append(v[t] - v[t + 1])
                moves.append((r["boards"][t], move, v[t] - v[t + 1]))
    if len(recent) < 50:
        return [], {"forward": 0, "back": 0}
    lo, hi = np.quantile(recent, [tail, 1 - tail])
    examples, counts = [], {"forward": 0, "back": 0}
    for board, move, progress in moves:
        if (board, move) in covered or lo < progress < hi:
            continue
        sd = dict(board)
        legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
        idx = _move_index(sd, *move, size[1], max_slide)
        targets = [idx] if progress >= hi else [i for i in legal if i != idx]
        if targets:
            examples.append(Example(board, encoder.encode_numpy(board), None, legal, targets,
                                    np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.float32)))
            counts["forward" if progress >= hi else "back"] += 1
    return examples, counts


def online_update(agent, examples, steps=1, batch_size=256, weights=None, agency_step="off", rng=None,
                  even_batches=False):
    """Learns from `examples` the way a person learns from one puzzle: `steps`
    passes over just these examples (one gradient step per batch), no replay
    of anything earlier, no validation, nothing to rewind to. `weights`: one
    per example (see _weighted_loss); None trains them all alike (_loss).

    Adam divides each step by the recent size of the gradients, so it cancels a
    weight shared by a whole batch: with `agency_step` "off" the weights only
    set how examples count against each other, and a batch of shown moves at
    weight 0.1 learns as much as one of the student's own. The other settings
    make the weights also set how much is learned, by the batch's mean move
    weight m (the number, or a pair's first entry); its loss is divided by m,
    so only the step depends on it:
    - "scale": the step is scaled by m (a tenth of a step at m = 0.1);
    - "skip": the step is taken with probability m, else skipped (a full step
      one time in ten at m = 0.1) - the same learning on average.
    At m = 0 no step is taken. `rng` (default: the random module) shuffles and
    skips. With agent.train_on(device), the steps run on a copy of the network
    on that device, and the new weights are copied back at the end.

    Batches: by default `batch_size` examples each, the last holding what is left - which Adam steps as far
    as a full one, so a pass over 517 examples takes two steps on 256 and a third on just 5. `even_batches`
    splits each pass into the same number of batches, ceil(n / batch_size), of equal size instead (3 of
    about 172): the same number of steps, none of them on a handful of examples."""
    assert agency_step in ("off", "scale", "skip"), agency_step
    rng = rng or random
    examples = list(examples) if weights is None else list(zip(examples, weights))
    costly = agency_step != "off" and weights is not None
    player, agent = agent, agent._learner()  # with train_on, train the copy on the training device
    base = [g["lr"] for g in agent.optimizer.param_groups]
    agent.net.train()
    try:
        for _ in range(steps):
            rng.shuffle(examples)
            if even_batches:
                k = max(1, math.ceil(len(examples) / batch_size))
                bounds = [round(j * len(examples) / k) for j in range(k + 1)]
            else:
                bounds = list(range(0, len(examples), batch_size)) + [len(examples)]
            for i, j in zip(bounds[:-1], bounds[1:]):
                batch = examples[i:j]
                if costly:
                    mean = float(np.mean([w[0] if isinstance(w, tuple) else w for _, w in batch]))
                    if mean <= 0 or agency_step == "skip" and rng.random() >= mean:
                        continue
                    for g, lr in zip(agent.optimizer.param_groups, base):
                        g["lr"] = lr * (mean if agency_step == "scale" else 1.0)
                if weights is None:
                    loss = _loss(agent, batch)
                else:
                    loss = _weighted_loss(agent, [e for e, _ in batch], [w for _, w in batch])
                    if costly:
                        loss = loss / mean
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
    finally:
        for g, lr in zip(agent.optimizer.param_groups, base):
            g["lr"] = lr
    if agent is not player:
        player.net.load_state_dict({k: v.to(player.device) for k, v in agent.net.state_dict().items()})
    player._scores = {}


def learning_session(agent, puzzles, teacher=None, max_attempts=1, max_steps=200, memory=10, explore=1.0,
                     window=10, encode=1.0, steps=1, block=1, seed=0, on_puzzle=None,
                     surrender=None, shown_weight=1.0, judge=None, progress_weight=0.0, progress_tail=0.2,
                     labels="own", agency_step="off", even_batches=False):
    """A session in which a pure-intuition student (`agent`) learns online, a
    few puzzles at a time, under human-like limits on learning:

    - online: after each block of `block` puzzles it learns from those
      puzzles' attempts only (online_update, `steps` passes), and plays the
      next block with what it has become - nothing earlier is replayed;
    - encoding lapses: each label and move shown is learned with
      probability `encode` (see intuition_examples);
    - limited hindsight: loops are cut out of a solved attempt only when it
      returned to a board seen within the last `window` moves (None: any);
      or, with `labels` "path" / "all", shortcut labels instead: shortest
      routes through every board the attempt visited (shortcut_routes;
      `window` then limits which connections between visited boards are noticed);
    - no rewinding: no validation, no keeping an earlier, better version.

    `block` 1 learns after every puzzle. A larger block gives the network
    more data per update - a way to grant it more experience than a person
    would need for the same progress (see session_experiment.py's
    --multiplier).

    Each puzzle (a starting state) gets up to `max_attempts` attempts
    (solve_intuition, sampling at temperature `explore`, avoiding the last
    `memory` boards, with `teacher` if given), stopping at the first solve -
    a restart is a new attempt. `max_steps` caps each attempt: a number or a
    callable giving a puzzle's cap from its state. With `surrender` (a
    SurrenderRule), the student may also give up on a puzzle - counting the
    moves of all its attempts at it - which ends the puzzle: no restart.

    Agency (`shown_weight`, see agency_weights): what a shown move teaches
    counts this much, the student's own play 1. Progress labels (`judge`, an
    agent whose value head rates moves - see progress_examples): with
    `progress_weight` > 0 the student also learns from its own moves, solved
    or not, the clearest `progress_tail` share each way of its last 1000,
    each label weighing `progress_weight`. At the defaults (1, no judge) it
    learns as before. `agency_step` "scale" or "skip" (see online_update): weights below 1
    also shrink the update, not only how examples count against each other -
    with "off", a block where every move was shown learns at full strength.
    `even_batches` (see online_update): each update's batches of equal size, not 256 and a remainder.

    Returns one dict per puzzle, in order: puzzle (index), solved, surrendered, attempts,
    moves (all attempts), taught, interventions, examples learned
    from (progress labels included), progress_labels (by direction, with a
    judge). `on_puzzle(n, agent)`, if given, is called after each update with
    the number of puzzles played so far - e.g. to snapshot the student."""
    rng = random.Random(seed)
    # the updates' own shuffling and skipping, apart from play; "off" keeps the random module, as before
    update_rng = random.Random(seed + 104729) if agency_step != "off" else None
    record = []
    pending, pending_w = [], []
    progress = judge is not None and progress_weight > 0
    weighted = shown_weight != 1 or progress
    recent = deque(maxlen=1000)
    for i, state in enumerate(puzzles):
        results = []
        if teacher is not None:
            teacher.counts = Counter()
        for _ in range(max_attempts):
            if teacher is not None:
                teacher.start(state, rng)
            gives_up = surrender.for_puzzle(state, sum(len(r["moves"]) for r in results)) if surrender else None
            results.append(solve_intuition(agent, state, _cap(max_steps, state), memory, explore, teacher, rng,
                                           gives_up))
            if teacher is not None:
                teacher.end_attempt(results[-1])
            if results[-1]["solved"] or results[-1]["surrendered"]:
                break
        examples = intuition_examples(results, agent.size, agent.max_slide, window, encode, rng, labels)
        if weighted:
            pending_w += agency_weights(examples, results, shown_weight, agent.size, agent.max_slide, window, labels)
        extra, counts = progress_examples(results, judge, recent, progress_tail, agent.size, agent.max_slide,
                                          window, labels) if progress else ([], None)
        examples += extra
        pending_w += [progress_weight] * len(extra)
        pending += examples
        record.append({"puzzle": i, "solved": results[-1]["solved"], "surrendered": results[-1]["surrendered"],
                       "attempts": len(results),
                       "moves": sum(len(r["moves"]) for r in results),
                       "taught": sum(sum(r["taught"]) for r in results),
                       "interventions": sum(sum(r["taught"]) for r in results), "examples": len(examples),
                       "teacher_counts": dict(teacher.counts) if teacher is not None else {}}
                      | ({"progress_labels": counts} if progress else {}))
        if (i + 1) % block == 0 or i + 1 == len(puzzles):
            if pending:
                online_update(agent, pending, steps, weights=pending_w if weighted else None,
                              agency_step=agency_step, rng=update_rng, even_batches=even_batches)
            pending, pending_w = [], []
            if on_puzzle is not None:
                on_puzzle(i + 1, agent)
    return record


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


def solve_arbitrated(agent, state, max_steps=200, gamma=0.0, threshold=0.5, memory=10, temperature=1.0):
    """Solves from `state` by arbitrating, every turn, between intuition and
    deliberation - the human-like solver. Returns (moves if solved else
    None, stats).

    - Intuition: a move sampled from softmax(logits / temperature) with the
      random module - at the default 1, in proportion to the policy head's
      own probabilities; at 0, always its top legal move. Played when the
      top move's probability (softmax over legal moves, the confidence) is
      at least `threshold` and the chosen move doesn't lead back to one of
      the last `memory` boards (None = every board this attempt). One move.
    - Deliberation: otherwise, one and_or.solve pass ordered by the agent's
      heuristic, with lapse probability `gamma`; its moves are played. A
      move that would revisit a recent board counts as low confidence -
      going in circles is a reason to stop and think.

    `temperature` is intuition's noise only; the agent's own `temperature`
    is deliberation's (see PolicyAgent). threshold 0 is pure intuition
    (apart from revisits); above 1, pure deliberation. stats counts
    "intuitive" moves, "deliberations" (passes) and "deliberate" moves."""
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
            choice = best
            if temperature > 0:
                choice = random.choices(range(len(options)), torch.softmax(logits / temperature, dim=0).tolist())[0]
            move, nxt = options[choice]
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


def solve_intuition(agent, state, max_steps=200, memory=10, temperature=0.0, teacher=None, rng=None,
                    surrender=None):
    """Pure intuition, no search: every turn the policy head chooses among the
    legal moves that don't return to one of the last `memory` boards (all
    legal moves if every one does) - its top move at `temperature` 0, else a
    move sampled from softmax(logits / temperature) with `rng` (default: the
    random module). With a `teacher` (see teach.py), each turn the teacher
    may show the optimal move instead, which is played; it can look at the
    move the student would have played (the archived CorrectMistakes did).

    `surrender`, if given, is a function of the number of moves played so
    far: after each move that doesn't solve the puzzle, the student gives
    up with that probability (drawn from `rng`) - see SurrenderRule.

    Returns {"solved", "surrendered", "boards" (start first), "moves",
    "taught" (per move, whether the teacher showed it)}."""
    rng = rng or random
    orientations = car_orientations(state)
    recent = deque([state], maxlen=memory)
    boards, moves, taught = [state], [], []
    while len(moves) < max_steps and _red_steps(state) > 0:
        options = dict(neighbors(state, orientations))  # move -> board it leads to
        proposed = []

        def propose():
            """The student's own move on this board, chosen once, when first asked."""
            if not proposed:
                fresh = [m for m, nxt in options.items() if nxt not in recent] or list(options)
                logits = agent._evaluate(state)[0][[agent.action_id(state, *m) for m in fresh]]
                if temperature > 0:
                    proposed.append(rng.choices(fresh, torch.softmax(logits / temperature, dim=0).tolist())[0])
                else:
                    proposed.append(fresh[int(logits.argmax())])
            return proposed[0]

        move = teacher.advise(state, len(moves), propose) if teacher is not None else None
        taught.append(move is not None)
        if move is None:
            move = propose()
        if teacher is not None:
            teacher.observe(state, move, options[move], taught[-1])
        state = options[move]
        boards.append(state)
        moves.append(move)
        recent.append(state)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(len(moves)):
            return {"solved": False, "surrendered": True, "boards": boards, "moves": moves, "taught": taught}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "boards": boards, "moves": moves,
            "taught": taught}


class SurrenderRule():
    """When a student gives up on a puzzle, as people do: after each move, it
    surrenders with a probability that depends on how many moves it has made
    on the puzzle so far relative to the optimal length - `rates[i]` while
    moves / optimal is in [edges[i], edges[i + 1]) (the last rate beyond the
    last edge). The rates are fitted to the human data
    (human_data.surrender_hazard). `distances` maps each puzzle's starting
    state to its optimal length. Called as rule(state, moves): the
    probability of giving up after `moves` moves on the puzzle starting at
    `state`. `scale` multiplies every rate (capped at 1): the human shape
    with a student-specific level, fitted so the student's starting solve
    rate and efficiency match people's. A plain object, so it can be sent
    to worker processes."""

    def __init__(self, edges, rates, distances, scale=1.0):
        self.edges, self.distances, self.scale = list(edges), dict(distances), scale
        self.rates = [min(1.0, scale * r) for r in rates]

    def __call__(self, state, moves):
        ratio = moves / self.distances[state]
        i = 0
        while i + 1 < len(self.edges) and ratio >= self.edges[i + 1]:
            i += 1
        return self.rates[min(i, len(self.rates) - 1)]

    def for_puzzle(self, state, offset=0):
        """The surrender function solve_intuition takes, for an attempt at
        `state` after `offset` moves already made on it (earlier attempts)."""
        return lambda moves: self(state, offset + moves)


def _intuition_job(job):
    """One untaught solve_intuition attempt by the worker's agent."""
    state, distance, memory, max_steps, temperature, seed, *rest = job
    surrender = rest[0].for_puzzle(state) if rest and rest[0] is not None else None
    result = solve_intuition(_WORKER["agent"], state, max_steps, memory, temperature, rng=random.Random(seed),
                             surrender=surrender)
    return {"solved": result["solved"], "moves": len(result["moves"]) if result["solved"] else None,
            "distance": distance}


def evaluate_intuition(agent, puzzles, n_runs=3, temperature=1.0, memory=10, max_steps=200, seed=0,
                       processes=None, surrender=None):
    """`n_runs` untaught solve_intuition attempts on every (distance, state) in
    `puzzles`, in parallel (each worker gets a CPU copy of the agent): the
    student's behavior with no teacher, sampling its moves at `temperature`
    as it does while learning (0 = always its top move, deterministic).
    `max_steps` is a number or a callable giving each puzzle's cap from its
    state. `surrender`: a SurrenderRule, to let the student give up as
    people do (None: it never does). Returns one dict per attempt, puzzle by
    puzzle (`n_runs` each): solved, moves (None if unsolved), distance."""
    jobs = [(state, d, memory, _cap(max_steps, state), temperature, hash((seed, i, r)) & 0xFFFFFFFF, surrender)
            for i, (d, state) in enumerate(puzzles) for r in range(n_runs)]
    with Pool(processes, initializer=_init_attempt_worker, initargs=_agent_payload(agent)) as pool:
        return pool.map(_intuition_job, jobs, chunksize=2)


def _arbitrated_job(job):
    """One solve_arbitrated attempt by the worker's agent (see label_attempts' Pool setup)."""
    state, distance, gamma, threshold, memory, max_steps, temperature, seed = job
    random.seed(seed)
    moves, stats = solve_arbitrated(_WORKER["agent"], state, max_steps, gamma, threshold, memory, temperature)
    return {"solved": moves is not None, "moves": len(moves) if moves is not None else None,
            "distance": distance, **stats}


def evaluate_arbitrated(agent, puzzles, n_runs=3, gamma=0.0, threshold=0.5, memory=10, max_steps=200, seed=0,
                        processes=None, temperature=1.0):
    """solve_arbitrated `n_runs` times on every (distance, state) in `puzzles`,
    in parallel (each worker gets a CPU copy of the agent, at its
    temperature for deliberation; `temperature` is intuition's). Returns one
    dict per attempt, puzzle by puzzle: solved, moves (None if unsolved),
    distance, and the stats of solve_arbitrated."""
    jobs = [(state, d, gamma, threshold, memory, max_steps, temperature, hash((seed, i, r)) & 0xFFFFFFFF)
            for i, (d, state) in enumerate(puzzles) for r in range(n_runs)]
    with Pool(processes, initializer=_init_attempt_worker, initargs=_agent_payload(agent)) as pool:
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
