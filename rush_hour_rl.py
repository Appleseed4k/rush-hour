import copy
import random

from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rush_hour_and_or import solve as and_or_solve, legal_moves
from rush_hour_lib import MOVES, DELTAS

OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}


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
    """Encodes an Environment state as a fixed-size (channels, n_rows, n_cols)
    grid - each cell's occupant category (empty / red / horizontal car /
    vertical car) as a one-hot channel - so both the encoding and its shape
    depend only on board size, never on a puzzle's specific cars, letting one
    agent train across many puzzles."""
    CATEGORIES = ("empty", "red", "h", "v")

    def __init__(self, size=(6, 6)):
        self.n_rows, self.n_cols = size

    @property
    def shape(self):
        return len(self.CATEGORIES), self.n_rows, self.n_cols

    def encode(self, state):
        grid = torch.zeros(self.n_rows, self.n_cols, len(self.CATEGORIES))
        grid[:, :, 0] = 1.0
        for name, positions in state:
            horizontal = positions[0][0] == positions[1][0]
            category = 1 if name == "red" else (2 if horizontal else 3)
            for i, j in positions:
                grid[i, j, 0] = 0.0
                grid[i, j, category] = 1.0
        return grid.permute(2, 0, 1)  # (channels, n_rows, n_cols), as Conv2d expects


def valid_action_mask(environment, action_space):
    """Boolean mask over action_space of the moves currently legal in `environment`."""
    legal = set(legal_actions(environment, action_space))
    return torch.tensor([action in legal for action in action_space], dtype=torch.bool)


class PolicyNetwork(nn.Module):
    """Convolutional trunk over the board grid feeding a single policy head
    (action logits). Convolution's translation-invariant filters let a
    pattern like "this car can slide because the cells ahead are clear" be
    learned once and applied at every board position, instead of needing to
    be re-learned independently at each input the way a flat MLP would -
    the property that lets weights trained on one puzzle's layout transfer to
    another's.
    """
    def __init__(self, in_channels, n_rows, n_cols, out_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden * n_rows * n_cols, out_dim)

    def forward(self, x):
        features = self.trunk(x).flatten(start_dim=1)
        return self.policy_head(features)


class PolicyAgent():
    """Trained by supervised cross-entropy against AND-OR traces (see
    train_policy) rather than TD bootstrapping. Its action space is keyed by
    board cell rather than car name (see car_anchor), so the same agent -
    same input/output dimensions, same weights - applies to any puzzle of
    this board size regardless of how many cars it has or what they're named.
    """
    def __init__(self, size=(6, 6), max_slide=None, lr=1e-3, hidden=128):
        self.size = size
        self.encoder = StateEncoder(size)
        self.max_slide = max_slide if max_slide is not None else max(size) - 1
        self.hidden = hidden
        cells = [(i, j) for i in range(size[0]) for j in range(size[1])]
        self.action_space = [(cell, direction, steps)
                              for direction in "lrud" for cell in cells
                              for steps in range(1, self.max_slide + 1)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        in_channels, n_rows, n_cols = self.encoder.shape
        self.net = PolicyNetwork(in_channels, n_rows, n_cols, len(self.action_space), hidden)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def save(self, path):
        """Saves this agent's weights plus the hyperparameters (size,
        max_slide, hidden) load() needs to reconstruct a PolicyNetwork/action
        space of the exact same shape before loading them back in. Optimizer
        state (Adam's momentum) is deliberately not saved - a loaded agent is
        meant to be dropped into a fresh optimization problem (e.g.
        fine-tuning on a puzzle it never trained on) rather than resume
        mid-trajectory on the puzzles it first trained on.
        """
        torch.save({
            "size": self.size,
            "max_slide": self.max_slide,
            "hidden": self.hidden,
            "state_dict": self.net.state_dict(),
        }, path)

    @classmethod
    def load(cls, path, lr=1e-3):
        """Reconstructs a PolicyAgent from a save() checkpoint: same size,
        max_slide and hidden (so the action space and network shape match
        exactly) with the saved weights loaded in, and a fresh Adam optimizer
        (see save()'s docstring for why optimizer state isn't persisted)."""
        checkpoint = torch.load(path, weights_only=False)
        agent = cls(size=checkpoint["size"], max_slide=checkpoint["max_slide"],
                    hidden=checkpoint["hidden"], lr=lr)
        agent.net.load_state_dict(checkpoint["state_dict"])
        return agent

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
        with torch.no_grad():
            logits = self.net(self.encoder.encode(state).unsqueeze(0)).squeeze(0)

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
        """Policy-head logits for `actions` (full (car_name, direction, steps)
        tuples, possibly not yet legal) from the current state - one forward
        pass regardless of how many actions are scored. Used to bias AND-OR's
        own candidate ordering (rush_hour_and_or.py's `heuristic` parameter)
        without ever applying a move, since AND-OR's internal candidates can be
        geometrically invalid until their blockers clear. Returns a dict mapping
        each action to its logit (-inf if outside this agent's action space).
        """
        with torch.no_grad():
            logits = self.net(self.encoder.encode(state).unsqueeze(0))
        logits = logits.squeeze(0)
        scores = {}
        for action in actions:
            car_name, direction, steps = action
            key = (car_anchor(state, car_name), direction, steps)
            scores[action] = logits[self.action_index[key]].item() if key in self.action_index else float("-inf")
        return scores


def _solve_trace(state, max_steps, heuristic=None, gamma=0.0):
    """One full AND-OR resolution attempt from `state`: repeatedly calls
    and_or_solve until it returns a plain move list (solved) or gives up (a
    dead end, or the move budget runs out). Returns the move list if solved,
    else None.

    `heuristic`, if given (typically a trained PolicyAgent's `.heuristic`),
    guides AND-OR's candidate ordering instead of the plain random search.

    `gamma`, passed straight through to and_or_solve - see rush_hour_and_or.solve.
    """
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


def _replay_trace(state, moves, size):
    """Replays `moves` against a fresh Environment seeded at `state`, returning
    (states_before_each_move, solved)."""
    environment = Environment(size, state)
    trace_states = []
    for move in moves:
        trace_states.append(environment.get_state())
        environment.move(*move)
    return trace_states, environment.check_win()


def generate_and_or_traces(initial_cars, size, n_traces, max_steps=200, heuristic=None, gamma=0.0, progress=None):
    """Harvests supervised imitation examples from `n_traces` independent
    AND-OR solve attempts from the same puzzle, discarding any attempt that
    doesn't reach the goal within max_steps. Returns (examples, solved_count):
    examples is a flat list of (state, action, cost_to_go) tuples, cost_to_go
    being moves remaining in that trace once `action` is taken.

    `heuristic`, if given, is passed through to _solve_trace to guide AND-OR's
    search instead of the plain random order.

    `gamma`, passed straight through to _solve_trace - see rush_hour_and_or.solve.

    `progress`, if given, is an existing tqdm bar to advance by one call per
    trace attempt instead of creating a fresh one here - this is what lets
    PuzzleTrainer.generate_examples show one bar for an entire round's trace
    generation across every training puzzle, rather than one completed bar
    per puzzle. When omitted, a standalone bar is created as before, for
    ad-hoc calls outside PuzzleTrainer.
    """
    state0 = tuple((name, tuple(positions)) for name, positions in initial_cars)
    examples = []
    solved_count = 0
    own_bar = progress is None
    pbar = tqdm(desc="AND-OR traces", total=n_traces) if own_bar else progress
    for i in range(n_traces):
        moves = _solve_trace(state0, max_steps, heuristic=heuristic, gamma=gamma)
        if moves:
            trace_states, solved = _replay_trace(state0, moves, size)
            if solved:
                solved_count += 1
                n = len(moves)
                examples.extend((s, move, n - j) for j, (s, move) in enumerate(zip(trace_states, moves)))
        if own_bar:
            pbar.set_postfix(solved=f"{solved_count}/{i + 1}", examples=len(examples))
        pbar.update(1)
    if own_bar:
        pbar.close()
    return examples, solved_count


def dedupe_examples(examples):
    """Collapses `examples` to one entry per distinct state, keeping only the
    action with the lowest realized cost_to_go seen for that state - the single
    most efficient demonstrated continuation, rather than the full noisy mixture
    of everything AND-OR's randomized search took from there. This is what
    keeps the policy head's argmax rollouts from reproducing AND-OR's own
    self-reversing detours as if they were legitimate labels.
    """
    best = {}
    for state, action, cost in examples:
        current = best.get(state)
        if current is None or cost < current[1]:
            best[state] = (action, cost)
    return [(state, action, cost) for state, (action, cost) in best.items()]


def _cap_examples(examples, max_examples):
    """Uniformly subsamples `examples` down to `max_examples` (no-op if
    max_examples is None or already satisfied)."""
    if max_examples is not None and len(examples) > max_examples:
        return random.sample(examples, max_examples)
    return examples


def train_policy(agent, examples, epochs=20, batch_size=64, bar_position=0,
                  dedupe=True, on_epoch=None, ema_decay=0.9, desc="training epochs"):
    """Supervised imitation training: cross-entropy between agent's masked
    softmax over the action space and the move actually taken, for every
    (state, action, cost_to_go) example. cost_to_go itself isn't a training
    target, only dedupe_examples' way of picking which action to keep per state.

    `dedupe` (default True) collapses `examples` to one row per distinct state
    via dedupe_examples before training - this is what keeps argmax rollouts
    from getting stuck in small self-reversing loops.

    `on_epoch`, if given, is called as `on_epoch(epoch, policy_loss)` after
    each epoch's weight update, on that epoch's raw, pre-averaging weights.

    `ema_decay`, if given (default 0.9), maintains an exponential moving
    average of agent.net's weights across epochs and loads it into agent.net
    once training finishes, in place of whichever raw epoch's gradient steps
    the loop happened to end on - smooths out single-epoch noise that could
    otherwise flip one decision in an otherwise-correct greedy chain. Effective
    averaging window is roughly 1 / (1 - ema_decay). Pass None to disable.

    `desc`, if given, overrides the progress bar's label - PuzzleTrainer.train
    sets this to include the current round number, so a multi-round run's
    epoch bars are distinguishable at a glance instead of all reading the
    same generic "training epochs".

    Returns a list of per-epoch policy_loss averages.
    """
    if dedupe:
        examples = dedupe_examples(examples)
    encoded_states = torch.stack([agent.encoder.encode(s) for s, _, _ in examples])
    targets = torch.tensor([
        agent.action_index[(car_anchor(s, car), direction, steps)]
        for s, (car, direction, steps), _ in examples
    ])

    # legal_moves(s) is recomputed here per example rather than reused from
    # wherever the example first came from, since examples arrive as bare
    # (state, action, cost_to_go) tuples with no mask attached - but distinct
    # examples very often share the same state, so caching by state matters:
    # it's what took this from the dominant cost in train_policy down to a
    # sub-second pass in the undeduped case.
    mask_cache = {}
    masks = []
    for s, _, _ in examples:
        if s not in mask_cache:
            legal = {(car_anchor(s, car), direction, steps) for car, direction, steps in legal_moves(s)}
            mask_cache[s] = torch.tensor([a in legal for a in agent.action_space], dtype=torch.bool)
        masks.append(mask_cache[s])
    masks = torch.stack(masks)

    n = len(examples)
    losses = []
    ema_state = {k: v.clone() for k, v in agent.net.state_dict().items()} if ema_decay is not None else None
    for epoch in tqdm(range(epochs), desc=desc, position=bar_position, leave=bar_position == 0):
        perm = torch.randperm(n)
        total_policy_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = agent.net(encoded_states[idx])
            logits = logits.masked_fill(~masks[idx], float("-inf"))
            policy_loss = nn.functional.cross_entropy(logits, targets[idx])
            agent.optimizer.zero_grad()
            policy_loss.backward()
            agent.optimizer.step()
            total_policy_loss += policy_loss.item() * len(idx)
        losses.append(total_policy_loss / n)
        if ema_state is not None:
            with torch.no_grad():
                for k, v in agent.net.state_dict().items():
                    ema_state[k].mul_(ema_decay).add_(v, alpha=1 - ema_decay)
        if on_epoch is not None:
            on_epoch(epoch, losses[-1])
    if ema_state is not None:
        agent.net.load_state_dict(ema_state)
    return losses


def _train_score(results, distances=None):
    """Comparable summary of one evaluate_all()-style (solved, steps) list:
    (solved_count, -mean_ratio_among_solved). Tuple comparison then does the
    right thing without needing to weigh the two dimensions against each
    other - solving strictly more puzzles is always better regardless of
    step count, and among equal solve counts, a smaller mean ratio (a
    larger, less negative second element) wins. Used by PuzzleTrainer.train's
    round-level regression guard.

    `distances`, if given (each puzzle's own optimal BFS distance, aligned to
    `results`), normalizes each solved puzzle's steps to steps/optimal before
    averaging - without it, a pool spanning multiple distances lets a round
    that trades a few short puzzles for one long one look like an
    improvement by raw step count alone, when it's actually worse on
    average. Omit it (the default) to compare raw step counts, e.g. for a
    single-DIST pool where every puzzle's optimal distance is the same
    constant and normalizing wouldn't change any comparison.
    """
    if distances is None:
        ratios = [s for solved, s in results if solved]
    else:
        ratios = [s / d for (solved, s), d in zip(results, distances) if solved]
    return (len(ratios), -float(np.mean(ratios)) if ratios else 0.0)


def evaluate_and_or(initial_cars, size, n_trials, max_steps=200, heuristic=None, gamma=0.0):
    """Solve rate and average move count over `n_trials` independent AND-OR
    attempts from initial_cars. Compare heuristic=None against a trained
    agent's heuristic on the same puzzle to see whether the apprentice makes
    the teacher itself solve more often / find shorter solutions. Returns
    (solve_rate, avg_moves) - avg_moves is nan when nothing solved.

    `gamma`, passed straight through to _solve_trace - see rush_hour_and_or.solve.
    """
    state0 = tuple((name, tuple(positions)) for name, positions in initial_cars)
    lengths = [
        len(moves)
        for moves in (_solve_trace(state0, max_steps, heuristic=heuristic, gamma=gamma) for _ in range(n_trials))
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


class PuzzleTrainer():
    """Trains one shared PolicyAgent across multiple puzzles at once: AND-OR
    traces from every `train_puzzles` entry are pooled into a single example
    set and shuffled together every epoch (train_policy's ordinary minibatch
    loop), rather than training on puzzles one at a time - which risks
    catastrophic forgetting of earlier puzzles as a small MLP's weights get
    overwritten by later ones. This only works because PolicyAgent's action
    space and encoder are keyed by board position rather than by a puzzle's
    own car names (see car_anchor) - one agent applies to any puzzle of the
    same board size.

    `test_puzzles`, if given, never contribute training examples - only
    evaluate()/evaluate_all() ever touch them, to measure how well the pooled
    representation generalizes to puzzles the agent never trained on.

    Puzzles are in the (name, positions) state format Environment and
    read_puzzles() already use - e.g. entries from lib.read_puzzles()[distance].

    `train_distances`/`test_distances`, if given (each puzzle's own optimal
    BFS distance, aligned to train_puzzles/test_puzzles), let train()'s
    round-level regression guard compare steps-to-solve as a ratio to
    optimal rather than a raw count - see _train_score. Omit them for a
    single-DIST pool, where every puzzle shares one optimal distance and
    normalizing wouldn't change any comparison.
    """
    def __init__(self, train_puzzles, test_puzzles=(), size=(6, 6), lr=1e-3, hidden=128,
                 train_distances=None, test_distances=None):
        self.train_puzzles = list(train_puzzles)
        self.test_puzzles = list(test_puzzles)
        self.train_distances = list(train_distances) if train_distances is not None else None
        self.test_distances = list(test_distances) if test_distances is not None else None
        self.size = size
        self.agent = PolicyAgent(size, lr=lr, hidden=hidden)
        self.examples = []
        self.losses = []
        self.history = []

    def generate_examples(self, n_traces=500, max_steps=200, heuristic=None, gamma=0.0, round_label=None):
        """Pools fresh AND-OR traces from every training puzzle (never the
        held-out test puzzles) into one example list, replacing whatever this
        trainer generated before. Stored on self.examples and returned.

        `gamma`, passed straight through to generate_and_or_traces - see
        rush_hour_and_or.solve.

        Shows a single progress bar for the whole call - spanning every
        training puzzle's `n_traces` attempts - rather than one completed bar
        per puzzle, with postfix tracking which puzzle is in flight and the
        running solved/example counts. `round_label`, if given (e.g.
        "round 2/6" from PuzzleTrainer.train), is folded into the bar's
        description so a multi-round run's bars are distinguishable.
        """
        self.examples = []
        desc = "AND-OR traces" if round_label is None else f"AND-OR traces ({round_label})"
        solved_total = 0
        with tqdm(total=len(self.train_puzzles) * n_traces, desc=desc) as pbar:
            for puzzle_idx, puzzle in enumerate(self.train_puzzles):
                examples, solved = generate_and_or_traces(
                    puzzle, self.size, n_traces, max_steps, heuristic, gamma, progress=pbar)
                self.examples.extend(examples)
                solved_total += solved
                pbar.set_postfix(puzzle=f"{puzzle_idx + 1}/{len(self.train_puzzles)}",
                                  solved=f"{solved_total}/{(puzzle_idx + 1) * n_traces}",
                                  examples=len(self.examples))
        return self.examples

    def evaluate(self, puzzle, max_steps=200):
        """(solved, steps, moves) for the current agent on one puzzle - see
        evaluate_policy."""
        return evaluate_policy(self.agent, puzzle, self.size, max_steps=max_steps)

    def evaluate_all(self, max_steps=200):
        """(train_results, test_results): each a list of (solved, steps)
        aligned to train_puzzles/test_puzzles - steps is nan when unsolved -
        from the agent's current weights."""
        def results(puzzles):
            return [
                (solved, steps if solved else float("nan"))
                for solved, steps, _ in (self.evaluate(p, max_steps) for p in puzzles)
            ]
        return results(self.train_puzzles), results(self.test_puzzles)

    def train(self, n_traces=500, epochs=20, batch_size=128, max_steps=200,
              ema_decay=0.9, max_examples=None, and_or_eval_trials=0, eval_every=1,
              n_rounds=1, use_heuristic=True, gamma=0.0):
        """Generates fresh pooled AND-OR traces from every training puzzle,
        then runs one train_policy pass over the pool - repeated for
        `n_rounds` rounds (Expert Iteration).

        Round 1 generates traces with plain, heuristic-free AND-OR search,
        same as the original single-round behavior. From round 2 onward,
        when `use_heuristic` is True (the default), generate_examples is
        passed the current agent's own `.heuristic` instead of None, so
        AND-OR's search is guided by the *last round's* trained apprentice.
        This targets the ceiling plain behavior cloning runs into: AND-OR's
        raw random search (and any gamma lapses in it, see `gamma`) is itself
        a noisy teacher, and a heuristic-guided search finds shorter, more
        consistent solutions for the next round's policy to imitate - each
        round's example pool (self.examples) fully replaces the previous
        round's, and train_policy continues from the agent's already-trained
        weights rather than reinitializing the network.

        `gamma`, passed straight through to generate_examples and (when
        `and_or_eval_trials` > 0) evaluate_and_or - see rush_hour_and_or.solve.
        A tunable parameter here (rather than a module constant) so callers
        like rush_hour.ipynb can vary it directly.

        `and_or_eval_trials`, if > 0, also runs that many independent AND-OR-
        only solves (see evaluate_and_or) per training puzzle, to compare the
        plain teacher's own solve rate/move count against the trained
        apprentice's. This always uses heuristic=None, regardless of
        `n_rounds`/`use_heuristic`, so it stays a fixed baseline rather than
        comparing the apprentice against a version of the teacher it already
        biased.

        `eval_every` controls how often self.history gets a checkpoint: every
        evaluate_all() call rolls every train + test puzzle all the way out
        (up to max_steps, scanning the full action space at each step), so
        doing it every single epoch dominates runtime once there are more
        than a handful of puzzles. Raise this (e.g. to 10 or 20) to trade
        history resolution for speed; it never affects training itself, only
        how densely self.history gets sampled.

        Returns (n_examples, policy_loss, train_results, test_results,
        and_or_stats) from the final round only. `train_results`/
        `test_results` are evaluate_all()'s (solved, steps) lists, evaluated
        once more against the final, EMA-smoothed weights. `and_or_stats` is
        a list of (solve_rate, avg_moves) per training puzzle, or None when
        and_or_eval_trials is 0. Per-epoch (train_results, test_results)
        checkpoints - one evaluate_all() every `eval_every` epochs, on that
        epoch's raw, pre-averaging weights - accumulate in self.history as
        (epoch, train_results, test_results) on one continuous epoch axis
        spanning all rounds (round * epochs + epoch), so a learning-curve
        plot reads as one continuous line across round boundaries.

        A round whose `n_traces` attempts all fail to solve within
        `max_steps` (always possible for a stochastic search, more likely
        the smaller `n_traces` is or the harder the puzzle) yields zero
        examples - rather than crashing train_policy on an empty batch, that
        round's weight update is skipped entirely (self.agent, self.losses
        and `heuristic` all carry over unchanged) and a warning is printed,
        so an unlucky round costs progress but not the whole run.

        Round-level regression guard: heuristic-guided rounds (2+) have no
        guarantee of improving the apprentice - a confidently-wrong heuristic
        can bias AND-OR's search (and thus the next round's training labels)
        worse than the previous round's, not just noisier (see
        rush_hour_and_or.AndNode._order_blockers, which commits to one
        blocker order per attempt with no backtracking if it dead-ends).
        After each round's train_policy call, this evaluates the candidate
        weights on train_puzzles *and* test_puzzles (via _train_score: solved
        count, then mean steps-to-optimal ratio among solved - see
        train_distances/test_distances) and compares each against its own
        best score seen so far this call. A round that regresses on *either*
        reverts self.agent's weights *and* optimizer state to the best
        checkpoint instead of carrying the regression into the next round; a
        round that holds or improves on both becomes the new baseline to
        beat. Checking test_puzzles too (not just train_puzzles) matters
        because solved-count-first scoring alone can't see a round that
        picks up one new solve while making several already-solved puzzles
        measurably less efficient - that regression wouldn't necessarily
        show up in train's solved-count at all, and previously went
        completely unguarded on test. Because a rejected round always
        reverts before the next one starts, "best so far" and "the previous
        (accepted) round" are the same checkpoint throughout - this
        guarantees neither train nor test performance gets worse across
        rounds, at the cost of one extra evaluate_all() per round.

        A rejected (or skipped, see above) round's weights are, by
        construction, identical to the last accepted checkpoint's - so
        rather than leaving a gap in self.history for that round's nominal
        epochs, they're backfilled with the last-accepted checkpoint's own
        (train_results, test_results), the same value evaluate_all() would
        actually return if called mid-round. This keeps self.history
        contiguous across every nominal epoch regardless of how many rounds
        get rejected - notably, it's what lets average_curves compare two
        independently fine-tuned curves that reject different rounds without
        either ending up with epochs the other doesn't have.
        """
        self.history = []
        heuristic = None
        best_net_state = copy.deepcopy(self.agent.net.state_dict())
        best_optim_state = copy.deepcopy(self.agent.optimizer.state_dict())
        best_train_results, best_test_results = self.evaluate_all(max_steps)
        best_train_score = _train_score(best_train_results, self.train_distances)
        best_test_score = _train_score(best_test_results, self.test_distances)

        def backfill(round_idx):
            self.history.extend(
                (round_idx * epochs + epoch, best_train_results, best_test_results)
                for epoch in range(0, epochs, eval_every)
            )

        for round_idx in range(n_rounds):
            round_label = f"round {round_idx + 1}/{n_rounds}"
            examples = self.generate_examples(n_traces, max_steps, heuristic=heuristic, gamma=gamma,
                                               round_label=round_label)
            examples = _cap_examples(examples, max_examples)
            if not examples:
                print(f"Warning: no AND-OR traces solved in {round_label} - skipping this round's training")
                backfill(round_idx)
                continue

            history_checkpoint = len(self.history)

            def on_epoch(epoch, loss, round_idx=round_idx):
                if epoch % eval_every == 0:
                    train_results, test_results = self.evaluate_all(max_steps)
                    self.history.append((round_idx * epochs + epoch, train_results, test_results))

            self.losses = train_policy(self.agent, examples, epochs=epochs, batch_size=batch_size,
                                        ema_decay=ema_decay, on_epoch=on_epoch,
                                        desc=f"training epochs ({round_label})")

            train_results, test_results = self.evaluate_all(max_steps)
            round_train_score = _train_score(train_results, self.train_distances)
            round_test_score = _train_score(test_results, self.test_distances)
            regressions = []
            if round_train_score < best_train_score:
                regressions.append(f"train {round_train_score} worse than {best_train_score}")
            if round_test_score < best_test_score:
                regressions.append(f"test {round_test_score} worse than {best_test_score}")
            if regressions:
                print(f"Warning: {round_label} regressed ({'; '.join(regressions)}) - "
                      f"reverting to the previous round's weights")
                self.agent.net.load_state_dict(best_net_state)
                self.agent.optimizer.load_state_dict(best_optim_state)
                self.history = self.history[:history_checkpoint]
                backfill(round_idx)
            else:
                best_train_score, best_test_score = round_train_score, round_test_score
                best_train_results, best_test_results = train_results, test_results
                best_net_state = copy.deepcopy(self.agent.net.state_dict())
                best_optim_state = copy.deepcopy(self.agent.optimizer.state_dict())
            if use_heuristic:
                heuristic = self.agent.heuristic

        train_results, test_results = self.evaluate_all(max_steps)
        if and_or_eval_trials > 0:
            and_or_stats = [
                evaluate_and_or(puzzle, self.size, and_or_eval_trials, max_steps=max_steps, gamma=gamma)
                for puzzle in self.train_puzzles
            ]
        else:
            and_or_stats = None

        policy_loss = self.losses[-1] if self.losses else float("nan")
        return len(examples), policy_loss, train_results, test_results, and_or_stats


def finetune_curve(agent, puzzle, size=(6, 6), n_traces=40, epochs=5, n_rounds=10,
                    max_examples=None, eval_every=1, use_heuristic=True, gamma=0.0):
    """Fine-tunes `agent` in place on a single `puzzle`, via PuzzleTrainer's
    ExIt loop (n_rounds rounds of heuristic-guided AND-OR trace generation +
    train_policy) - `puzzle` is this trainer's sole training puzzle, so
    (unlike PuzzleTrainer's ordinary test_puzzles) it *does* contribute
    gradients: the point here is to measure how fast an agent can adapt to
    one specific puzzle when allowed to, not to score it zero-shot.

    Meant to compare a puzzle-diverse-pretrained agent (see PolicyAgent.load)
    against one starting from scratch (a fresh PolicyAgent(size)) on the same
    puzzle - the pretrained agent's shared representation should let it climb
    toward an optimal solution in fewer fine-tuning epochs.

    `gamma`, passed straight through to PuzzleTrainer.train - see
    rush_hour_and_or.solve.

    Returns the (epoch, solved, steps) curve, one entry per evaluate_all()
    checkpoint (steps is nan while unsolved), on the same continuous
    round-spanning epoch axis PuzzleTrainer.train() uses.
    """
    trainer = PuzzleTrainer([puzzle], [], size)
    trainer.agent = agent
    trainer.train(n_traces=n_traces, epochs=epochs, n_rounds=n_rounds,
                  max_examples=max_examples, eval_every=eval_every, use_heuristic=use_heuristic, gamma=gamma)
    return [(epoch, train_results[0][0], train_results[0][1]) for epoch, train_results, _ in trainer.history]


def average_curves(curves, optimal=None):
    """Averages `curves` - a list of finetune_curve() results, one per
    puzzle - into a single (epochs, mean_steps, solve_rate) triple:
    mean_steps is the mean steps-to-solve among whichever puzzles were
    solved at that epoch (nan if none were), solve_rate the fraction solved.
    Built for comparing a pretrained vs. scratch agent's convergence across a
    whole held-out test set at once, rather than reading a noisy per-puzzle
    scatter plot.

    Curves are aligned by actual epoch value (the intersection of epochs
    every curve has a checkpoint for), not list position - PuzzleTrainer.
    train's round-level regression guard strips history entries for rounds
    it rejects, so independent finetune_curve() calls (different puzzles,
    different seeds) can reject different rounds and end up with gaps at
    different epochs. Zipping by position would silently average mismatched
    epochs together whenever that happens.

    `optimal`, if given, is each curve's own optimal length (e.g. its
    puzzle's BFS distance), aligned to `curves` - steps are then normalized
    to a fraction of that puzzle's optimal before averaging (1.0 = optimal),
    so puzzles of different optimal lengths can share one comparable scale
    instead of assuming every puzzle in the set solves in the same number of
    moves.
    """
    by_epoch = [{epoch: (solved, steps) for epoch, solved, steps in curve} for curve in curves]
    epochs = sorted(set.intersection(*(set(d) for d in by_epoch)))
    mean_steps, solve_rate = [], []
    for epoch in epochs:
        if optimal is None:
            solved_steps = [steps for solved, steps in (d[epoch] for d in by_epoch) if solved]
        else:
            solved_steps = [steps / opt for (solved, steps), opt in zip((d[epoch] for d in by_epoch), optimal)
                             if solved]
        solve_rate.append(len(solved_steps) / len(curves))
        mean_steps.append(float(np.mean(solved_steps)) if solved_steps else float("nan"))
    return epochs, mean_steps, solve_rate
