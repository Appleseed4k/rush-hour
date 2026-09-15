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


def _solve_trace(state, max_steps, heuristic=None):
    """One full AND-OR resolution attempt from `state`: repeatedly calls
    and_or_solve until it returns a plain move list (solved) or gives up (a
    dead end, or the move budget runs out). Returns the move list if solved,
    else None.

    `heuristic`, if given (typically a trained PolicyAgent's `.heuristic`),
    guides AND-OR's candidate ordering instead of the plain random search.
    """
    moves = []
    while len(moves) < max_steps:
        att = and_or_solve(state, heuristic=heuristic)
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


def generate_and_or_traces(initial_cars, size, n_traces, max_steps=200, heuristic=None):
    """Harvests supervised imitation examples from `n_traces` independent
    AND-OR solve attempts from the same puzzle, discarding any attempt that
    doesn't reach the goal within max_steps. Returns a flat list of (state,
    action, cost_to_go) tuples, cost_to_go being moves remaining in that trace
    once `action` is taken.

    `heuristic`, if given, is passed through to _solve_trace to guide AND-OR's
    search instead of the plain random order.
    """
    state0 = tuple((name, tuple(positions)) for name, positions in initial_cars)
    examples = []
    solved_count = 0
    with tqdm(range(n_traces), desc="AND-OR traces") as pbar:
        for i in pbar:
            moves = _solve_trace(state0, max_steps, heuristic=heuristic)
            if moves:
                trace_states, solved = _replay_trace(state0, moves, size)
                if solved:
                    solved_count += 1
                    n = len(moves)
                    examples.extend((s, move, n - j) for j, (s, move) in enumerate(zip(trace_states, moves)))
            pbar.set_postfix(solved=f"{solved_count}/{i + 1}", examples=len(examples))
    return examples


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
                  dedupe=True, on_epoch=None, ema_decay=0.9):
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
    for epoch in tqdm(range(epochs), desc="training epochs", position=bar_position, leave=bar_position == 0):
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


def evaluate_and_or(initial_cars, size, n_trials, max_steps=200, heuristic=None):
    """Solve rate and average move count over `n_trials` independent AND-OR
    attempts from initial_cars. Compare heuristic=None against a trained
    agent's heuristic on the same puzzle to see whether the apprentice makes
    the teacher itself solve more often / find shorter solutions. Returns
    (solve_rate, avg_moves) - avg_moves is nan when nothing solved.
    """
    state0 = tuple((name, tuple(positions)) for name, positions in initial_cars)
    lengths = [
        len(moves)
        for moves in (_solve_trace(state0, max_steps, heuristic=heuristic) for _ in range(n_trials))
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
    """
    def __init__(self, train_puzzles, test_puzzles=(), size=(6, 6), lr=1e-3, hidden=128):
        self.train_puzzles = list(train_puzzles)
        self.test_puzzles = list(test_puzzles)
        self.size = size
        self.agent = PolicyAgent(size, lr=lr, hidden=hidden)
        self.examples = []
        self.losses = []
        self.history = []

    def generate_examples(self, n_traces=500, max_steps=200, heuristic=None):
        """Pools fresh AND-OR traces from every training puzzle (never the
        held-out test puzzles) into one example list, replacing whatever this
        trainer generated before. Stored on self.examples and returned."""
        self.examples = []
        for puzzle in self.train_puzzles:
            self.examples.extend(generate_and_or_traces(puzzle, self.size, n_traces, max_steps, heuristic))
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
              n_rounds=1, use_heuristic=True):
        """Generates fresh pooled AND-OR traces from every training puzzle,
        then runs one train_policy pass over the pool - repeated for
        `n_rounds` rounds (Expert Iteration).

        Round 1 generates traces with plain, heuristic-free AND-OR search,
        same as the original single-round behavior. From round 2 onward,
        when `use_heuristic` is True (the default), generate_examples is
        passed the current agent's own `.heuristic` instead of None, so
        AND-OR's search is guided by the *last round's* trained apprentice.
        This targets the ceiling plain behavior cloning runs into: AND-OR's
        raw random search (and any GAMMA lapses in it) is itself a noisy
        teacher, and a heuristic-guided search finds shorter, more consistent
        solutions for the next round's policy to imitate - each round's
        example pool (self.examples) fully replaces the previous round's, and
        train_policy continues from the agent's already-trained weights
        rather than reinitializing the network.

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
        """
        self.history = []
        heuristic = None
        for round_idx in range(n_rounds):
            examples = self.generate_examples(n_traces, max_steps, heuristic=heuristic)
            examples = _cap_examples(examples, max_examples)

            def on_epoch(epoch, loss, round_idx=round_idx):
                if epoch % eval_every == 0:
                    train_results, test_results = self.evaluate_all(max_steps)
                    self.history.append((round_idx * epochs + epoch, train_results, test_results))

            self.losses = train_policy(self.agent, examples, epochs=epochs, batch_size=batch_size,
                                        ema_decay=ema_decay, on_epoch=on_epoch)
            if use_heuristic:
                heuristic = self.agent.heuristic

        train_results, test_results = self.evaluate_all(max_steps)
        if and_or_eval_trials > 0:
            and_or_stats = [
                evaluate_and_or(puzzle, self.size, and_or_eval_trials, max_steps=max_steps)
                for puzzle in self.train_puzzles
            ]
        else:
            and_or_stats = None

        return len(examples), self.losses[-1], train_results, test_results, and_or_stats


def finetune_curve(agent, puzzle, size=(6, 6), n_traces=40, epochs=5, n_rounds=10,
                    max_examples=None, eval_every=1, use_heuristic=True):
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

    Returns the (epoch, solved, steps) curve, one entry per evaluate_all()
    checkpoint (steps is nan while unsolved), on the same continuous
    round-spanning epoch axis PuzzleTrainer.train() uses.
    """
    trainer = PuzzleTrainer([puzzle], [], size)
    trainer.agent = agent
    trainer.train(n_traces=n_traces, epochs=epochs, n_rounds=n_rounds,
                  max_examples=max_examples, eval_every=eval_every, use_heuristic=use_heuristic)
    return [(epoch, train_results[0][0], train_results[0][1]) for epoch, train_results, _ in trainer.history]


def average_curves(curves):
    """Averages `curves` - a list of finetune_curve() results, one per
    puzzle, all sharing the same epoch axis - into a single (epochs,
    mean_steps, solve_rate) triple: mean_steps is the mean steps-to-solve
    among whichever puzzles were solved at that epoch (nan if none were),
    solve_rate the fraction solved. Built for comparing a pretrained vs.
    scratch agent's convergence across a whole held-out test set at once,
    rather than reading a noisy per-puzzle scatter plot.
    """
    epochs = [epoch for epoch, _, _ in curves[0]]
    mean_steps, solve_rate = [], []
    for i in range(len(epochs)):
        solved_steps = [curve[i][2] for curve in curves if curve[i][1]]
        solve_rate.append(len(solved_steps) / len(curves))
        mean_steps.append(float(np.mean(solved_steps)) if solved_steps else float("nan"))
    return epochs, mean_steps, solve_rate
