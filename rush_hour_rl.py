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
    for car, direction, steps in action_space:
        if environment.move(car, direction, steps):
            environment.move(car, OPPOSITE[direction], steps)
            legal.append((car, direction, steps))
    return legal


class StateEncoder():
    """Encodes an Environment state as a fixed-size, normalized feature vector -
    each car's anchor cell (min row, min col), since orientation never changes.
    """
    def __init__(self, cars, size=(6, 6)):
        self.car_order = list(cars)
        self.n_rows, self.n_cols = size

    @property
    def dim(self):
        return 2 * len(self.car_order)

    def encode(self, state):
        state_dict = dict(state)
        feats = []
        for name in self.car_order:
            positions = state_dict[name]
            i0 = min(i for i, j in positions)
            j0 = min(j for i, j in positions)
            feats.append(i0 / (self.n_rows - 1))
            feats.append(j0 / (self.n_cols - 1))
        return torch.tensor(feats, dtype=torch.float32)


def valid_action_mask(environment, action_space):
    """Boolean mask over action_space of the moves currently legal in `environment`."""
    legal = set(legal_actions(environment, action_space))
    return torch.tensor([action in legal for action in action_space], dtype=torch.bool)


class PolicyNetwork(nn.Module):
    """Shared trunk feeding a single policy head (action logits)."""
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.policy_head(self.trunk(x))


class PolicyAgent():
    """Trained by supervised cross-entropy against AND-OR traces (see
    train_policy) rather than TD bootstrapping."""
    def __init__(self, cars, size=(6, 6), max_slide=None, lr=1e-3, hidden=128):
        self.encoder = StateEncoder(cars, size)
        max_slide = max_slide if max_slide is not None else max(size) - 1
        self.action_space = [(car, direction, steps)
                              for direction in "lrud" for car in cars
                              for steps in range(1, max_slide + 1)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        self.net = PolicyNetwork(self.encoder.dim, len(self.action_space), hidden)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def policy_avoiding_cycles(self, environment, mask, visited=frozenset()):
        """Ranks every currently legal action by the policy head's logit (highest
        first) and returns the first one whose resulting state isn't already in
        `visited`, falling back to the plain top-ranked action only when every
        candidate leads somewhere already visited.

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
            car, direction, steps = self.action_space[i]
            environment.move(car, direction, steps)
            next_state = environment.get_state()
            environment.move(car, OPPOSITE[direction], steps)
            scored.append(((car, direction, steps), logits[i].item(), next_state))

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
        return {
            action: (logits[self.action_index[action]].item() if action in self.action_index else float("-inf"))
            for action in actions
        }


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


def train_policy_bc(cars, initial_cars, size, n_traces=500, epochs=20, batch_size=128,
                     max_steps=200, lr=1e-3, hidden=128, ema_decay=0.9,
                     max_examples=None, and_or_eval_trials=0):
    """Plain behavior cloning: one batch of AND-OR traces from the puzzle's
    initial state, trained once on a freshly-initialized agent.

    `ema_decay` is passed through to train_policy to smooth epoch-to-epoch
    instability in whether the trained policy solves independently.

    `and_or_eval_trials`, if > 0, also runs that many independent AND-OR-only
    solves (see evaluate_and_or) to report the plain teacher's own solve
    rate/move count alongside the trained apprentice's.

    Returns (agent, examples, steps_history, stats). `steps_history` is a list
    of (epoch, steps) checkpoints - one evaluate_policy rollout after each
    epoch, on that epoch's raw, pre-averaging weights (nan if unsolved within
    max_steps). `stats` is (n_examples, policy_loss, policy_solved,
    policy_steps, and_or_solve_rate, and_or_avg_moves), evaluated once more
    against the final EMA-smoothed weights (and_or_* is None, None when
    and_or_eval_trials is 0).
    """
    examples = generate_and_or_traces(initial_cars, size, n_traces, max_steps)
    examples = _cap_examples(examples, max_examples)
    agent = PolicyAgent(cars, size, lr=lr, hidden=hidden)

    steps_history = []

    def on_epoch(epoch, loss):
        solved, steps, _ = evaluate_policy(agent, initial_cars, size, max_steps=max_steps)
        steps_history.append((epoch, steps if solved else float("nan")))

    losses = train_policy(agent, examples, epochs=epochs, batch_size=batch_size,
                           ema_decay=ema_decay, on_epoch=on_epoch)
    policy_loss = losses[-1]
    policy_solved, policy_steps, _ = evaluate_policy(agent, initial_cars, size, max_steps=max_steps)
    if and_or_eval_trials > 0:
        and_or_solve_rate, and_or_avg_moves = evaluate_and_or(
            initial_cars, size, and_or_eval_trials, max_steps=max_steps)
    else:
        and_or_solve_rate, and_or_avg_moves = None, None
    stats = (len(examples), policy_loss, policy_solved, policy_steps,
              and_or_solve_rate, and_or_avg_moves)
    return agent, examples, steps_history, stats


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
    targets = torch.tensor([agent.action_index[a] for _, a, _ in examples])

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
            legal = set(legal_moves(s))
            mask_cache[s] = torch.tensor([m in legal for m in agent.action_space], dtype=torch.bool)
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
