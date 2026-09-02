import os
import random
import shutil
import warnings
from collections import deque, namedtuple
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rush_hour_and_or import solve as and_or_solve, legal_moves, blockers_in_path
from rush_hour_lib import multi_bfs, MOVES, DELTAS

OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}


class Car():
    def __init__(self, positions, name):
        self.name = name
        self.positions = positions
        self.orientation = "h" if positions[0][0] == positions[1][0] else "v"


class Environment():
    """
    Puzzle grid environment. Contains a grid with car objects that can be moved until the red car is at its goal state.
    """
    def __init__(self, size, cars):
        """Initializes grid with specified size and places every car. `cars` is a
        sequence of (name, positions) pairs, red first — the same shape this
        class's own get_state() returns, so a captured state can be fed straight
        back in to rebuild an equivalent grid.
        """
        self.puzzle = [[0 for j in range(size[1])] for i in range(size[0])]
        self.goal = (2, 5)
        self.cars = {}
        for name, positions in cars:
            self.cars[name] = self.add_car(positions, name)
        self._solution = None

    @property
    def solution(self):
        """Distance-to-goal for every state reachable from this puzzle's initial
        layout, via multi_bfs - expensive (a full BFS over the reachable state
        graph), so computed once on first access and cached rather than redone
        per DQNTrain built on this same Environment.
        """
        if self._solution is None:
            self._solution = multi_bfs(self.get_state())
        return self._solution

    def add_car(self, positions, name):
        car = Car(positions, name)
        for i, j in positions:
            self.puzzle[i][j] = car
        return car

    def move(self, car_name, direction, steps=1):
        """Slides car_name `steps` cells in `direction` (single-cell by default).
        Every cell the car sweeps through along the way must be in bounds and
        clear of other cars, not just the final resting cells — a multi-cell
        slide can't jump over a blocker the way checking only the destination
        would allow.
        """
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

    def visualize(self, save_path=None, step_number=None, solution=None):
        n_rows, n_cols = len(self.puzzle), len(self.puzzle[0])

        fig, ax = plt.subplots()
        ax.add_patch(plt.Rectangle((0, 0), n_cols, n_rows, facecolor="white", edgecolor="none"))

        for car in self.cars.values():
            rows = [i for i, j in car.positions]
            cols = [j for i, j in car.positions]
            i0, i1 = min(rows), max(rows)
            j0, j1 = min(cols), max(cols)
            facecolor = "red" if car.name == "red" else "gray"
            ax.add_patch(plt.Rectangle(
                (j0, i0), j1 - j0 + 1, i1 - i0 + 1,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.5,
            ))
            if car.name != "red":
                ax.text(
                    (j0 + j1 + 1) / 2, (i0 + i1 + 1) / 2, car.name,
                    ha="center", va="center", color="white", fontsize=12,
                )

        ax.add_patch(plt.Rectangle(
            (0, 0), n_cols, n_rows,
            facecolor="none",
            edgecolor="black",
            linewidth=1.5,
        ))

        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        title_parts = []
        if step_number is not None:
            title_parts.append(f"Step {step_number}")
        if solution is not None:
            title_parts.append(f"{solution[self.get_state()]} moves to solve")
        if title_parts:
            ax.set_title("  —  ".join(title_parts))

        if save_path:
            fig.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()


def legal_actions(environment, action_space):
    """Returns the subset of action_space currently executable, without leaving
    the environment mutated (each candidate move is tried and immediately
    undone). Each action is a (car, direction, steps) tuple.
    """
    legal = []
    for car, direction, steps in action_space:
        if environment.move(car, direction, steps):
            environment.move(car, OPPOSITE[direction], steps)
            legal.append((car, direction, steps))
    return legal


def average_histories(histories):
    """Averages several `history` lists (each a list of matching
    (episode, solve_rate, avg_steps) checkpoints, e.g. from repeated runs with
    the same `eval_every`) into one. `avg_steps` is nan-aware, since a
    checkpoint where a given run didn't solve records `nan` there.
    """
    episodes = [episode for episode, _, _ in histories[0]]
    solve_rates = np.array([[sr for _, sr, _ in history] for history in histories])
    avg_steps = np.array([[st for _, _, st in history] for history in histories])
    mean_solve_rate = solve_rates.mean(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_avg_steps = np.nanmean(avg_steps, axis=0)
    return list(zip(episodes, mean_solve_rate, mean_avg_steps))


def plot_training_results(runs, file_name=None):
    """Plots solve rate and avg steps per run, comparing multiple training histories.

    `runs` maps a label (e.g. "teach") to a trainer's `history` list of
    (episode, solve_rate, avg_steps) tuples.
    """
    runs = {label: history for label, history in runs.items() if history}
    if not runs:
        print("No training history to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for label, history in runs.items():
        episodes, solve_rates, avg_steps = zip(*history)
        ax1.plot(episodes, solve_rates, marker="o", label=label, linestyle="--")
        ax2.plot(episodes, avg_steps, marker="o", label=label, linestyle="--")

    ax1.set_xlabel("Training episode")
    ax1.set_ylabel("Solve rate")
    ax1.set_ylim(0, 1)
    ax1.set_title("Greedy solve rate")
    ax1.legend()

    ax2.set_xlabel("Training episode")
    ax2.set_ylabel("Avg steps to solve")
    ax2.set_title("Avg steps (solved episodes)")
    ax2.legend()

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.show()


class StateEncoder():
    """Encodes an Environment state as a fixed-size, normalized feature vector.

    Each car's anchor cell (min row, min col) is enough to reconstruct its
    full occupancy, since a car's orientation never changes.
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


class QNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


Transition = namedtuple("Transition", "state action reward next_state done next_mask")


class ReplayBuffer():
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DQNAgent():
    def __init__(self, cars, size=(6, 6), max_slide=None, gamma=0.95, epsilon=1.0, lr=1e-3,
                 hidden=128, buffer_size=20000, batch_size=64, tau=0.01):
        """`max_slide` caps how many cells a single action can move a car — the
        action space is every (car, direction, steps) combination for steps in
        1..max_slide, mirroring the and-or solver's multi-cell slides rather
        than a single fixed cell per action. Defaults to the longest possible
        slide on this board, `max(size) - 1`. Illegal slides at a given state
        (out of bounds, or blocked partway through) are masked out by
        valid_action_mask exactly like illegal single-cell moves always were.
        """
        self.encoder = StateEncoder(cars, size)
        max_slide = max_slide if max_slide is not None else max(size) - 1
        self.action_space = [(car, direction, steps)
                              for direction in "lrud" for car in cars
                              for steps in range(1, max_slide + 1)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.tau = tau
        self.buffer = ReplayBuffer(buffer_size)
        self.train_steps = 0

        self.policy_net = QNetwork(self.encoder.dim, len(self.action_space), hidden)
        self.target_net = QNetwork(self.encoder.dim, len(self.action_space), hidden)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def policy(self, state, mask, greedy=False):
        valid_idx = mask.nonzero(as_tuple=True)[0]
        if not greedy and random.random() < self.epsilon:
            return self.action_space[valid_idx[random.randrange(len(valid_idx))].item()]
        with torch.no_grad():
            q = self.policy_net(self.encoder.encode(state).unsqueeze(0)).squeeze(0)
        q = q.masked_fill(~mask, float("-inf"))
        return self.action_space[int(q.argmax())]

    def remember(self, s0, a, r, s1, done, next_mask):
        self.buffer.push(
            self.encoder.encode(s0), self.action_index[a], r,
            self.encoder.encode(s1), done, next_mask,
        )

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size)
        states = torch.stack(batch.state)
        actions = torch.tensor(batch.action).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.stack(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32)
        next_masks = torch.stack(batch.next_mask)

        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_q = next_q.masked_fill(~next_masks, float("-inf"))
            no_moves = next_masks.sum(dim=1) == 0
            max_next_q = next_q.max(dim=1).values
            max_next_q[no_moves] = 0.0
        target = rewards + self.gamma * max_next_q * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        with torch.no_grad():
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.mul_(1 - self.tau).add_(self.tau * policy_param)


class PolicyValueNetwork(nn.Module):
    """Shared trunk, two heads - the same shape the paper uses once it adds
    a value network on top of TPT (§5.2): a policy head (action logits) and
    a value head (here, predicted cost-to-go, i.e. moves remaining to the
    goal, rather than their win/loss probability, since our targets are
    exact realized move-counts from AND-OR traces rather than binary game
    outcomes).
    """
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, out_dim)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.trunk(x)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


class PolicyAgent():
    """Imitation-learning counterpart to DQNAgent: same action-space/encoder
    setup and the same policy(state, mask, greedy) interface, but net(x) is
    read as (action logits, predicted cost-to-go) rather than Q-values, and
    it's trained by supervised cross-entropy + regression against AND-OR
    traces (see train_policy) instead of TD bootstrapping. Sharing
    DQNAgent's interface means anywhere an agent is passed a masked (state,
    greedy) query - e.g. valid_action_mask-based rollouts, or eventually a
    policy callback inside rush_hour_and_or.py - this can be dropped in
    unchanged; value_policy below is the separate, environment-based
    decision rule that actually consumes the value head.
    """
    def __init__(self, cars, size=(6, 6), max_slide=None, lr=1e-3, hidden=128):
        self.encoder = StateEncoder(cars, size)
        max_slide = max_slide if max_slide is not None else max(size) - 1
        self.action_space = [(car, direction, steps)
                              for direction in "lrud" for car in cars
                              for steps in range(1, max_slide + 1)]
        self.action_index = {a: i for i, a in enumerate(self.action_space)}
        self.net = PolicyValueNetwork(self.encoder.dim, len(self.action_space), hidden)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

    def policy(self, state, mask, greedy=False):
        logits, _ = self.net(self.encoder.encode(state).unsqueeze(0))
        logits = logits.squeeze(0).masked_fill(~mask, float("-inf"))
        if greedy:
            return self.action_space[int(logits.argmax())]
        probs = torch.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, 1).item()
        return self.action_space[idx]

    def value_policy(self, environment, mask, visited=frozenset()):
        """Greedy 1-ply, value-guided action choice: applies every currently
        legal action to `environment`, scores the resulting state with the
        value head, undoes the move, and returns whichever action minimizes
        predicted total remaining moves (1 + cost-to-go of the resulting
        state) - rather than the policy head's raw argmax. Every candidate
        here is already a fully legal, directly-executable move (this is
        the legal-moves-only regime that doubles as GammaLapse's fallback),
        so unlike AND-OR's own internal AND/OR-node candidates, this never
        has to evaluate a hypothetical invalid/overlapping board - exactly
        the state-conditioned-on-legal-action distinction that makes value
        scoring safe here but not (yet) inside AND-OR's own recursion.
        This is meant to catch the kind of move a pure policy-argmax can't:
        e.g. undoing the last move nets zero progress, so its resulting
        state's predicted cost-to-go should be no better than - and, with
        the extra step counted, strictly worse than - continuing forward.

        `visited` (states already seen this rollout, supplied by the
        caller) is a cheap guard against the value head's own remaining
        blind spot: it isn't trained to be Bellman-consistent, so two
        neighboring states can each look slightly better than the other
        and form a shallow trap even though the head is otherwise well
        fit - the endgame wobble found earlier. Among the scored
        candidates, this returns the lowest-cost action whose resulting
        state _isn't_ already in `visited`, falling back to the plain
        lowest-cost action only when every legal candidate leads
        somewhere already seen (a genuine dead end no choice can avoid).
        """
        scored = []
        for i in mask.nonzero(as_tuple=True)[0].tolist():
            car, direction, steps = self.action_space[i]
            environment.move(car, direction, steps)
            next_state = environment.get_state()
            with torch.no_grad():
                _, value = self.net(self.encoder.encode(next_state).unsqueeze(0))
            environment.move(car, OPPOSITE[direction], steps)
            scored.append(((car, direction, steps), 1 + value.item(), next_state))

        scored.sort(key=lambda entry: entry[1])
        for action, _cost, next_state in scored:
            if next_state not in visited:
                return action
        return scored[0][0]

    def heuristic(self, state, actions):
        """Policy-head logits for `actions` (full (car_name, direction, steps)
        tuples, possibly not yet legal) from the current, already-valid
        `state` - one forward pass regardless of how many actions are scored.
        This is the AND-OR-side counterpart to policy()/value_policy(): where
        those two only ever choose among moves that are already directly
        executable (legal_moves, or environment.move()-confirmed), AND-OR's
        own internal candidates at an AndNode/OrNode are geometric
        possibilities whose blockers may not be cleared yet, so scoring the
        state a candidate would produce would mean evaluating an invalid,
        overlapping board. Scoring the action's raw logit against the
        *current* state's encoding sidesteps that entirely - no move is ever
        applied here. Returns a dict mapping every action in `actions` to a
        score (its logit, or -inf if the action isn't in this agent's action
        space, e.g. a slide longer than max_slide) - the shape
        rush_hour_and_or.py's order_candidates/heuristic parameter expects.
        """
        with torch.no_grad():
            logits, _ = self.net(self.encoder.encode(state).unsqueeze(0))
        logits = logits.squeeze(0)
        return {
            action: (logits[self.action_index[action]].item() if action in self.action_index else float("-inf"))
            for action in actions
        }


def _solve_trace(state, max_steps, heuristic=None):
    """One full AND-OR resolution attempt from `state`: repeatedly calls
    and_or_solve until it returns a plain list (solved) or gives up (a
    dead end with no legal moves, or the move budget runs out). Returns
    the move list if solved, else None. Shared by generate_and_or_traces
    (starting from the puzzle's initial state) and dagger_round (starting
    from wherever a rollout actually ends up).

    `heuristic`, if given (typically a trained PolicyAgent's own
    `.heuristic` method), guides AND-OR's internal candidate ordering
    instead of the plain random search - this is the "teacher improves
    from the apprentice" half of the loop: once a network exists, later
    rounds can generate sharper, more network-consistent expert traces
    instead of forever re-deriving supervision from an undirected search.
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
    """Replays `moves` against a fresh Environment seeded at `state`,
    returning (states_before_each_move, solved) - solved confirms the
    replayed moves actually reach the goal, since and_or_solve's own
    (state, moves) bookkeeping isn't re-validated against real Environment
    move semantics until this point.
    """
    environment = Environment(size, state)
    trace_states = []
    for move in moves:
        trace_states.append(environment.get_state())
        environment.move(*move)
    return trace_states, environment.check_win()


def generate_and_or_traces(initial_cars, size, n_traces, max_steps=200, heuristic=None):
    """Harvests supervised imitation examples from AND-OR solves: `n_traces`
    independent attempts from the same puzzle (`initial_cars`, in
    Environment's (name, positions) format), each replayed to recover the
    exact state before every move and confirm the puzzle was actually won.
    Discards any attempt that doesn't reach the goal within max_steps (an
    unlucky run of GammaLapses/dead-ends) - this is expected to throw away
    a large fraction of attempts on harder puzzles, same as the plain
    solver's own known solve-rate noise. Returns a flat list of (state,
    action, cost_to_go) tuples, where cost_to_go is the number of moves
    remaining in that trace once `action` is taken.

    `heuristic`, if given, is passed straight through to _solve_trace to
    guide AND-OR's search instead of the plain random order - omit it (the
    default) for the undirected traces round 0 of imitation needs, since an
    untrained/freshly-initialized network's heuristic scores are just noise.
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


def dagger_round(agent, initial_cars, size, n_rollouts, max_steps=200, greedy=True, use_value=True,
                  guide_and_or=False, bar_position=0):
    """One DAgger correction pass (Ross, Gordon & Bagnell 2011; the
    aggregation approach behind the paper's DAgger-improved TPT network in
    §4.3): rolls the current agent out from initial_cars n_rollouts times,
    and at every state the rollout actually visits, re-queries AND-OR
    fresh from that exact state for its own preferred resolution - not
    whatever the agent just did - recording the full resulting (state,
    action, cost_to_go) chain as supervision. The rollout itself still
    advances via the agent's own decision rule, so later states in the same
    rollout reflect wherever the agent's own mistakes actually lead it -
    that's what a single round of generate_and_or_traces can't see, since
    it only ever labels states along AND-OR's own trajectories.

    `use_value` rolls out via value_policy (1-ply value-guided lookahead),
    the actually-deployed decision rule once a value head exists - so the
    states that get corrected are the ones the real, near-solving rollout
    visits (e.g. the small endgame wobble the value head still falls into),
    rather than states along the policy head's argmax, which on a hard
    puzzle can fail outright within a couple of moves and so never even
    reaches the states most worth correcting. Set False to fall back to
    the policy head (`greedy` then selects argmax vs. sampled) instead.

    `guide_and_or` passes agent.heuristic into each correction's AND-OR
    search (see _solve_trace), instead of AND-OR's plain random candidate
    order - the teacher half of the ExIt loop: by this point (round >= 1)
    the agent already has something learned from round 0, so its heuristic
    is a real signal, not noise. GAMMA's own lapse probability still keeps
    corrections from collapsing to one deterministic trajectory.

    `bar_position` is the progress bar's tqdm `position` (and, via that,
    whether it's left on screen after finishing - see train_policy's
    matching parameter): 0 when this is the only active bar, 1+ when
    train_policy_dagger already has its own round-level bar open, so the
    two don't fight over the same terminal line.
    """
    examples = []
    heuristic = agent.heuristic if guide_and_or else None
    with tqdm(range(n_rollouts), desc="DAgger rollouts", position=bar_position, leave=bar_position == 0) as pbar:
        for _ in pbar:
            environment = Environment(size, initial_cars)
            visited = set()
            for step in range(max_steps):
                state = environment.get_state()
                if state in visited:
                    break  # the rollout itself is cycling; no new states to learn from
                visited.add(state)

                correction = _solve_trace(state, max_steps - step, heuristic=heuristic)
                if correction:
                    trace_states, solved = _replay_trace(state, correction, size)
                    if solved:
                        n = len(correction)
                        examples.extend(
                            (s, move, n - i) for i, (s, move) in enumerate(zip(trace_states, correction))
                        )

                mask = valid_action_mask(environment, agent.action_space)
                if not mask.any():
                    break
                if use_value:
                    move = agent.value_policy(environment, mask, visited=visited)
                else:
                    move = agent.policy(state, mask, greedy=greedy)
                environment.move(*move)
                if environment.check_win():
                    break
            pbar.set_postfix(examples=len(examples))
    return examples


def train_policy_dagger(cars, initial_cars, size, n_iterations=5, n_initial_traces=500,
                         n_rollouts_per_round=30, epochs_per_round=20, batch_size=128,
                         max_steps=200, use_value_rollout=True, greedy_rollout=True,
                         lr=1e-3, hidden=128, value_weight=1.0, warm_start=True,
                         guide_and_or=False, and_or_eval_trials=0):
    """Round 0 is plain behavior cloning (generate_and_or_traces + train_policy)
    on demonstrations from the puzzle's start state alone. Every later round
    rolls the *current* agent out (dagger_round), aggregates the newly
    labeled states into the growing dataset - mirroring the paper's "online"
    ExIt/DAgger variant (§3.3), which aggregates every dataset generated so
    far rather than training on only the latest batch.

    `warm_start` (default True) keeps training the same agent across rounds
    rather than reinitializing a fresh network each time: each round's
    train_policy call only has to absorb that round's new examples on top
    of what it already knows, rather than re-deriving everything from a
    blank slate on an ever-larger dataset every round with a fixed epoch
    budget. This matches the actual thing being modeled - a system that
    gets better through accumulated experience, not a fresh apprentice
    re-reading a growing textbook from page one each round. Set False to
    reinitialize a fresh PolicyAgent every round instead (the original
    behavior), which isolates what the aggregated dataset alone teaches a
    network, independent of any particular training trajectory - a cleaner
    read on the dataset's quality, at the cost of not modeling compounding
    improvement.

    `use_value_rollout` (default True) rolls DAgger's exploration out via
    the trained value head (value_policy) rather than the policy head's
    argmax - since value_policy is the decision rule that actually gets
    close to solving hard puzzles, this targets corrections at the states
    it actually struggles with (e.g. an endgame wobble), rather than
    states along the policy head's argmax, which can fail outright within
    a couple of moves and never reach the states most worth correcting.
    `greedy_rollout` only matters when use_value_rollout is False.

    `guide_and_or` (default False) is passed to dagger_round: from round 1
    onward, each correction's AND-OR search is guided by the *current*
    agent's own heuristic instead of plain random search - the teacher
    improving from the apprentice, closing the loop the other direction
    (apprentice-from-teacher is what generate_and_or_traces/dagger_round's
    supervision already does). `and_or_eval_trials`, if > 0, measures that
    effect directly each round: runs that many independent AND-OR-only
    solves (see evaluate_and_or) with the current agent's heuristic (if
    guide_and_or) or plain random search (if not), so and_or_solve_rate/
    and_or_avg_moves in round_stats show whether the teacher's own search
    is getting better as the apprentice does, alongside the apprentice's
    own policy_*/value_* performance.

    Returns (agent, examples, round_stats), where round_stats is a list of
    (round, n_examples, policy_loss, value_loss, policy_solved, policy_steps,
    value_solved, value_steps, and_or_solve_rate, and_or_avg_moves) -
    policy_* comes from evaluate_policy (argmax over the policy head),
    value_* from evaluate_value_policy (1-ply value-guided lookahead), and
    and_or_* from evaluate_and_or (None, None when and_or_eval_trials is 0).
    """
    examples = generate_and_or_traces(initial_cars, size, n_initial_traces, max_steps)
    agent = PolicyAgent(cars, size, lr=lr, hidden=hidden)
    round_stats = []

    # position=0/leave=True for the round bar, position=1/leave=False for
    # train_policy's and dagger_round's nested bars (via bar_position=1
    # below) - distinct terminal lines, so the round bar's postfix and a
    # nested bar's own updates never overwrite each other mid-line.
    round_bar = tqdm(range(n_iterations), desc="DAgger rounds", position=0, leave=True)
    for round_idx in round_bar:
        losses = train_policy(agent, examples, epochs=epochs_per_round, batch_size=batch_size,
                               value_weight=value_weight, bar_position=1)
        policy_loss, value_loss = losses[-1]
        policy_solved, policy_steps, _ = evaluate_policy(
            agent, initial_cars, size, max_steps=max_steps, greedy=True)
        value_solved, value_steps, _ = evaluate_value_policy(
            agent, initial_cars, size, max_steps=max_steps)
        if and_or_eval_trials > 0:
            and_or_solve_rate, and_or_avg_moves = evaluate_and_or(
                initial_cars, size, and_or_eval_trials, max_steps=max_steps,
                heuristic=agent.heuristic if guide_and_or else None)
        else:
            and_or_solve_rate, and_or_avg_moves = None, None
        round_stats.append((round_idx, len(examples), policy_loss, value_loss,
                             policy_solved, policy_steps, value_solved, value_steps,
                             and_or_solve_rate, and_or_avg_moves))
        round_bar.set_postfix(examples=len(examples), policy_solved=policy_solved, value_solved=value_solved)
        if round_idx == n_iterations - 1:
            break
        new_examples = dagger_round(agent, initial_cars, size, n_rollouts_per_round,
                                     max_steps=max_steps, greedy=greedy_rollout,
                                     use_value=use_value_rollout, guide_and_or=guide_and_or,
                                     bar_position=1)
        examples = examples + new_examples
        if not warm_start:
            agent = PolicyAgent(cars, size, lr=lr, hidden=hidden)

    return agent, examples, round_stats


def print_round_stats(round_stats):
    """Pretty-prints train_policy_dagger's round_stats - (round, n_examples,
    policy_loss, value_loss, policy_solved, policy_steps, value_solved,
    value_steps, and_or_solve_rate, and_or_avg_moves) per round - as a
    plain-text, right-aligned table instead of one repr'd tuple per line.
    None (an unsolved round's steps) and and_or_* when and_or_eval_trials
    was 0 both render as "-".
    """
    headers = ["round", "examples", "pol_loss", "val_loss", "pol_solved", "pol_steps",
               "val_solved", "val_steps", "ao_rate", "ao_moves"]

    def fmt(value):
        if value is None or (isinstance(value, float) and value != value):  # nan != nan
            return "-"
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value)

    rows = [[fmt(v) for v in row] for row in round_stats]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(c.rjust(w) for c, w in zip(cells, widths))

    print(line(headers))
    print(line(["-" * w for w in widths]))
    for row in rows:
        print(line(row))


def train_policy(agent, examples, epochs=20, batch_size=64, value_weight=1.0, bar_position=0):
    """Supervised imitation training: cross-entropy between agent's masked
    softmax over the action space and the move actually taken, plus a
    Huber (smooth L1) regression loss between the value head and
    cost_to_go, for every (state, action, cost_to_go) example (see
    generate_and_or_traces). Repeated states across independent traces
    naturally recover an empirical action distribution rather than a
    single hard label, since minimizing per-example cross-entropy over
    the whole dataset converges toward matching each state's observed
    action frequencies - the same cost-sensitivity a tree-policy target
    gets from visit counts, without needing to build that distribution by
    hand. The two losses share the trunk and are summed (`value_weight`
    scales the regression term, since raw move-counts and log-probabilities
    sit on different scales). Returns a list of (policy_loss, value_loss)
    per-epoch averages.

    `bar_position` is the training-epochs progress bar's tqdm `position` -
    see dagger_round's matching parameter for why (avoids fighting
    train_policy_dagger's own round-level bar for the same terminal line).
    """
    encoded_states = torch.stack([agent.encoder.encode(s) for s, _, _ in examples])
    targets = torch.tensor([agent.action_index[a] for _, a, _ in examples])
    costs = torch.tensor([c for _, _, c in examples], dtype=torch.float32)

    # legal_moves(s) is recomputed here per example rather than reused from
    # wherever the example first came from, since examples arrive as bare
    # (state, action, cost_to_go) tuples with no mask attached - but distinct
    # examples very often share the same state (repeated states within one
    # AND-OR trace, or across independent traces/rollouts), so caching by
    # state (rather than recomputing legal_moves unconditionally) matters:
    # it's what took this from the dominant cost in train_policy (~95% of
    # wall time on a distance-16 puzzle) down to a sub-second, no-bar-needed
    # pass.
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
    for _ in tqdm(range(epochs), desc="training epochs", position=bar_position, leave=bar_position == 0):
        perm = torch.randperm(n)
        total_policy_loss = 0.0
        total_value_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits, values = agent.net(encoded_states[idx])
            logits = logits.masked_fill(~masks[idx], float("-inf"))
            policy_loss = nn.functional.cross_entropy(logits, targets[idx])
            value_loss = nn.functional.smooth_l1_loss(values, costs[idx])
            loss = policy_loss + value_weight * value_loss
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
            total_policy_loss += policy_loss.item() * len(idx)
            total_value_loss += value_loss.item() * len(idx)
        losses.append((total_policy_loss / n, total_value_loss / n))
    return losses


def evaluate_and_or(initial_cars, size, n_trials, max_steps=200, heuristic=None):
    """Solve rate and average move count over `n_trials` independent AND-OR
    attempts from initial_cars (see _solve_trace) - the direct before/after
    read on whether `heuristic` (typically a trained PolicyAgent's own
    `.heuristic`) actually makes the *teacher* itself solve more often
    and/or find shorter solutions, not just whatever the apprentice's own
    rollouts do. Compare a call with heuristic=None against one with a
    trained agent's heuristic on the same puzzle to see the teacher-side
    half of the ExIt loop directly. Returns (solve_rate, avg_moves) -
    avg_moves is nan when nothing solved, matching evaluate_policy's
    convention for an unsolved run.
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


def evaluate_policy(agent, initial_cars, size, max_steps=200, greedy=True):
    """Rolls the agent's policy forward from initial_cars purely over the
    legal-move action space - no AND-OR involved, i.e. every decision is
    made the way a GammaLapse fallback move is chosen today. This is the
    "as if every move was a gamma lapse" validation: it exercises the
    policy network on its own, independent of any later AO*-style
    candidate-ranking integration into rush_hour_and_or.py. Returns
    (solved, steps, moves).
    """
    environment = Environment(size, initial_cars)
    moves = []
    for step in range(max_steps):
        state = environment.get_state()
        mask = valid_action_mask(environment, agent.action_space)
        if not mask.any():
            break
        move = agent.policy(state, mask, greedy=greedy)
        environment.move(*move)
        moves.append(move)
        if environment.check_win():
            return True, step + 1, moves
    return False, None, moves


def evaluate_value_policy(agent, initial_cars, size, max_steps=200):
    """Like evaluate_policy, but decisions come from value_policy's 1-ply
    value-guided lookahead instead of the policy head's argmax - the
    comparison that shows whether the value head resolves failures the
    policy head's argmax can't, like a self-reversing 2-cycle. Tracks
    every state seen this rollout and passes it to value_policy as
    `visited`, so a shallow value trap (e.g. a 2-state wobble) gets routed
    around via the next-best action instead of looping until max_steps.
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
        move = agent.value_policy(environment, mask, visited=visited)
        environment.move(*move)
        moves.append(move)
        if environment.check_win():
            return True, step + 1, moves
    return False, None, moves


class DQNTrain():
    def __init__(self, agent, environment, blocker_threshold=5):
        """`environment`'s current configuration is captured as the puzzle to
        train on — pass in a freshly-built `Environment` (its board size and
        car layout are copied; the instance itself isn't mutated or reused).

        `blocker_threshold` governs solve_and_or()'s hybrid rollout: at each
        state, once blockers_in_path(state) (cars currently in red's direct
        path to the exit - see rush_hour_and_or.py) is at or below this
        count, the move is handed to AND-OR's backward-chaining solver;
        above it, the agent picks the move directly. Modeling a human
        running on instinct while a puzzle still looks unresolved, then
        switching to careful calculation once the finish is visibly close.
        """
        self.agent = agent
        self.size = (len(environment.puzzle), len(environment.puzzle[0]))
        self.initial = [(name, list(positions)) for name, positions in environment.get_state()]
        self.history = []
        self.last_moves = None
        self.cars = list(environment.cars.keys())
        self.solution = environment.solution
        self.blocker_threshold = blocker_threshold

    def _new_environment(self):
        return Environment(self.size, self.initial)

    def solve(self, environment, max_steps=200, greedy=False, penalty=0.01, track_moves=False):
        solved = False
        steps = 0
        moves = [] if track_moves else None
        mask = valid_action_mask(environment, self.agent.action_space)
        while not solved and steps < max_steps and mask.any():
            s0 = environment.get_state()
            a = self.agent.policy(s0, mask, greedy=greedy)
            s1 = environment.move(*a)
            steps += 1
            if track_moves:
                moves.append(a)
            solved = environment.check_win()
            next_mask = valid_action_mask(environment, self.agent.action_space)
            if not greedy:
                r = (1 - steps * penalty) if solved else 0
                self.agent.remember(s0, a, r, s1, solved, next_mask)
                self.agent.train_step()
            mask = next_mask
        if track_moves:
            self.last_moves = moves
        return solved, steps

    def solve_and_or(self, environment, max_steps=200, greedy=False, penalty=0.01, track_moves=False):
        """Default rollout for both training and evaluation: a hybrid of the agent's
        own instinct and AND-OR's calculated backward-chaining (rush_hour_and_or.solve),
        switching per move based on blockers_in_path (see `blocker_threshold` on
        __init__) rather than always deferring to one or the other. Below the
        threshold, AND-OR plans the (possibly multi-move) resolution, still handing
        control back to the agent's own policy at any point it would otherwise give
        up and act randomly (a GammaLapse, or the terminal dead-end fallback); above
        it, the agent picks the single next move directly. Unless `greedy`, every
        move along the way - AND-OR's or the agent's - is replayed against
        `environment` and remembered/trained on identically, so the agent bootstraps
        through the whole trajectory rather than only the states where it chose the
        action. `greedy` runs the agent's policy deterministically and skips
        training, for evaluation.
        """
        def agent_policy(state):
            legal = set(legal_moves(state))
            mask = torch.tensor([a in legal for a in self.agent.action_space], dtype=torch.bool)
            return self.agent.policy(state, mask, greedy=greedy)

        solved = False
        steps = 0
        moves = [] if track_moves else None
        while not solved and steps < max_steps:
            state = environment.get_state()
            if blockers_in_path(state) <= self.blocker_threshold:
                result = and_or_solve(state, policy=agent_policy)
                step_moves = result if isinstance(result, list) else result[1]
            elif legal_moves(state):
                step_moves = [agent_policy(state)]
            else:
                step_moves = None
            if not step_moves:
                break
            for move in step_moves:
                if steps >= max_steps:
                    break
                s0 = environment.get_state()
                s1 = environment.move(*move)
                steps += 1
                if track_moves:
                    moves.append(move)
                solved = environment.check_win()
                if not greedy:
                    next_mask = valid_action_mask(environment, self.agent.action_space)
                    r = (1 - steps * penalty) if solved else 0
                    self.agent.remember(s0, move, r, s1, solved, next_mask)
                    self.agent.train_step()
                if solved:
                    break
        if track_moves:
            self.last_moves = moves
        return solved, steps

    def evaluate(self, max_steps=200):
        environment = self._new_environment()
        solved, steps = self.solve_and_or(environment, max_steps=max_steps, greedy=True, track_moves=True)
        return float(solved), (steps if solved else float("nan"))

    def teach(self, teacher, max_steps=200, penalty=0.01):
        """Runs one training session steered by `teacher`: at each state the
        teacher may supply a hint, otherwise the agent's masked policy chooses.
        Once the teacher is exhausted (or the puzzle is solved), any remaining
        steps are handed off to a generic solve() call.
        """
        environment = self._new_environment()
        solved = False
        steps = 0
        mask = valid_action_mask(environment, self.agent.action_space)
        while not solved and steps < max_steps and mask.any() and not teacher.exhausted:
            s0 = environment.get_state()
            advice = teacher.advise(s0)
            advised = advice is not None
            a = advice if advised else self.agent.policy(s0, mask, greedy=False)
            s1 = environment.move(*a)
            steps += 1
            solved = environment.check_win()
            next_mask = valid_action_mask(environment, self.agent.action_space)
            r = teacher.reward(advised, steps, solved, penalty)
            self.agent.remember(s0, a, r, s1, solved, next_mask)
            self.agent.train_step()
            teacher.observe(s1, r)
            mask = next_mask
        if not solved:
            self.solve_and_or(environment, max_steps=max_steps, penalty=penalty)
        return solved, steps

    def learn(self, n, eval_every=50, eval_max_steps=200, max_steps=200, penalty=0.01,
              epsilon_decay=0.95, epsilon_min=0.05,
              teach_every=None, teacher=None, policy="random",
              max_hints=1, hint_prob=0.5, hint_reward=0.1):
        """`teacher`, if given, is reused (and its Q-table kept) across every
        teaching session in this run — pass in the same Teacher across several
        `learn()` calls to let a "q"-policy teacher keep learning across
        multiple generations of student agent. If omitted and `teach_every` is
        set, a fresh Teacher is built from `policy`/... and returned.
        """
        if teach_every and teacher is None:
            from rush_hour_teacher import Teacher
            teacher = Teacher(self.cars, self.solution, policy=policy,
                               max_hints=max_hints, hint_prob=hint_prob, hint_reward=hint_reward)
        for episode in tqdm(range(1, n + 1)):
            environment = self._new_environment()
            self.solve_and_or(environment, max_steps=max_steps, penalty=penalty)
            self.agent.epsilon = max(epsilon_min, self.agent.epsilon * epsilon_decay)
            if episode % eval_every == 0:
                solve_rate, avg_steps = self.evaluate(eval_max_steps)
                self.history.append((episode, solve_rate, avg_steps))
            if teach_every and episode % teach_every == 0:
                teacher.new_session()
                self.teach(teacher, max_steps=max_steps, penalty=penalty)
        return teacher

    def visualize_evaluation(self, folder="visualization/eval_frames_dqn"):
        """Reconstructs the last evaluation episode move-by-move and saves a figure per move."""
        if not self.last_moves:
            print("No evaluation moves recorded yet — call evaluate() (or learn()) first.")
            return

        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

        environment = self._new_environment()
        environment.visualize(save_path=os.path.join(folder, "move_000.png"),
                               step_number=0, solution=self.solution)

        for i, action in enumerate(self.last_moves, start=1):
            environment.move(*action)
            environment.visualize(save_path=os.path.join(folder, f"move_{i:03d}.png"),
                                   step_number=i, solution=self.solution)

        print(f"Saved {len(self.last_moves) + 1} frames to {folder}/")
