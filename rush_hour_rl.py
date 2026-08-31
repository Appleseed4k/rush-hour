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
