import itertools
import os
import random
import shutil
import warnings
from collections import defaultdict, deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from rush_hour_bfs import multi_bfs, hint, MOVES, DELTAS

counter = itertools.count(start=0, step=1)

OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}


def sample_puzzle(path="rush_nw.txt"):
    """Samples a random puzzle from a `rush_nw.txt`-format file (see conv.ipynb's
    Data Preparation section): each line is "<distance> <bitboard> <count>",
    where `bitboard` is a 36-character, row-major layout of a 6x6 grid ("o" for
    empty, "A" for the target car, and one other letter per remaining car).
    Returns a list of blocks in the format `Environment` expects: the target
    car's positions first, followed by every other car's, each a list of
    (row, col) tuples.
    """
    with open(path) as f:
        bitboard = random.choice(f.readlines()).split()[1]

    cars = defaultdict(list)
    for idx, cell in enumerate(bitboard):
        if cell != "o":
            row, col = divmod(idx, 6)
            cars[cell].append((row, col))

    other_names = sorted(name for name in cars if name != "A")
    return [cars["A"]] + [cars[name] for name in other_names]


def sample_unique(path="rush_nw_unique.txt"):
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


class Environment():
    """
    Puzzle grid environment. Contains a grid with block objects that can be moved until the red block is at its goal state.
    """
    def __init__(self, size, blocks):
        """Initializes grid with specified size and places the red block."""
        self.puzzle = [[0 for j in range(size[1])] for i in range(size[0])]
        self.goal = (2, 5)
        self.blocks = {"red": self.add_block(blocks[0], red=True)}
        global counter
        counter = itertools.count(start=0, step=1)
        for block_p in blocks[1:]:
            block = self.add_block(block_p)
            self.blocks[block.name] = block

    def add_block(self, positions, red=False):
        block = Block(positions, name="red" if red else None)
        for i, j in positions:
            self.puzzle[i][j] = block
        return block

    def move(self, block_name, direction):
        block = self.blocks[block_name]
        if direction not in MOVES[block.orientation]:
            return False
        old_pos = block.positions
        di, dj = DELTAS[direction]
        new_pos = [(i + di, j + dj) for i, j in old_pos]
        n_rows, n_cols = len(self.puzzle), len(self.puzzle[0])
        for new_i, new_j in new_pos:
            if not (0 <= new_i < n_rows and 0 <= new_j < n_cols):
                return False
            if self.puzzle[new_i][new_j] not in [0, block]:
                return False
        for i, j in old_pos:
            self.puzzle[i][j] = 0
        for new_i, new_j in new_pos:
            self.puzzle[new_i][new_j] = block
        block.positions = new_pos
        return self.get_state()

    def get_state(self):
        return tuple((name, tuple(block.positions)) for name, block in self.blocks.items())

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

        for block in self.blocks.values():
            rows = [i for i, j in block.positions]
            cols = [j for i, j in block.positions]
            i0, i1 = min(rows), max(rows)
            j0, j1 = min(cols), max(cols)
            facecolor = "red" if block.name == "red" else "gray"
            ax.add_patch(plt.Rectangle(
                (j0, i0), j1 - j0 + 1, i1 - i0 + 1,
                facecolor=facecolor,
                edgecolor="black",
                linewidth=1.5,
            ))
            if block.name != "red":
                ax.text(
                    (j0 + j1 + 1) / 2, (i0 + i1 + 1) / 2, block.name,
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


class Block():
    def __init__(self, positions, name=None):
        self.name = name if name is not None else str(next(counter))
        self.positions = positions
        self.orientation = "h" if positions[0][0] == positions[1][0] else "v"


def legal_actions(environment, action_space):
    """Returns the subset of action_space currently executable, without leaving
    the environment mutated (each candidate move is tried and immediately undone).
    """
    legal = []
    for block, direction in action_space:
        if environment.move(block, direction):
            environment.move(block, OPPOSITE[direction])
            legal.append((block, direction))
    return legal


class Teacher():
    """Decides, state by state, whether a training session should be steered by
    a hint instead of letting the agent's own policy choose.

    Two orthogonal settings:
    - `mode`: where a hint comes from. "automatic" pulls from the puzzle's
      BFS-optimal solution; "manual" prompts a human for typed input (`block
      direction`, or "skip" to let the agent choose that state, or "solve" to
      end the session early).
    - `policy`: how "automatic" mode decides *whether* to hint at a given
      state. "random" flips a `hint_prob`-weighted coin. "q" learns the
      decision with tabular Q-learning: the state is just distance-to-goal,
      the action is hint-or-not, and the reward is the student's own per-step
      reward, bootstrapped one step at a time exactly like `Agent.q_learning`
      — a hint's credit comes from what happens immediately after it, with
      any longer-run effect propagating backward through the chain of
      updates rather than a hand-widened reward window.

    Once `max_hints` hints have been given in a session, `exhausted` becomes
    True, at which point the calling training loop is expected to hand off to
    a generic solve() call for the remainder of the session. Call
    `new_session()` before each teaching session to reset the hint budget
    while keeping any learned Q-table intact across sessions.

    A "q"-policy teacher starts out `training`: each session's outcome updates
    its Q-table and decays its epsilon. Call `freeze()` once you're done
    training it (e.g. after a warm-up run over several student generations)
    to stop both, so it can be reused as a fixed, fully-evaluated policy
    across a separate group of students without drifting mid-comparison.
    """
    def __init__(self, blocks, solution, mode="automatic", policy="random", max_hints=1,
                 hint_prob=0.5, hint_reward=0.1, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.99):
        self.action_space = [(block, direction) for direction in "lrud" for block in blocks]
        self.solution = solution
        self.mode = mode
        self.policy = policy
        self.max_hints = max_hints
        self.hint_prob = hint_prob
        self.hint_reward = hint_reward
        self.hints_used = 0

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.qtable = defaultdict(lambda: {True: 0.0, False: 0.0})
        self._pending = None
        self.training = True

    @property
    def exhausted(self):
        return self.hints_used >= self.max_hints

    def freeze(self, epsilon=0.0):
        """Stops further Q-table updates and epsilon decay, fixing the "q" policy
        as-is (fully greedy by default) so it can be evaluated across a fresh
        group of students without continuing to learn mid-comparison.
        """
        self.training = False
        self.epsilon = epsilon

    def new_session(self):
        """Resets the per-session hint budget; keeps the learned Q-table (if any)."""
        self.hints_used = 0
        self._pending = None
        if self.policy == "q" and self.training:
            self.epsilon *= self.epsilon_decay

    def _context(self, state):
        return self.solution[state]

    def advise(self, environment, state, step_number=None):
        """Returns an advised (block, direction) action, or None to defer to the agent."""
        if self.exhausted:
            return None
        if self.mode == "manual":
            action = self._advise_manual(environment, state, step_number)
        elif self.policy == "q":
            action = self._advise_q(state)
        else:
            action = self._advise_automatic(state)
        if action is not None:
            self.hints_used += 1
        return action

    def observe(self, next_state, reward):
        """Bootstrapped Q-update for the "q" policy's last decision. No-op for
        "random" policy or manual mode (only "q" has anything to learn), and
        for a frozen teacher (see `freeze()`).
        """
        if self._pending is None or not self.training:
            return
        ctx0, decision = self._pending
        self._pending = None
        ctx1 = self._context(next_state)
        q1 = self.qtable[ctx1]
        rpe = reward + self.gamma * max(q1.values()) - self.qtable[ctx0][decision]
        self.qtable[ctx0][decision] += self.alpha * rpe

    def reward(self, advised, steps, solved, penalty):
        """Reward for the action just taken. Random-policy automatic hints get a
        flat bonus; everything else (agent moves, manual hints, and "q"-policy
        hints) uses the normal shaped reward, so the "q" policy learns from the
        same signal the student does.
        """
        if advised and self.mode == "automatic" and self.policy == "random":
            return self.hint_reward
        return (1 - steps * penalty) if solved else 0

    def _advise_automatic(self, state):
        if random.random() >= self.hint_prob:
            return None
        return hint(state, self.solution)

    def _advise_q(self, state):
        ctx = self._context(state)
        q = self.qtable[ctx]
        if random.random() < self.epsilon:
            decision = random.choice([True, False])
        elif q[True] == q[False]:
            # Untrained/tied contexts fall back to the same hint_prob-weighted
            # coin flip as the "random" policy, instead of silently favoring
            # True via dict insertion order.
            decision = random.random() < self.hint_prob
        else:
            decision = max(q, key=q.get)
        action = hint(state, self.solution) if decision else None
        decision = action is not None
        self._pending = (ctx, decision)
        return action

    def _advise_manual(self, environment, state, step_number):
        environment.visualize(step_number=step_number, solution=self.solution)
        legal = set(legal_actions(environment, self.action_space))
        while True:
            prompt = input("What action should the agent take?")
            if prompt == "skip":
                return None
            if prompt == "solve":
                self.hints_used = self.max_hints
                return None
            action = tuple(prompt.split())
            if action not in legal:
                print("Invalid move, try again")
                continue
            return action


class Agent():
    def __init__(self, blocks, alpha=0.1, gamma=0.95, epsilon=1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        action_space = [(block, direction) for direction in "lrud" for block in blocks]
        self.qtable = defaultdict(lambda: {action: 0 for action in action_space})

    def policy(self, state, greedy=False):
        actions = self.qtable[state]
        if greedy or random.random() > self.epsilon:
            return max(actions, key=actions.get)
        else:
            return random.choice(list(actions.keys()))

    def q_learning(self, s0, s1, a, r):
        q1 = self.qtable[s1]
        rpe = r + self.gamma * max(q1.values()) - self.qtable[s0][a]
        self.qtable[s0][a] += self.alpha * rpe


class Train():
    def __init__(self, agent, environment):
        """`environment`'s current configuration is captured as the puzzle to
        train on — pass in a freshly-built `Environment` (its board size and
        block layout are copied; the instance itself isn't mutated or reused).
        """
        self.agent = agent
        self.size = (len(environment.puzzle), len(environment.puzzle[0]))
        self.initial = [list(positions) for _, positions in environment.get_state()]
        self.history = []
        self.last_moves = None
        self.blocks = list(environment.blocks.keys())
        self.solution = multi_bfs(environment.get_state())

    def _new_environment(self):
        return Environment(self.size, self.initial)

    def _prune(self, s0, a, excluded=None):
        """Mark move `a` from state `s0` as invalid so it isn't tried again."""
        if excluded is not None:
            excluded[s0].add(a)
        else:
            del self.agent.qtable[s0][a]

    def solve(self, environment, max_steps=10000, decay=0.99, greedy=False, penalty=0.001, track_moves=False):
        solved = False
        steps = 0
        moves = [] if track_moves else None
        excluded = defaultdict(set)
        while not solved and max_steps > steps:
            s0 = environment.get_state()
            if greedy:
                actions = self.agent.qtable[s0]
                candidates = [action for action in actions if action not in excluded[s0]]
                if not candidates:
                    break
                a = max(candidates, key=actions.get)
            else:
                a = self.agent.policy(s0, greedy=False)
            s1 = environment.move(a[0], a[1])
            steps += 1
            if not s1:
                self._prune(s0, a, excluded if greedy else None)
                continue
            if track_moves:
                moves.append(a)
            solved = environment.check_win()
            if not greedy:
                r = (1 - steps * penalty) if solved else 0
                self.agent.q_learning(s0, s1, a, r)
        if not greedy:
            self.agent.epsilon *= decay
        if track_moves:
            self.last_moves = moves
        return solved, steps

    def evaluate(self, max_steps=200):
        environment = self._new_environment()
        solved, steps = self.solve(environment, max_steps=max_steps, greedy=True, track_moves=True)
        return float(solved), (steps if solved else float("nan"))

    def teach(self, teacher, penalty=0.01):
        """Runs one training session steered by `teacher`: at each state the
        teacher may supply a hint, otherwise the agent's own policy chooses.
        Once the teacher is exhausted (or the puzzle is solved), any remaining
        steps are handed off to a generic solve() call.
        """
        environment = self._new_environment()
        solved = False
        steps = 0
        while not solved and not teacher.exhausted:
            s0 = environment.get_state()
            advice = teacher.advise(environment, s0, step_number=steps)
            advised = advice is not None
            a = advice if advised else self.agent.policy(s0, greedy=False)
            s1 = environment.move(a[0], a[1])
            steps += 1
            if not s1:
                self._prune(s0, a)
                continue
            solved = environment.check_win()
            r = teacher.reward(advised, steps, solved, penalty)
            self.agent.q_learning(s0, s1, a, r)
            teacher.observe(s1, r)
        if not solved:
            self.solve(environment, penalty=penalty)

    def learn(self, n, eval_every=50, eval_max_steps=200, teach_every=None, teacher=None,
              mode="automatic", policy="random", max_hints=1, hint_prob=0.5, hint_reward=0.1,
              decay=0.99, penalty=0.01):
        """`teacher`, if given, is reused (and its Q-table kept) across every
        teaching session in this run — pass in the same Teacher across several
        `learn()` calls to let a "q"-policy teacher keep learning across
        multiple generations of student agent. If omitted and `teach_every` is
        set, a fresh Teacher is built from `mode`/`policy`/... and returned.
        """
        if teach_every and teacher is None:
            teacher = Teacher(self.blocks, self.solution, mode=mode, policy=policy,
                               max_hints=max_hints, hint_prob=hint_prob, hint_reward=hint_reward)
        for episode in range(1, n + 1):
            environment = self._new_environment()
            self.solve(environment, decay=decay, penalty=penalty)
            if episode % eval_every == 0:
                solve_rate, avg_steps = self.evaluate(eval_max_steps)
                self.history.append((episode, solve_rate, avg_steps))
            if teach_every and episode % teach_every == 0:
                teacher.new_session()
                self.teach(teacher, penalty=penalty)
        return teacher

    def visualize_evaluation(self, folder="eval_frames"):
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

        for i, (block, direction) in enumerate(self.last_moves, start=1):
            environment.move(block, direction)
            environment.visualize(save_path=os.path.join(folder, f"move_{i:03d}.png"),
                                   step_number=i, solution=self.solution)

        print(f"Saved {len(self.last_moves) + 1} frames to {folder}/")


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

    Each block's anchor cell (min row, min col) is enough to reconstruct its
    full occupancy, since a block's orientation never changes.
    """
    def __init__(self, blocks, size=(6, 6)):
        self.block_order = list(blocks)
        self.n_rows, self.n_cols = size

    @property
    def dim(self):
        return 2 * len(self.block_order)

    def encode(self, state):
        state_dict = dict(state)
        feats = []
        for name in self.block_order:
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
    def __init__(self, blocks, size=(6, 6), gamma=0.95, epsilon=1.0, lr=1e-3,
                 hidden=128, buffer_size=20000, batch_size=64, tau=0.01):
        self.encoder = StateEncoder(blocks, size)
        self.action_space = [(block, direction) for direction in "lrud" for block in blocks]
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
    def __init__(self, agent, environment):
        """`environment`'s current configuration is captured as the puzzle to
        train on — pass in a freshly-built `Environment` (its board size and
        block layout are copied; the instance itself isn't mutated or reused).
        """
        self.agent = agent
        self.size = (len(environment.puzzle), len(environment.puzzle[0]))
        self.initial = [list(positions) for _, positions in environment.get_state()]
        self.history = []
        self.last_moves = None
        self.blocks = list(environment.blocks.keys())
        self.solution = multi_bfs(environment.get_state())

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
            s1 = environment.move(a[0], a[1])
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

    def evaluate(self, max_steps=200):
        environment = self._new_environment()
        solved, steps = self.solve(environment, max_steps=max_steps, greedy=True, track_moves=True)
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
            advice = teacher.advise(environment, s0, step_number=steps)
            advised = advice is not None
            a = advice if advised else self.agent.policy(s0, mask, greedy=False)
            s1 = environment.move(a[0], a[1])
            steps += 1
            solved = environment.check_win()
            next_mask = valid_action_mask(environment, self.agent.action_space)
            r = teacher.reward(advised, steps, solved, penalty)
            self.agent.remember(s0, a, r, s1, solved, next_mask)
            self.agent.train_step()
            teacher.observe(s1, r)
            mask = next_mask
        if not solved:
            self.solve(environment, max_steps=max_steps, penalty=penalty)
        return solved, steps

    def learn(self, n, eval_every=50, eval_max_steps=200, max_steps=200, penalty=0.01,
              epsilon_decay=0.95, epsilon_min=0.05,
              teach_every=None, teacher=None, mode="automatic", policy="random",
              max_hints=1, hint_prob=0.5, hint_reward=0.1):
        """`teacher`, if given, is reused (and its Q-table kept) across every
        teaching session in this run — pass in the same Teacher across several
        `learn()` calls to let a "q"-policy teacher keep learning across
        multiple generations of student agent. If omitted and `teach_every` is
        set, a fresh Teacher is built from `mode`/`policy`/... and returned.
        """
        if teach_every and teacher is None:
            teacher = Teacher(self.blocks, self.solution, mode=mode, policy=policy,
                               max_hints=max_hints, hint_prob=hint_prob, hint_reward=hint_reward)
        for episode in range(1, n + 1):
            environment = self._new_environment()
            self.solve(environment, max_steps=max_steps, penalty=penalty)
            self.agent.epsilon = max(epsilon_min, self.agent.epsilon * epsilon_decay)
            if episode % eval_every == 0:
                solve_rate, avg_steps = self.evaluate(eval_max_steps)
                self.history.append((episode, solve_rate, avg_steps))
            if teach_every and episode % teach_every == 0:
                teacher.new_session()
                self.teach(teacher, max_steps=max_steps, penalty=penalty)
        return teacher

    def visualize_evaluation(self, folder="eval_frames_dqn"):
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

        for i, (block, direction) in enumerate(self.last_moves, start=1):
            environment.move(block, direction)
            environment.visualize(save_path=os.path.join(folder, f"move_{i:03d}.png"),
                                   step_number=i, solution=self.solution)

        print(f"Saved {len(self.last_moves) + 1} frames to {folder}/")
