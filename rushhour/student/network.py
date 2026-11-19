"""The student's network: board encoding, the policy network (policy, dest and value heads) and PolicyAgent,
which plays moves (intuition) and orders AND-OR's options (the heuristic deliberation uses)."""
import copy
import random
from types import SimpleNamespace


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



DIRECTIONS = "lrud"


# Target for a destination first reached k moves into an optimal solution:
# DEST_DECAY ** (k - 1), so the destinations of optimal next moves score 1.
DEST_DECAY = 0.8


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
