"""A student that learns Sokoban online, the counterpart of the pure-intuition
student in scripts/rl.py (PolicyAgent, solve_intuition, learning_session).

The student is a policy network over pushes. Each turn it samples a push from
the network's softmax (at a temperature), avoiding pushes that return to one
of its last `memory` boards when it has another. Walking is free, so the
network sees the player only through the region it can reach. Pushes can't be
undone: when the student's quick deadlock test (Level.deadlocked: a box on a
dead cell, or frozen off a goal) fires, or no push is legal, it restarts the
level. Restarts count toward nothing but its push cap. A teacher (teach.py)
may show the optimal push instead on any turn.

After each attempt the student learns from what it saw, as in Rush Hour:
- labels "own": a solved attempt teaches its own winning path (since the last
  restart), loops cut out;
- labels "all" (shortcut labels): hindsight over every board the attempt
  visited, restarts included. Two visited boards are joined when one legal
  push turns one into the other (and, with `window`, the student was at them
  within `window` pushes of each other). Every visited board that reaches the
  solved board is labelled with the first pushes of its shortest routes to
  it, and its distance along them.
- every push a teacher showed teaches too (its board with that push as the
  target), unless a solved attempt already labels the board.

Updates are online (online_update): a few passes over the latest block of
puzzles' examples, nothing replayed. Everything is CPU-sized: Boxoban levels
are 10 x 10, and smaller levels are padded with walls.
"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from level import DIRECTIONS

SIZE = (10, 10)
CHANNELS = 4  # wall, goal, box, reachable


def _plane(mask, level):
    """A level-sized bitmask as a SIZE float array (padding is 0)."""
    n = level.n_cells
    bits = np.unpackbits(np.frombuffer(mask.to_bytes((n + 7) // 8, "little"), dtype=np.uint8),
                         bitorder="little")[:n].astype(np.float32)
    out = np.zeros(SIZE, dtype=np.float32)
    out[:level.height, :level.width] = bits.reshape(level.height, level.width)
    return out


class Encoder:
    """Boards as network input: walls (padding counts as wall), goals, boxes
    and the player's reachable region. The static planes are cached per level."""

    def __init__(self):
        self._static = {}

    def static(self, level):
        key = id(level)
        if key not in self._static:
            if len(self._static) > 64:
                self._static.clear()
            wall = np.ones(SIZE, dtype=np.float32)
            wall[:level.height, :level.width] = 1 - _plane(level.floor, level)[:level.height, :level.width]
            self._static[key] = (level, wall, _plane(level.goals, level))
        return self._static[key][1:]

    def encode(self, level, boxes, region):
        wall, goal = self.static(level)
        return np.stack([wall, goal, _plane(boxes, level), _plane(region, level)])


def action_index(level, push):
    """The network's index for push (box cell, direction): padded cell x 4 + direction."""
    b, d = push
    i, j = divmod(b, level.width)
    return (i * SIZE[1] + j) * 4 + DIRECTIONS.index(d)


class LaneBlock(nn.Module):
    """Residual block with full-row and full-column convolutions, as in
    scripts/rl.py: what a push needs (a clear lane, a goal down the corridor)
    is often several cells away."""

    def __init__(self, hidden):
        super().__init__()
        h, w = SIZE
        self.row = nn.Conv2d(hidden, hidden, (1, 2 * w - 1), padding=(0, w - 1))
        self.col = nn.Conv2d(hidden, hidden, (2 * h - 1, 1), padding=(h - 1, 0))
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.norm = nn.BatchNorm2d(hidden)

    def forward(self, x):
        return F.relu(x + self.norm(self.local(F.relu(self.row(x) + self.col(x)))))


class PolicyNetwork(nn.Module):
    """Policy logits per (cell, direction), and a value head: pushes to go, in tens."""

    def __init__(self, hidden=32, blocks=3):
        super().__init__()
        h, w = SIZE
        rows = torch.linspace(-1, 1, h)[:, None].expand(h, w)
        cols = torch.linspace(-1, 1, w)[None, :].expand(h, w)
        self.register_buffer("coords", torch.stack([rows, cols]))
        self.stem = nn.Conv2d(CHANNELS + 2, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[LaneBlock(hidden) for _ in range(blocks)])
        self.policy_head = nn.Conv2d(hidden, 4, 1)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        h = self.blocks(F.relu(self.stem(x)))
        policy = self.policy_head(h).permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return policy, self.value_head(h.mean((2, 3))).squeeze(1)


class Agent:
    """The network, its optimizer and encoder."""

    def __init__(self, hidden=32, blocks=3, lr=1e-3):
        self.hidden, self.blocks = hidden, blocks
        self.net = PolicyNetwork(hidden, blocks)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.encoder = Encoder()

    def save(self, path):
        torch.save({"hidden": self.hidden, "blocks": self.blocks, "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, lr=1e-3):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        agent = cls(checkpoint["hidden"], checkpoint["blocks"], lr)
        agent.net.load_state_dict(checkpoint["state_dict"])
        return agent

    def logits(self, level, boxes, region):
        if self.net.training:
            self.net.eval()
        x = torch.from_numpy(self.encoder.encode(level, boxes, region)).unsqueeze(0)
        with torch.no_grad():
            return self.net(x)[0][0]


# --- playing ---------------------------------------------------------------------------------------

def solve_intuition(agent, level, max_pushes, memory=10, temperature=1.0, teacher=None, rng=None):
    """One attempt at `level`: up to `max_pushes` pushes (restarts don't reset
    the count), the network's push sampled at `temperature` (0: its top push)
    unless `teacher` shows one. Returns {"solved", "pushes", "restarts",
    "taught" (pushes shown), "segments"}: per stretch between restarts, its
    boards (Level.key, start first) and its pushes as (push, taught) pairs."""
    rng = rng or random
    state = level.start
    start_key = level.key(state)
    segments = [{"boards": [start_key], "moves": []}]
    recent = deque([start_key], maxlen=memory)
    pushes = restarts = taught = 0
    if teacher is not None:
        teacher.start(level, rng)
    while pushes < max_pushes and not level.solved(state):
        boxes, player = state
        region = level.reach(boxes, player)
        legal = level.pushes_from(boxes, region)
        if not legal or level.deadlocked(boxes):
            state, restarts = level.start, restarts + 1
            segments.append({"boards": [start_key], "moves": []})
            recent = deque([start_key], maxlen=memory)
            if teacher is not None:
                teacher.restart(state)
            continue
        proposed = []

        def propose():
            if not proposed:
                nexts = [level.key(level.apply(state, p)) for p in legal]
                fresh = [i for i, k in enumerate(nexts) if k not in recent] or list(range(len(legal)))
                logits = agent.logits(level, boxes, region)[[action_index(level, legal[i]) for i in fresh]]
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=0).tolist()
                    proposed.append(legal[rng.choices(fresh, probs)[0]])
                else:
                    proposed.append(legal[fresh[int(logits.argmax())]])
            return proposed[0]

        push = teacher.advise(state, pushes, propose) if teacher is not None else None
        shown = push is not None
        if not shown:
            push = propose()
        nxt = level.apply(state, push)
        if teacher is not None:
            teacher.observe(state, push, nxt, shown)
        state = nxt
        pushes += 1
        taught += shown
        key = level.key(state)
        segments[-1]["boards"].append(key)
        segments[-1]["moves"].append((push, shown))
        recent.append(key)
    result = {"solved": level.solved(state), "pushes": pushes, "restarts": restarts, "taught": taught,
              "segments": segments}
    if teacher is not None:
        teacher.end_attempt(result)
    return result


# --- labels ----------------------------------------------------------------------------------------

LABELS = ("own", "all")


def _cut_loops(boards, moves, window=None):
    """A path with loops cut out: a board seen again rewinds the path to its
    earlier visit (only if seen within `window` pushes; None: any)."""
    kept_boards, kept_moves, position, seen = [boards[0]], [], {boards[0]: 0}, {boards[0]: 0}
    for t, (board, move) in enumerate(zip(boards[1:], moves), 1):
        if board in position and (window is None or t - seen[board] <= window):
            del kept_boards[position[board] + 1:]
            del kept_moves[position[board]:]
            position = {b: i for i, b in enumerate(kept_boards)}
        else:
            position[board] = len(kept_boards)
            kept_boards.append(board)
            kept_moves.append(move)
        seen[board] = t
    return kept_boards, kept_moves


def labels_of(level, result, labels="all", window=None):
    """{board key: (distance, set of target pushes)} that a solved attempt
    teaches (see the module docstring); {} for an unsolved one."""
    if not result["solved"]:
        return {}
    if labels == "own":
        last = result["segments"][-1]
        boards, moves = _cut_loops(last["boards"], [p for p, _ in last["moves"]], window)
        return {b: (len(moves) - t, {moves[t]}) for t, b in enumerate(boards[:-1])}
    if labels != "all":
        raise ValueError(f"labels: one of {LABELS}, not {labels!r}")
    times, t = {}, 0
    for seg in result["segments"]:
        for i, b in enumerate(seg["boards"]):
            times.setdefault(b, []).append(t + i)
        t += len(seg["moves"])
    goal = result["segments"][-1]["boards"][-1]

    def near(a, b):
        return window is None or any(abs(x - y) <= window for x in times[a] for y in times[b])

    into = {b: [] for b in times}  # board -> [(earlier board, push)] joined into it
    for b in times:
        state = level.from_key(b)
        for p in level.pushes(state):
            n = level.key(level.apply(state, p))
            if n in times and n != b and near(b, n):
                into[n].append((b, p))
    dist, queue = {goal: 0}, [goal]
    for n in queue:
        for b, _ in into[n]:
            if b not in dist:
                dist[b] = dist[n] + 1
                queue.append(b)
    out = {}
    for n in queue:
        for b, p in into[n]:
            if dist[b] == dist[n] + 1:
                out.setdefault(b, (dist[b], set()))[1].add(p)
    return out


class Example:
    __slots__ = ("x", "legal", "targets", "distance")

    def __init__(self, x, legal, targets, distance):
        self.x, self.legal, self.targets, self.distance = x, legal, targets, distance


def examples_from(agent, level, result, labels="all", window=None):
    """Examples one attempt teaches: its labels (labels_of), then the pushes
    a teacher showed on boards the labels don't cover (no distance target)."""
    found = labels_of(level, result, labels, window)
    shown = {}
    for seg in result["segments"]:
        for b, (p, was_taught) in zip(seg["boards"], seg["moves"]):
            if was_taught and b not in found:
                shown.setdefault(b, set()).add(p)
    examples = []
    for b, (distance, targets) in list(found.items()) + [(b, (None, ps)) for b, ps in shown.items()]:
        boxes, player = level.from_key(b)
        region = level.reach(boxes, player)
        legal = [action_index(level, p) for p in level.pushes_from(boxes, region)]
        examples.append(Example(agent.encoder.encode(level, boxes, region), legal,
                                [action_index(level, p) for p in targets], distance))
    return examples


# --- learning --------------------------------------------------------------------------------------

def _loss(agent, batch):
    x = torch.from_numpy(np.stack([e.x for e in batch]))
    n, n_actions = len(batch), SIZE[0] * SIZE[1] * 4
    legal = torch.zeros(n, n_actions, dtype=torch.bool)
    target = torch.zeros(n, n_actions, dtype=torch.bool)
    for r, e in enumerate(batch):
        legal[r, e.legal] = True
        target[r, e.targets] = True
    distance = torch.tensor([e.distance if e.distance is not None else float("nan") for e in batch])
    policy, value = agent.net(x)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    loss = -logp.masked_fill(~target, float("-inf")).logsumexp(1).mean()
    has = ~torch.isnan(distance)
    if has.any():
        loss = loss + 0.5 * F.smooth_l1_loss(value[has], distance[has] / 10)
    return loss


def online_update(agent, examples, steps=1, batch_size=256, rng=None):
    """`steps` passes over just these examples, one Adam step per batch."""
    rng = rng or random
    examples = list(examples)
    agent.net.train()
    for _ in range(steps):
        rng.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            loss = _loss(agent, examples[i:i + batch_size])
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()


def learning_session(agent, levels, teacher=None, max_pushes=200, memory=10, explore=1.0, labels="all",
                     window=None, steps=1, block=1, seed=0, on_puzzle=None, on_attempt=None):
    """The student plays `levels` in order, one attempt each (with `teacher`
    if given), and learns from each block of `block` puzzles' examples once
    they're played (online_update, `steps` passes). `max_pushes`: a number or
    a callable giving a level's cap. Returns one dict per puzzle: solved,
    pushes, taught, restarts, examples. `on_puzzle(n, agent)` is called after
    each update with the puzzles played so far, `on_attempt(i, result)` after
    each attempt."""
    rng = random.Random(seed)
    update_rng = random.Random(seed + 104729)
    record, pending = [], []
    for i, level in enumerate(levels):
        cap = max_pushes(level) if callable(max_pushes) else max_pushes
        result = solve_intuition(agent, level, cap, memory, explore, teacher, rng)
        if on_attempt is not None:
            on_attempt(i, result)
        examples = examples_from(agent, level, result, labels, window)
        pending += examples
        record.append({"solved": result["solved"], "pushes": result["pushes"], "taught": result["taught"],
                       "restarts": result["restarts"], "examples": len(examples)})
        if (i + 1) % block == 0 or i + 1 == len(levels):
            if pending:
                online_update(agent, pending, steps, rng=update_rng)
            pending = []
            if on_puzzle is not None:
                on_puzzle(i + 1, agent)
    return record


def test_attempts(agent, levels, caps, runs=3, temperature=1.0, memory=10, seed=0):
    """Untaught attempts at held-out `levels` ((optimal, level) pairs), `runs`
    each, with random numbers fixed by (seed, puzzle, run) so every student and
    teacher meets the same draws. Returns [{"solved", "pushes", "optimal"}]."""
    out = []
    for i, (optimal, level) in enumerate(levels):
        for r in range(runs):
            rng = random.Random(hash((seed, i, r)) & 0xFFFFFFFF)
            result = solve_intuition(agent, level, caps[i], memory, temperature, None, rng)
            out.append({"solved": result["solved"], "pushes": result["pushes"], "optimal": optimal,
                        "cap": caps[i]})
    return out


def score(attempts):
    """(score, solve rate): score is pushes / optimal with an unsolved attempt
    counted at its cap, as in the Rush Hour notebook's measures(); lower is better."""
    s = [(a["pushes"] if a["solved"] else a["cap"]) / a["optimal"] for a in attempts]
    return float(np.mean(s)), float(np.mean([a["solved"] for a in attempts]))
