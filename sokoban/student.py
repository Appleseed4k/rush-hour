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
import math
import random
from collections import Counter, deque

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
    """Policy logits per (cell, direction), a value head (pushes to go, in tens), and a dead head: the logit that
    the board can no longer be solved, the maximum over cells of a per-cell score (one stuck box is enough). The
    dead head starts at zero weights and bias -3 (every board looks alive), so a network that hasn't learned it plays
    as one without it; checkpoints from before it load with it at that start."""

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
        self.dead_head = nn.Conv2d(hidden, 1, 1)
        nn.init.zeros_(self.dead_head.weight)
        nn.init.constant_(self.dead_head.bias, -3.0)

    def forward(self, x):
        x = torch.cat([x, self.coords.expand(x.shape[0], -1, -1, -1)], dim=1)
        h = self.blocks(F.relu(self.stem(x)))
        policy = self.policy_head(h).permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        dead = self.dead_head(h).flatten(1).max(1).values
        return policy, self.value_head(h.mean((2, 3))).squeeze(1), dead


class Agent:
    """The network, its optimizer and encoder."""

    DEAD_LR = 1e-2  # the dead head is learned from scratch during a session, so it gets its own, faster rate

    def __init__(self, hidden=32, blocks=3, lr=1e-3):
        self.hidden, self.blocks = hidden, blocks
        self.net = PolicyNetwork(hidden, blocks)
        dead = list(self.net.dead_head.parameters())
        rest = [q for q in self.net.parameters() if all(q is not d for d in dead)]
        # (the dead head's parameters get no gradient unless the student learns it, so Adam leaves them alone)
        self.optimizer = torch.optim.Adam([{"params": rest}, {"params": dead, "lr": self.DEAD_LR}], lr=lr)
        self.encoder = Encoder()
        self.freeze_norm = False  # see online_update
        self.dead = False  # learn the dead head from detected deadlocks, and look a push ahead with it in play
        self.critic_beta = 0.0  # > 0: weight a win's labels by the value head's view of each push (examples_from)

    def train_mode(self):
        """Training mode, with BatchNorm kept on its running statistics when
        `freeze_norm` (small online batches would otherwise drag them about)."""
        self.net.train()
        if self.freeze_norm:
            for m in self.net.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

    def save(self, path):
        torch.save({"hidden": self.hidden, "blocks": self.blocks, "state_dict": self.net.state_dict()}, path)

    @classmethod
    def load(cls, path, lr=1e-3):
        checkpoint = torch.load(path, weights_only=False, map_location="cpu")
        agent = cls(checkpoint["hidden"], checkpoint["blocks"], lr)
        missing, unexpected = agent.net.load_state_dict(checkpoint["state_dict"], strict=False)
        if unexpected or any(not k.startswith("dead_head.") for k in missing):
            raise RuntimeError(f"checkpoint {path}: missing {missing}, unexpected {unexpected}")
        return agent

    def logits(self, level, boxes, region):
        if self.net.training:
            self.net.eval()
        x = torch.from_numpy(self.encoder.encode(level, boxes, region)).unsqueeze(0)
        with torch.no_grad():
            return self.net(x)[0][0]

    def evaluate(self, level, states):
        """(value in pushes, chance of dead) for each (boxes, player) state."""
        if self.net.training:
            self.net.eval()
        x = torch.from_numpy(np.stack([self.encoder.encode(level, b, level.reach(b, p)) for b, p in states]))
        with torch.no_grad():
            _, value, dead = self.net(x)
        return (value * 10).numpy(), torch.sigmoid(dead).numpy()


# --- playing ---------------------------------------------------------------------------------------

def solve_intuition(agent, level, max_pushes, memory=10, temperature=1.0, teacher=None, rng=None):
    """One attempt at `level`: up to `max_pushes` pushes (restarts don't reset
    the count), the network's push sampled at `temperature` (0: its top push)
    unless `teacher` shows one. Returns {"solved", "pushes", "restarts",
    "taught" (pushes shown), "segments"}: per stretch between restarts, its
    boards (Level.key, start first), its pushes as (push, taught) pairs, and
    "dead": True if it ended in a detected deadlock (the student restarted);
    and "undone": pushes the teacher took back (Teacher.veto) before they were
    played, which don't count toward `max_pushes`; "forced": restarts the
    teacher forced (Teacher.force_restart; also in "restarts", and the
    stretch they end is marked "forced")."""
    rng = rng or random
    state = level.start
    start_key = level.key(state)
    segments = [{"boards": [start_key], "moves": []}]
    recent = deque([start_key], maxlen=memory)
    pushes = restarts = taught = undone = forced = 0
    if teacher is not None:
        teacher.start(level, rng)
    while pushes < max_pushes and not level.solved(state):
        boxes, player = state
        region = level.reach(boxes, player)
        legal = level.pushes_from(boxes, region)
        if not legal or level.deadlocked(boxes):
            segments[-1]["dead"] = True  # the stretch ended in a detected deadlock (or no legal push)
            state, restarts = level.start, restarts + 1
            segments.append({"boards": [start_key], "moves": []})
            recent = deque([start_key], maxlen=memory)
            if teacher is not None:
                teacher.restart(state)
            continue
        if teacher is not None and teacher.force_restart(state, len(segments[-1]["moves"])):
            segments[-1]["forced"] = True
            state, restarts, forced = level.start, restarts + 1, forced + 1
            segments.append({"boards": [start_key], "moves": []})
            recent = deque([start_key], maxlen=memory)
            teacher.restart(state)
            continue
        proposed, vetoed = [], set()

        def propose():
            if not proposed:
                nexts = [level.key(level.apply(state, p)) for p in legal]
                allowed = [i for i in range(len(legal)) if legal[i] not in vetoed]
                fresh = [i for i in allowed if nexts[i] not in recent] or allowed
                if agent.dead and len(fresh) > 1:  # skip pushes onto boards that look dead, unless all do
                    _, dead = agent.evaluate(level, [level.apply(state, legal[i]) for i in fresh])
                    fresh = [i for i, p in zip(fresh, dead) if p < 0.5] or fresh
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
            # A push the teacher takes back: the student chooses again without it. A veto only stops a push that
            # would deadlock a solvable board, and such a board always has a push that doesn't, so this ends.
            while teacher is not None and teacher.veto(state, push):
                undone += 1
                vetoed.add(push)
                proposed.clear()
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
              "undone": undone, "forced": forced, "segments": segments}
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
    """A labelled board: push targets (and a distance) to learn toward, or -
    a negative label - pushes to `avoid`, weighing `weight` in the loss."""
    __slots__ = ("x", "legal", "targets", "distance", "avoid", "weight", "dead", "critic")

    def __init__(self, x, legal, targets, distance, avoid=(), weight=1.0, dead=None, critic=1.0):
        self.x, self.legal, self.targets, self.distance = x, legal, targets, distance
        self.avoid, self.weight, self.dead = list(avoid), weight, dead  # dead: a target for the dead head, or None
        self.critic = critic  # a further weight on the policy target only (critic_weights)


def fatal_pushes(result):
    """{board key: pushes} the student played into a deadlock its quick test
    then caught: the last push of every stretch that ended in a restart. (A
    deadlock the test catches late is blamed on the push just before it.)"""
    out = {}
    for seg in result["segments"]:
        if seg.get("dead") and seg["moves"]:
            push, taught = seg["moves"][-1]
            if not taught:
                out.setdefault(seg["boards"][-2], set()).add(push)
    return out


def optimal_only(level, found):
    """`found` (labels_of's result) with every target push the oracle doesn't call optimal taken out, and boards
    left with none dropped: a check on how much the wrong labels in a win cost. Distance targets are kept."""
    if not found:
        return found
    from teach import oracle
    space = oracle(level.to_line())
    out = {}
    for b, (distance, targets) in found.items():
        state = level.from_key(b)
        d = space.distance(state)
        if d is None or d <= 0:
            continue
        keep = {p for p in targets if space.distance(level.apply(state, p)) == d - 1}
        if keep:
            out[b] = (distance, keep)
    return out


def critic_weights(agent, level, found):
    """{board: weight} for a win's labelled boards: the value head's view of the labelled push, advantage =
    value before - value after - 1 (0: a push that brings the board one closer; below: one that wastes pushes, or
    leads somewhere that looks dead), weight = min(1, exp(advantage / agent.critic_beta)). Several target pushes:
    the best of them. The student judges with the network it has before learning from this attempt."""
    boards = list(found)
    if not boards:
        return {}
    states = [level.from_key(b) for b in boards]
    nexts = [(i, level.apply(states[i], p)) for i, b in enumerate(boards) for p in found[b][1]]
    values, dead = agent.evaluate(level, states + [s for _, s in nexts])
    before, after, after_dead = values[:len(states)], values[len(states):], dead[len(states):]
    best = np.full(len(boards), -np.inf)
    for (i, _), v, pd in zip(nexts, after, after_dead):
        if agent.dead and pd >= 0.5:
            v = max(v, before[i] + 10)  # looks dead: as bad as ten pushes wasted
        best[i] = max(best[i], before[i] - v - 1)
    return {b: float(min(1.0, math.exp(a / agent.critic_beta))) for b, a in zip(boards, best)}


def dead_examples(agent, level, result):
    """Dead-head examples from one attempt: every board its quick test caught as deadlocked (or with no legal
    push) is dead; the start board is alive (labelled boards of a win are marked alive in examples_from). No policy
    or distance targets."""
    out = []
    boards = [(b, True) for seg in result["segments"] if seg.get("dead") for b in seg["boards"][-1:]]
    boards.append((result["segments"][0]["boards"][0], False))
    for b, dead in dict(boards).items():
        boxes, player = level.from_key(b)
        out.append(Example(agent.encoder.encode(level, boxes, level.reach(boxes, player)), [], [], None, dead=dead))
    return out


def examples_from(agent, level, result, labels="all", window=None, shown_unsolved=True, negative_weight=0.0,
                  optimal=False):
    """Examples one attempt teaches: its labels (labels_of; with `optimal`,
    only the oracle-optimal ones, see optimal_only), then the pushes
    a teacher showed on boards the labels don't cover (no distance target) -
    with `shown_unsolved` False, only in solved attempts. With
    `negative_weight` > 0, also a negative label for every push the student
    played into a detected deadlock (fatal_pushes), on a board with another
    legal push: learned as "not this push here", weighing `negative_weight`."""
    found = labels_of(level, result, labels, window)
    if optimal:
        found = optimal_only(level, found)
    shown = {}
    for seg in (result["segments"] if shown_unsolved or result["solved"] else []):
        for b, (p, was_taught) in zip(seg["boards"], seg["moves"]):
            if was_taught and b not in found:
                shown.setdefault(b, set()).add(p)
    weights = critic_weights(agent, level, found) if agent.critic_beta > 0 else {}
    examples = []
    for b, (distance, targets) in list(found.items()) + [(b, (None, ps)) for b, ps in shown.items()]:
        boxes, player = level.from_key(b)
        region = level.reach(boxes, player)
        legal = [action_index(level, p) for p in level.pushes_from(boxes, region)]
        examples.append(Example(agent.encoder.encode(level, boxes, region), legal,
                                [action_index(level, p) for p in targets], distance, critic=weights.get(b, 1.0),
                                dead=False if agent.dead and b in found else None))
    if agent.dead:
        examples += dead_examples(agent, level, result)
    if negative_weight > 0:
        for b, pushes in fatal_pushes(result).items():
            boxes, player = level.from_key(b)
            region = level.reach(boxes, player)
            legal = [action_index(level, p) for p in level.pushes_from(boxes, region)]
            if len(legal) > 1:
                examples.append(Example(agent.encoder.encode(level, boxes, region), legal, [], None,
                                        [action_index(level, p) for p in pushes], negative_weight))
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
    policy, value, dead = agent.net(x)
    aux = [r for r, e in enumerate(batch) if not e.targets and not e.avoid]  # dead-head examples only
    if aux:
        keep = [r for r in range(n) if r not in set(aux)]
        dead_rows = [r for r, e in enumerate(batch) if e.dead is not None]
        dead_loss = F.binary_cross_entropy_with_logits(
            dead[dead_rows], torch.tensor([float(batch[r].dead) for r in dead_rows])) if dead_rows else 0.0
        rest = _loss_rows(batch, keep, policy, value, legal, target, distance) if keep else 0.0
        return rest + 0.5 * dead_loss
    dead_rows = [r for r, e in enumerate(batch) if e.dead is not None]
    loss = _loss_rows(batch, list(range(n)), policy, value, legal, target, distance)
    if dead_rows:
        loss = loss + 0.5 * F.binary_cross_entropy_with_logits(
            dead[dead_rows], torch.tensor([float(batch[r].dead) for r in dead_rows]))
    return loss


def _loss_rows(batch, rows, policy, value, legal, target, distance):
    """The policy and value loss over `rows` of the batch (all of them have policy targets or pushes to avoid)."""
    if len(rows) < len(batch):
        idx = torch.tensor(rows)
        batch = [batch[r] for r in rows]
        policy, value, legal, target, distance = policy[idx], value[idx], legal[idx], target[idx], distance[idx]
    n, n_actions = len(batch), policy.shape[1]
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    negative = [r for r, e in enumerate(batch) if e.avoid]
    weighted = not negative and any(e.weight != 1.0 or e.critic != 1.0 for e in batch)
    if weighted:  # positives with weights (balance, critic): weighted sum over the batch, divided by its size
        w = torch.tensor([e.weight for e in batch], dtype=torch.float32)
        wp = w * torch.tensor([e.critic for e in batch], dtype=torch.float32)
        loss = -(wp * logp.masked_fill(~target, float("-inf")).logsumexp(1)).sum() / n
    elif not negative:
        loss = -logp.masked_fill(~target, float("-inf")).logsumexp(1).mean()
    else:
        # Positive labels: probability on the targets. Negative labels (unlikelihood): -log(1 - probability on
        # the pushes to avoid), which lowers them without saying which other push is right.
        positive = [r for r, e in enumerate(batch) if not e.avoid]
        avoid = torch.zeros(n, n_actions, dtype=torch.bool)
        for r in negative:
            avoid[r, batch[r].avoid] = True
        terms = []
        if positive:
            terms.append(-logp[positive].masked_fill(~target[positive], float("-inf")).logsumexp(1))
        p_avoid = logp[negative].masked_fill(~avoid[negative], float("-inf")).logsumexp(1).exp()
        weights = torch.tensor([batch[r].weight for r in negative])
        terms.append(-weights * torch.log1p(-p_avoid.clamp(max=1 - 1e-6)))
        loss = torch.cat(terms).sum() / n
    has = ~torch.isnan(distance)
    if has.any() and weighted:
        loss = loss + 0.5 * (w[has] * F.smooth_l1_loss(value[has], distance[has] / 10, reduction="none")).sum() / n
    elif has.any():
        loss = loss + 0.5 * F.smooth_l1_loss(value[has], distance[has] / 10)
    return loss


def online_update(agent, examples, steps=1, batch_size=256, rng=None, even_batches=False):
    """`steps` passes over just these examples, one Adam step per batch.

    Batches: by default `batch_size` examples each, the last holding what is
    left - and Adam steps about as far on a remainder of 5 examples as on a
    full batch. `even_batches` (as in Rush Hour's rl.online_update) splits
    each pass into the same number of batches, ceil(n / batch_size), of equal
    size instead: the same number of steps, none on a handful of examples."""
    rng = rng or random
    examples = list(examples)
    # Dead-head examples (no policy target) ride along in the batches the others make, without adding steps.
    aux = [e for e in examples if not e.targets and not e.avoid]
    if aux:
        examples = [e for e in examples if e.targets or e.avoid] or aux
        if examples is aux:
            aux = []
    agent.train_mode()
    for _ in range(steps):
        rng.shuffle(examples)
        if even_batches:
            k = max(1, math.ceil(len(examples) / batch_size))
            bounds = [round(j * len(examples) / k) for j in range(k + 1)]
        else:
            bounds = list(range(0, len(examples), batch_size)) + [len(examples)]
        rng.shuffle(aux)
        k = len(bounds) - 1
        aux_bounds = [round(j * len(aux) / k) for j in range(k + 1)]
        for (i, j), (a, b) in zip(zip(bounds[:-1], bounds[1:]), zip(aux_bounds[:-1], aux_bounds[1:])):
            loss = _loss(agent, examples[i:j] + aux[a:b])
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()


def _balance(examples, groups, target, seen):
    """Sets each example's weight from its group: target share / the group's share so far, capped at 5,
    rescaled to average 1."""
    total = sum(seen.values())
    raw = [min(5.0, target[g] / (seen[g] / total)) for g in groups]
    mean = sum(raw) / len(raw)
    for e, w in zip(examples, raw):
        e.weight *= w / mean


def learning_session(agent, levels, teacher=None, max_pushes=200, memory=10, explore=1.0, labels="all",
                     window=None, steps=1, block=1, seed=0, on_puzzle=None, on_attempt=None, replay=None,
                     updates=0, shown_unsolved=True, negative_weight=0.0, batch_size=256, even_batches=False,
                     optimal_labels=False, balance=None):
    """The student plays `levels` in order, one attempt each (with `teacher`
    if given), and learns from each block of `block` puzzles' examples once
    they're played (online_update, `steps` passes). `max_pushes`: a number or
    a callable giving a level's cap. Returns one dict per puzzle: solved,
    pushes, taught, restarts, examples. `on_puzzle(n, agent)` is called after
    each update with the puzzles played so far, `on_attempt(i, result)` after
    each attempt. `negative_weight` > 0 adds negative labels for pushes played
    into a detected deadlock (see examples_from).

    `balance` = (group, target): `group(level)` names a level's group (a
    difficulty bin, say) and `target` {group: share} the share of the
    learning each group should get. Each example is weighted target / the
    group's share of all examples so far this session (capped at 5), the
    weights rescaled to average 1 in each update: a teacher that floods the
    student with one group's examples can't crowd the others out.

    `replay` (a number of examples) is for pretraining, not for sessions
    meant to be human-like: examples go into a buffer of the latest `replay`,
    and each update is instead `updates` gradient steps on batches drawn from
    the whole buffer."""
    rng = random.Random(seed)
    update_rng = random.Random(seed + 104729)
    record, pending = [], []
    seen, groups = Counter(), []  # examples so far per group, and each pending example's group (balance)
    buffer = deque(maxlen=replay) if replay else None
    for i, level in enumerate(levels):
        cap = max_pushes(level) if callable(max_pushes) else max_pushes
        result = solve_intuition(agent, level, cap, memory, explore, teacher, rng)
        if on_attempt is not None:
            on_attempt(i, result)
        examples = examples_from(agent, level, result, labels, window, shown_unsolved, negative_weight,
                                 optimal_labels)
        if balance is not None:
            g = balance[0](level)
            seen[g] += len(examples)
            groups += [g] * len(examples)
        pending += examples
        record.append({"solved": result["solved"], "pushes": result["pushes"], "taught": result["taught"],
                       "undone": result["undone"], "forced": result["forced"],
                       "restarts": result["restarts"],
                       "examples": sum(1 for e in examples if e.targets or e.avoid),
                       "dead_examples": sum(1 for e in examples if e.dead and not e.targets),
                       "negatives": sum(1 for e in examples if e.avoid)})
        if getattr(teacher, "predicted", None) is not None:  # a Bayes teacher's forecast of solving untaught
            record[-1]["predicted"] = teacher.predicted
        if (i + 1) % block == 0 or i + 1 == len(levels):
            if buffer is not None:
                buffer.extend(pending)
                if buffer:
                    agent.train_mode()
                    for _ in range(updates):
                        loss = _loss(agent, update_rng.sample(list(buffer), min(256, len(buffer))))
                        agent.optimizer.zero_grad()
                        loss.backward()
                        agent.optimizer.step()
            elif pending:
                if balance is not None:
                    _balance(pending, groups, balance[1], seen)
                online_update(agent, pending, steps, batch_size, update_rng, even_batches)
            pending, groups = [], []
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
