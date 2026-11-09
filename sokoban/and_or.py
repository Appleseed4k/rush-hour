"""AND-OR backward planning for Sokoban: the counterpart of scripts/and_or.py.

The Rush Hour model regresses from "red exits" through the cars in its way.
Here the root is "every box on a goal", an AND over deliveries:

  Root      every goal filled: pick a (box, goal) pair, deliver it, recurse on
            the rest. Pairs are tried best-first (or in random order), with
            backtracking, so a fill order that dead-ends is undone in planning.
  Deliver   one box to one goal along a shortest route on the empty board, as
            straight segments. Each segment is a PushNode; after one resolves,
            the next is planned from the new state.
  PushNode  (AND) push box b k cells in direction d. It needs every swept cell
            clear and the player able to reach b - d (the stand cell).
            Its blockers are
              * path blockers: boxes on the swept cells (always on);
              * access blockers: boxes on the fewest-box walk from the player to
                the stand cell (`access=True`).
            The blockers are resolved one at a time, trying every order, as in
            Rush Hour.
  OrNode    (OR) the box at x must leave the cells `keep_out`. The candidates
            are its straight pushes of any length that end outside keep_out on
            a live cell. With probability gamma the whole pass lapses
            instead (Lapse), exactly as in scripts/and_or.py.

`protected` holds the cells of boxes that nothing below may move: every box
whose own push is still being resolved, and (`protect_placed=True`) boxes
already on goals. Boxes have no names, so a box is the cell it stands on.
That is stable here, because nothing above a subgoal moves while it's being
resolved.

Pushes cannot be undone. Candidates never end on a dead cell, and with
`freeze=True` a push that freezes a box off a goal is rejected too. Lapse
moves and random fallbacks can still deadlock the board. `play()` then
restarts the level (the default `on_deadlock="restart"`).

Moves are single pushes `(box_cell, direction)`, as in level.py, so plan
lengths compare directly with label.py's optimal push counts.
"""
import heapq
import itertools
import random
from dataclasses import dataclass, field

from level import DIRECTIONS

MAX_DEPTH = 10      # most OrNodes on one chain below a PushNode
BUDGET = 3000       # most OrNode visits in one pass before it gives up

# Settings combinations test_level.py exercises.
VARIANTS_FOR_TESTS = [
    dict(access=False, freeze=False, protect_placed=False),
    dict(access=True, freeze=True, protect_placed=False),
    dict(access=True, freeze=True, protect_placed=True),
]


class Lapse(Exception):
    """An OrNode's stopping check fired: abandon the pass, play `moves` (the
    completed pushes so far plus one lapse push) and end at `state`."""

    def __init__(self, state, moves):
        self.state = state
        self.moves = moves


class OutOfBudget(Exception):
    pass


@dataclass
class Planner:
    """The settings one pass runs with, and its OrNode counter.

    `heuristic(state, pushes) -> {push: score}` orders every choice point
    best-first (as PolicyAgent.heuristic does in scripts/rl.py). Without it
    choices are shuffled with `rng`."""
    level: object
    gamma: float = 0.0
    heuristic: object = None
    access: bool = True
    freeze: bool = True
    protect_placed: bool = True
    max_depth: int = MAX_DEPTH
    budget: int = BUDGET
    rng: random.Random = field(default_factory=random.Random)
    or_visits: int = 0
    # For explaining passes (see push_roles): the (box, goal) delivery being
    # planned, and {(boxes before, push): (depth, delivery)} for every push
    # the pass's PushNodes played, kept or not.
    task: tuple = None
    roles: dict = field(default_factory=dict)

    def tick(self):
        self.or_visits += 1
        if self.or_visits > self.budget:
            raise OutOfBudget()

    def order(self, state, items, pushes_of):
        """`items` best-first by the best heuristic score among pushes_of(item)
        (a list of pushes), or shuffled."""
        items = list(items)
        if self.heuristic is None:
            self.rng.shuffle(items)
            return items
        per_item = [pushes_of(i) for i in items]
        scores = self.heuristic(state, sorted({push for pushes in per_item for push in pushes}))
        best = [max((scores[push] for push in pushes), default=float("-inf")) for pushes in per_item]
        return [item for _, item in sorted(zip(best, items), key=lambda t: t[0], reverse=True)]

    def lapse_move(self, state):
        """The heuristic's top legal push, or a uniformly random one; None if
        no push is legal. Unlike plan pushes, it may deadlock the board."""
        pushes = self.level.pushes(state)
        if not pushes:
            return None
        if self.heuristic is None:
            return self.rng.choice(pushes)
        scores = self.heuristic(state, pushes)
        return max(pushes, key=lambda p: scores[p])

    def placed(self, boxes):
        return frozenset(_cells(boxes & self.level.goals)) if self.protect_placed else frozenset()


def _cells(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def access_path(level, boxes, player, target, impassable):
    """(box cells, path cells) on a walk from the player to `target` that
    crosses the fewest boxes (then the fewest cells), never entering a wall or
    a cell in `impassable`. Box cells are listed nearest the player first.
    ((), cells) if no box is in the way; None if no walk exists."""
    if target in impassable:
        return None
    region = level.reach(boxes, player)
    if region >> target & 1:
        return (), frozenset()
    w = level.width
    best = {}
    heap = []
    for c in _cells(region):
        best[c] = (0, 0)
        heap.append((0, 0, c, None))
    heapq.heapify(heap)
    parent = {}
    while heap:
        nb, steps, c, _ = heapq.heappop(heap)
        if (nb, steps) > best[c]:
            continue
        if c == target:
            break
        for s in (1, -1, w, -w):
            n = c + s
            if not level.floor >> n & 1 or n in impassable:
                continue
            cost = (nb + (boxes >> n & 1), steps + 1)
            if cost < best.get(n, (1 << 30, 0)):
                best[n] = cost
                parent[n] = c
                heapq.heappush(heap, (*cost, n, c))
    if target not in best:
        return None
    path, c = [], target
    while c in parent:
        path.append(c)
        c = parent[c]
    path.reverse()
    return tuple(c for c in path if boxes >> c & 1), frozenset(path)


class PushNode:
    """AND: push the box at b k cells in direction d (see the module docstring)."""

    def __init__(self, planner, state, b, d, k, visited=frozenset(), protected=frozenset(), depth=0):
        self.p, self.state, self.b, self.d, self.k = planner, state, b, d, k
        self.visited, self.protected, self.depth = visited, protected, depth
        s = planner.level.shifts[d]
        self.swept = [b + s * i for i in range(1, k + 1)]
        self.stand = b - s

    def blockers(self, state):
        """{box cell: keep_out} for every box this push waits on in `state`, or
        None if one of them can't be moved or the stand cell can't be reached."""
        level, (boxes, player) = self.p.level, state
        keep_out = frozenset(self.swept) | {self.stand}
        found = {c: keep_out for c in self.swept if boxes >> c & 1}
        route = access_path(level, boxes, player, self.stand, self.protected | {self.b})
        if route is None:
            return None
        path_boxes, path_cells = route
        if path_boxes:
            if not self.p.access:
                return None
            for c in path_boxes:
                found[c] = found.get(c, keep_out) | keep_out | path_cells
        if self.protected & found.keys():
            return None
        return found

    def play(self, state, moves):
        """(new state, moves + the k pushes), or None if the final push would
        freeze a box off a goal (with `freeze`)."""
        level, s = self.p.level, self.p.level.shifts[self.d]
        moves = list(moves)
        for i in range(self.k):
            push = (self.b + s * i, self.d)
            self.p.roles[(state[0], push)] = (self.depth, self.p.task)
            state = level.apply(state, push)
            moves.append(push)
        if self.p.freeze:
            end = self.b + s * self.k
            if level.frozen_off_goal(state[0], [end, end + 1, end - 1, end + level.width, end - level.width]):
                return None
        return state, moves

    def solve(self):
        blockers = self.blockers(self.state)
        if blockers is None:
            return None
        if not blockers:
            return self.play(self.state, [])
        p, level = self.p, self.p.level
        next_protected = self.protected | {self.b}
        ranked = p.order(self.state, blockers.items(), lambda item: [(item[0], d) for d in DIRECTIONS])
        for order in itertools.permutations(ranked):
            working, moves = self.state, []
            for cell, keep_out in order:
                if not working[0] >> cell & 1:
                    continue  # an earlier resolution already moved it
                try:
                    result = OrNode(p, working, cell, keep_out, self.visited, next_protected, self.depth).solve()
                except Lapse as lapse:
                    raise Lapse(lapse.state, moves + lapse.moves) from None
                if result is None:
                    break
                working, sub = result
                moves.extend(sub)
            else:
                boxes, player = working
                if any(boxes >> c & 1 for c in self.swept) or not level.reach(boxes, player) >> self.stand & 1:
                    continue
                result = self.play(working, moves)
                if result is not None:
                    return result
        return None


class OrNode:
    """OR: the box at x must leave every cell in keep_out."""

    def __init__(self, planner, state, x, keep_out, visited=frozenset(), protected=frozenset(), depth=0):
        self.p, self.state, self.x, self.keep_out = planner, state, x, frozenset(keep_out)
        self.visited, self.protected, self.depth = visited, protected, depth

    def candidates(self):
        """(d, k) straight pushes that take the box out of keep_out onto a live
        cell (other boxes in the way are left for the PushNode to resolve)."""
        level, x = self.p.level, self.x
        out = []
        for d in DIRECTIONS:
            s = level.shifts[d]
            if not level.floor >> (x - s) & 1:
                continue
            k = 1
            while level.live >> (x + s * k) & 1:  # a dead cell ends the line: all beyond it are dead too
                if x + s * k not in self.keep_out:
                    out.append((d, k))
                k += 1
        return out

    def solve(self):
        key = (self.x, self.keep_out)
        if key in self.visited or self.depth >= self.p.max_depth:
            return None
        p = self.p
        p.tick()
        if p.gamma > 0 and p.rng.random() < p.gamma:
            move = p.lapse_move(self.state)
            if move is None:
                return None
            raise Lapse(p.level.apply(self.state, move), [move])
        visited = self.visited | {key}
        for d, k in p.order(self.state, self.candidates(), lambda c: [(self.x, c[0])]):
            result = PushNode(p, self.state, self.x, d, k, visited, self.protected, self.depth + 1).solve()
            if result is not None:
                return result
        return None


def segments(level, b, goal):
    """(d, k) straight runs from b along a shortest empty-board route to goal."""
    dist = level.box_distance(goal)
    if b not in dist:
        return []
    out = []
    for d in DIRECTIONS:
        s = level.shifts[d]
        if not level.floor >> (b - s) & 1:
            continue
        k = 1
        while dist.get(b + s * k) == dist[b] - k:
            out.append((d, k))
            k += 1
    return out


def deliver(p, state, b, goal, protected):
    """(state, moves) with the box at b pushed onto goal, segment by segment,
    or None. Backtracks over segment choices; each PushNode's own search is
    first-found, as in Rush Hour."""
    if b == goal:
        return state, []
    s_of = p.level.shifts
    for d, k in p.order(state, segments(p.level, b, goal), lambda c: [(b, c[0])]):
        result = PushNode(p, state, b, d, k, protected=protected).solve()
        if result is None:
            continue
        new_state, moves = result
        try:
            rest = deliver(p, new_state, b + s_of[d] * k, goal, protected)
        except Lapse as lapse:
            raise Lapse(lapse.state, moves + lapse.moves) from None
        if rest is not None:
            return rest[0], moves + rest[1]
    return None


def _pairs(p, state):
    """(loose box, goal) pairs to try delivering. Without protect_placed a
    loose box may also aim at an occupied goal: the box sitting there is then
    a path blocker, pushed on (often to another goal) by the PushNode."""
    level, boxes = p.level, state[0]
    loose = list(_cells(boxes & ~level.goals))
    targets = list(_cells(level.goals & ~boxes if p.protect_placed else level.goals))
    pairs = [(b, g) for b in loose for g in targets if b in level.box_distance(g)]
    return p.order(state, pairs, lambda pair: [(pair[0], d) for d in DIRECTIONS])


def plan_all(p, state, deliveries_left=None):
    """(state, moves) filling every goal, or None: an AND over deliveries,
    ordered and backtracked over (box, goal) pairs. At most twice as many
    deliveries as boxes, so goal shuffles can't cycle."""
    if p.level.solved(state):
        return state, []
    if deliveries_left is None:
        deliveries_left = 2 * p.level.n_boxes
    if deliveries_left == 0:
        return None
    for b, g in _pairs(p, state):
        p.task = (b, g)
        result = deliver(p, state, b, g, p.placed(state[0]))
        if result is None:
            continue
        new_state, moves = result
        try:
            rest = plan_all(p, new_state, deliveries_left - 1)
        except Lapse as lapse:
            raise Lapse(lapse.state, moves + lapse.moves) from None
        if rest is not None:
            return rest[0], moves + rest[1]
    return None


@dataclass
class Pass:
    """One pass from `start`: the pushes it plays, the state they reach, and
    how it ended - "plan" (a full solution), "lapse", "partial" (no full
    plan, one box delivered), "random" (no plan at all: one random push) or
    "stuck" (no legal push)."""
    state: tuple
    moves: list
    kind: str
    or_visits: int = 0
    roles: list = field(default_factory=list)


def push_roles(p, state, moves, kind):
    """What each of a pass's pushes was for, as dicts: role "deliver" (a
    segment of the delivery itself, depth 0) or "blocker" (clearing a box out
    of the way, depth = OrNodes below the delivery), with the delivery's
    "box" (its cell when the delivery was planned) and "goal"; or role
    "lapse" / "random" for the one unplanned push a lapse or a planless pass
    ends with."""
    out = []
    for i, push in enumerate(moves):
        if i == len(moves) - 1 and kind in ("lapse", "random"):
            out.append({"role": kind})
        else:
            depth, (box, goal) = p.roles[(state[0], push)]
            out.append({"role": "deliver" if depth == 0 else "blocker", "depth": depth, "box": box, "goal": goal})
        state = p.level.apply(state, push)
    return out


def solve_pass(p, state):
    """One pass from `state` (see Pass), with each push's role attached."""
    result = _solve_pass(p, state)
    result.roles = push_roles(p, state, result.moves, result.kind)
    return result


def _solve_pass(p, state):
    p.or_visits = 0
    p.roles = {}
    try:
        result = plan_all(p, state)
    except Lapse as lapse:
        return Pass(lapse.state, lapse.moves, "lapse", p.or_visits)
    except OutOfBudget:
        result = None
    if result is not None:
        return Pass(*result, "plan", p.or_visits)

    visits, p.or_visits = p.or_visits, 0
    try:
        for b, g in _pairs(p, state):
            p.task = (b, g)
            result = deliver(p, state, b, g, p.placed(state[0]))
            if result is not None:
                return Pass(*result, "partial", visits + p.or_visits)
    except Lapse as lapse:
        return Pass(lapse.state, lapse.moves, "lapse", visits + p.or_visits)
    except OutOfBudget:
        pass

    pushes = p.level.pushes(state)
    if not pushes:
        return Pass(state, [], "stuck", visits + p.or_visits)
    move = p.rng.choice(pushes)
    return Pass(p.level.apply(state, move), [move], "random", visits + p.or_visits)


def play(level, gamma=0.0, max_pushes=500, on_deadlock="restart", patience=20, seed=None, **settings):
    """Repeated passes from the start until the level is solved or
    `max_pushes` pushes have been played (restarts don't reset the count).

    When a pass leaves the board in a detected deadlock (Level.deadlocked),
    or with no legal push, or after `patience` passes in a row that found no
    plan at all (a deadlock the quick test misses, most often), the level
    restarts (`on_deadlock="restart"`) or play stops ("stop").

    Returns a dict: solved, pushes, restarts, passes ({kind: count}), moves
    (the pushes since the last restart) and log: one dict per pass with its
    kind, the moves it played, their roles (see push_roles) and whether the
    level restarted after it."""
    p = Planner(level, gamma=gamma, rng=random.Random(seed), **settings)
    state, moves, pushes, restarts, idle = level.start, [], 0, 0, 0
    kinds, log = {}, []
    while not level.solved(state) and pushes < max_pushes:
        result = solve_pass(p, state)
        kinds[result.kind] = kinds.get(result.kind, 0) + 1
        room = max_pushes - pushes
        played = result.moves[:room]
        for push in played:
            state = level.apply(state, push)
        moves.extend(played)
        pushes += len(played)
        log.append({"kind": result.kind, "moves": played, "roles": result.roles[:len(played)], "restart": False})
        idle = idle + 1 if result.kind == "random" else 0
        if level.solved(state):
            break
        if result.kind == "stuck" or level.deadlocked(state[0]) or idle >= patience:
            if on_deadlock != "restart":
                break
            log[-1]["restart"] = True
            state, moves, idle, restarts = level.start, [], 0, restarts + 1
    return {"solved": level.solved(state), "pushes": pushes, "restarts": restarts, "passes": kinds,
            "moves": moves, "log": log}
