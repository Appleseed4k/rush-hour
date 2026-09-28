import functools
import itertools
import math
import random

from lib import DELTAS, read_puzzles, visualize

BOARD_SIZE = 6
# Largest single-pass plan (in OrNodes) the plan analyses below search for.
# Uncapped, their search wanders into exponentially deep dead ends on some
# boards; raising this from 12 to 15 changed no first path on a 1024-state
# sample of the training puzzles.
MAX_PLAN_SIZE = 12


class GammaLapse(Exception):
    """Raised by OrNode.solve() when its stopping probability fires, unwinding the
    whole in-progress search all the way to the top-level solve()."""

    def __init__(self, new_state, moves):
        self.new_state = new_state
        self.moves = moves


def legal_moves(state):
    """Every (car_name, direction, steps) currently legal for any car."""
    occupied = {cell: name for name, positions in state for cell in positions}
    moves = []
    for car_name, positions in state:
        horizontal = positions[0][0] == positions[1][0]
        axis_pos = [j for _, j in positions] if horizontal else [i for i, _ in positions]
        cross = positions[0][0] if horizontal else positions[0][1]
        lo, hi = min(axis_pos), max(axis_pos)
        for direction in (('l', 'r') if horizontal else ('u', 'd')):
            delta = DELTAS[direction][1 if horizontal else 0]
            steps = 1
            while True:
                edge = hi + steps if delta > 0 else lo - steps
                if not (0 <= edge < BOARD_SIZE):
                    break
                cell = (cross, edge) if horizontal else (edge, cross)
                occupant = occupied.get(cell)
                if occupant is not None and occupant != car_name:
                    break
                moves.append((car_name, direction, steps))
                steps += 1
    return moves


def order_candidates(state, car_name, candidates, heuristic):
    """(direction, steps) candidates for car_name, best-first per `heuristic` if
    given, else in random order. `heuristic(state, actions)` must return a dict
    scoring every action - the same shape PolicyAgent.heuristic returns."""
    if heuristic is None:
        random.shuffle(candidates)
        return candidates
    actions = [(car_name, direction, steps) for direction, steps in candidates]
    scores = heuristic(state, actions)
    actions.sort(key=lambda a: scores[a], reverse=True)
    return [(direction, steps) for _, direction, steps in actions]


def first_solve(state, car_name, candidates, visited=frozenset(), protected=frozenset(), heuristic=None, gamma=0.0):
    """Tries each (direction, steps) candidate for car_name, best-first per
    `heuristic` if given, returning the first (new_state, moves) that resolves,
    or None if every candidate dead-ends."""
    candidates = order_candidates(state, car_name, candidates, heuristic)
    for direction, steps in candidates:
        result = AndNode(state, car_name, direction, steps, visited, protected, heuristic, gamma).solve()
        if result is not None:
            return result
    return None


class OrNode:
    """Subgoal: car_name must vacate every cell in `collisions`. With probability
    `gamma`, abandons the entire search (not just this subgoal) for a single
    other move instead - the heuristic's top-scoring legal move if `heuristic`
    is given, else a uniformly random one.

    `heuristic`, if given, also orders this subgoal's own candidate
    resolutions best-first (see order_candidates) instead of randomly.
    """

    def __init__(self, state, car_name, collisions, visited=frozenset(), protected=frozenset(), heuristic=None,
                 gamma=0.0):
        self.state = state
        self.car_name = car_name
        self.car = dict(state)[car_name]
        self.collisions = frozenset(collisions)
        self.visited = visited
        self.protected = protected
        self.heuristic = heuristic
        self.gamma = gamma

    def directions(self):
        """(direction, steps) candidates that clear the car off every collision cell."""
        horizontal = self.car[0][0] == self.car[1][0]
        if horizontal:
            axis_pos = [j for _, j in self.car]
            axes = [j for i, j in self.collisions if i == self.car[0][0]]
            neg, pos = "l", "r"
        else:
            axis_pos = [i for i, _ in self.car]
            axes = [i for i, j in self.collisions if j == self.car[0][1]]
            neg, pos = "u", "d"
        if not axes:
            return []

        lo, hi = min(axis_pos), max(axis_pos)
        near, far = min(axes), max(axes)
        candidates = []
        # Positive (right/down): low edge must pass the farthest collision.
        min_steps, max_steps = far - lo + 1, (BOARD_SIZE - 1) - hi
        candidates += [(pos, s) for s in range(max(1, min_steps), max_steps + 1)]
        # Negative (left/up): high edge must pass the nearest collision.
        min_steps, max_steps = hi - near + 1, lo
        candidates += [(neg, s) for s in range(max(1, min_steps), max_steps + 1)]
        return candidates

    def solve(self):
        """(new_state, moves) via depth-first search over candidate actions, or
        None if every candidate dead-ends."""
        key = (self.car_name, self.collisions)
        if key in self.visited:
            return None
        next_visited = self.visited | {key}

        if random.random() < self.gamma:
            moves = legal_moves(self.state)
            if not moves:
                return None
            if self.heuristic is None:
                car_name, direction, steps = random.choice(moves)
            else:
                # Defer to the apprentice's own greedy pick instead of a blind
                # random move, so a lapse still yields a demonstration worth
                # imitating once a heuristic is actually available (round 2+) -
                # otherwise every lapse injects pure noise regardless of how
                # good the trained policy already is.
                scores = self.heuristic(self.state, moves)
                car_name, direction, steps = max(moves, key=lambda move: scores[move])
            node = AndNode(self.state, car_name, direction, steps)
            raise GammaLapse(node.apply(self.state), [(car_name, direction, steps)])

        return first_solve(self.state, self.car_name, self.directions(), next_visited, self.protected,
                            self.heuristic, self.gamma)


class AndNode:
    """Action: move car_name `steps` cells in `direction`. Solvable once every car
    occupying a swept cell has vacated it (AND semantics).

    `heuristic`, if given, also decides the order multiple simultaneous
    blockers are tried in first (see solve()) - doesn't change whether a
    state solves, only which order of candidate moves the search commits to
    first, and which it falls back to when that order dead-ends.
    """

    def __init__(self, state, car_name, direction, steps, visited=frozenset(), protected=frozenset(), heuristic=None,
                 gamma=0.0):
        self.state = state
        self.car_name = car_name
        self.direction = direction
        self.steps = steps
        self.visited = visited
        self.protected = protected
        self.heuristic = heuristic
        self.gamma = gamma

    def swept_cells(self):
        """Cells this move newly enters, nearest first."""
        car = dict(self.state)[self.car_name]
        di, dj = DELTAS[self.direction]
        if di:
            axis_pos, cross, delta = [i for i, _ in car], car[0][1], di
        else:
            axis_pos, cross, delta = [j for _, j in car], car[0][0], dj
        lo, hi = min(axis_pos), max(axis_pos)
        if delta > 0:
            axis_cells = range(hi + 1, hi + self.steps + 1)
        else:
            axis_cells = range(lo - 1, lo - self.steps - 1, -1)
        return [(pos, cross) if di else (cross, pos) for pos in axis_cells]

    def blockers(self):
        """Every swept cell each other car occupies, grouped by car."""
        occupied = {
            cell: name
            for name, positions in self.state
            if name != self.car_name
            for cell in positions
        }
        blockers = {}
        for cell in self.swept_cells():
            occupant = occupied.get(cell)
            if occupant is not None:
                blockers.setdefault(occupant, []).append(cell)
        return blockers

    def apply(self, state):
        di, dj = DELTAS[self.direction]
        car = dict(state)[self.car_name]
        new_car = tuple((i + di * self.steps, j + dj * self.steps) for i, j in car)
        return tuple((n, new_car if n == self.car_name else pos) for n, pos in state)

    def _order_blockers(self, order):
        """`order` (blocker_name, collisions) pairs, best-first per self.heuristic:
        ranks each blocker by its single best-scoring candidate. Falls back to a
        random order when self.heuristic is None."""
        if self.heuristic is None:
            random.shuffle(order)
            return order

        all_actions = []
        blocker_actions = {}
        for blocker_name, collisions in order:
            candidates = OrNode(self.state, blocker_name, collisions, self.visited, self.protected).directions()
            actions = [(blocker_name, direction, steps) for direction, steps in candidates]
            blocker_actions[blocker_name] = actions
            all_actions.extend(actions)
        if not all_actions:
            return order

        scores = self.heuristic(self.state, all_actions)
        best_score = {
            blocker_name: max((scores[a] for a in actions), default=float("-inf"))
            for blocker_name, actions in blocker_actions.items()
        }
        return sorted(order, key=lambda item: best_score[item[0]], reverse=True)

    def solve(self):
        """(new_state, moves) if some order of resolving every blocker leaves
        the path clear, else None.

        Tries every permutation of the blockers, best-first per
        _order_blockers (or shuffled, without a heuristic) - not just that
        one preferred order. Resolving blockers in a particular order can
        dead-end even when a different order would succeed (e.g. one
        blocker's own fix re-obstructs a cell a later blocker still needs),
        so committing to a single order and giving up on failure would make
        heuristic guidance strictly less capable than plain random search:
        random search gets a fresh shuffled order (and thus a fresh chance)
        on every independent top-level attempt, while a heuristic computes
        the same "best" order for the same state every time, with no
        built-in retry diversity of its own. Backtracking here restores that
        floor - within a single attempt, a heuristic can therefore never
        solve fewer states than exhaustive search over blocker orders would.
        """
        blockers = self.blockers()
        if self.protected & blockers.keys():
            return None

        ranked = self._order_blockers(list(blockers.items()))
        next_protected = self.protected | {self.car_name}
        for order in itertools.permutations(ranked):
            working_state = self.state
            moves = []
            for blocker_name, collisions in order:
                try:
                    result = OrNode(working_state, blocker_name, collisions, self.visited, next_protected,
                                     self.heuristic, self.gamma).solve()
                except GammaLapse as lapse:
                    raise GammaLapse(lapse.new_state, moves + lapse.moves) from None
                if result is None:
                    break
                working_state, blocker_moves = result
                moves.extend(blocker_moves)
            else:
                if not AndNode(working_state, self.car_name, self.direction, self.steps, self.visited).blockers():
                    final_state = self.apply(working_state)
                    moves.append((self.car_name, self.direction, self.steps))
                    return final_state, moves
        return None


def blockers_in_path(state):
    """Cars currently occupying red's direct slide to the exit."""
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    if steps <= 0:
        return 0
    return len(AndNode(state, 'red', 'r', steps).blockers())


def red_candidates(state, exclude_steps):
    """(direction, steps) options for repositioning red itself, tried once the
    direct slide to the exit (`exclude_steps`) has failed."""
    red = dict(state)['red']
    axis_pos = [j for _, j in red]
    lo, hi = min(axis_pos), max(axis_pos)
    max_right = (BOARD_SIZE - 1) - hi
    candidates = [('r', s) for s in range(1, max_right + 1) if s != exclude_steps]
    candidates += [('l', s) for s in range(1, lo + 1)]
    return candidates


def _match_and(state, car_name, direction, steps, visited, protected, target, pos):
    """Yields (new_state, new_pos) for every way AndNode(...).solve() could,
    with some choice of shuffles and no lapse, return exactly
    target[pos:new_pos]. Mirrors AndNode.solve's structure step for step."""
    move = (car_name, direction, steps)
    # This action's own move comes last in its plan, so it must still be ahead.
    if move not in target[pos:]:
        return
    node = AndNode(state, car_name, direction, steps)
    blockers = node.blockers()
    if protected & blockers.keys():
        return
    next_protected = protected | {car_name}
    for order in itertools.permutations(blockers.items()):
        yield from _match_order(state, order, move, visited, next_protected, target, pos)


def _match_order(state, order, move, visited, protected, target, pos):
    """Resolves `order`'s blockers in sequence against target, then plays `move`."""
    if not order:
        if pos < len(target) and target[pos] == move and not AndNode(state, *move).blockers():
            yield AndNode(state, *move).apply(state), pos + 1
        return
    (blocker_name, collisions), rest = order[0], order[1:]
    for new_state, new_pos in _match_or(state, blocker_name, collisions, visited, protected, target, pos):
        yield from _match_order(new_state, rest, move, visited, protected, target, new_pos)


def _match_or(state, car_name, collisions, visited, protected, target, pos):
    """OrNode.solve's counterpart to _match_and."""
    key = (car_name, frozenset(collisions))
    if key in visited:
        return
    next_visited = visited | {key}
    node = OrNode(state, car_name, collisions)
    for direction, steps in node.directions():
        yield from _match_and(state, car_name, direction, steps, next_visited, protected, target, pos)


def is_single_pass(state, moves):
    """True if `moves` is a complete plan that one lapse-free solve() pass from
    `state` could return (for some ordering of its random choices) - i.e. the
    direct red-to-exit search resolving with exactly these moves.

    Any successful branch can be made the one the depth-first search returns
    by ordering it first at every choice point, so this only asks whether
    such a branch exists, never how likely it is: no gamma needed."""
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    if steps <= 0:
        return not moves
    moves = list(moves)
    return any(new_pos == len(moves)
               for _, new_pos in _match_and(state, 'red', 'r', steps, frozenset(), frozenset(), moves, 0))


def final_pass_start(state, moves):
    """Estimated index into `moves` where the final (successful) solve() pass
    began, from the moves alone: the earliest t whose remaining moves[t:]
    could have been one lapse-free pass from the state reached after
    moves[:t]. Every move from there on is consistent with a single committed
    plan.

    The true start always qualifies, so this never lands after it. It can land
    before it when the last pre-boundary moves happen to fit the final plan
    too (a lapse's random move matching the plan's first move, say) - those
    cases are indistinguishable from the moves alone. None if no suffix
    qualifies (e.g. `moves` never reaches the goal)."""
    states = [state]
    for car_name, direction, steps in moves:
        states.append(AndNode(states[-1], car_name, direction, steps).apply(states[-1]))
    for t in range(len(moves) + 1):
        if is_single_pass(states[t], moves[t:]):
            return t
    return None


def _plans_and(state, car_name, direction, steps, visited, protected, depth_left, size_left, first_left=None):
    """Yields (new_state, moves, depth, ors, first_path) for every plan
    AndNode(...).solve() could return without lapsing whose OR-node tree
    fits the limits: depth is the most OrNodes on any root-to-leaf chain,
    ors the car of every OrNode in the order the depth-first search visits
    them (one lapse check each, so len(ors) is the plan's size), and
    first_path the cars on the chain down to the first move the plan
    reaches - the OrNodes a pass must get through before it has any
    completed branch to play out on a lapse (see AndNode.solve's GammaLapse
    handling). `first_left`, if given, caps len(first_path). Mirrors
    AndNode.solve's structure step for step."""
    blockers = AndNode(state, car_name, direction, steps).blockers()
    if protected & blockers.keys():
        return
    next_protected = protected | {car_name}
    for order in itertools.permutations(blockers.items()):
        yield from _plans_order(state, order, (car_name, direction, steps), visited, next_protected,
                                depth_left, size_left, first_left, [], 0, [], [])


def _plans_order(state, order, move, visited, protected, depth_left, size_left, first_left,
                 moves, depth, ors, first_path):
    """Resolves `order`'s blockers in sequence, then plays `move`. Only the
    first blocker lies on the first path down, so only it gets `first_left`."""
    if not order:
        if not AndNode(state, *move).blockers():
            yield AndNode(state, *move).apply(state), moves + [move], depth, ors, first_path
        return
    (blocker_name, collisions), rest = order[0], order[1:]
    is_first = not ors
    for new_state, blocker_moves, sub_depth, sub_ors, sub_first in _plans_or(
            state, blocker_name, collisions, visited, protected, depth_left, size_left - len(ors),
            first_left if is_first else None):
        yield from _plans_order(new_state, rest, move, visited, protected, depth_left, size_left, first_left,
                                moves + blocker_moves, max(depth, sub_depth), ors + sub_ors,
                                sub_first if is_first else first_path)


def _plans_or(state, car_name, collisions, visited, protected, depth_left, size_left, first_left=None):
    """OrNode.solve's counterpart to _plans_and; this node itself counts once
    toward depth and size, visited before everything beneath it."""
    if depth_left < 1 or size_left < 1 or (first_left is not None and first_left < 1):
        return
    key = (car_name, frozenset(collisions))
    if key in visited:
        return
    next_visited = visited | {key}
    for direction, steps in OrNode(state, car_name, collisions).directions():
        for new_state, moves, depth, ors, first_path in _plans_and(
                state, car_name, direction, steps, next_visited, protected, depth_left - 1, size_left - 1,
                None if first_left is None else first_left - 1):
            yield new_state, moves, depth + 1, [car_name] + ors, [car_name] + first_path


def _first_plan(state, depth_left, size_left, first_left=None):
    """First lapse-free single-pass plan from `state` within the limits, as
    _plans_and yields it, or None."""
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    return next(_plans_and(state, 'red', 'r', steps, frozenset(), frozenset(), depth_left, size_left, first_left),
                None)


def plan_metrics(state, max_size=MAX_PLAN_SIZE):
    """(depth, size, depth_plan, size_plan) for the lapse-free plans one
    solve() pass from `state` could return: depth is the fewest OrNodes on
    the longest backward chain any plan needs (how far back from red the
    reasoning must reach before a first move), size the fewest OrNodes any
    plan visits in total (lapse checks it must survive, ignoring dead ends),
    each with a plan achieving it. The two minima can come from different
    plans. Depth-first search order and wasted exploration don't count.

    None if no single pass reaches the exit within max_size OrNodes - e.g.
    when red itself must first move out of the way (solve()'s red_candidates
    fallback), which takes more than one pass."""
    red = dict(state)['red']
    if max(j for _, j in red) >= BOARD_SIZE - 1:
        return 0, 0, [], []
    if _first_plan(state, max_size, max_size) is None:
        return None
    depth_plan = next(p for d in range(max_size + 1) if (p := _first_plan(state, d, max_size)) is not None)
    size_plan = next(p for n in range(max_size + 1) if (p := _first_plan(state, max_size, n)) is not None)
    return depth_plan[2], len(size_plan[3]), depth_plan[1], size_plan[1]


def or_budget(gamma, q=0.5):
    """How many OrNodes a pass can visit before its chance of no lapse drops
    below `q`: the largest n with (1 - gamma)**n >= q. The reasoning horizon
    that horizon_car measures against; infinite when gamma is 0."""
    if gamma <= 0:
        return math.inf
    return math.floor(math.log(q) / math.log(1 - gamma))


@functools.lru_cache(maxsize=100_000)
def first_path(state, max_size=MAX_PLAN_SIZE):
    """The shortest chain of OrNode cars, red's blocker first, that a pass
    from `state` must reason down before reaching a move some successful
    plan starts with. A pass that gets that far has progress it keeps even
    if it lapses later - a completed branch gets played out
    (AndNode.solve's GammaLapse handling) - so this, not the whole plan, is
    what must fit within the reasoning budget. () if red can already exit,
    None if no single pass within max_size OrNodes can reach the exit (see
    plan_metrics). Cached per state, since wandering revisits states often."""
    red = dict(state)['red']
    if max(j for _, j in red) >= BOARD_SIZE - 1:
        return ()
    if _first_plan(state, max_size, max_size) is None:
        return None
    return tuple(next(p[4] for n in range(max_size + 1)
                      if (p := _first_plan(state, max_size, max_size, n)) is not None))


def chain_first_moves(state, max_size=MAX_PLAN_SIZE, max_plans=20):
    """The moves that start first_path(state)'s chain: the distinct first
    moves of lapse-free plans whose first path is that short (the deepest
    subgoal's resolving move, played before anything above it), from up to
    `max_plans` such plans. Empty if red can already exit or no single pass
    can reach the exit."""
    path = first_path(state, max_size)
    if not path:
        return set()
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    plans = _plans_and(state, 'red', 'r', steps, frozenset(), frozenset(), max_size, max_size, len(path))
    return {plan[1][0] for plan in itertools.islice(plans, max_plans)}


def horizon_car(state, budget, max_size=MAX_PLAN_SIZE):
    """The car just outside the reasoning horizon from `state`: on
    first_path(state), the car one past the `budget` OrNodes a pass can
    afford (see or_budget) - the first subgoal it can't reason back to
    before lapsing. In a linear chain red <- K <- B <- I <- D with a budget
    of 3, that's D. None if the path fits within the budget, or if no single
    pass can reach the exit - the solver can't plan its way out of such
    states at all, whatever the budget. Wasted search (dead ends, abandoned
    orders) isn't counted, so this horizon is optimistic."""
    path = first_path(state, max_size)
    if path is None or len(path) <= budget:
        return None
    return path[budget]


def solve(state, heuristic=None, gamma=0.0):
    """One stochastic pass of AND-OR subgoal decomposition (see GammaLapse)
    driving red to the exit. Returns a plain move list once red reaches the
    exit; otherwise a (new_state, moves) pair reflecting a partial attempt,
    for the caller to feed back in.

    `heuristic`, if given, is a `heuristic(state, actions) -> {action: score}`
    callable (see PolicyAgent.heuristic in rl.py) that orders every
    candidate-selection point best-first instead of randomly. It never changes
    whether a state solves, only which solution is found and how quickly.

    `gamma`, the probability an OrNode abandons its subgoal for a random
    legal move instead (see GammaLapse) - 0 (the default) never lapses. Left
    as an explicit parameter rather than a module constant so callers (e.g.
    rush_hour.ipynb) can tune it directly instead of editing this file.
    """
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    if steps <= 0:
        return []

    try:
        result = AndNode(state, 'red', 'r', steps, heuristic=heuristic, gamma=gamma).solve()
    except GammaLapse as lapse:
        return lapse.new_state, lapse.moves

    if result is not None:
        _, moves = result
        return moves

    try:
        result = first_solve(state, 'red', red_candidates(state, steps), heuristic=heuristic, gamma=gamma)
    except GammaLapse as lapse:
        return lapse.new_state, lapse.moves
    if result is not None:
        return result

    moves = legal_moves(state)
    if not moves:
        return state, None
    move = random.choice(moves)
    car_name, direction, steps = move
    return AndNode(state, car_name, direction, steps).apply(state), [move]
