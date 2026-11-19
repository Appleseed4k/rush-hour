import itertools
import os
import random

from lib import DELTAS

BOARD_SIZE = 6

# Whether a subgoal may move a car whose own action is still being resolved
# (every car above it in the chain, red included). The default for solve(); it
# is read when this module is imported, not on each call.
#
# True protects them (nothing above a subgoal can be moved): searches stay
# small, but a state that needs a car to step aside and
# come back (red most often) has no direct plan. False protects nobody, as in
# the original model: such plans are found, solve rates are higher on mid-range
# puzzles, and the cost is search size - with lapses off, a ~200x larger search
# per pass. (With lapses on, gamma > 0, False costs no
# more than True: scripts/p3_session/and_or_gamma.py.)
#
# Off by default since 2026-11-17: it fits people better
# (notebooks/archive/p3_session.ipynb's gamma section) and is the expert of the
# apprentice in models/apprentice/gamma0.07_free. Earlier work (phase 1, tag
# pre-cleanup) used True, so rerunning it as it was needs it on. With lapses off (gamma = 0, the default of
# rl.solve_and_or and its relatives) an unprotected search can grow without
# bound on some boards: give gamma > 0, or turn protection on.
#
# The environment variable AND_OR_PROTECT_MOVERS=1 turns it on for a whole
# run, worker processes included (they inherit the environment) - e.g.
# scripts/p3_session/apprentice.py --protect-movers on.
PROTECT_MOVERS = os.environ.get("AND_OR_PROTECT_MOVERS", "0") != "0"


# A `protected` set carries its own on/off switch: protect() only adds cars to it
# while it holds this marker, which seed_protected() puts there. That keeps the
# switch travelling with the set through every call that already passes it down
# (including chunk_library's), without a new argument anywhere. It can't clash
# with a car name, so `protected & blockers.keys()` is unaffected.
PROTECTING = "<protecting>"


def protect(protected, car_name):
    """`protected` plus `car_name`, the car whose action is now being resolved:
    while protection is on, nothing beneath that action may move it. A no-op
    when it is off (see solve()'s `protect_movers`) - then any car, red
    included, can be moved by a subgoal, as in the original model."""
    return protected | {car_name} if PROTECTING in protected else protected


def seed_protected(protect_movers):
    """The `protected` set a root action starts from."""
    return frozenset({PROTECTING}) if protect_movers else frozenset()


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


def lapse_move(state, heuristic=None):
    """The single move a lapse plays from `state`, or None if no move is legal:
    the heuristic's top-scoring legal move if `heuristic` is given, else a
    uniformly random one.

    Deferring to the apprentice's own greedy pick instead of a blind random
    move means a lapse still yields a demonstration worth imitating once a
    heuristic is actually available (round 2+) - otherwise every lapse injects
    pure noise regardless of how good the trained policy already is."""
    moves = legal_moves(state)
    if not moves:
        return None
    if heuristic is None:
        return random.choice(moves)
    scores = heuristic(state, moves)
    return max(moves, key=lambda move: scores[move])


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
            move = lapse_move(self.state, self.heuristic)
            if move is None:
                return None
            raise GammaLapse(AndNode(self.state, *move).apply(self.state), [move])

        return first_solve(self.state, self.car_name, self.directions(), next_visited, self.protected,
                           self.heuristic, self.gamma)


class AndNode:
    """Action: move car_name `steps` cells in `direction`. Solvable once every car
    occupying a swept cell has vacated it (AND semantics).

    The action is a destination, not a slide length: it is the cell the car
    reaches from where it stands in `state`. Resolving a blocker can move the
    car (a car stepping aside so another can pass, say), so solve() re-derives
    the final slide from the car's place at that point (see move_from) - a
    plan can step a car away and then still bring it to its destination. (That
    can only happen with protection off; see PROTECT_MOVERS and solve()'s `protect_movers`.)

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

    def move_from(self, state):
        """(direction, steps) that takes the car from its place in `state` to the
        destination this action has from self.state, or None if it is already
        there."""
        di, dj = DELTAS[self.direction]
        axis_of = (lambda st: [i for i, _ in dict(st)[self.car_name]]) if di else \
                  (lambda st: [j for _, j in dict(st)[self.car_name]])
        delta = min(axis_of(self.state)) + (di or dj) * self.steps - min(axis_of(state))
        if delta == 0:
            return None
        if di:
            return ('d' if delta > 0 else 'u'), abs(delta)
        return ('r' if delta > 0 else 'l'), abs(delta)

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
        next_protected = protect(self.protected, self.car_name)
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
                final = self.move_from(working_state)
                if final is None:  # resolving the blockers already put the car on its destination
                    return working_state, moves
                if not AndNode(working_state, self.car_name, *final, self.visited).blockers():
                    final_state = AndNode(working_state, self.car_name, *final).apply(working_state)
                    moves.append((self.car_name, *final))
                    return final_state, moves
        return None


def _red_steps(state):
    """Cells red still has to slide to reach the exit."""
    return (BOARD_SIZE - 1) - max(j for _, j in dict(state)['red'])


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


def solve(state, heuristic=None, gamma=0.0, protect_movers=PROTECT_MOVERS):
    """One stochastic pass of AND-OR subgoal decomposition (see GammaLapse)
    driving red to the exit. Returns a plain move list once red reaches the
    exit; otherwise a (new_state, moves) pair reflecting a partial attempt,
    for the caller to feed back in.

    `heuristic`, if given, is a `heuristic(state, actions) -> {action: score}`
    callable (see PolicyAgent.heuristic in rl.py) that orders every
    candidate-selection point best-first instead of randomly. It never changes
    whether a state solves, only which solution is found and how quickly.

    `gamma`, the probability an OrNode abandons its subgoal for a random
    legal move instead (see GammaLapse) - 0 (the default) never lapses.

    `protect_movers` forbids a subgoal from moving any car whose own action is
    still being resolved - every car above it in the chain, red included. It
    defaults to the module's PROTECT_MOVERS (see there for the trade-off). Off,
    a plan can have a car step aside and come back (red most often), see
    AndNode; on, a state that needs that has no direct plan and falls back to
    repositioning red at random (red_candidates).
    """
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    if steps <= 0:
        return []

    try:
        result = AndNode(state, 'red', 'r', steps, protected=seed_protected(protect_movers), heuristic=heuristic,
                         gamma=gamma).solve()
    except GammaLapse as lapse:
        return lapse.new_state, lapse.moves

    if result is not None:
        _, moves = result
        return moves

    try:
        result = first_solve(state, 'red', red_candidates(state, steps), protected=seed_protected(protect_movers),
                             heuristic=heuristic, gamma=gamma)
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

