import random

from rush_hour_bfs import DELTAS
from rush_hour_lib import sample_unique, visualize

BOARD_SIZE = 6
GAMMA = 0.01  # probability of abandoning a subgoal at an OrNode for a random action


class GammaLapse(Exception):
    """Raised by OrNode.solve() when its stopping probability fires, unwinding the
    whole in-progress search all the way to the top-level solve() - a lapse
    abandons the entire train of thought, not just the subgoal being considered
    when it happened, so no ancestor should backtrack and try another candidate
    after one of these.

    `moves`/`origins` carry every move actually made on the way to `new_state` -
    not just the lapse's own random move, but also any blockers an enclosing
    AndNode had already resolved before one of ITS OWN blockers lapsed. Those
    already-committed moves are real and did happen (self.state, threaded through
    the recursion, already reflects them), so AndNode.solve() must catch this,
    prepend its own so-far-accumulated moves/origins, and re-raise - only the
    outermost solve() should let it stop propagating."""

    def __init__(self, new_state, moves, origins):
        self.new_state = new_state
        self.moves = moves
        self.origins = origins


def legal_moves(state):
    """Every (car_name, direction, steps) currently legal for any car - the pool a
    person picks from when they abandon a subgoal (see GAMMA) or when the tree
    search dead-ends outright (see solve())."""
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


class OrNode:
    """Subgoal: car_name must vacate every cell in `collisions`. With probability
    GAMMA, abandons the entire search (not just this subgoal) for a uniformly
    random legal move instead - see GammaLapse. Otherwise, tries its candidate
    actions one at a time in random order (depth-first), backtracking to the next
    candidate whenever one dead-ends, until one works or all have been tried. This
    keeps the search bounded to what a single train of thought could hold - only
    the current path's choices, not the whole tree - unlike breadth-first
    propagation over every leaf at once."""

    def __init__(self, state, car_name, collisions, visited=frozenset(), protected=frozenset()):
        self.state = state
        self.car_name = car_name
        self.car = dict(state)[car_name]
        self.collisions = frozenset(collisions)
        self.visited = visited
        self.protected = protected

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
        """(new_state, moves, origins) via depth-first search over candidate
        actions (see class docstring), or None if every candidate dead-ends (a
        local failure the calling AndNode's OWN caller backtracks from - not a
        GammaLapse, which instead propagates straight through uncaught). `origins`
        is parallel to `moves`, tagging every move "tree" (all of them are, since
        a GammaLapse's single random move never joins a `moves` list - it replaces
        the whole search's result instead - see GAMMA)."""
        key = (self.car_name, self.collisions)
        if key in self.visited:
            return None
        next_visited = self.visited | {key}

        if random.random() < GAMMA:
            moves = legal_moves(self.state)
            if not moves:
                return None
            car_name, direction, steps = random.choice(moves)
            node = AndNode(self.state, car_name, direction, steps)
            raise GammaLapse(node.apply(self.state), [(car_name, direction, steps)], ["gamma"])

        candidates = self.directions()
        random.shuffle(candidates)
        for direction, steps in candidates:
            result = AndNode(self.state, self.car_name, direction, steps, next_visited, self.protected).solve()
            if result is not None:
                return result
        return None


class AndNode:
    """Action: move car_name `steps` cells in `direction`. Solvable once every car
    occupying a swept cell has vacated it (AND semantics)."""

    def __init__(self, state, car_name, direction, steps, visited=frozenset(), protected=frozenset()):
        self.state = state
        self.car_name = car_name
        self.direction = direction
        self.steps = steps
        self.visited = visited
        self.protected = protected

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
        """Every swept cell each other car occupies, grouped by car - a same-line
        blocker can overlap several swept cells at once, and all of them are needed
        so OrNode requires clearing the whole overlap, not just the nearest cell."""
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

    def solve(self):
        """(new_state, moves, origins) if every blocker resolves and the path stays
        clear, else None - a local failure the calling OrNode backtracks from by
        trying its next candidate. When there are multiple simultaneous blockers,
        the order they're addressed in is sampled once and not reconsidered -
        deliberately not a full CSP search - though each individual blocker's own
        OrNode does backtrack over ITS candidates. `origins` is parallel to
        `moves`: each blocker's own tags pass through unchanged, and this node's
        own final move (the action it was asked to solve) is tagged "tree".

        If a blocker's OrNode.solve() raises GammaLapse, the blockers resolved
        before it in this loop are real moves that already happened (self.state
        stays untouched, but working_state was threaded into that OrNode, so its
        new_state already reflects them) - so they're prepended onto the lapse's
        moves/origins before re-raising, rather than silently dropped. Letting the
        exception through unmerged would desync a caller replaying `moves` from
        the original state, since new_state would then reflect more moves than
        got reported.

        No car already committed to a move higher up the call stack (self.car_name
        included) is ever nominated as someone else's blocker: self.protected tracks
        every such car, growing by self.car_name before it's passed to blockers'
        subgoals. Without this, a blocker nested arbitrarily deep could nudge one of
        those committed cars aside to solve its own problem, leaving the commitment
        stale - the final apply() below trusts working_state's position for
        self.car_name unconditionally, so a stale one silently produces a bogus (even
        out-of-bounds) move."""
        blockers = self.blockers()
        if self.protected & blockers.keys():
            return None

        working_state = self.state
        moves = []
        origins = []
        order = list(blockers.items())
        random.shuffle(order)
        next_protected = self.protected | {self.car_name}
        for blocker_name, collisions in order:
            try:
                result = OrNode(working_state, blocker_name, collisions, self.visited, next_protected).solve()
            except GammaLapse as lapse:
                raise GammaLapse(lapse.new_state, moves + lapse.moves, origins + lapse.origins) from None
            if result is None:
                return None
            working_state, blocker_moves, blocker_origins = result
            moves.extend(blocker_moves)
            origins.extend(blocker_origins)

        if AndNode(working_state, self.car_name, self.direction, self.steps, self.visited).blockers():
            return None  # a blocker's own fix left the path obstructed after all

        final_state = self.apply(working_state)
        moves.append((self.car_name, self.direction, self.steps))
        origins.append("tree")
        return final_state, moves, origins


def red_candidates(state, exclude_steps):
    """(direction, steps) options for repositioning red itself, tried once the
    direct slide to the exit (`exclude_steps`) has failed: shorter rightward
    slides, plus leftward slides - lets solve() find puzzles that need red to pull
    back before a later slide can clear, which trying only the maximal rightward
    move can never discover."""
    red = dict(state)['red']
    axis_pos = [j for _, j in red]
    lo, hi = min(axis_pos), max(axis_pos)
    max_right = (BOARD_SIZE - 1) - hi
    candidates = [('r', s) for s in range(1, max_right + 1) if s != exclude_steps]
    candidates += [('l', s) for s in range(1, lo + 1)]
    return candidates


def solve(state, origins=None):
    """Sequence of (car_name, direction, steps) moves driving red to the exit (its
    rightmost cell reaching column BOARD_SIZE - 1), via one stochastic pass of AND-OR
    subgoal decomposition (see GAMMA) - modeling a single bounded round of human
    backward reasoning, not an exhaustive solver. Expected to fail far from the
    goal, and results vary run to run.

    If the direct slide to the exit can't be resolved, tries repositioning red
    itself (see red_candidates) before falling back to a fully random legal move
    (see legal_moves). A fallback returns (new_state, moves) instead of a bare move
    list, since a GammaLapse can carry real moves an AndNode had already committed
    to before it fired (see GammaLapse). Callers should check which shape they got
    back (bare list vs. tuple) and call solve() again on new_state to keep trying.

    If `origins` is given (a list), one tag per move actually returned is appended
    to it: "tree" for a move the search deliberately found (direct slide, red
    repositioning, or blocker resolution), "gamma" for a lapse's random move, or
    "abort" for the final fallback's random move once every candidate at every
    level backtracked through without finding a chain or lapsing."""
    red = dict(state)['red']
    steps = (BOARD_SIZE - 1) - max(j for _, j in red)
    if steps <= 0:
        return []

    try:
        result = AndNode(state, 'red', 'r', steps).solve()
    except GammaLapse as lapse:
        if origins is not None:
            origins.extend(lapse.origins)
        return lapse.new_state, lapse.moves

    if result is not None:
        _, moves, move_origins = result
        if origins is not None:
            origins.extend(move_origins)
        return moves

    candidates = red_candidates(state, steps)
    random.shuffle(candidates)
    for direction, cand_steps in candidates:
        try:
            result = AndNode(state, 'red', direction, cand_steps).solve()
        except GammaLapse as lapse:
            if origins is not None:
                origins.extend(lapse.origins)
            return lapse.new_state, lapse.moves
        if result is not None:
            new_state, moves, move_origins = result
            if origins is not None:
                origins.extend(move_origins)
            return new_state, moves

    moves = legal_moves(state)
    if not moves:
        return state, None
    move = random.choice(moves)
    car_name, direction, steps = move
    if origins is not None:
        origins.append("abort")
    return AndNode(state, car_name, direction, steps).apply(state), [move]


if __name__ == "__main__":
    puzzles = sample_unique()

    puzzle = puzzles[25]
    state = puzzle
    all_moves = []
    all_origins = []
    att = solve(state, all_origins)
    while type(att) is not list:
        state, moves = att
        all_moves.extend(moves)
        att = solve(state, all_origins)
    all_moves += att
    visualize("and_or", puzzle, all_moves)
