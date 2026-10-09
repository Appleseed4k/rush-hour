"""Python port of the AND-OR model in the Olieslagers et al. (2023) archive
(code/rush_hour/{and_or_trees,and_or_model,rush_hour}.jl, supplement
Algorithms 2-9), for comparison with and_or.py / paper_model.py. Written from
the Julia, quirks included; `faithful=False` switches the quirks off.

The model is a per-state, one-step distribution over the person's next action:
  1. build the AND-OR tree on the *current* board only (get_and_or_tree);
  2. propagate probability to the leaves with gamma = 0 (propagate_ps),
     splitting uniformly over children at every node;
  3. weight each leaf by (1 - gamma)^depth, give the remainder to "stop"
     (apply_gamma);
  4. treat moves by the previously moved car as cycles, redistribute cycle
     mass proportionally, spread "stop" uniformly over every legal move
     (process_dict).

Quirks reproduced when `faithful`:
  * unblocking_moves! gives a blocker's candidate amounts signed (-behind,
    +ahead), and get_blocking_nodes! declares the whole move impossible when
    they sum to 0 - which also happens when e.g. {-1, +1} are both available.
  * propagate! pushes each sibling OR node onto the shared train of thought,
    so a later sibling's subtree sees earlier siblings as ancestors.
"""
import collections

from and_or import legal_moves

class TreeTooLarge(Exception):
    """Raised when building or walking the tree exceeds `max_visits` node visits.
    The Julia re-expands shared subtrees once per path, which is exponential on
    congested boards (no `protected`-style pruning), so a cap keeps rollouts
    finite and lets callers count how often it bites."""


BOARD = 6
MAX_AMOUNT = 4  # possible_moves! only looks at slides of 1..4 cells
CYCLE = "cycle"


def _cars(state):
    """({name: (vertical, lane, low_end, length)}, {cell: name}) with 0-based
    coordinates; `lane` is the column of a vertical car or the row of a
    horizontal one."""
    cars, arr = {}, {}
    for name, pos in state:
        vertical = pos[0][1] == pos[1][1]
        lane = pos[0][1] if vertical else pos[0][0]
        axis = [i for i, _ in pos] if vertical else [j for _, j in pos]
        cars[name] = (vertical, lane, min(axis), len(pos))
        for cell in pos:
            arr[cell] = name
    return cars, arr


def _blocked_by(cars, arr, move):
    """Distinct cars on the cells `move` = (name, signed amount) sweeps
    (move_blocked_by!), nearest first."""
    name, m = move
    vertical, lane, lo, length = cars[name]
    out, prev = [], None
    for i in range(1, abs(m) + 1):
        a = lo + length - 1 + i if m > 0 else lo - i
        car = arr.get((a, lane) if vertical else (lane, a))
        if car is None or car == prev:
            continue
        out.append(car)
        prev = car
    return out


def _unblocking_amounts(cars, move, blocker):
    """Signed slides of `blocker` that clear `move`'s path (unblocking_moves!).
    Slides are capped by the room in the lane after setting aside the other
    cars that share it."""
    name, m = move
    v1, lane1, lo1, len1 = cars[name]
    v2, lane2, lo2, len2 = cars[blocker]
    behind, ahead = lo2, (BOARD - 1) - (lo2 + len2 - 1)
    for other, (v, lane, lo, length) in cars.items():
        if other != blocker and v == v2 and lane == lane2:
            if lo > lo2:
                ahead -= length
            else:
                behind -= length
    amounts = []
    if v1 == v2:  # aligned: the blocker has to get out of the mover's way along the lane
        if m < 0:
            need = (lo2 + len2 - 1) - (lo1 + m)
            if need < behind:
                amounts += [-i for i in range(need + 1, behind + 1)]
        else:
            need = (lo1 + len1 - 1 + m) - lo2
            if need < ahead:
                amounts += list(range(need + 1, ahead + 1))
    else:  # crossing: the blocker has to clear the mover's lane
        need = lo2 + len2 - 1 - lane1
        if need < behind:
            amounts += [-i for i in range(need + 1, behind + 1)]
        need = lane1 - lo2
        if need < ahead:
            amounts += list(range(need + 1, ahead + 1))
    return amounts


def _blocking_rows(cars, arr, move, faithful):
    """[(blocker, amount, ...)] subgoal rows for `move`, or None if some blocker
    has no way out (get_blocking_nodes!)."""
    rows = []
    for blocker in _blocked_by(cars, arr, move):
        amounts = _unblocking_amounts(cars, move, blocker)
        if not amounts or (faithful and sum(amounts) == 0):
            return None
        rows.append((blocker, *amounts))
    return rows


def build_tree(state, faithful=True, max_iter=100, max_visits=60_000):
    """(root_move, AND, OR, parents): AND[(d, move)] -> [(d + 1, row)] subgoals,
    OR[(d, row)] -> [(d, move)] resolving actions, a leaf being an AND whose
    only child is (d, None); parents[move] -> AND nodes that are leaves."""
    cars, arr = _cars(state)
    red = cars["red"]
    root_move = ("red", (BOARD - 1) - (red[2] + red[3] - 1))
    AND, OR, parents = {}, {}, {}
    visits = [0]

    def add(prev_and, rows, train, rec):
        visits[0] += 1
        if visits[0] > max_visits:
            raise TreeTooLarge(max_visits)
        if rec > max_iter:
            raise RecursionError("max_iter depth reached")
        d, _ = prev_and
        for row in rows:
            or_node = (d + 1, row)
            children = AND.setdefault(prev_and, [])
            if or_node not in children:
                children.append(or_node)
            if row in train:
                continue
            for amount in row[1:]:
                next_move = (row[0], amount)
                sub = _blocking_rows(cars, arr, next_move, faithful)
                if sub is None:
                    continue
                and_node = (d + 1, next_move)
                options = OR.setdefault(or_node, [])
                if and_node not in options:
                    options.append(and_node)
                if not sub:
                    leaf = AND.setdefault(and_node, [])
                    if (d + 1, None) not in leaf:
                        leaf.append((d + 1, None))
                    holders = parents.setdefault(next_move, [])
                    if and_node not in holders:
                        holders.append(and_node)
                    continue
                add(and_node, sub, train + [row], rec + 1)

    root_and = (1, root_move)
    AND.setdefault(root_and, [])
    rows = _blocking_rows(cars, arr, root_move, faithful)
    if rows:
        add(root_and, rows, [(root_move[0], root_move[1])], 0)
    return root_and, AND, OR, parents


def propagate(root_and, AND, OR, faithful=True, max_visits=200_000):
    """{(depth, move or CYCLE): probability}, split uniformly with gamma = 0
    (propagate_ps)."""
    out = collections.defaultdict(float)
    visits = [0]

    def is_leaf(node):
        return AND[node][0][1] is None

    def walk(p, train, node):
        visits[0] += 1
        if visits[0] > max_visits:
            raise TreeTooLarge(max_visits)
        base = list(train)  # ancestors only; `train` also collects siblings
        children = AND[node]
        for or_node in children:
            p_or = p / len(children)
            if or_node[1] in (train if faithful else base) or or_node not in OR:
                out[(or_node[0], CYCLE)] += p_or
                continue
            train.append(or_node[1])
            options = OR[or_node]
            for nxt in options:
                p_and = p_or / len(options)
                if is_leaf(nxt):
                    out[nxt] += p_and
                else:
                    walk(p_and, list(train) if faithful else base + [or_node[1]], nxt)

    walk(1.0, [(root_and[1][0], root_and[1][1])], root_and)
    return out


def _to_repo_move(state, move):
    """(name, signed amount) -> and_or.py's (name, direction, steps)."""
    name, amount = move
    horizontal = dict(state)[name][0][0] == dict(state)[name][1][0]
    if horizontal:
        return name, "r" if amount > 0 else "l", abs(amount)
    return name, "d" if amount > 0 else "u", abs(amount)


def action_distribution(state, gamma, prev_car=None, exclude_same_car=True, faithful=True):
    """({move: probability}, stop, info): the original's distribution over the
    legal moves (and_or.py's (name, direction, steps) form) at `state`, as
    gamma_only_model computes it, with `stop` the total probability of a lapse
    (already spread over the moves). `prev_car`, if given and
    `exclude_same_car`, is the car that moved last: leaves moving it count as
    cycles. `info` carries diagnostics (leaf moves outside the legal set)."""
    root_and, AND, OR, parents = build_tree(state, faithful)
    raw = propagate(root_and, AND, OR, faithful)
    scaled = {k: v * (1 - gamma) ** k[0] for k, v in raw.items() if v > 0}
    stop = 1 - sum(scaled.values())

    legal = [m for m in legal_moves(state) if m[2] <= MAX_AMOUNT]
    probs = {m: 0.0 for m in legal}
    cycle, stray = 0.0, 0.0
    for (depth, move), v in scaled.items():
        if move == CYCLE:
            cycle += v
            continue
        repo_move = _to_repo_move(state, move)
        if exclude_same_car and prev_car is not None and move[0] == prev_car:
            cycle += v
        elif repo_move in probs:
            probs[repo_move] += v
        else:
            stray += v
    z = sum(probs.values()) + stop
    if z > 0:
        scale = 1 + cycle / z
        probs = {m: p * scale for m, p in probs.items()}
        stop *= scale
    else:
        probs = {m: cycle / len(probs) for m in probs}
    probs = {m: p + stop / len(probs) for m, p in probs.items()}
    return probs, stop, {"stray_leaf_mass": stray, "cycle_mass": cycle, "n_leaves": len(parents)}
