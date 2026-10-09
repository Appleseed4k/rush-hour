"""The AND-OR model as Olieslagers et al. (2023) specify it, for comparison with and_or.solve.

Same tree (and_or.AndNode / OrNode: blockers, candidate directions, the
`protected` and `visited` rules), different execution. The paper treats every
board state independently: it builds the state's AND-OR tree, stops at any OR
node with probability gamma (and then takes a uniformly random legal action),
otherwise ends on a leaf - an action that is available right now. Exactly one
action is played, then the tree is rebuilt from the new state. and_or.solve
instead plays out a whole plan (every blocker's move) and only checks for a
lapse while planning.

Choices the paper leaves open, made here with its uniform-heuristic setup:
  * AND node -> one blocking car's subgoal, uniformly (Fig. 3A middle).
  * OR node -> one resolving action, uniformly (Fig. 3A bottom).
  * Mass reaching a cycle (`visited`) or a dead end (`protected`, or no
    resolving action) is spread proportionally over the leaves, as the paper
    does for cycles; the lapse mass is left alone.
"""
import random

from and_or import AndNode, OrNode, _red_steps, legal_moves, red_candidates


def _mix(parts):
    """Uniform mixture of (leaves, lapse) results, or None if there are none."""
    if not parts:
        return None
    leaves, lapse = {}, 0.0
    for part_leaves, part_lapse in parts:
        for move, p in part_leaves.items():
            leaves[move] = leaves.get(move, 0.0) + p / len(parts)
        lapse += part_lapse / len(parts)
    return leaves, lapse


def _and_dist(state, car_name, direction, steps, visited, protected, gamma):
    """(leaves, lapse) - the distribution over leaf actions (and the lapse
    probability) of playing out this AND node's subtree, or None if no leaf
    is reachable beneath it."""
    blockers = AndNode(state, car_name, direction, steps).blockers()
    if protected & blockers.keys():
        return None
    if not blockers:
        return {(car_name, direction, steps): 1.0}, 0.0
    subgoals = (_or_dist(state, name, cells, visited, protected | {car_name}, gamma)
                for name, cells in blockers.items())
    return _mix([r for r in subgoals if r is not None])


def _or_dist(state, car_name, collisions, visited, protected, gamma):
    """OR counterpart to _and_dist: a stopping check (probability gamma of
    lapsing here), then one viable resolving action, uniformly."""
    key = (car_name, frozenset(collisions))
    if key in visited:
        return None
    options = (_and_dist(state, car_name, direction, steps, visited | {key}, protected, gamma)
               for direction, steps in OrNode(state, car_name, collisions).directions())
    mixed = _mix([r for r in options if r is not None])
    if mixed is None:
        return None
    leaves, lapse = mixed
    return {move: p * (1 - gamma) for move, p in leaves.items()}, gamma + (1 - gamma) * lapse


def action_distribution(state, gamma):
    """(lapse, leaves, used_fallback): the paper's one-step distribution at
    `state` - `lapse`, the probability of a uniformly random legal action;
    `leaves`, {move: probability} over the tree's leaf actions; they sum to 1.

    The root is the OR node "red exits": one stopping check, then the AND
    node moving red to the exit. If that tree has no leaf at all (red must
    first reposition itself - and_or.solve's red_candidates fallback), the
    root's candidates become those repositioning moves, `used_fallback` True.
    Empty `leaves` means even that has none: the caller plays a random action."""
    steps = _red_steps(state)
    result, used_fallback = _and_dist(state, 'red', 'r', steps, frozenset(), frozenset(), gamma), False
    if result is None:
        used_fallback = True
        options = (_and_dist(state, 'red', direction, s, frozenset(), frozenset(), gamma)
                   for direction, s in red_candidates(state, steps))
        result = _mix([r for r in options if r is not None])
        if result is None:
            return 1.0, {}, used_fallback
    leaves, lapse = result
    return gamma + (1 - gamma) * lapse, {move: p * (1 - gamma) for move, p in leaves.items()}, used_fallback


def step(state, gamma, rng=random):
    """(move, lapsed): one action sampled from action_distribution(state, gamma)."""
    lapse, leaves, _ = action_distribution(state, gamma)
    if not leaves or rng.random() < lapse:
        return rng.choice(legal_moves(state)), True
    moves = list(leaves)
    return rng.choices(moves, weights=[leaves[m] for m in moves])[0], False


def solve_paper(state, gamma, max_steps=200, rng=random):
    """(moves, n_lapses): replans from scratch after every single action
    (see step) until red reaches the exit; moves is None if that takes more
    than `max_steps` actions."""
    moves, lapses = [], 0
    while len(moves) < max_steps:
        if _red_steps(state) <= 0:
            return moves, lapses
        move, lapsed = step(state, gamma, rng)
        lapses += lapsed
        state = AndNode(state, *move).apply(state)
        moves.append(move)
    return (moves, lapses) if _red_steps(state) <= 0 else (None, lapses)
