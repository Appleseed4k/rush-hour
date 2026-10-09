"""Chunk recognition inside and_or's search: a library of sub-plan templates
that OrNode.solve consults after its own lapse check, so a recognized
sub-plan costs that one check instead of one per OrNode it covers.

A template is a move-level tree, (rel, children), where rel is
chunks.relation()'s description of a move relative to the move it unblocks,
and each child is either another template or HOLE - a blocker the chunk
leaves to ordinary reasoning (a real OrNode, with its own lapse checks).
Children are unordered, like blockers.

Matching never looks at absolute positions or at cars the plan doesn't
touch. A template is instantiated top-down on the current board: at each
node, pick the move whose relation to its parent's move equals the
template's rel, then require that move's blockers - recomputed on this
board - to pair up one-to-one with the template's children (a hole takes
any blocker). Cars elsewhere never block one of these moves, so they never
enter the match; a car that does block one but isn't in the template makes
the blocker count wrong and the match fails.

Recognition is two-phase. First a free, perceptual match on the current
board, with each hole's car simply lifted off as if already cleared; the
largest template that matches is the one recognized. Then it is committed
to: played out for real, holes searched by ordinary OrNodes (with their
own lapse checks). If the real attempt fails (a hole can't be cleared, or
clearing it re-blocks the plan), the subgoal falls back to ordinary search
- no second template is tried.

Alternatively, rank() uses recognition only to order a subgoal's own
candidate moves (recognized ones first, largest and most-supported template
first), leaving the search and its lapse checks otherwise unchanged."""
import itertools
import random
from collections import Counter, defaultdict

import chunks
import nested_chunks
from and_or import AndNode, GammaLapse, OrNode, protect

HOLE = None


def template(node, covered=None):
    """Template for an annotated tree node (see chunks.annotate). Children not
    in `covered` (a set of node ids; None = all) become holes."""
    kids = [template(c, covered) if covered is None or id(c) in covered else HOLE for c in node["children"]]
    return node["rel"], tuple(sorted(kids, key=render))


def render(t):
    if t is HOLE:
        return "?"
    rel, kids = t
    return rel + (f"({','.join(render(k) for k in kids)})" if kids else "")


def moves_in(t):
    """Moves a template plays itself (holes excluded)."""
    return 0 if t is HOLE else 1 + sum(moves_in(k) for k in t[1])


def holes_in(t):
    return 1 if t is HOLE else sum(holes_in(k) for k in t[1])


def flat_templates(train, min_size=2, max_size=6, min_support=3):
    """(template, support) for chunks.py's library: whole subtrees, no holes."""
    mined = chunks.mine(train, min_size, max_size)
    return [(template(v["example"][1]), len(v["support"])) for v in mined.values()
            if len(v["support"]) >= min_support]


def nested_templates(learned):
    """Templates for nested_chunks.learn()'s chunks, each expanded to moves
    from its recorded example, with its holes kept."""
    out = []
    for chunk in learned:
        unit = chunk["example"]
        if unit is not None:
            out.append((template(unit["nodes"][0], {id(n) for n in unit["nodes"]}), chunk["support"]))
    return out


def _rel(state, car_name, collisions, parent_direction, direction, steps):
    return chunks.relation(parent_direction, {"state": state, "car": car_name, "collisions": collisions,
                                              "action": (direction, steps)})


class ChunkLibrary:
    """(template, weight) pairs indexed by root rel, tried largest first, then
    by weight (e.g. how many puzzles a template was seen in). Templates under
    `min_moves` moves are dropped. `uses` counts successful instantiations
    per template, `covered` the moves they played, `failures` recognized
    templates whose real attempt failed."""

    def __init__(self, entries, min_moves=2):
        self.weight = {}
        unique = {}
        for t, w in entries:
            if moves_in(t) >= min_moves:
                unique[render(t)] = t
                self.weight[render(t)] = max(w, self.weight.get(render(t), 0))
        self.by_root = defaultdict(list)
        for t in unique.values():
            self.by_root[t[0]].append(t)
        for ts in self.by_root.values():
            ts.sort(key=lambda t: (-moves_in(t), -self.weight[render(t)], holes_in(t), render(t)))
        self.size = len(unique)
        self.reset_stats()

    def reset_stats(self):
        self.uses = Counter()
        self.covered = 0
        self.failures = 0

    def resolve(self, node, visited):
        """(new_state, moves) from the template recognized at OrNode `node`
        (whose own key is already in `visited`), else None. A lapse inside a
        hole propagates as GammaLapse, carrying the chunk's moves completed
        before it."""
        if node.parent_direction is None:
            return None
        candidates = []
        for direction, steps in node.directions():
            rel = _rel(node.state, node.car_name, node.collisions, node.parent_direction, direction, steps)
            candidates += [(t, direction, steps) for t in self.by_root.get(rel, ())]
        candidates.sort(key=lambda c: (-moves_in(c[0]), holes_in(c[0])))
        for t, direction, steps in candidates:
            args = (node.car_name, direction, steps, t[1], visited, node.protected)
            if next(_and(node.state, *args, None, []), None) is None:
                continue
            result = next(_and(node.state, *args, node, []), None)
            if result is None:
                self.failures += 1
                return None
            self.uses[render(t)] += 1
            self.covered += moves_in(t)
            return result
        return None

    def rank(self, node, candidates, visited):
        """`candidates` ((direction, steps) for OrNode `node`) reordered:
        those starting a template that dry-matches on the current board
        first, by the largest and then most-supported such template; the
        rest after, in random order."""
        scored = []
        for direction, steps in candidates:
            rel = _rel(node.state, node.car_name, node.collisions, node.parent_direction, direction, steps)
            best = None
            for t in self.by_root.get(rel, ()):
                args = (node.car_name, direction, steps, t[1], visited, node.protected)
                if next(_and(node.state, *args, None, []), None) is not None:
                    best = (moves_in(t), self.weight[render(t)])
                    self.uses[render(t)] += 1
                    break
            scored.append((best is not None, best or (0, 0), random.random(), (direction, steps)))
        scored.sort(reverse=True)
        return [c for *_, c in scored]


def _or(state, car_name, collisions, parent_direction, t, visited, protected, ctx, done):
    """Instantiations of template `t` for the subgoal "car_name must vacate
    `collisions`", as (new_state, done + its moves). With ctx None, a dry
    perceptual match: holes are cleared by lifting their car off the board
    instead of being searched."""
    key = (car_name, frozenset(collisions))
    if key in visited or all(name != car_name for name, _ in state):  # lifted off in a dry match
        return
    visited = visited | {key}
    for direction, steps in OrNode(state, car_name, collisions).directions():
        if _rel(state, car_name, collisions, parent_direction, direction, steps) == t[0]:
            yield from _and(state, car_name, direction, steps, t[1], visited, protected, ctx, done)


def _and(state, car_name, direction, steps, kids, visited, protected, ctx, done):
    """Plays car_name's move once every blocker is cleared, each blocker by a
    distinct child of `kids`, in any order (as AndNode.solve tries them). Like
    AndNode.solve, the move is made from wherever clearing the blockers left
    the car (see AndNode.move_from), and skipped if that is its destination."""
    node = AndNode(state, car_name, direction, steps)
    blockers = node.blockers()
    if len(blockers) != len(kids) or protected & blockers.keys():
        return
    protected = protect(protected, car_name)
    for order in itertools.permutations(blockers.items()):
        for working_state, moves in _order(state, order, kids, direction, visited, protected, ctx, done):
            if all(name != car_name for name, _ in working_state):  # lifted off in a dry match
                continue
            final = node.move_from(working_state)
            if final is None:
                yield working_state, moves
            elif not AndNode(working_state, car_name, *final).blockers():
                yield AndNode(working_state, car_name, *final).apply(working_state), moves + [(car_name, *final)]


def _order(state, order, kids, direction, visited, protected, ctx, done):
    if not order:
        yield state, done
        return
    (blocker, collisions), rest = order[0], order[1:]
    tried = set()
    for i, kid in enumerate(kids):
        if render(kid) in tried:
            continue
        tried.add(render(kid))
        others = kids[:i] + kids[i + 1:]
        if kid is HOLE and ctx is None:
            subs = [(tuple(entry for entry in state if entry[0] != blocker), done)]
        elif kid is HOLE:
            try:
                result = OrNode(state, blocker, collisions, visited, protected, ctx.heuristic, ctx.gamma,
                                ctx.library, direction).solve()
            except GammaLapse as lapse:
                raise GammaLapse(lapse.new_state, done + lapse.moves) from None
            if result is None:
                continue
            subs = [(result[0], done + result[1])]
        else:
            subs = _or(state, blocker, collisions, direction, kid, visited, protected, ctx, done)
        for new_state, new_done in subs:
            yield from _order(new_state, rest, others, direction, visited, protected, ctx, new_done)


def build(split_train, kind, min_moves=2, **learn_kwargs):
    """ChunkLibrary of `kind` ("flat" or "nested") from annotated forests,
    as (puzzle_index, forest) pairs (e.g. a chunks.solve_and_split() split)."""
    if kind == "flat":
        return ChunkLibrary(flat_templates(split_train, min_size=min_moves), min_moves)
    learned, _ = nested_chunks.learn(nested_chunks.as_passes(split_train), **learn_kwargs)
    return ChunkLibrary(nested_templates(learned), min_moves)
