"""What the student learns from: examples from its own attempts (solved ones with loops cut out, shortcuts
found in hindsight, the moves a teacher showed), and the AND-OR attempts expert iteration labels."""
import random
from collections import Counter
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np
import torch

from ..and_or import AndNode, OrNode, _red_steps, solve as and_or_solve
from ..board import DELTAS, car_orientations, neighbors
from .network import DEST_DECAY, DIRECTIONS, StateEncoder
from .solve import _WORKER, _agent_payload, _cap, _init_attempt_worker, solve_intuition


# The network learns only what a solver itself could know: its own solved
# AND-OR attempts, with loops cut out, and which option resolved each
# subgoal in the plans that worked (see expert_iteration). Exact labels from
# a breadth-first search of each puzzle's state space (label_puzzles,
# label_subgoals) are for measuring the network, not for training it.

class Example():
    """One labelled board: its encoding, distance to the goal, legal and
    target move indices, and dest-head targets (action indices and values).
    From attempts (attempt_examples), the distance is the moves the best
    attempt still took and the target move is the one it played."""
    __slots__ = ("state", "x", "distance", "legal", "optimal", "dest_idx", "dest_target")

    def __init__(self, state, x, distance, legal, optimal, dest_idx, dest_target):
        self.state, self.x, self.distance = state, x, distance
        self.legal, self.optimal = legal, optimal
        self.dest_idx, self.dest_target = dest_idx, dest_target


class SubgoalExample():
    """One subgoal from an AND-OR search: the board, the action indices of
    the options AND-OR compares there (the subgoal car's candidate moves),
    and which of them are targets - the option that resolved it in a plan
    that worked, or the options an optimal solution prefers."""
    __slots__ = ("state", "x", "candidates", "targets")

    def __init__(self, state, x, candidates, targets):
        self.state, self.x, self.candidates, self.targets = state, x, candidates, targets


def _move_index(state_dict, car, direction, steps, n_cols, max_slide):
    """Action index of a move, keyed by the car's top/left cell (the order of
    PolicyAgent.action_space)."""
    r = min(i for i, _ in state_dict[car])
    c = min(j for _, j in state_dict[car])
    return ((r * n_cols + c) * 4 + DIRECTIONS.index(direction)) * max_slide + steps - 1


def _dest_labels(board, reached, size, max_slide):
    """Dest-head (action indices, targets) for `board`: every on-board
    destination of every car, with target DEST_DECAY ** (k - 1) if
    reached[(car, positions)] = k, else 0."""
    sd = dict(board)
    dest_idx, dest_target = [], []
    for car, direction, steps, moved in _destinations(board, size, max_slide):
        k = reached.get((car, moved))
        dest_idx.append(_move_index(sd, car, direction, steps, size[1], max_slide))
        dest_target.append(DEST_DECAY ** (k - 1) if k is not None else 0.0)
    return np.array(dest_idx, dtype=np.int16), np.array(dest_target, dtype=np.float32)


def _destinations(board, size, max_slide):
    """(car, direction, steps, positions moved to) for every on-board destination of every car, in the order of
    _dest_labels's targets."""
    n_rows, n_cols = size
    for car, cells in board:
        horizontal = cells[0][0] == cells[1][0]
        for direction in ("lr" if horizontal else "ud"):
            di, dj = DELTAS[direction]
            for steps in range(1, max_slide + 1):
                moved = tuple((i + di * steps, j + dj * steps) for i, j in cells)
                if not all(0 <= i < n_rows and 0 <= j < n_cols for i, j in moved):
                    break
                yield car, direction, steps, moved


class SearchRecorder():
    """While active (a `with` block), records the search tree of every
    and_or.solve pass: one frame per OrNode.solve call (its attempted
    actions) and per AndNode.solve call (its subgoals, and whether it
    succeeded). Call reset() before each pass."""

    def __enter__(self):
        self._or, self._and = OrNode.solve, AndNode.solve
        recorder = self

        def or_solve(node):
            frame = {"node": node, "attempts": [], "ok": False,
                     "free": (node.car_name, node.collisions) in node.visited}
            if recorder.stack:
                recorder.stack[-1]["children"].append(frame)
            recorder.subgoals.append(frame)
            recorder.stack.append(frame)
            try:
                result = recorder._or(node)
            finally:
                recorder.stack.pop()
            frame["ok"] = result is not None
            return result

        def and_solve(node):
            frame = {"node": node, "children": [], "ok": False}
            (recorder.stack[-1]["attempts"] if recorder.stack else recorder.roots).append(frame)
            recorder.stack.append(frame)
            try:
                result = recorder._and(node)
            finally:
                recorder.stack.pop()
            frame["ok"] = result is not None
            return result

        OrNode.solve, AndNode.solve = or_solve, and_solve
        self.reset()
        return self

    def __exit__(self, *exc):
        OrNode.solve, AndNode.solve = self._or, self._and

    def reset(self):
        self.roots, self.stack, self.subgoals = [], [], []

    def plan_choices(self):
        """[(OrNode, (direction, steps))] - every subgoal on the plan the last
        pass returned and the option that resolved it; empty if the pass
        lapsed or found no plan. A successful AndNode's subgoals are its last
        k OrNode calls (k = its blocker count): it returns right after the
        blocker order that worked."""
        root = next((r for r in reversed(self.roots) if r["ok"]), None)
        choices = []

        def walk(and_frame):
            k = len(and_frame["node"].blockers())
            for or_frame in (and_frame["children"][-k:] if k else []):
                chosen = next(a for a in reversed(or_frame["attempts"]) if a["ok"])
                choices.append((or_frame["node"], (chosen["node"].direction, chosen["node"].steps)))
                walk(chosen)

        if root is not None:
            walk(root)
        return choices


def _subgoal_example(node, targets, size, max_slide):
    """SubgoalExample for an OrNode; `targets` is the set of its (direction,
    steps) options to rank first. None if it has fewer than two options."""
    options = node.directions()
    if len(options) < 2:
        return None
    sd = dict(node.state)
    candidates = [_move_index(sd, node.car_name, d, s, size[1], max_slide) for d, s in options]
    return SubgoalExample(node.state, StateEncoder(size).encode_numpy(node.state),
                          np.array(candidates, dtype=np.int16), np.array([o in targets for o in options]))


def erase_loops(boards, moves, window=None):
    """An attempt's boards (start first) and moves with loops cut out: a
    board seen again rewinds the path to its earlier visit. With `window`,
    only a board last seen at most `window` moves earlier is recognized - a
    longer loop goes unnoticed and stays in the path (hindsight limited to
    recent boards); None recognizes every board."""
    kept = _kept_moves(boards, window)
    return [boards[0]] + [boards[t + 1] for t in kept], [moves[t] for t in kept]


def _kept_moves(boards, window=None):
    """The indices of the moves erase_loops keeps, in order."""
    kept, position, seen = [], {boards[0]: 0}, {boards[0]: 0}
    for t, board in enumerate(boards[1:], 1):
        if board in position and (window is None or t - seen[board] <= window):
            kept = kept[:position[board]]
            position = {boards[0]: 0} | {boards[k + 1]: i + 1 for i, k in enumerate(kept)}
        else:
            position[board] = len(kept) + 1
            kept.append(t - 1)
        seen[board] = t
    return kept


def shortcut_routes(boards, moves, taught=None, everywhere=False, window=None):
    """Shortcut labels for a solved attempt (`boards`, start first, ending solved; `moves`; `taught`, per move,
    whether a teacher showed it): hindsight over everything the student saw. The visited boards form a graph - two
    are joined when one legal move turns one into the other (moves are reversible) - and the shortest route from a
    board to the solved board through it is often far shorter than the path played, even with loops cut out.
    `window` limits the hindsight: two boards are joined only if the student was at them within `window` moves of
    each other (None: any), so a connection between boards far apart in the attempt goes unnoticed; the moves
    played are always joined, so every visited board still gets a label.
    Returns (boards, moves, taught) routes for attempt_examples: from each labelled board, one shortest route per
    first move that starts one (so moves tied for shortest are all targets). Labelled boards: the ones on shortest
    routes from the start, or with `everywhere` every visited board that reaches the solved board. A route's move is
    taught when the teacher showed that move on that board in the attempt."""
    taught = taught or [False] * len(moves)
    shown = {(b, m) for b, m, t in zip(boards, moves, taught) if t}
    visited = set(boards)
    goal = boards[-1]
    orient = car_orientations(boards[0])
    if window is None:
        adj = {b: [(m, n) for m, n in neighbors(b, orient) if n in visited] for b in visited}
    else:
        times = {}
        for t, b in enumerate(boards):
            times.setdefault(b, []).append(t)

        def near(a, b):
            return any(abs(x - y) <= window for x in times[a] for y in times[b])

        adj = {b: [(m, n) for m, n in neighbors(b, orient) if n in visited and near(b, n)] for b in visited}
    dist, queue = {goal: 0}, [goal]
    for b in queue:
        for _, n in adj[b]:
            if n not in dist:
                dist[n] = dist[b] + 1
                queue.append(n)

    def route(b):
        bs, ms = [b], []
        while dist[b] > 0:
            m, b = next((m, n) for m, n in adj[b] if dist.get(n) == dist[b] - 1)
            ms.append(m)
            bs.append(b)
        return bs, ms

    if everywhere:
        starts = [b for b in visited if dist.get(b, 0) > 0]
    else:
        starts, frontier = set(), {boards[0]} if dist[boards[0]] > 0 else set()
        while frontier:
            starts |= frontier
            frontier = {n for b in frontier for _, n in adj[b] if dist.get(n) == dist[b] - 1 and dist[n] > 0}
    routes = []
    for b in starts:
        for m, n in adj[b]:
            if dist.get(n) == dist[b] - 1:
                bs, ms = route(n)
                bs, ms = [b] + bs, [m] + ms
                routes.append((bs, ms, [(x, y) in shown for x, y in zip(bs, ms)]))
    return routes


LABELS = ("own", "path", "all")


def solved_routes(result, labels="own", window=None):
    """What a solved attempt (a solve_intuition result) labels, as (boards, moves, taught) routes for
    attempt_examples. `labels` "own": the path it played with loops cut out (erase_loops, recognizing boards within
    `window` moves; None = all); "path" / "all": shortcut labels (shortcut_routes - the boards on shortest routes from
    the start, or every visited board; connections noticed only between boards visited within `window` moves of
    each other, None = any)."""
    boards, moves, taught = result["boards"], result["moves"], result["taught"]
    if labels == "own":
        kept = _kept_moves(boards, window)
        return [([boards[0]] + [boards[t + 1] for t in kept], [moves[t] for t in kept], [taught[t] for t in kept])]
    if labels not in LABELS:
        raise ValueError(f"labels: one of {LABELS}, not {labels!r}")
    return shortcut_routes(boards, moves, taught, everywhere=labels == "all", window=window)


def attempt_examples(attempts, size=(6, 6), max_slide=5):
    """Examples from solved attempts ((boards, moves) pairs): per board, the
    attempt with the fewest moves left from it supplies the labels - the
    move it played (moves tied for fewest all count), the moves it still
    took as the distance, and how many moves later it had each car on each
    destination as dest targets."""
    best = {}  # board -> (moves left, set of move indices, {(car, positions): moves until reached})
    for boards, moves in attempts:
        n = len(moves)
        for t in range(n):
            board, left = boards[t], n - t
            idx = _move_index(dict(board), *moves[t], size[1], max_slide)
            if board in best and best[board][0] < left:
                continue
            if board in best and best[board][0] == left:
                best[board][1].add(idx)
                continue
            reached = {}
            for k in range(t + 1, n + 1):
                for entry in boards[k]:
                    reached.setdefault(entry, k - t)
            for entry in board:
                reached.pop(entry, None)
            best[board] = (left, {idx}, reached)

    encoder = StateEncoder(size)
    examples = []
    for board, (left, targets, reached) in best.items():
        sd = dict(board)
        legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
        dest_idx, dest_target = _dest_labels(board, reached, size, max_slide)
        examples.append(Example(board, encoder.encode_numpy(board), left, legal, sorted(targets), dest_idx,
                                dest_target))
    return examples


def _shown_example(board, moves, size, max_slide):
    """Example from a teacher showing `moves` on `board`: the moves are the
    targets; there is no distance or destination target."""
    sd = dict(board)
    legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
    optimal = sorted(_move_index(sd, *m, size[1], max_slide) for m in moves)
    return Example(board, StateEncoder(size).encode_numpy(board), None, legal, optimal,
                   np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.float32))


def intuition_examples(results, size=(6, 6), max_slide=5, window=None, encode=1.0, rng=None, labels="own"):
    """Examples from one puzzle's solve_intuition attempts (`results`). Solved
    attempts teach their boards along solved_routes(..., `labels`, `window`):
    by default the path played with loops cut out (erase_loops, recognizing
    boards within `window` moves; None = all), or shortcut labels ("path",
    "all") - see attempt_examples. Every
    move the teacher showed teaches too, solved or not: its board with that
    move as the target, unless a solved attempt already labels the board.

    `encode` < 1: encoding lapses - each board label from a solved attempt
    and each move shown is learned only with probability `encode` (drawn
    from `rng`); at 1 no random numbers are drawn."""
    rng = rng or random

    def kept():
        return encode >= 1 or rng.random() < encode

    attempts, shown = [], {}
    for result in results:
        for board, move, was_taught in zip(result["boards"], result["moves"], result["taught"]):
            if was_taught and kept():
                shown.setdefault(board, set()).add(move)
        if result["solved"]:
            attempts += [(bs, ms) for bs, ms, _ in solved_routes(result, labels, window)]
    examples = [e for e in attempt_examples(attempts, size, max_slide) if kept()]
    covered = {e.state for e in examples}
    examples += [_shown_example(board, moves, size, max_slide) for board, moves in shown.items()
                 if board not in covered]
    return examples


def _intuition_attempts(state0, n_attempts, seed, max_steps, size, max_slide, agent, mode):
    """attempt_puzzle for the pure-intuition student (see solve_intuition),
    with the mode's teacher, if any, showing optimal moves along the way."""
    rng = random.Random(seed)
    teacher = mode.get("teacher")
    if teacher is not None:
        teacher.counts = Counter()  # this puzzle's own tally (jobs can share a teacher object)
    results = []
    for _ in range(n_attempts):
        if teacher is not None:
            teacher.start(state0, rng)
        results.append(solve_intuition(agent, state0, max_steps, mode["memory"], mode["explore"], teacher, rng))
        if teacher is not None:
            teacher.end_attempt(results[-1])
    examples = intuition_examples(results, size, max_slide)
    return examples, [], {"solved": sum(r["solved"] for r in results),
                          "taught": sum(sum(r["taught"]) for r in results),
                          "moves": sum(len(r["moves"]) for r in results),
                          "interventions": sum(sum(r["taught"]) for r in results),
                          "teacher_counts": dict(teacher.counts) if teacher is not None else {}}


def attempt_puzzle(job):
    """(Examples, SubgoalExamples, info) for one puzzle from `n_attempts`
    attempts by the student `mode` describes (see label_attempts). info
    counts solved attempts, moves the teacher showed and moves played.

    - "and_or" student: AND-OR attempts at lapse probability `gamma`, guided
      by the worker's agent if it has one. Only solved attempts teach: their
      boards, loops cut out (attempt_examples), and every subgoal of every
      plan they completed, with the option that resolved it as the target.
    - "intuition" student: the worker's agent playing by intuition alone
      (solve_intuition), sampling at the mode's "explore" temperature, with
        the mode's "teacher" (see teachers.py) free to show optimal moves.
      Solved attempts teach as above (no subgoals: there is no search); every
      move the teacher showed teaches too, solved or not - its board with
      that move as the target, unless a solved attempt already labels the
      board."""
    state0, n_attempts, gamma, seed, max_steps, (size, max_slide), mode = job
    random.seed(seed)
    agent = _WORKER.get("agent")
    if mode["student"] == "intuition":
        return _intuition_attempts(state0, n_attempts, seed, max_steps, size, max_slide, agent, mode)
    heuristic = agent.heuristic if agent is not None else None
    attempts, subgoals, played = [], [], 0
    with SearchRecorder() as recorder:
        for _ in range(n_attempts):
            state, boards, moves, choices = state0, [state0], [], []
            while len(moves) < max_steps and _red_steps(state) > 0:
                recorder.reset()
                att = and_or_solve(state, heuristic=heuristic, gamma=gamma)
                step = att if type(att) is list else att[1]
                if not step:
                    break
                choices += recorder.plan_choices()
                for move in step:
                    state = AndNode(state, *move).apply(state)
                    boards.append(state)
                    moves.append(move)
            played += len(moves)
            if _red_steps(state) <= 0:
                attempts.append(erase_loops(boards, moves))
                for node, chosen in choices:
                    example = _subgoal_example(node, {chosen}, size, max_slide)
                    if example is not None:
                        subgoals.append(example)
    return attempt_examples(attempts, size, max_slide), subgoals, {"solved": len(attempts), "taught": 0,
                                                                   "moves": played}


def label_attempts(puzzles, n_attempts=4, gamma=0.07, agent=None, seed=0, max_steps=200, processes=None,
                   size=(6, 6), max_slide=5, student="and_or", teacher=None, explore=1.0, memory=10):
    """Labels from the student's own attempts on every puzzle (in parallel):
    (Examples, SubgoalExamples, info) - see attempt_puzzle. info sums
    attempts made, attempts solved, moves the teacher showed ("taught"; also
    "interventions", the teaching budget spent), moves played, and the
    teacher's own counts ("teacher_counts", see teachers.py).

    `student` "and_or": with `agent`, attempts are made by the human-like
    solver it guides (its heuristic, at its temperature); without, by plain
    AND-OR. `student` "intuition": `agent` (required) plays by intuition
    alone, sampling at temperature `explore` with recent-board memory
    `memory`, and `teacher` (see teachers.py), if given, may show optimal moves.
    `size` and `max_slide` (the agent's, when one is given) set the action
    indexing. `max_steps` caps each attempt's moves: a number, or a callable
    giving each puzzle's cap from its starting state."""
    if student == "intuition" and agent is None:
        raise ValueError("the intuition student needs an agent")
    if teacher is not None and student != "intuition":
        raise ValueError("teachers are only wired to the intuition student")
    weights, config = None, None
    if agent is not None:
        size, max_slide = agent.size, agent.max_slide
        weights, config = _agent_payload(agent)
    mode = {"student": student, "teacher": teacher, "explore": explore, "memory": memory}
    jobs = [(state, n_attempts, gamma, hash((seed, i)) & 0xFFFFFFFF, _cap(max_steps, state), (size, max_slide), mode)
            for i, state in enumerate(puzzles)]
    examples, subgoals = [], []
    info = {"attempts": len(puzzles) * n_attempts, "solved": 0, "taught": 0, "moves": 0, "interventions": 0,
            "teacher_counts": Counter()}
    with Pool(processes, initializer=_init_attempt_worker, initargs=(weights, config)) as pool:
        for ex, sg, counts in tqdm(pool.imap(attempt_puzzle, jobs, chunksize=2), total=len(jobs),
                                   desc="attempts"):
            examples += ex
            subgoals += sg
            for k, v in counts.items():
                if k == "teacher_counts":
                    info[k].update(v)
                else:
                    info[k] += v
    info["teacher_counts"] = dict(info["teacher_counts"])
    return examples, subgoals, info


def agency_weights(examples, results, shown_weight, size=(6, 6), max_slide=5, window=None, labels="own"):
    """Loss weights (see _weighted_loss) for intuition_examples(results, ...): what a teacher's shown move taught
    counts `shown_weight`, the student's own play 1 - with `shown_weight` < 1 it learns less from being shown a move
    than from finding one itself (the generation effect), at 0 nothing at all. An example on a board where a move
    was shown weighs `shown_weight` throughout; on a solved path's other boards, so do the dest targets of
    destinations a shown move first brought a car to (the rest of the example weighs 1; with several routes from a
    board, as shortcut labels have, a destination any of them got from a shown move). `size`, `max_slide`, `window`
    and `labels` as given to intuition_examples."""
    shown = {b for r in results for b, t in zip(r["boards"], r["taught"]) if t}
    given = {}  # path board -> destinations a shown move first brought a car to, later on the path
    for r in results:
        if r["solved"]:
            for path, moves, taught in solved_routes(r, labels, window):
                for i, board in enumerate(path[:-1]):
                    first = {}
                    for j in range(i, len(moves)):
                        car = moves[j][0]
                        first.setdefault((car, dict(path[j + 1])[car]), taught[j])
                    given.setdefault(board, set()).update(entry for entry, t in first.items() if t)
    weights = []
    for e in examples:
        if e.state in shown:
            weights.append(shown_weight)
        elif given.get(e.state):
            dest = [shown_weight if (car, moved) in given[e.state] else 1.0
                    for car, _, _, moved in _destinations(e.state, size, max_slide)]
            weights.append((1.0, np.array(dest, dtype=np.float32)))
        else:
            weights.append(1.0)
    return weights


def progress_examples(results, judge, recent, tail, size=(6, 6), max_slide=5, window=None, labels="own"):
    """Learning from its own moves, solved or not, by its sense of progress: `judge`'s value head (moves to go; a
    frozen copy of the student's starting network) rates each move the student made itself by the moves to go
    before it minus after. Among its last moves (`recent`, a deque this adds to), the clearest `tail` share each way
    are labelled: a step back teaches "not that move here" (the targets are every other legal move - the policy loss
    then pushes that move's probability down), a step forward "that move here" (it is the target). Moves a solved
    path already labels (solved_routes with `labels` and `window`) and moves a teacher showed get no label; a step
    back needs another legal move. Returns (examples, {"forward": n, "back": n})."""
    covered = set()
    for r in results:
        if r["solved"]:
            covered |= {(b, m) for bs, ms, _ in solved_routes(r, labels, window) for b, m in zip(bs, ms)}
    encoder = StateEncoder(size)
    judge.net.eval()
    moves = []
    for r in results:
        with torch.no_grad():
            v = judge.net.heads(torch.from_numpy(np.stack([encoder.encode_numpy(b) for b in r["boards"]])).float()
                                .to(judge.device))[2].cpu().numpy() * 10
        for t, (move, taught) in enumerate(zip(r["moves"], r["taught"])):
            if not taught:
                recent.append(v[t] - v[t + 1])
                moves.append((r["boards"][t], move, v[t] - v[t + 1]))
    if len(recent) < 50:
        return [], {"forward": 0, "back": 0}
    lo, hi = np.quantile(recent, [tail, 1 - tail])
    examples, counts = [], {"forward": 0, "back": 0}
    for board, move, progress in moves:
        if (board, move) in covered or lo < progress < hi:
            continue
        sd = dict(board)
        legal = [_move_index(sd, *m, size[1], max_slide) for m, _ in neighbors(board, car_orientations(board))]
        idx = _move_index(sd, *move, size[1], max_slide)
        targets = [idx] if progress >= hi else [i for i in legal if i != idx]
        if targets:
            examples.append(Example(board, encoder.encode_numpy(board), None, legal, targets,
                                    np.zeros(0, dtype=np.int16), np.zeros(0, dtype=np.float32)))
            counts["forward" if progress >= hi else "back"] += 1
    return examples, counts
