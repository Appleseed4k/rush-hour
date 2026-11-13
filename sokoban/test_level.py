"""Checks for level.py and label.py. Run from the sokoban folder:

    python test_level.py            # quick checks plus the first 40 Microban levels
    python test_level.py --boxoban 20

The main check compares label.StateSpace's optimal push counts with a
deliberately naive solver written straight from the rules: a 0-1 BFS over
(player cell, box cells) in which a step costs 0 and a push costs 1, with no
bitmasks, no player normalization and no dead-cell pruning.
"""
import argparse
import glob
import os
import random
import sys
from collections import deque

from label import StateSpace
from level import DELTAS, Level, read_levels

HERE = os.path.dirname(os.path.abspath(__file__))


def naive_pushes(rows, max_nodes=3_000_000):
    """Optimal pushes by 0-1 BFS on the raw grid, or -1 if unsolvable (None if
    the search exceeds max_nodes)."""
    grid = [list(r) for r in rows]
    walls, goals, boxes, player = set(), set(), set(), None
    for i, r in enumerate(grid):
        for j, c in enumerate(r):
            if c == "#":
                walls.add((i, j))
            if c in ".+*":
                goals.add((i, j))
            if c in "$*":
                boxes.add((i, j))
            if c in "@+":
                player = (i, j)
    start = (player, frozenset(boxes))
    best = {start: 0}
    dq = deque([(0, start)])
    while dq:
        cost, (p, bx) = dq.popleft()
        if cost > best[(p, bx)]:
            continue
        if bx == goals:
            return cost
        if len(best) > max_nodes:
            return None
        for di, dj in DELTAS.values():
            q = (p[0] + di, p[1] + dj)
            if q in walls:
                continue
            if q in bx:
                beyond = (q[0] + di, q[1] + dj)
                if beyond in walls or beyond in bx:
                    continue
                new, c = (q, (bx - {q}) | {beyond}), cost + 1
            else:
                new, c = (q, bx), cost
            if c < best.get(new, 1 << 30):
                best[new] = c
                (dq.appendleft if c == cost else dq.append)((c, new))
    return -1


def check_tiny():
    cases = [
        ("#####/#@$.#/#####", 1),
        ("######/#@$-.#/######", 2),
        ("#####/#$@.#/#####", -1),           # box in a corner: dead from the start
        ("#####/#*@-#/#####", 0),            # already solved
        ("#######/#.@-$-#/#######", -1),     # player can't get behind the box in a corridor
        ("######/#-.--#/#-$--#/#-@--#/######", 1),
    ]
    for board, expected in cases:
        level = Level.from_line(board)
        space = StateSpace(level)
        naive = naive_pushes(level.render())
        assert space.dist[0] == naive == expected, (board, space.dist[0], naive, expected)
        assert Level.from_line(level.to_line()).to_line() == level.to_line()
    print("tiny levels ok")


def check_level(level, rows):
    """StateSpace agrees with the naive solver, its solution replays to a solved
    state, and every solvable non-goal state has a successor one push closer."""
    space = StateSpace(level)
    naive = naive_pushes(rows)
    if naive is None:
        return "naive search too large"
    assert space.dist[0] == naive, (level.name, space.dist[0], naive)
    sol = space.solution()
    if sol is not None:
        state = level.start
        for push in sol:
            assert push in level.pushes(state), (level.name, push)
            state = level.apply(state, push)
        assert level.solved(state) and len(sol) == naive
    for i, d in enumerate(space.dist):
        if d > 0:
            assert any(space.dist[j] == d - 1 for _, j in space.succ[i]), (level.name, i)
        if d == 0:
            assert level.solved(level.from_key(space.keys[i]))
    return f"{naive} pushes, {len(space)} states"


def check_freeze_and_planner(levels):
    """Level.deadlocked never flags a solvable state (checked against the
    oracle), and every pass the planner returns replays legally, with a
    "plan" pass ending solved."""
    from and_or import VARIANTS_FOR_TESTS, Planner, play, solve_pass
    flagged = unsolvable = 0
    for level in levels:
        space = StateSpace(level)
        for key, d in zip(space.keys, space.dist):
            boxes, _ = level.from_key(key)
            if level.deadlocked(boxes):
                flagged += 1
                assert d < 0, (level.name, level.render((boxes, None)))
            unsolvable += d < 0
        for settings in VARIANTS_FOR_TESTS:
            for seed in range(3):
                result = solve_pass(Planner(level, gamma=0.1, rng=random.Random(seed), **settings), level.start)
                state = level.start
                for push in result.moves:
                    assert push in level.pushes(state), (level.name, settings, push)
                    state = level.apply(state, push)
                assert state == result.state and (result.kind != "plan" or level.solved(state))
        outcome = play(level, gamma=0.05, max_pushes=150, seed=0)
        assert outcome["solved"] == level.solved(_replay(level, outcome["moves"]))
    print(f"deadlock test sound: flagged {flagged} of {unsolvable} unsolvable states; planner passes replay")


def check_teacher(levels):
    """Random(0) plays exactly like no teacher; Random(1) solves every level
    in its optimal number of pushes; at p = 0.5 every push it shows is
    optimal, and the log, moves and taught count agree with each other."""
    from and_or import play
    from teach import Random, Teacher
    for level in levels:
        space = StateSpace(level)
        optimal = space.dist[0]
        if optimal < 0:
            continue
        plain = play(level, gamma=0.05, max_pushes=150, seed=0)
        zero = play(level, gamma=0.05, max_pushes=150, seed=0, teacher=Random(0))
        assert zero == plain and zero["taught"] == 0, level.name
        full = play(level, gamma=0.05, max_pushes=500, seed=0, teacher=Random(1))
        assert full["solved"] and full["pushes"] == full["taught"] == optimal and full["restarts"] == 0, level.name

        class Checking(Random):
            def observe(self, state, push, next_state, taught):
                assert not taught or self.mistake_kind(state, next_state) is None, level.name

        teacher = Checking(0.5)
        half = play(level, gamma=0.05, max_pushes=150, seed=1, teacher=teacher)
        assert half["taught"] == teacher.counts["shown"]
        assert half["taught"] == sum(e["kind"] == "taught" for e in half["log"])
        assert half["pushes"] == sum(len(e["moves"]) for e in half["log"])
        assert half["solved"] == level.solved(_replay(level, half["moves"])), level.name
        assert Teacher.distance(teacher, level.start) == optimal
    print(f"teachers ok on {len(levels)} levels")


def check_student_labels(levels):
    """student.labels_of: every label is legal and no shorter than the
    oracle's distance; on an optimal attempt (a teacher showing every push)
    shortcut labels are exactly the oracle's distances and optimal pushes."""
    import torch
    from student import Agent, labels_of, solve_intuition
    from teach import Random
    torch.manual_seed(0)
    agent = Agent()
    n = 0
    for level in levels:
        space = StateSpace(level)
        optimal = space.dist[0]
        if optimal <= 0:
            continue
        for teacher in (None, Random(0.5), Random(1.0)):
            result = solve_intuition(agent, level, max(50, 5 * optimal), teacher=teacher, rng=random.Random(0))
            exact = teacher is not None and teacher.p == 1.0
            assert not exact or (result["solved"] and result["pushes"] == optimal)
            for labels in ("own", "all"):
                for window in (None, 3):
                    for key, (d, targets) in labels_of(level, result, labels, window).items():
                        state = level.from_key(key)
                        true = space.distance(state)
                        assert true is not None and d >= true and targets <= set(level.pushes(state)), level.name
                        if exact and labels == "all" and window is None:
                            assert d == true and all(space.distance(level.apply(state, p)) == true - 1
                                                     for p in targets), level.name
                        n += 1
    print(f"student labels ok ({n} labels)")


def check_guard(levels):
    """Guard(1) never lets the student reach a deadlocked board, so it never
    restarts; every push it shows is optimal and comes right where the
    student's own push would have deadlocked."""
    import torch
    from student import Agent, solve_intuition
    from teach import Guard
    torch.manual_seed(0)
    agent = Agent()
    caught = 0
    for level in levels:
        space = StateSpace(level)
        optimal = space.dist[0]
        if optimal <= 0:
            continue

        class Checking(Guard):
            def observe(self, state, push, next_state, taught):
                assert self.distance(next_state) >= 0, level.name
                assert not taught or self.mistake_kind(state, next_state) is None, level.name

        for seed in range(3):
            teacher = Checking(1.0)
            result = solve_intuition(agent, level, max(50, 5 * optimal), teacher=teacher, rng=random.Random(seed))
            assert result["restarts"] == 0 and result["taught"] == teacher.counts["shown"], level.name
            assert teacher.counts["caught"] == teacher.counts["shown"] + teacher.counts["no_move"]
            caught += teacher.counts["caught"]
    print(f"guard ok ({caught} deadlocking pushes caught)")


def check_undo(levels):
    """Undo(1) never lets the student reach a deadlocked board and shows
    nothing: every push played is the student's own and never one it took
    back on that board. Undo(0) plays exactly as no teacher does."""
    import torch
    from student import Agent, solve_intuition
    from teach import Undo
    torch.manual_seed(0)
    agent = Agent()
    undone = 0
    for level in levels:
        optimal = StateSpace(level).dist[0]
        if optimal <= 0:
            continue
        cap = max(50, 5 * optimal)

        class Checking(Undo):
            def start(self, level, rng):
                super().start(level, rng)
                self.taken = {}

            def veto(self, state, push):
                out = super().veto(state, push)
                if out:
                    self.taken.setdefault(state, set()).add(push)
                return out

            def observe(self, state, push, next_state, taught):
                assert not taught and self.distance(next_state) >= 0, level.name
                assert push not in self.taken.get(state, ()), level.name

        for seed in range(3):
            teacher = Checking(1.0)
            result = solve_intuition(agent, level, cap, teacher=teacher, rng=random.Random(seed))
            assert result["restarts"] == 0 and result["taught"] == 0, level.name
            assert result["undone"] == teacher.counts["undone"] == teacher.counts["caught"], level.name
            undone += result["undone"]
            alone = solve_intuition(agent, level, cap, rng=random.Random(seed))
            zero = solve_intuition(agent, level, cap, teacher=Undo(0.0), rng=random.Random(seed))
            assert alone["segments"] == zero["segments"] and zero["undone"] == 0, level.name
    print(f"undo ok ({undone} deadlocking pushes taken back)")


def check_bayes(levels):
    """Bayes: its backward induction matches a simulation of its own model of
    the student (after it has learned from a few levels, so the fitted
    parameters are in use); bayes-undo with theta 1 acts as Undo(1) and with
    theta 0 as no teacher; bayes-show with theta 1 shows every push on a
    solvable board."""
    import numpy as np
    import torch
    from student import Agent, solve_intuition
    from teach import Bayes, Undo
    torch.manual_seed(0)
    agent = Agent()
    caps = {}
    teachers = {theta: Bayes(theta, "undo") for theta in (0.0, 1.0)}
    for level in levels:
        optimal = StateSpace(level).dist[0]
        if optimal <= 0:
            continue
        cap = caps[level.name] = max(50, 5 * optimal)
        for seed in range(2):
            alone = solve_intuition(agent, level, cap, rng=random.Random(seed))
            zero = solve_intuition(agent, level, cap, teacher=teachers[0.0], rng=random.Random(seed))
            assert alone["segments"] == zero["segments"] and zero["undone"] == 0, level.name
            undo = solve_intuition(agent, level, cap, teacher=Undo(1.0), rng=random.Random(seed))
            full = solve_intuition(agent, level, cap, teacher=teachers[1.0], rng=random.Random(seed))
            assert full["segments"] == undo["segments"] and full["undone"] == undo["undone"], level.name
            show = Bayes(1.0, "show")
            shown = solve_intuition(agent, level, cap, teacher=show, rng=random.Random(seed))
            assert shown["solved"] and shown["taught"] == shown["pushes"] == optimal, level.name

    # The forecast against a simulation of the same walk, on the last level, with what theta 0 learned.
    teacher = teachers[0.0]
    level = next(lv for lv in reversed(levels) if lv.name in caps)
    teacher.start(level, random.Random(0))
    kinds, jumps, (r0, r1) = teacher.mistake_shares()
    assert teacher._cap == caps[level.name] and 0 < r0 < 1 and 0 < r1 < 1
    rng = np.random.default_rng(0)
    d, K, cap = teacher._d, teacher._K, teacher._cap
    for zi in (3, 6, 9):
        logit = teacher.skill + np.exp(teacher.log_tau) * teacher.Z[zi]
        wins, n = 0, 20000
        for _ in range(n):
            k, m, dead = d, 0, None  # dead: None, "landed" (the test checks next) or "missed"
            while m < cap and k != 0:
                if dead == "landed":
                    dead = None if rng.random() < r0 else "missed"
                    if dead is None:
                        k = d
                    continue
                m += 1
                if dead == "missed":
                    if m < cap and rng.random() < r1:
                        dead, k = None, d
                    continue
                b = min(k, teacher.BINS) - 1
                if rng.random() < 1 / (1 + np.exp(-logit[b])):
                    k -= 1
                    continue
                kind = rng.choice(3, p=kinds[b])
                if kind == 0:
                    dead = "landed"
                elif kind == 1:
                    k = min(k + 1 + rng.choice(teacher.JUMPS, p=jumps), K)
            wins += k == 0 and dead is None
        planned = teacher._alone[0, zi, d]
        assert abs(wins / n - planned) < 4 * (planned * (1 - planned) / n) ** 0.5 + 1e-3, (zi, wins / n, planned)
    # The restart plan: its value against a simulation of the same walk with the teacher restarting where its
    # plan says (per difficulty), after learning from the same levels; and the restarts it forces in play.
    restarter = Bayes(0.0, "restart")
    forced = {"dead": 0, "far": 0}
    for lv in levels:
        if lv.name not in caps:
            continue
        r = solve_intuition(agent, lv, caps[lv.name], teacher=restarter, rng=random.Random(1))
        assert r["forced"] == restarter.counts["dead"] + restarter.counts["far"] - forced["dead"] - forced["far"]
        forced["dead"], forced["far"] = restarter.counts["dead"], restarter.counts["far"]
    restarter.start(level, random.Random(0))
    kinds, jumps, (r0, r1) = restarter.mistake_shares()
    d, K, cap = restarter._d, restarter._K, restarter._cap
    for zi in (3, 6, 9):
        logit = restarter.skill + np.exp(restarter.log_tau) * restarter.Z[zi]
        gain = restarter._gain[:, zi, :]
        wins, n = 0, 20000
        for _ in range(n):
            k, m, dead = d, 0, None
            while m < cap and k != 0:
                if dead == "landed":
                    dead = None if rng.random() < r0 else "missed"
                    if dead is None:
                        k = d
                    continue
                if gain[m, K + 1 if dead else min(k, K)] > 0:  # the teacher restarts the student
                    dead, k = None, d
                    if gain[m, d] > 0:
                        break  # (never: restarting at the start gains nothing)
                m += 1
                if dead == "missed":
                    if m < cap and rng.random() < r1:
                        dead, k = None, d
                    continue
                b = min(k, restarter.BINS) - 1
                if rng.random() < 1 / (1 + np.exp(-logit[b])):
                    k -= 1
                    continue
                kind = rng.choice(3, p=kinds[b])
                if kind == 0:
                    dead = "landed"
                elif kind == 1:
                    k = min(k + 1 + rng.choice(restarter.JUMPS, p=jumps), K)
            wins += k == 0 and dead is None
        planned = restarter._value[0, zi, d]
        assert abs(wins / n - planned) < 4 * (planned * (1 - planned) / n) ** 0.5 + 1e-3, (zi, wins / n, planned)
        assert planned >= restarter._alone[0, zi, d] - 1e-12
    print(f"bayes restart ok (plan matches simulation; forced {forced['dead']} on deadlocked boards, "
          f"{forced['far']} on far ones)")

    # The undo-value plan: its value against a simulation with the teacher undoing where its plan says.
    undoer = Bayes(0.1, "undo-value")
    for lv in levels:
        if lv.name in caps:
            solve_intuition(agent, lv, caps[lv.name], teacher=undoer, rng=random.Random(2))
    spent = undoer.counts["undone"]
    undoer.start(level, random.Random(0))
    kinds, jumps, (r0, r1) = undoer.mistake_shares()
    d, K, cap = undoer._d, undoer._K, undoer._cap
    for zi in (3, 6, 9):
        logit = undoer.skill + np.exp(undoer.log_tau) * undoer.Z[zi]
        gain = undoer._undo_gain[:, zi, :]
        scores, n = [], 20000
        for _ in range(n):
            k, m, dead, undos = d, 0, None, 0
            while m < cap and k != 0:
                if dead == "landed":
                    dead = None if rng.random() < r0 else "missed"
                    if dead is None:
                        k = d
                    continue
                m += 1
                if dead == "missed":
                    if m < cap and rng.random() < r1:
                        dead, k = None, d
                    continue
                b = min(k, undoer.BINS) - 1
                q = 1 / (1 + np.exp(-logit[b]))
                good = rng.random() < q
                kind = None if good else rng.choice(3, p=kinds[b])
                if kind == 0 and gain[m - 1, min(k, K)] > 0:  # taken back: the student plays a non-deadlocking push
                    keep = np.array([q, (1 - q) * kinds[b, 1], (1 - q) * kinds[b, 2]])
                    choice = rng.choice(3, p=keep / keep.sum())
                    undos += 1
                    good, kind = choice == 0, (None if choice == 0 else choice)
                if good:
                    k -= 1
                elif kind == 0:
                    dead = "landed"
                elif kind == 1:
                    k = min(k + 1 + rng.choice(undoer.JUMPS, p=jumps), K)
            scores.append(float(k == 0 and dead is None) - undoer.cost * undos)  # the plan's value: solving less undos
        planned = undoer._value[0, zi, d]
        assert abs(np.mean(scores) - planned) < 4 * np.std(scores) / n ** 0.5 + 1e-3, (zi, np.mean(scores), planned)
    print(f"bayes undo-value ok (plan matches simulation; {spent} pushes taken back, cost now {undoer.cost:.3g})")
    print(f"bayes ok (forecast matches simulation; skill at distance 1-3 "
          f"{', '.join(f'{1 / (1 + np.exp(-s)):.2f}' for s in teacher.skill[:3])}, r0 {r0:.2f}, r1 {r1:.2f})")


def check_restart(levels):
    """Restart teachers only send the student back to the start: the student
    never plays a push from a board the oracle calls deadlocked; with a
    finite k every stretch it plays to the end (a win included) fits in
    k * optimal pushes; with undo it never deadlocks, so it restarts only
    for straying; and nothing is ever shown."""
    import torch
    from student import Agent, solve_intuition
    from teach import Restart, make_teacher
    torch.manual_seed(0)
    agent = Agent()
    forced = 0
    for level in levels:
        optimal = StateSpace(level).dist[0]
        if optimal <= 0:
            continue
        cap = max(50, 5 * optimal)
        for spec in ("restart:inf", "restart:1.5", "undo+restart:1.5"):
            teacher = make_teacher(spec)
            assert isinstance(teacher, Restart)
            seen = []
            teacher.observe = lambda state, push, next_state, taught: seen.append((state, taught))
            result = solve_intuition(agent, level, cap, teacher=teacher, rng=random.Random(0))
            assert result["taught"] == 0 and not any(t for _, t in seen), (spec, level.name)
            assert all(teacher.distance(s) >= 0 for s, _ in seen), (spec, level.name)
            assert result["forced"] == teacher.counts["dead"] + teacher.counts["stray"], (spec, level.name)
            if spec != "restart:inf":
                limit = 1.5 * optimal
                for seg in result["segments"]:
                    assert len(seg["moves"]) <= limit or not seg.get("forced") and seg is not result["segments"][-1], \
                        (spec, level.name)
                if result["solved"]:
                    assert len(result["segments"][-1]["moves"]) <= limit, (spec, level.name)
            if spec.startswith("undo"):
                assert teacher.counts["dead"] == 0 and not any(s.get("dead") for s in result["segments"])
            forced += result["forced"]
    print(f"restart ok ({forced} restarts forced)")


def check_negative_labels(levels):
    """student.fatal_pushes names only pushes that led straight into a board
    the quick test calls deadlocked (or with no legal push); negative weight 0
    adds no examples; and learning a negative label lowers that push's
    probability on its board."""
    import torch
    from student import Agent, _loss, action_index, examples_from, fatal_pushes, solve_intuition
    torch.manual_seed(0)
    agent = Agent()
    found = 0
    negatives = []
    for level in levels:
        optimal = StateSpace(level).dist[0]
        if optimal <= 0:
            continue
        result = solve_intuition(agent, level, max(50, 5 * optimal), rng=random.Random(0))
        for key, pushes in fatal_pushes(result).items():
            state = level.from_key(key)
            for p in pushes:
                boxes, player = level.apply(state, p)
                assert level.deadlocked(boxes) or not level.pushes((boxes, player)), level.name
                found += 1
        plain = examples_from(agent, level, result, "all")
        assert not any(e.avoid for e in plain)
        with_negative = examples_from(agent, level, result, "all", negative_weight=1.0)
        assert len(with_negative) >= len(plain)
        negatives += [e for e in with_negative if e.avoid]
    assert negatives, "no deadlocks to label"
    e = negatives[0]

    def p_avoid():
        agent.net.eval()
        with torch.no_grad():
            logits = agent.net(torch.from_numpy(e.x).unsqueeze(0))[0][0]
        return torch.softmax(logits[e.legal], 0)[[e.legal.index(i) for i in e.avoid]].sum().item()

    before = p_avoid()
    agent.train_mode()
    for _ in range(20):
        loss = _loss(agent, [e])
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
    assert p_avoid() < before, (before, p_avoid())
    print(f"negative labels ok ({found} fatal pushes; one board's fatal push {before:.2f} -> {p_avoid():.2f})")


def _replay(level, moves):
    state = level.start
    for push in moves:
        state = level.apply(state, push)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--microban", type=int, default=40)
    parser.add_argument("--boxoban", type=int, default=0, help="random Boxoban medium levels to check")
    args = parser.parse_args()

    check_tiny()
    path = os.path.join(HERE, "data", "raw", "microban", "microban1.xsb")
    microban = read_levels(path)[:args.microban]
    for name, rows in microban:
        print(f"microban {name}: {check_level(Level(rows, name), rows)}")
    check_freeze_and_planner([Level(rows, name) for name, rows in microban])
    check_teacher([Level(rows, name) for name, rows in microban])
    check_student_labels([lv for lv in (Level(rows, name) for name, rows in microban) if lv.height <= 10 and lv.width <= 10][:20])
    check_guard([lv for lv in (Level(rows, name) for name, rows in microban) if lv.height <= 10 and lv.width <= 10][:20])
    check_negative_labels([lv for lv in (Level(rows, name) for name, rows in microban)
                           if lv.height <= 10 and lv.width <= 10][:20])
    check_undo([lv for lv in (Level(rows, name) for name, rows in microban) if lv.height <= 10 and lv.width <= 10][:20])
    check_restart([lv for lv in (Level(rows, name) for name, rows in microban)
                   if lv.height <= 10 and lv.width <= 10][:20])
    check_bayes([lv for lv in (Level(rows, name) for name, rows in microban) if lv.height <= 10 and lv.width <= 10][:20])
    if args.boxoban:
        files = glob.glob(os.path.join(HERE, "data", "raw", "boxoban-levels", "medium", "valid", "*.txt"))
        rng = random.Random(0)
        levels = [lv for f in rng.sample(files, 2) for lv in read_levels(f)]
        for name, rows in rng.sample(levels, args.boxoban):
            print(f"boxoban medium {name}: {check_level(Level(rows, name), rows)}")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
