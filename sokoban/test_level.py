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
    if args.boxoban:
        files = glob.glob(os.path.join(HERE, "data", "raw", "boxoban-levels", "medium", "valid", "*.txt"))
        rng = random.Random(0)
        levels = [lv for f in rng.sample(files, 2) for lv in read_levels(f)]
        for name, rows in rng.sample(levels, args.boxoban):
            print(f"boxoban medium {name}: {check_level(Level(rows, name), rows)}")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
