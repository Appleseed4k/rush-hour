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
    if args.boxoban:
        files = glob.glob(os.path.join(HERE, "data", "raw", "boxoban-levels", "medium", "valid", "*.txt"))
        rng = random.Random(0)
        levels = [lv for f in rng.sample(files, 2) for lv in read_levels(f)]
        for name, rows in rng.sample(levels, args.boxoban):
            print(f"boxoban medium {name}: {check_level(Level(rows, name), rows)}")
    print("all checks passed")


if __name__ == "__main__":
    sys.exit(main())
