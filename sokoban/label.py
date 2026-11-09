"""Full state spaces and optimal push distances for Sokoban levels, and the
command that turns raw level files into dataset files.

Mirrors scripts/lib.py's multi_bfs: a forward search lists every state
reachable from the start, then a search backward from the solved states
gives each one its distance to the goal. Unlike Rush Hour, pushes can't be
undone, so the backward search runs over predecessor edges, and a state
from which no solved state is reachable gets no distance (it is a deadlock).

By default pushes onto dead cells (Level.dead) are skipped while
enumerating: such a state is always a deadlock, and leaving them out keeps
the state space several times smaller. Every distance is unchanged.

Dataset line format (one level per line, sorted by decreasing distance as in
data/rush_nw.txt):

    <pushes> <board> <n_states> <n_solvable> <source>

`pushes` is the optimal number of pushes (-1 if the level has no solution),
`board` is Level.to_line(), `n_states` the number of distinct positions
reachable from the start (with live_only) and `n_solvable` how many of them
can still be solved. `source` is "<file>:<level name>".

Usage (from the sokoban folder):
    python label.py data/raw/microban/microban1.xsb -o data/microban1.txt
    python label.py "data/raw/boxoban-levels/hard/*.txt" -o data/boxoban_hard.txt --processes 12
"""
import argparse
import glob
import os
import sys
import time
from collections import deque
from multiprocessing import Pool

from level import Level, read_levels


class StateSpace:
    """Every position reachable from `start` (default: the level's start), as
    Level.key() ids, with push edges and distances to the goal.

    keys[i]   - the i-th state's key, in forward-search order (keys[0] is the start)
    index     - {key: i}
    succ[i]   - [(push, j)]: each push from state i and the state j it leads to
    dist[i]   - pushes to the goal along an optimal line, or -1 if unsolvable
    """

    def __init__(self, level, start=None, live_only=True, max_states=None):
        self.level = level
        start = level.start if start is None else start
        boxes, player = start
        region = level.reach(boxes, player)
        k0 = level.key(start, region)
        self.keys, self.index, self.succ = [k0], {k0: 0}, []
        frontier = [(boxes, region)]
        i = 0
        while i < len(frontier):
            boxes, region = frontier[i]
            edges = []
            for push in level.pushes_from(boxes, region, live_only):
                new_boxes, new_player = level.apply((boxes, None), push)
                new_region = level.reach(new_boxes, new_player)
                k = level.key((new_boxes, new_player), new_region)
                j = self.index.get(k)
                if j is None:
                    j = len(self.keys)
                    self.index[k] = j
                    self.keys.append(k)
                    frontier.append((new_boxes, new_region))
                    if max_states is not None and len(self.keys) > max_states:
                        raise StateSpaceTooLarge(len(self.keys))
                edges.append((push, j))
            self.succ.append(edges)
            i += 1

        n = len(self.keys)
        pred = [[] for _ in range(n)]
        for i, edges in enumerate(self.succ):
            for _, j in edges:
                pred[j].append(i)
        goals = level.goals
        dist = [-1] * n
        queue = deque()
        for i, (b, _) in enumerate(frontier):
            if b == goals:
                dist[i] = 0
                queue.append(i)
        while queue:
            j = queue.popleft()
            d = dist[j] + 1
            for i in pred[j]:
                if dist[i] < 0:
                    dist[i] = d
                    queue.append(i)
        self.dist = dist

    def __len__(self):
        return len(self.keys)

    @property
    def n_solvable(self):
        return sum(d >= 0 for d in self.dist)

    def distance(self, state):
        """Pushes to the goal from `state` (-1 for a deadlock, None if it is
        not in this space - e.g. a box on a dead cell when live_only)."""
        i = self.index.get(self.level.key(state))
        return None if i is None else self.dist[i]

    def solution(self, i=0):
        """An optimal push sequence from state i, or None if it has none. Pushes
        name the box's cell in the state they are played from."""
        if self.dist[i] < 0:
            return None
        pushes = []
        while self.dist[i] > 0:
            push, i = next((p, j) for p, j in self.succ[i] if self.dist[j] == self.dist[i] - 1)
            pushes.append(push)
        return pushes

    def distances(self):
        """{key: distance} for every state, like multi_bfs' result."""
        return dict(zip(self.keys, self.dist))


class StateSpaceTooLarge(Exception):
    pass


def label_level(job):
    """(line or None, seconds, error) for one (source, name, rows, max_states) job."""
    source, name, rows, max_states = job
    t0 = time.perf_counter()
    try:
        level = Level(rows, name)
        space = StateSpace(level, max_states=max_states)
    except (ValueError, StateSpaceTooLarge) as e:
        return None, time.perf_counter() - t0, f"{source}:{name}: {type(e).__name__} {e}"
    pushes = space.dist[0]
    line = f"{pushes} {level.to_line()} {len(space)} {space.n_solvable} {source}:{name}"
    return line, time.perf_counter() - t0, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", help="level files (globs allowed)")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--limit", type=int, default=None, help="label only the first N levels")
    parser.add_argument("--max-states", type=int, default=5_000_000,
                        help="skip a level whose state space exceeds this")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args(argv)

    paths = sorted(p for pattern in args.inputs for p in (glob.glob(pattern) or [pattern]))
    raw = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
    jobs = []
    for path in paths:
        path = os.path.abspath(path)
        # files under data/raw are named by their path there (hard/000), others by file name
        rel = os.path.relpath(path, raw) if path.startswith(raw + os.sep) else os.path.basename(path)
        source = os.path.splitext(rel)[0].replace("\\", "/").removeprefix("boxoban-levels/")
        for name, rows in read_levels(path):
            jobs.append((source, name, rows, args.max_states))
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} levels from {len(paths)} file(s)", file=sys.stderr)

    lines, errors, total = [], [], 0.0
    t0 = time.perf_counter()
    if args.processes > 1:
        with Pool(args.processes) as pool:
            results = pool.imap(label_level, jobs, chunksize=4)
            for n, (line, seconds, error) in enumerate(results, 1):
                total += seconds
                (lines.append(line) if line else errors.append(error))
                if n % 500 == 0:
                    print(f"  {n}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    else:
        for n, job in enumerate(jobs, 1):
            line, seconds, error = label_level(job)
            total += seconds
            (lines.append(line) if line else errors.append(error))
            if n % 100 == 0:
                print(f"  {n}/{len(jobs)}  {time.perf_counter() - t0:.0f}s", file=sys.stderr)

    lines.sort(key=lambda l: -int(l.split(" ", 1)[0]))
    with open(args.output, "w", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"wrote {len(lines)} levels to {args.output} in {time.perf_counter() - t0:.0f}s "
          f"({total / max(len(jobs), 1):.2f}s per level of CPU)", file=sys.stderr)
    for error in errors:
        print("skipped", error, file=sys.stderr)


if __name__ == "__main__":
    main()
