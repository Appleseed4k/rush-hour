"""Teachers against the AND-OR student, within an attempt: how much does
showing optimal pushes help the planner solve a level, and how many pushes
does that cost the teaching budget?

    python teach_experiment.py data/microban1.txt data/boxoban_hard.txt --n 100 --processes 12
    python teach_experiment.py data/boxoban_hard.txt --teachers none random:0.1 random:0.3 --gammas 0.05

The counterpart of scripts/teach_experiment.py, but there the student is a
policy network that learns from the moves it's shown, across rounds of
expert iteration. The Sokoban student doesn't learn yet, so this measures
only the immediate effect: each (level, teacher, gamma, seed) is one
and_or.play() attempt, and every teacher plays the same levels with the same
seeds. Per attempt it records
  solved    play() solved the level within --max-pushes pushes
  pushes    pushes played (restarts included); own = the student's, taught = the teacher's
  restarts  times the student went back to the start
  capped    the attempt hit --max-visits (search effort, nodes visited)
            before solving; it counts as unsolved
and writes one CSV row per attempt to --out (each level's rows as soon as it
finishes), then prints a summary by teacher, and solve rates by
optimal-distance bin. `--summarize` reprints the summary from a saved CSV.
"""
import argparse
import csv
import os
import statistics
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from and_or import play
from evaluate import bin_of
from level import Level
from teach import make_teacher

EDGES = (0, 10, 20, 30, 40, 60)


def read_levels(path, n=None, seed=0, max_states=None):
    """[(optimal, board, source)]: a random sample of n solvable levels with
    at most max_states states (the teacher's oracle has to fit in memory)."""
    import random
    rows = []
    with open(path) as f:
        for line in f:
            pushes, board, n_states, _, source = line.rstrip("\n").split(" ", 4)
            if int(pushes) >= 0 and (max_states is None or int(n_states) <= max_states):
                rows.append((int(pushes), board, source))
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def run(job):
    """Every (teacher, gamma, seed) attempt at one level."""
    optimal, board, source, dataset, teachers, gammas, seeds, max_pushes, max_visits, settings = job
    level = Level.from_line(board, source)
    rows = []
    for spec in teachers:
        for gamma in gammas:
            for seed in seeds:
                teacher = make_teacher(spec)
                t0 = time.perf_counter()
                result = play(level, gamma=gamma, max_pushes=max_pushes, seed=seed, teacher=teacher,
                              max_visits=max_visits, **settings)
                rows.append({
                    "dataset": dataset, "source": source, "optimal": optimal, "teacher": spec, "gamma": gamma,
                    "seed": seed, "solved": int(result["solved"]), "pushes": result["pushes"],
                    "own": result["pushes"] - result["taught"], "taught": result["taught"],
                    "restarts": result["restarts"], "visits": result["visits"], "capped": int(result["capped"]),
                    "no_move": teacher.counts["no_move"] if teacher is not None else 0,
                    "seconds": round(time.perf_counter() - t0, 3),
                })
    return rows


def _mean_se(xs):
    if not xs:
        return float("nan"), float("nan")
    m = statistics.fmean(xs)
    return m, (statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def summarize(rows, teachers, edges=EDGES):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["gamma"], r["teacher"])].append(r)
    bins = [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    print(f"\n{'dataset':24s} {'gamma':>5s} {'teacher':12s} {'n':>5s} {'solved':>12s} {'pushes/opt':>10s} "
          f"{'taught':>7s} {'share':>6s} {'restarts':>8s} {'capped':>6s} |" + " ".join(f"{b:>5s}" for b in bins))
    order = {t: i for i, t in enumerate(teachers)}
    for (dataset, gamma, spec), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], order[kv[0][2]])):
        solved, se = _mean_se([r["solved"] for r in rs])
        ratio = statistics.median([r["pushes"] / r["optimal"] for r in rs if r["solved"] and r["optimal"] > 0] or [float("nan")])
        taught = statistics.fmean(r["taught"] for r in rs)
        share = sum(r["taught"] for r in rs) / max(1, sum(r["pushes"] for r in rs))
        by_bin = defaultdict(list)
        for r in rs:
            by_bin[bin_of(r["optimal"], edges)].append(r["solved"])
        cells = " ".join(f"{sum(v) / len(v):5.2f}" if v else f"{'-':>5s}" for v in (by_bin[b] for b in bins))
        print(f"{dataset:24s} {gamma:5.2f} {spec:12s} {len(rs):5d} {solved:6.2f}+-{se:4.2f} {ratio:10.2f} "
              f"{taught:7.1f} {share:6.2f} {statistics.fmean(r['restarts'] for r in rs):8.1f} {statistics.fmean(r['capped'] for r in rs):6.2f} |{cells}")
    print("\nsolved: share of attempts solved (+- standard error); pushes/opt: median pushes over optimal, solved "
          "attempts;\ntaught: pushes shown per attempt; share: of all pushes played; restarts: per attempt;\ncapped: share of attempts stopped by --max-visits;\n"
          "after |: solve rate by the level's optimal pushes")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--n", type=int, default=100, help="levels sampled per dataset")
    parser.add_argument("--teachers", nargs="+", default=["none", "random:0.05", "random:0.1", "random:0.2",
                                                          "random:0.5", "random:1"],
                        help="kind:parameter for each teacher, or none")
    parser.add_argument("--gammas", default="0,0.05")
    parser.add_argument("--seeds", type=int, default=3, help="attempts per level and teacher")
    parser.add_argument("--max-pushes", type=int, default=400)
    parser.add_argument("--max-visits", type=int, default=300_000,
                        help="search cap per attempt, in OrNodes + PushNodes visited (about 4 s; 0 = none)")
    parser.add_argument("--max-states", type=int, default=1_000_000,
                        help="skip levels whose oracle is larger than this")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--out", default="data/eval_teach.csv")
    parser.add_argument("--summarize", action="store_true", help="only reprint the summary of --out")
    args = parser.parse_args(argv)

    if args.summarize:
        with open(args.out, newline="") as f:
            rows = [{k: (float(v) if k in ("gamma", "seconds") else v if k in ("dataset", "source", "teacher")
                         else int(v)) for k, v in r.items()} for r in csv.DictReader(f)]
        summarize(rows, list(dict.fromkeys(r["teacher"] for r in rows)))
        return

    for spec in args.teachers:
        make_teacher(spec)  # fail early on a bad spec
    gammas = [float(g) for g in args.gammas.split(",")]
    jobs = []
    for path in args.datasets:
        dataset = os.path.splitext(os.path.basename(path))[0]
        for optimal, board, source in read_levels(path, args.n, max_states=args.max_states):
            jobs.append((optimal, board, source, dataset, args.teachers, gammas, range(args.seeds),
                         args.max_pushes, args.max_visits or None, {}))
    print(f"{len(jobs)} levels x {len(args.teachers)} teachers x {len(gammas)} gammas x {args.seeds} seeds",
          file=sys.stderr)
    jobs.sort(key=lambda job: -job[0])  # long levels first, so a slow one doesn't start last
    t0 = time.perf_counter()
    rows = []
    with open(args.out, "w", newline="") as f, (Pool(args.processes) if args.processes > 1 else _Serial()) as pool:
        writer = None
        for n, out in enumerate(pool.imap_unordered(run, jobs), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(out[0]))
                writer.writeheader()
            writer.writerows(out)
            f.flush()
            rows.extend(out)
            slow = max(r["seconds"] for r in out)
            print(f"  {n}/{len(jobs)} levels  {time.perf_counter() - t0:.0f}s  ({out[0]['source']}: "
                  f"{sum(r['seconds'] for r in out):.0f}s, slowest attempt {slow:.0f}s)", file=sys.stderr, flush=True)
    print(f"done in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    summarize(rows, args.teachers)


class _Serial:
    """Pool's imap_unordered, in this process."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, jobs):
        return map(fn, jobs)


if __name__ == "__main__":
    main()
