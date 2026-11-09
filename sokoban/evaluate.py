"""Planner variants against optimal push counts: how often one lapse-free pass
plans a whole solution, and how often play() solves the level.

    python evaluate.py data/microban1.txt data/boxoban_unfiltered_test.txt --n 200 --processes 12

For each level (a dataset line, see label.py) and variant it records
  plan      one pass with gamma 0 returned a full plan ("plan" kind)
  plan_len  that plan's pushes (compare with `optimal`)
  solved    play(gamma) solved it within --max-pushes pushes
  pushes    pushes play() used (restarts included), restarts, and pass kinds
and writes one CSV row per (level, variant, gamma) to --out, then prints
solve rates by optimal-distance bin.
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

from and_or import Planner, play, solve_pass
from level import Level

VARIANTS = {
    "path": dict(access=False, freeze=False, protect_placed=False),
    "access": dict(access=True, freeze=False, protect_placed=False),
    "access+freeze": dict(access=True, freeze=True, protect_placed=False),
    "access+freeze+protect": dict(access=True, freeze=True, protect_placed=True),
}


def read_dataset(path, n=None, seed=0):
    """[(optimal, board, source)] - a random sample of n solvable levels, or all."""
    rows = []
    with open(path) as f:
        for line in f:
            pushes, board, _, _, source = line.rstrip("\n").split(" ", 4)
            if int(pushes) >= 0:
                rows.append((int(pushes), board, source))
    if n is not None and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def run(job):
    optimal, board, source, dataset, variant, gamma, max_pushes, seed = job
    level = Level.from_line(board, source)
    settings = VARIANTS[variant]
    t0 = time.perf_counter()
    first = solve_pass(Planner(level, rng=random.Random(seed), **settings), level.start)
    result = play(level, gamma=gamma, max_pushes=max_pushes, seed=seed, **settings)
    return {
        "dataset": dataset, "source": source, "optimal": optimal, "variant": variant, "gamma": gamma,
        "plan": int(first.kind == "plan"), "plan_len": len(first.moves) if first.kind == "plan" else "",
        "solved": int(result["solved"]), "pushes": result["pushes"], "restarts": result["restarts"],
        "passes": ";".join(f"{k}={v}" for k, v in sorted(result["passes"].items())),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def bin_of(optimal, edges):
    for lo, hi in zip(edges, edges[1:]):
        if lo <= optimal < hi:
            return f"{lo}-{hi - 1}"
    return f"{edges[-1]}+"


def summarize(rows, edges=(0, 10, 20, 30, 40, 60)):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["gamma"], r["variant"])].append(r)
    bins = [f"{lo}-{hi - 1}" for lo, hi in zip(edges, edges[1:])] + [f"{edges[-1]}+"]
    print(f"\n{'dataset':28s} {'gamma':>5s} {'variant':22s} {'n':>4s} {'plan':>5s} {'solved':>6s} "
          f"{'len/opt':>7s} {'restarts':>8s}  solved by optimal pushes " + " ".join(f"{b:>6s}" for b in bins))
    for (dataset, gamma, variant), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1],
                                                                                list(VARIANTS).index(kv[0][2]))):
        planned = [r for r in rs if r["plan"] and r["optimal"] > 0]
        ratio = sum(r["plan_len"] / r["optimal"] for r in planned) / len(planned) if planned else float("nan")
        by_bin = defaultdict(list)
        for r in rs:
            by_bin[bin_of(r["optimal"], edges)].append(r["solved"])
        cells = " ".join(f"{sum(v) / len(v):6.2f}" if v else f"{'-':>6s}" for v in (by_bin[b] for b in bins))
        print(f"{dataset:28s} {gamma:5.2f} {variant:22s} {len(rs):4d} {sum(r['plan'] for r in rs) / len(rs):5.2f} "
              f"{sum(r['solved'] for r in rs) / len(rs):6.2f} {ratio:7.2f} "
              f"{sum(r['restarts'] for r in rs) / len(rs):8.1f}  {'':24s} {cells}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+")
    parser.add_argument("--n", type=int, default=200, help="levels sampled per dataset (default 200)")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--gammas", default="0,0.05")
    parser.add_argument("--max-pushes", type=int, default=400)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--out", default="data/eval.csv")
    args = parser.parse_args(argv)

    jobs = []
    for path in args.datasets:
        dataset = os.path.splitext(os.path.basename(path))[0]
        for optimal, board, source in read_dataset(path, args.n):
            for variant in args.variants.split(","):
                for gamma in map(float, args.gammas.split(",")):
                    jobs.append((optimal, board, source, dataset, variant, gamma, args.max_pushes, 0))
    print(f"{len(jobs)} runs", file=sys.stderr)
    t0 = time.perf_counter()
    if args.processes > 1:
        with Pool(args.processes) as pool:
            rows = list(pool.imap_unordered(run, jobs, chunksize=2))
    else:
        rows = [run(job) for job in jobs]
    print(f"done in {time.perf_counter() - t0:.0f}s", file=sys.stderr)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
