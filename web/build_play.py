"""Builds the Rush Hour play page: samples puzzles from the analysis pool and
writes them into play_template.html.

    python web/build_play.py              # -> web/rush_hour_play.html
    python web/build_play.py --teacher none

The teacher is "none", "delay:<seconds>" (the default, delay:10: after that
long without a move, the page makes an optimal move for the player) or
"bayes:<seconds>[:<name>=<value>,...]" (after that long, a Bayesian teacher
decides whether to make the move; its parameters a, b, cost, quit and worse
are described in the template's teacher section), e.g. bayes:10:cost=0.05.

The default sample (250 puzzles from each of the distance bins 1-8, 9-12,
13-16 and 17-20 of data/rush_nw.txt, seed 2026) is the one the published
page uses, so poolIndex in recorded games points into it. Distances are this
repo's (red car reaches the edge), as in rush_nw.txt.
"""
import argparse
import collections
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "..", "data", "rush_nw.txt")
TEMPLATE = os.path.join(HERE, "play_template.html")
MARKER = "/*PUZZLES*/"
TEACHER_MARKER = "/*TEACHER*/"
BINS = ((1, 8), (9, 12), (13, 16), (17, 20))


def sample(pool=POOL, bins=BINS, per_bin=250, seed=2026):
    """[(distance, board)] - `per_bin` puzzles from each bin, bins in order."""
    by_bin = collections.defaultdict(list)
    with open(pool) as f:
        for line in f:
            if not line.strip():
                continue
            d, board, _ = line.split()
            d = int(d)
            for lo, hi in bins:
                if lo <= d <= hi:
                    by_bin[(lo, hi)].append((d, board))
    rng = random.Random(seed)
    chosen = []
    for b in sorted(by_bin):
        chosen += rng.sample(by_bin[b], per_bin)
    return chosen


BAYES_PARAMETERS = {"a", "b", "cost", "quit", "worse"}


def check_teacher(spec):
    def number(x):
        try:
            return float(x) >= 0
        except ValueError:
            return False

    kind, _, rest = spec.partition(":")
    seconds, _, params = rest.partition(":")
    ok = spec == "none" or (kind == "delay" and not params and number(seconds) and float(seconds) > 0) or (
        kind == "bayes" and number(seconds) and float(seconds) > 0 and all(
            name in BAYES_PARAMETERS and number(value)
            for name, _, value in (item.partition("=") for item in params.split(",") if params)))
    if not ok:
        raise argparse.ArgumentTypeError(
            f"teacher {spec!r}: use none, delay:<seconds> or bayes:<seconds>[:<name>=<value>,...] "
            f"with names from {sorted(BAYES_PARAMETERS)}")
    return spec


def build(puzzles, teacher="delay:10", template=TEMPLATE):
    with open(template) as f:
        page = f.read()
    for marker in (MARKER, TEACHER_MARKER):
        assert page.count(marker) == 1, "template needs exactly one " + marker
    page = page.replace(TEACHER_MARKER, check_teacher(teacher))
    return page.replace(MARKER, " ".join(f"{d}:{b}" for d, b in puzzles))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-bin", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--teacher", type=check_teacher, default="delay:10")
    ap.add_argument("--out", default=os.path.join(HERE, "rush_hour_play.html"))
    args = ap.parse_args(argv)
    puzzles = sample(per_bin=args.per_bin, seed=args.seed)
    with open(args.out, "w") as f:
        f.write(build(puzzles, args.teacher))
    print(f"{len(puzzles)} puzzles, teacher {args.teacher} -> {args.out}")


if __name__ == "__main__":
    main()
