"""Phase 1's teaching result on the two-mode student (experiments/replication_queue.py), scored as phase 1's x10 agency
sweep was (experiments/phase2_analysis.py, agency_seeds): each teacher's end score against no teacher, paired by
student, per seed and pooled over seeds 0-2 (24 students); the score is the end test's moves / optimal with an
unsolved puzzle at its cap. The same for phase 1's intuition student on the same start, and phase 1's table beside
them.

    python experiments/replication_analysis.py

Tables: models/summary/replication_{twomode,intuition}.csv (columns as ao10_agency_seeds.csv), and
replication_theta.csv: bayes4's theta for the two-mode student (replication_queue.py --theta), as bayes4_theta.csv.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phase2_analysis as p2  # noqa: E402
from replication_queue import STUDENTS, name  # noqa: E402

PHASE1 = os.path.join(p2.SUMMARY, "ao10_agency_seeds.csv")


def seeds(student):
    return {seed: name(student, seed, 0).replace("agency0", "agency{}") for seed in (0, 1, 2)}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    tables = {"phase 1": pd.read_csv(PHASE1)}
    for student in STUDENTS:
        ag, none = p2.agency_seeds(seeds(student))
        ag.to_csv(os.path.join(p2.SUMMARY, f"replication_{student}.csv"), index=False)
        tables[student] = ag
        print(f"== {student}: no teacher's end score by seed")
        print(none.round(3).to_string(index=False))
        print(ag[["agency", "teacher", "seed 0", "seed 1", "seed 2", "pooled", "pooled se", "z", "better", "end",
                  "shown"]].round(3).to_string(index=False))
        print()
    keys = ["agency", "teacher"]
    side = None
    for label, t in tables.items():
        t = t[keys + ["pooled", "pooled se"]].rename(columns={"pooled": label, "pooled se": f"{label} se"})
        side = t if side is None else side.merge(t, on=keys, how="outer")
    print("== teacher - no teacher, pooled over seeds (moves / optimal; negative helps)")
    print(side.round(3).to_string(index=False))
    th = p2.theta(seeds("twomode")[0])
    th.to_csv(os.path.join(p2.SUMMARY, "replication_theta.csv"), index=False)
    print()
    print("== bayes4's theta, two-mode student (seed 0)")
    print(th.round(3).to_string(index=False))
