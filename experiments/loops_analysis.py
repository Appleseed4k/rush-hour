"""Scores stage 1 of the salience and loop-aware bayes4 test (experiments/loops_queue.py): the comparisons fixed there,
paired by student, per agency, on the end test's 9-16-move puzzles (primary) and on all of them (secondary).

End score: moves / optimal with an unsolved attempt at its cap (lower is better); a negative difference favours the
first teacher named.

    python experiments/loops_analysis.py     # writes models/summary/loops_stage1.csv and prints it
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import phase2_analysis as p2  # noqa: E402
from loops_queue import AGENCIES, LOOPS, MATCHED, seed_part  # noqa: E402
from rushhour import analysis as an  # noqa: E402

OUT = os.path.join(p2.SUMMARY, "loops_stage1.csv")
PEOPLE_MAX = 16
STUDENTS = {"salient": "salient_m10{}_agency{:g}", "soft": "twomode_m10{}_agency{:g}"}
SEEDS = (0, 1, 2)


def scores(student, agency, teacher, max_distance):
    """{(seed, student): end score} over the seeds; no teacher from the agency-1 run."""
    out = {}
    for seed in SEEDS:
        run = STUDENTS[student].format(seed_part(seed), 1.0 if teacher == "none" else agency)
        e = an.entry(run, teacher)
        if e is not None:
            out |= {(seed, s): v for s, v in p2.end_scores(e, max_distance).items()}
    return out


def spend(student, agency, teacher):
    shares = [an.spend(e) for seed in SEEDS
              if (e := an.entry(STUDENTS[student].format(seed_part(seed), agency), teacher)) is not None]
    return np.mean(shares) if shares else np.nan


def paired(a, b):
    d = [a[k] - b[k] for k in a if k in b]
    m, se = p2.mean_se(d)
    return {"diff": m, "se": se, "z": m / se, "better": f"{sum(x < 0 for x in d)}/{len(d)}", "n": len(d)}


def table():
    rows = []
    for max_distance, scope in ((PEOPLE_MAX, "9-16"), (None, "all")):
        for a in AGENCIES:
            sal = {t: scores("salient", a, t, max_distance) for t in ("none", "bayes4", LOOPS, MATCHED)}
            soft = {t: scores("soft", a, t, max_distance) for t in ("none", "bayes4", LOOPS)}
            comparisons = [
                ("1. bayes4 - none", "salient", sal["bayes4"], sal["none"]),
                ("2. loops - bayes4", "salient", sal[LOOPS], sal["bayes4"]),
                ("2. loops - bayes4", "soft", soft[LOOPS], soft["bayes4"]),
                ("3. loops - random 10%", "salient", sal[LOOPS], sal[MATCHED]),
                ("(bayes4 - none)", "soft", soft["bayes4"], soft["none"]),
                ("(loops - none)", "salient", sal[LOOPS], sal["none"]),
                ("(loops - none)", "soft", soft[LOOPS], soft["none"]),
                ("(random 10% - none)", "salient", sal[MATCHED], sal["none"]),
            ]
            for name, student, x, y in comparisons:
                rows.append({"scope": scope, "agency": a, "comparison": name, "student": student, **paired(x, y)})
            # the interaction: (loops - bayes4) on the salient student minus on the soft one; different students,
            # so unpaired
            ds = [[x[k] - y[k] for k in x if k in y] for x, y in ((sal[LOOPS], sal["bayes4"]),
                                                                (soft[LOOPS], soft["bayes4"]))]
            (m1, s1), (m2, s2) = p2.mean_se(ds[0]), p2.mean_se(ds[1])
            rows.append({"scope": scope, "agency": a, "comparison": "2. interaction (salient - soft)",
                         "student": "both", "diff": m1 - m2, "se": np.hypot(s1, s2), "z": (m1 - m2) / np.hypot(s1, s2),
                         "n": len(ds[0]) + len(ds[1])})
    return pd.DataFrame(rows)


def ends():
    """Mean end score per student, teacher and agency, with the teacher's spend."""
    rows = []
    for max_distance, scope in ((PEOPLE_MAX, "9-16"), (None, "all")):
        for student, teachers in (("salient", ("none", "bayes4", LOOPS, MATCHED)), ("soft", ("none", "bayes4", LOOPS))):
            for a in AGENCIES:
                for t in teachers:
                    s = scores(student, a, t, max_distance)
                    rows.append({"scope": scope, "student": student, "agency": a, "teacher": t,
                                 "end": np.mean(list(s.values())), "n": len(s),
                                 "shown": np.nan if t == "none" else spend(student, a, t)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t, e = table(), ends()
    t.to_csv(OUT, index=False)
    e.to_csv(OUT.replace(".csv", "_ends.csv"), index=False)
    print(e.round(3).to_string(index=False))
    print()
    print(t.round(3).to_string(index=False))
