"""Phase 2 of the switch to AND-OR's apprentice (experiments/phase2_queue.py): the three questions, as tables in
models/summary.

1. hindsight_fit.csv: each hindsight (own, W=3, W=10, full) and multiplier against people's learning (p1.6's fit: the
   within-learner change in log moves / optimal per 10 trials over trials 1-30 on short (6, 10) and long (13, 15)
   puzzles, and moves / optimal in trials 1-10; misfit = sum of the three z squared).
2. ao10_agency_seeds.csv: each teacher's end score against no teacher, paired by student, per seed and pooled over
   seeds 0-2 (24 students); the score is the end test's moves / optimal with an unsolved puzzle at its cap.
3. bayes4_theta.csv: bayes4's theta at agencies 0, 0.3 and 1 (seed 0), the same way.

    python experiments/phase2_analysis.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import analysis as an, human_data as hd, paths  # noqa: E402
from rushhour.student import spec  # noqa: E402

SUMMARY = os.path.join(paths.MODELS, "summary")
SPLIT, BLOCK = 30, 10
SHORT, LONG = (6, 10), (13, 15)
STATS = ("short", "long", "start")
HINDSIGHT = {"own": "own", "W=3": "w3", "W=10": "w10", "full": "all"}
MULTIPLIERS = (5, 10, 20, 30, 60, 120)
SEEDS = {0: "aostart_free_m10_agency{}", 1: "aostart_free_m10_s1_agency{}", 2: "aostart_free_m10_s2_agency{}"}
NONE_AT = "1"  # no teacher plays only in the agency-1 run: it never sees a shown move, so it is the same at every agency
TEACHERS = ["bayes4", "random:0.3", "random:1.0"]
THETAS = ["bayes4:0.25", "bayes4", "bayes4:0.75", "bayes4:0.9"]


def mean_se(x):
    x = np.asarray(list(x), float)
    return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def played(run):
    e = an.entry(run, "none")
    if e is None:
        return None
    return pd.DataFrame([x | {"student": s} for s, rec in e["sessions"].items() for x in rec])


def fit_stats(df, person, puzzle):
    early = df[(df.trial <= SPLIT) & df.solved].assign(log_ratio=lambda d: np.log(d.moves / d.distance))
    out = {name: hd.trend(early[early.distance.isin(ds)], "log_ratio", person=person, puzzle=puzzle)
           for name, ds in (("short", SHORT), ("long", LONG))}
    first = df[(df.trial <= BLOCK) & df.solved]
    out["start"] = mean_se((first.moves / first.distance).groupby(first[person]).mean())
    return out


def hindsight_fit():
    human = fit_stats(hd.trials(), "subject", "instance")
    rows = [{"hindsight": "people (42)", "m": None, **{k: human[k][0] for k in STATS},
             **{f"{k} se": human[k][1] for k in STATS}}]
    for name, key in HINDSIGHT.items():
        for m in MULTIPLIERS:
            df = played(f"ao_people_{key}_m{m}")
            if df is None:
                continue
            s = fit_stats(df, "student", "puzzle")
            row = {"hindsight": name, "m": m}
            for k in STATS:
                (b, se), (hb, hse) = s[k], human[k]
                row[k], row[f"{k} se"], row[f"z {k}"] = b, se, (b - hb) / np.hypot(se, hse)
            first = df[df.trial <= BLOCK]
            row["solved 1-10"] = first.solved.mean()
            row["misfit"] = sum(row[f"z {k}"] ** 2 for k in STATS)
            rows.append(row)
    return pd.DataFrame(rows)


def end_scores(e, max_distance=None):
    """{student: end score} of a teacher's entry; with `max_distance`, over the end test's puzzles of at most that
    many moves only."""
    if max_distance is None:
        return {s: v[0] for s, v in an.final(e).items()}
    return {s: an.measures([a for a in by_n[max(by_n)] if a["distance"] <= max_distance], e["config"])[0]
            for s, by_n in e["results"].items()}


def ends(run, teacher, max_distance=None):
    e = an.entry(run, teacher)
    return None if e is None else end_scores(e, max_distance)


def versus(run, teacher, none, max_distance=None):
    """(end, [teacher - none per student], share shown) or None."""
    e = an.entry(run, teacher)
    if e is None or none is None:
        return None
    end = end_scores(e, max_distance)
    return np.mean(list(end.values())), [end[s] - none[s] for s in end if s in none], an.spend(e)


def agency_seeds(seeds=SEEDS, max_distance=None):
    """seeds: {seed: run name pattern with {} for the agency} (default: phase 1's x10 sweep); max_distance: score
    only the end test's puzzles of at most that many moves."""
    rows = []
    for a in spec.AGENCIES:
        tag = f"{a:g}"
        for teacher in TEACHERS:
            pooled, ends_, shown = [], [], []
            row = {"agency": a, "teacher": teacher}
            for seed, pattern in seeds.items():
                none = ends(pattern.format(NONE_AT), "none", max_distance)
                v = versus(pattern.format(tag), teacher, none, max_distance)
                if v is None:
                    continue
                end, d, sh = v
                m, se = mean_se(d)
                row[f"seed {seed}"] = m
                row[f"seed {seed} se"] = se
                pooled += d
                ends_.append(end)
                shown.append(sh)
            if not pooled:
                continue
            m, se = mean_se(pooled)
            row |= {"pooled": m, "pooled se": se, "z": m / se, "better": f"{sum(x < 0 for x in pooled)}/{len(pooled)}",
                    "end": np.mean(ends_), "shown": np.mean(shown)}
            rows.append(row)
    none_rows = []
    for seed, pattern in seeds.items():
        none = ends(pattern.format(NONE_AT), "none", max_distance)
        none_rows.append({"seed": seed, "none end": np.mean(list(none.values())) if none else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(none_rows)


def theta(pattern=SEEDS[0]):
    """pattern: seed 0's run names with {} for the agency (default: phase 1's)."""
    none = ends(pattern.format(NONE_AT), "none")
    rows = []
    for a in (0.0, 0.3, 1.0):
        for teacher in THETAS:
            v = versus(pattern.format(f"{a:g}"), teacher, none)
            if v is None:
                continue
            end, d, sh = v
            m, se = mean_se(d)
            theta_ = float(teacher.split(":")[1]) if ":" in teacher else 0.45
            rows.append({"agency": a, "theta": theta_, "end": end, "vs none": m, "se": se,
                         "better": f"{sum(x < 0 for x in d)}/{len(d)}", "shown": sh})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    fit = hindsight_fit()
    fit.to_csv(os.path.join(SUMMARY, "hindsight_fit.csv"), index=False)
    print(fit[["hindsight", "m", "short", "long", "start", "z short", "z long", "z start", "solved 1-10", "misfit"]]
          .round(3).to_string(index=False))
    print()
    print(fit.pivot(index="hindsight", columns="m", values="misfit").reindex(list(HINDSIGHT)).round(1))
    ag, none = agency_seeds()
    ag.to_csv(os.path.join(SUMMARY, "ao10_agency_seeds.csv"), index=False)
    print()
    print(none.round(3).to_string(index=False))
    print(ag.round(3).to_string(index=False))
    th = theta()
    th.to_csv(os.path.join(SUMMARY, "bayes4_theta.csv"), index=False)
    print()
    print(th.round(3).to_string(index=False))
