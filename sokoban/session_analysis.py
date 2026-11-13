"""Reading sessions.py runs back for notebooks: per-teacher tables, test
curves, paired comparisons, and the figures of notebooks/s1_teaching.ipynb.

Scores are as in sessions.py: a held-out attempt scores pushes / optimal, an
unsolved one counted at its cap (lower is better); a student's learning is
its start score minus its end score.
"""
import statistics

import numpy as np
import pandas as pd

from sessions import OUT, load_run
from student import score

NEUTRAL = "#8a8a85"
# Help is a magnitude, so random teachers share one hue, light (little help) to dark (all of it).
HELP_RAMP = {"random:0.1": "#86b6ee", "random:0.3": "#3f86da", "random:1": "#184f94",
             # the guard is another teacher (an identity), in categorical slot 3, its dose light to dark
             "guard:0.3": "#7fd3b2", "guard:1": "#14855d",
             # the undo teacher, categorical slot 4, the same light-to-dark dose
             "undo:0.3": "#a59cec", "undo:1": "#4535b0",
             # restart teachers: their own hue (dead only light, with straying dark), undo+restart magenta
             "restart:inf": "#f2a37f", "restart:1.5": "#c4471a", "undo+restart:1.5": "#c2357a",
             # the Bayesian undo teacher (teach.Bayes): its own hue, light (low threshold) to dark
             "bayes-undo:0.1": "#e3b448", "bayes-undo:0.5": "#9a6a00",
             # its other rules: the planned restart (dark orange, beside restart's hue), the value-of-undo rule (teal)
             "bayes-restart:0": "#8f2d0b", "bayes-undo-value:0.04": "#6cc3c9", "bayes-undo-value:0.18": "#1f7f86"}
GUARD_COLOR = "#1baf7a"
UNDO_COLOR = "#6250d6"  # categorical slot 4
# The two label settings are identities: categorical slots 1 and 2.
LABEL_COLORS = {"all": "#2a78d6", "own": "#eb6834"}
LABEL_NAMES = {"all": "shortcut labels", "own": "own labels"}


def teacher_name(spec):
    if spec == "none":
        return "no teacher"
    kind, _, arg = spec.partition(":")
    if kind.endswith("restart") and arg == "inf":
        return f"{kind} dead"
    return f"{kind} {arg or 1}"


def mean_se(xs):
    xs = list(xs)
    if not xs:
        return np.nan, np.nan
    return statistics.fmean(xs), (statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def load(name, out=OUT):
    """A run, or None if it hasn't been tested from the start yet."""
    try:
        run = load_run(name, out)
    except FileNotFoundError:
        return None
    return run if run["initial"] is not None else None


def start(run):
    return score(run["initial"])


def end_scores(run, spec):
    """{student: (score, solve rate)} at the last test point."""
    out = {}
    for s, st in run["teachers"][spec].items():
        last = max(st["tests"], key=int)
        out[s] = score(st["tests"][last])
    return out


def spend(run, spec):
    """Teacher interventions per session push: pushes shown, taken back (undo) or restarts forced (restart)."""
    records = [e for st in run["teachers"][spec].values() for e in st["record"]]
    return (sum(e["taught"] + e.get("undone", 0) + e.get("forced", 0) for e in records)
            / max(1, sum(e["pushes"] for e in records)))


def teacher_table(run):
    """One row per teacher with every student finished."""
    s0, solved0 = start(run)
    rows = []
    for spec, students in run["teachers"].items():
        if len(students) < run["config"]["students"]:
            continue
        records = [e for st in students.values() for e in st["record"]]
        end = end_scores(run, spec)
        learn = mean_se(s0 - v[0] for v in end.values())
        solved = mean_se(v[1] for v in end.values())
        rows.append({"teacher": teacher_name(spec), "pushes shown": spend(run, spec),
                     "solved as played": statistics.fmean(e["solved"] for e in records),
                     "end solve rate": solved[0], "end solve rate se": solved[1],
                     "learning": learn[0], "learning se": learn[1]})
    table = pd.DataFrame(rows).set_index("teacher")
    table.attrs["start"] = (s0, solved0)
    return table


def paired(run, spec, base):
    """End score of `spec` minus `base`'s, student by student: (mean, se, students better under `spec`)."""
    a, b = end_scores(run, spec), end_scores(run, base)
    d = [a[s][0] - b[s][0] for s in a if s in b]
    m, se = mean_se(d)
    return m, se, sum(x < 0 for x in d), len(d)


def paired_table(run, specs=None):
    """Row minus column end score for every pair of finished teachers (below 0: the row learned more)."""
    specs = specs or [t for t, st in run["teachers"].items() if len(st) == run["config"]["students"]]
    cells = {}
    for a in specs:
        cells[teacher_name(a)] = {}
        for b in specs:
            if a == b:
                cells[teacher_name(a)][teacher_name(b)] = ""
                continue
            m, se, better, n = paired(run, a, b)
            cells[teacher_name(a)][teacher_name(b)] = f"{m:+.2f} ± {se:.2f} ({better}/{n})"
    return pd.DataFrame(cells).T


def curve(run, spec, measure):
    """[(levels played, mean, se)] of `measure` ("score" or "solve rate") across students, start included."""
    k = 0 if measure == "score" else 1
    out = [(0, start(run)[k], 0.0)]
    students = run["teachers"][spec]
    points = sorted({int(p) for st in students.values() for p in st["tests"]})
    for p in points:
        vals = [score(st["tests"][str(p)])[k] for st in students.values() if str(p) in st["tests"]]
        m, se = mean_se(vals)
        out.append((p, m, se))
    return out


def plot_curves(ax, run, measure, specs=None):
    """Held-out `measure` along the session, one line per teacher, direct-labelled at the end."""
    specs = specs or list(run["teachers"])
    for spec in specs:
        if not run["teachers"].get(spec):
            continue
        pts = curve(run, spec, measure)
        x = [p for p, _, _ in pts]
        y = np.array([m for _, m, _ in pts])
        e = np.array([s for _, _, s in pts])
        color = HELP_RAMP.get(spec, NEUTRAL)
        ax.plot(x, y, color=color, lw=2, marker="o", ms=5, label=teacher_name(spec))
        ax.fill_between(x, y - e, y + e, color=color, alpha=0.15, lw=0)
    ax.set_xlabel("session levels played")
    ax.set_ylabel({"score": "held-out score (pushes / optimal, lower is better)",
                   "solve rate": "held-out solve rate"}[measure])
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)


def plot_dilemma(ax, runs):
    """Learning against the share of pushes shown, one line per run (label setting); no teacher as a
    dashed reference line in the run's color. `runs`: {label setting: run}."""
    labelled = False
    for labels, run in runs.items():
        if run is None:
            continue
        table = teacher_table(run)
        color = LABEL_COLORS[labels]
        taught = table[table.index.str.startswith("random")]
        for kind, kcolor, marker, label in (("guard", GUARD_COLOR, "D", "guard: shows the right push"),
                                            ("undo", UNDO_COLOR, "s", "undo: takes the push back")):
            rows = table[table.index.str.startswith(kind + " ")]
            if len(rows) and labels == "all":
                ax.errorbar(rows["pushes shown"], rows["learning"], yerr=rows["learning se"], color=kcolor,
                            lw=0, elinewidth=1.5, marker=marker, ms=8, capsize=3, label=f"{label} (shortcut labels)")
                for name, row in rows.iterrows():
                    ax.annotate(name.replace(f"{kind} ", f"{kind} q = "), (row["pushes shown"], row["learning"]),
                                textcoords="offset points", xytext=(8, 4), fontsize=8, color="#52514e")
        if len(taught):
            ax.errorbar(taught["pushes shown"], taught["learning"], yerr=taught["learning se"], color=color,
                        lw=2, marker="o", ms=7, capsize=3, label=LABEL_NAMES[labels])
            if not labelled:  # every run's teachers sit at the same shares: name them once
                for name, row in taught.iterrows():
                    ax.annotate(name.replace("random ", "p = "), (row["pushes shown"], row["learning"]),
                                textcoords="offset points", xytext=(8, -14), fontsize=8, color="#52514e")
                labelled = True
        if "no teacher" in table.index:
            ax.axhline(table.loc["no teacher", "learning"], color=color, ls="--", lw=1.2,
                       label=f"no teacher ({LABEL_NAMES[labels]})")
    ax.set_xscale("log")
    ax.set_xlabel("teacher interventions per session push (log scale)")
    ax.set_ylabel("learning (start score − end score)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))


def cross_run_table(base, other, specs=None):
    """Each teacher in both runs (same students, levels and seeds; one setting differs): learning in `base` and in
    `other`, and `other` minus `base`, paired by student (above 0: `other` learned more; count = students who did)."""
    specs = specs or [t for t in other["teachers"] if len(other["teachers"][t]) == other["config"]["students"]
                      and len(base["teachers"].get(t, {})) == base["config"]["students"]]
    s0 = start(base)[0]
    rows = []
    for spec in specs:
        a, b = end_scores(base, spec), end_scores(other, spec)
        d = [a[s][0] - b[s][0] for s in a if s in b]  # learning difference: lower end score = more learning
        m, se = mean_se(d)
        la, lb = mean_se(s0 - v[0] for v in a.values()), mean_se(s0 - v[0] for v in b.values())
        rows.append({"teacher": teacher_name(spec), "pushes shown": spend(other, spec),
                     "learning": la[0], "learning, other": lb[0], "difference": m, "difference se": se,
                     "students better": f"{sum(x > 0 for x in d)}/{len(d)}"})
    return pd.DataFrame(rows).set_index("teacher")


def plot_cross_run(ax, table, base_name, other_name, base_color=LABEL_COLORS["all"], other_color="#6250d6"):
    """Learning per teacher in two runs, side by side (a dot per run, joined per teacher)."""
    x = np.arange(len(table))
    for i, (_, row) in enumerate(table.iterrows()):
        ax.plot([i, i], [row["learning"], row["learning, other"]], color="#c9c8c2", lw=1.5, zorder=1)
    ax.scatter(x, table["learning"], color=base_color, s=60, zorder=2, label=base_name)
    ax.scatter(x, table["learning, other"], color=other_color, s=60, marker="D", zorder=2, label=other_name)
    ax.set_xticks(x, table.index)
    ax.set_ylabel("learning (start score − end score)")
    ax.grid(axis="y", color="#e6e5e1", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
