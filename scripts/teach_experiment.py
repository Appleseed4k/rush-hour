"""What the session experiments (session_experiment.py) share with the earlier offline-teaching experiments:
where puzzles and runs live, the teacher registry (spec "kind:p,key=value" -> a teach.py teacher, and the name
its results are saved under), the move cap, and mean / standard-error summaries.

The offline-teaching runner itself (expert iteration with a teacher over rounds, tested after each round;
results in models/teach/) is archived in scripts/old/p2_teach/teach_experiment_rounds.py.
"""
import os
import statistics

import teach

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
POOL = os.path.join(ROOT, "data", "rush_nw.txt")
OUT = os.path.join(ROOT, "models", "teach")
TEACHERS = {"random": teach.Random,
            "bayes": teach.Bayesian, "bayes-learn": teach.LearningBayesian,
            "bayes2": teach.Bayesian2, "bayes-learn2": teach.LearningBayesian2, "trace": teach.Trace,
            "wall": teach.Wall, "stuck": teach.Stuck, "stuck-bayes": teach.StuckBayes, "stuck-bayes1": teach.StuckBayes1,
            "hint": teach.Hint, "hint-bayes": teach.HintBayes, "bayes3": teach.Bayesian3}
# Kinds no longer available here, by the name saved runs keep their results under. Their code is in
# scripts/old/p2_teach/ (correct-after, correct-before) or, for flag, in future_old/ (not in this repo).
RETIRED = {"flag": "flag mistakes", "correct-after": "correct after mistake", "correct-before": "correct mistakes"}


def make_teacher(spec):
    """`kind`, `kind:p` or `kind:p,key=value,...` (extra keyword arguments, as numbers, to the teacher class)."""
    head, *extra = spec.split(",")
    kind, _, p = head.partition(":")
    if kind in RETIRED:
        raise SystemExit(f"the {RETIRED[kind]} teacher is retired (see scripts/old/ or future_old/): drop {spec} from the teachers")
    kwargs = {k: float(v) for k, v in (e.split("=") for e in extra)}
    cls = TEACHERS[kind]
    return cls(float(p) if p else getattr(cls, "DEFAULT", 1.0), **kwargs)  # a kind may set its own default


def teacher_name(spec):
    """The name a teacher's results are saved under - for retired kinds too."""
    kind, _, p = spec.partition(":")
    if kind in RETIRED:
        q = float(p) if p else 1.0
        return RETIRED[kind] if q >= 1 else f"{RETIRED[kind]} {q}"
    return make_teacher(spec).name


def bins(specs):
    return [tuple(int(x) for x in spec.split("-")) for spec in specs]


def step_cap(distance, config):
    if config["max_steps"] is not None:
        return config["max_steps"]
    return max(config["min_steps"], int(round(config["step_factor"] * distance)))


def mean_se(values):
    """(mean, standard error) of the non-nan values."""
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan")
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")
    return statistics.mean(values), se


def summarize(runs, n_runs, distance_bins=()):
    """Solve rate and moves / optimal as (mean, standard error across
    puzzles), overall and per (lo, hi) distance bin, from test attempts that
    come puzzle by puzzle, `n_runs` each - averaged per puzzle first."""
    groups = [runs[i:i + n_runs] for i in range(0, len(runs), n_runs)]
    solve = [(g[0]["distance"], statistics.mean(float(r["solved"]) for r in g)) for g in groups]
    ratio = [statistics.mean(r["moves"] / r["distance"] for r in g if r["solved"])
             for g in groups if any(r["solved"] for r in g)]
    out = {"solve rate": mean_se([v for _, v in solve]), "moves / optimal": mean_se(ratio)}
    for lo, hi in distance_bins:
        out[(lo, hi)] = mean_se([v for d, v in solve if lo <= d <= hi])
    return out
