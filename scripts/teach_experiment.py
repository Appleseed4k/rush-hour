"""What the session experiments (session_experiment.py) share: where puzzles and runs live, the teacher registry
(spec "kind:p,key=value" -> a teach.py teacher, and the name its results are saved under), the move cap, and mean /
standard-error summaries.
"""
import os
import statistics

import teach

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
POOL = os.path.join(ROOT, "data", "rush_nw.txt")
OUT = os.path.join(ROOT, "models", "teach")
TEACHERS = {"random": teach.Random, "bayes": teach.Bayesian, "stuck-bayes1": teach.StuckBayes1, "bayes3": teach.Bayesian3}
# Retired kinds whose results the summary still reads, by the name their results are saved under (their names
# don't need the class). Every other retired kind is at tag pre-cleanup; runs that used one still load, without it.
RETIRED = {"flag": "flag mistakes", "correct-after": "correct after mistake", "correct-before": "correct mistakes"}


class RetiredTeacher(ValueError):
    """A spec whose kind is no longer in TEACHERS."""


def make_teacher(spec):
    """`kind`, `kind:p` or `kind:p,key=value,...` (extra keyword arguments, as numbers, to the teacher class)."""
    head, *extra = spec.split(",")
    kind, _, p = head.partition(":")
    if kind not in TEACHERS:
        raise RetiredTeacher(f"teacher kind {kind!r} is retired or unknown (kinds: {', '.join(TEACHERS)}; the "
                             f"others are at tag pre-cleanup): drop {spec} from the teachers")
    kwargs = {k: float(v) for k, v in (e.split("=") for e in extra)}
    cls = TEACHERS[kind]
    return cls(float(p) if p else getattr(cls, "DEFAULT", 1.0), **kwargs)  # a kind may set its own default


def teacher_name(spec):
    """The name a teacher's results are saved under - for the RETIRED kinds too (others raise RetiredTeacher)."""
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

