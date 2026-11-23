"""Offline test of how a teacher should carry its estimate of the student's skill from one puzzle to the next:
refit it from the stored history after every puzzle (bayes3), or update a running summary (a filter)?

Data: the trace run (p1.4_dilemma, "Calibrating the Bayesian teacher's model of the student"): 8 unhelped simulated
students x 2160 puzzles at agency 0, each puzzle's distance-to-goal path. Every model sees each student's paths in
order and, before every move, predicts the chance that the move is optimal (one closer to the goal). They differ
only in how skill is carried across puzzles:

- refit: bayes3 itself (the default teacher before bayes4, retired: run at tag pre-cleanup). Per-puzzle counts of
  the last 300 puzzles are stored, each weighted by decay^age, and skill per distance, the overall level and tau are
  refitted (MAP) after every puzzle. Its rows are in the original sweep's results, next to the trace run
  (models/sessions/teach_shortcut_agency0/trace/skill_tracking.json), with every model below.
- filter: a normal belief over the 11 skill logits, a mean and a covariance, with bayes3's hierarchical prior
  (the overall level integrated out). After each puzzle its moves are folded in by a Laplace approximation, z
  averaged over bayes3's grid, and the covariance is widened by `q` (the student may have changed since). Nothing
  else is stored.
  - filter: tau fixed, predicts from the mean, as bayes3 does;
  - filter+var: also averages over its uncertainty about skill;
  - filter+tau: tau learned as well (bayes4's model of the student, teachers.Bayesian4) - one filter per tau on a grid, each weighed by how well it has predicted
    (Laplace marginal likelihood), the weights forgetting by `forget` per puzzle.
  - `trend_sd` > 0 (either filter): the student is expected to improve. A learning rate v (logits per puzzle,
    shared by every distance; prior N(0, trend_sd^2)) joins the belief, skill moves by v between puzzles, and v
    itself may change by `trend_q` (variance) per puzzle. The belief over v is learned from how skill moves.

Score: log loss per move (lower is better), and calibration (mean predicted minus actual share of optimal moves),
by quarter of the session (540 puzzles each). Within a puzzle every model updates its belief about the puzzle's
difficulty z move by move, as bayes3 does, so a puzzle's summed log loss is minus the log probability of its whole
path under the belief held before it: a proper score of that belief.

    python experiments/skill_tracking.py        # the default sweep, saved to models/summary/skill_tracking.json
"""
import argparse
import json
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rushhour import paths, teachers as tr  # noqa: E402

TRACE = Path(paths.SESSIONS) / "teach_shortcut_agency0" / "trace" / "sessions.json"
OUT = Path(paths.MODELS) / "summary" / "skill_tracking.json"  # the original sweep's is next to TRACE (see above)
SkillFilter = tr.SkillFilter
B, Z, LOG_PZ = SkillFilter.B, SkillFilter.Z, SkillFilter.LOG_PZ
QUARTERS = 4


def load(path=TRACE):
    """Per student: its puzzles in order, each (start distance, farthest distance, bin of every move, whether it
    was optimal, whether it went backward)."""
    with open(path) as f:
        sessions = json.load(f)
    out = {}
    for s, records in sessions.items():
        puzzles = []
        for r in records:
            p = np.array(r["teacher_counts"].get("path", [r["distance"]]))
            before, after = p[:-1], p[1:]
            puzzles.append((int(p[0]), int(r["teacher_counts"].get("max_distance", p.max())),
                            np.minimum(before, B) - 1, after == before - 1, after == before + 1))
        out[int(s)] = puzzles
    return out


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(components, b, good):
    """The chance each move of a puzzle is optimal, before it is played. `components`: the belief about the
    student, as (log weight, skill logits, tau, variances or None) - one for a single model, one per tau for a bank;
    with variances, skill's uncertainty is averaged over (probit approximation). The puzzle's difficulty is tau * z,
    the belief over (component, z) updated by the moves before each one, as in teachers.Bayesian4.observe."""
    if len(b) == 0:
        return np.zeros(0)
    logit = []
    for _, mu, tau, var in components:
        x = mu[b][:, None] + tau * Z[None, :]  # (moves, z)
        if var is not None:
            x = x / np.sqrt(1 + math.pi * var[b][:, None] / 8)
        logit.append(x)
    logit = np.stack(logit, 1)  # (moves, component, z)
    log_w = np.array([c[0] for c in components])
    prior = (log_w - np.logaddexp.reduce(log_w))[:, None] + LOG_PZ[None, :]
    ll = np.where(good[:, None, None], tr._log_sigmoid(logit), tr._log_sigmoid(-logit))
    before = prior + np.concatenate([np.zeros((1,) + prior.shape), np.cumsum(ll, 0)[:-1]])
    post = np.exp(before - before.max((1, 2), keepdims=True))
    post /= post.sum((1, 2), keepdims=True)
    return (post * _sigmoid(logit)).sum((1, 2))


class _Identity(dict):
    """Stands in for a puzzle's BFS distances when states are just distances."""

    def __init__(self, farthest):
        super().__init__((k, k) for k in range(farthest + 1))


def run_bayes4(puzzles, **kwargs):
    """The default teacher's beliefs (teachers.Bayesian4: the filter+tau bank, deciding from its best tau)."""
    yield from _drive(tr.Bayesian4(**kwargs), puzzles)


def _drive(t, puzzles):
    """A Bayesian teacher's beliefs, fed each puzzle's moves offline; yields the belief before each puzzle."""
    t._plan = lambda: None  # deciding isn't tested here, only the beliefs
    for start, farthest, b, good, worse in puzzles:
        t._puzzle, t._distances = object(), _Identity(farthest)
        t._begin_puzzle(start)
        yield [(0.0, t.skill.copy(), math.exp(t.log_tau), None)]
        np.add.at(t._optimal, b, good)
        np.add.at(t._own, b, 1)
        np.add.at(t._mistakes, b, ~good)
        np.add.at(t._worse, b, worse)
        t._moves += len(b)
        t._end_puzzle()


def _counts(b, good):
    return np.bincount(b, good, minlength=B), np.bincount(b, minlength=B).astype(float)


def run_filter(puzzles, uncertainty=False, **kwargs):
    f = SkillFilter(**kwargs)
    for _, _, b, good, _ in puzzles:
        yield [(0.0, f.mu.copy(), f.tau, np.diag(f.cov).copy() if uncertainty else None)]
        f.update(*_counts(b, good))


TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2)


def run_bank(puzzles, taus=TAUS, forget=1.0, **kwargs):
    """tau learned too: one filter per tau on a grid, each weighed by how well it has predicted the puzzles so far
    (log weights times `forget` per puzzle, so tau may change over the session). Within a puzzle the belief is over
    (tau, z) jointly."""
    filters = [SkillFilter(tau=t, **kwargs) for t in taus]
    log_w = np.zeros(len(taus))
    for _, _, b, good, _ in puzzles:
        yield [(w, f.mu.copy(), f.tau, None) for w, f in zip(log_w, filters)]
        o, n = _counts(b, good)
        log_w = forget * log_w + np.array([f.update(o, n) for f in filters])


def score(puzzles, beliefs):
    """Per quarter of the session: summed log loss, moves, summed predicted, summed optimal."""
    n = len(puzzles)
    out = np.zeros((QUARTERS, 4))
    for i, ((_, _, b, good, _), components) in enumerate(zip(puzzles, beliefs)):
        p = np.clip(predict(components, b, good), 1e-9, 1 - 1e-9)
        out[i * QUARTERS // n] += [-(np.log(p[good]).sum() + np.log(1 - p[~good]).sum()), len(p), p.sum(), good.sum()]
    return out


MODELS = {"bayes4": run_bayes4, "filter": run_filter, "filter+var": lambda p, **k: run_filter(p, True, **k),
          "filter+tau": run_bank}


def job(args):
    model, kwargs, student = args
    puzzles = load()[student]
    t = time.perf_counter()
    table = score(puzzles, MODELS[model](puzzles, **kwargs))
    return model, kwargs, student, table.tolist(), time.perf_counter() - t


TREND = {"trend_sd": 0.001}  # expected learning: about 0.3 logits over the 2160 puzzles, faster early


def configs(quick=False):
    if quick:
        return [("bayes4", {}), ("filter", {"q": 0.003}), ("filter+var", {"q": 0.003})]
    out = [("bayes4", {})]
    for q in (0.0, 0.001, 0.003):
        out.append(("filter", {"tau": 0.5, "q": q, "q_shared": 0.0}))
        out.append(("filter+tau", {"forget": 0.99, "q": q, "q_shared": 0.0}))
    out.append(("filter+var", {"tau": 0.5, "q": 0.0, "q_shared": 0.0}))
    for trend_q in (0.0, 1e-8, 1e-7):
        out.append(("filter", {"tau": 0.5, **TREND, "trend_q": trend_q}))
        out.append(("filter+tau", {"forget": 0.99, **TREND, "trend_q": trend_q}))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="three models only")
    parser.add_argument("--students", type=int, default=8)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--out", default=str(OUT))
    a = parser.parse_args()
    jobs = [(m, k, s) for m, k in configs(a.quick) for s in range(a.students)]
    rows = []
    with Pool(a.processes) as pool:
        for i, (model, kwargs, student, table, seconds) in enumerate(pool.imap_unordered(job, jobs), 1):
            rows.append({"model": model, "kwargs": kwargs, "student": student, "quarters": table, "seconds": seconds})
            t = np.array(table)
            print(f"[{i}/{len(jobs)}] {model} {kwargs} student {student}: "
                  f"log loss {t[:, 0].sum() / t[:, 1].sum():.4f} ({seconds:.0f}s)", flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f)
    print("saved", a.out)


if __name__ == "__main__":
    main()
