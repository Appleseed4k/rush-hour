"""Teachers that decide, move by move, when to show a student the optimal
move during a solve attempt.

Terms: "the student" is the simulated student, the policy network that plays
and learns (student.PolicyAgent, student.learning_session). "Model of the student" (or
student model) always means a teacher's own beliefs about that student, as in
the Bayesian teachers below - never the network itself.

A teacher knows each puzzle's breadth-first-search distances (board.multi_bfs,
computed the first time it has to teach on that puzzle), so the move it shows
is always optimal: one that brings the board a move closer to the goal (one
of them at random when several do). What varies between teachers is *when*
they show it - the schedule an adaptive teacher would eventually learn. The
fixed schedules are Random(p): p = 0 never teaches (the student learns only
from its own attempts), p = 1 shows every move (the student only watches).
Each move shown is one unit of the teaching budget (an intervention).

Bayesian4 ("bayes4"), the default teacher, is adaptive: it keeps a model of
the student, updated from every move the student makes, and shows the
optimal move whenever that model gives the student under theta (0.45) chance
of solving alone from here.

Other teachers tried along the way (the earlier Bayesian teachers bayes,
stuck-bayes1 and bayes3, mistake correction, flagging, walls, hints, ...) are
at tag pre-cleanup; runs that used them still load (RETIRED).

The interface, used by student.solve_intuition during a student's attempt:
- start(state, rng): an attempt at the puzzle starting from `state` begins;
  `rng` (a random.Random) is the attempt's own randomness.
- advise(state, step, proposal): the move to show on board `state`, `step`
  moves into the attempt, or None to let the student play its own move.
  `proposal()` returns the move the student would play here (computed only
  if asked, so teachers that don't look leave the student's random draws
  untouched).
- observe(state, move, next_state, taught): `move` was just played on `state`,
  leading to `next_state`; `taught` says whether the teacher showed it.
  The student isn't told anything here: it's where a teacher notices a
  mistake to correct next turn, or updates its beliefs about the student.
- end_attempt(result): the attempt is over (student.solve_intuition's result:
  boards, moves, which moves were taught, whether it solved). Where an
  adaptive teacher would learn from how the student did; a no-op here.

`counts` (a Counter) tallies whatever a teacher wants reported; rl sums it
over a round's attempts into expert_iteration's history.
Specs: "kind", "kind:p" or "kind:p,key=value,..." name a teacher (make_teacher); TEACHERS lists the
kinds.
"""
import json
import math
import os
import random
from collections import Counter

import numpy as np

from .board import car_orientations, multi_bfs, neighbors


class Teacher():
    """Base class: shows the optimal move whenever wants_to_teach says so."""
    name = "teacher"

    def __init__(self):
        self.counts = Counter()
        self.rng = random.Random()
        self._puzzle = None
        self._distances = None
        self._orientations = None

    def start(self, state, rng):
        self.rng = rng
        if state != self._puzzle:
            self._puzzle, self._distances = state, None
            self._orientations = car_orientations(state)

    def wants_to_teach(self, state, step, proposal):
        raise NotImplementedError

    def distances(self):
        """Every board's distance to the goal for the current puzzle (the
        search runs the first time it's needed)."""
        if self._distances is None:
            self._distances = multi_bfs(self._puzzle)
        return self._distances

    def optimal_move(self, state):
        """A move that brings `state` one move closer to the goal (one of them
        at random), or None if red is already out."""
        distances = self.distances()
        d = distances[state]
        options = [move for move, nxt in neighbors(state, self._orientations) if distances.get(nxt) == d - 1]
        return self.rng.choice(options) if options else None

    def advise(self, state, step, proposal=None):
        return self.optimal_move(state) if self.wants_to_teach(state, step, proposal) else None

    def observe(self, state, move, next_state, taught):
        pass

    def end_attempt(self, result):
        pass

    def __repr__(self):
        return self.name


class Random(Teacher):
    """Shows the optimal move on each turn with probability `p`, independently.
    p = 0 and p = 1 decide without drawing from the attempt's random numbers,
    so they leave the student's own sampling exactly as it would be with no
    teacher deciding at all."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.name = f"random {p}"

    def wants_to_teach(self, state, step, proposal=None):
        if self.p <= 0:
            return False
        if self.p >= 1:
            return True
        return self.rng.random() < self.p




def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class SkillFilter:
    """A normal belief over the skill logits per distance bin, carried from puzzle to puzzle: a hierarchical prior
    (skill ~ N(level, spread^2), level ~ N(prior_mean, prior_sd^2), the level integrated out), each puzzle folded in
    by a Laplace approximation with its difficulty z averaged over the grid, then `q` x I + `q_shared` x 11^T added to
    the covariance (independent and shared drift: the student may have changed). With `trend_sd` > 0 the belief
    also covers a learning rate v (logits per puzzle, every distance alike): skill moves by v between puzzles, and
    v by noise of variance `trend_q` - a Kalman filter with a local linear trend. Used by Bayesian4, and tested
    offline in experiments/skill_tracking.py."""

    B = 11  # distance bins: 1, 2, ..., 10, 11 or more
    Z = np.linspace(-3, 3, 13)  # quadrature points for a puzzle's difficulty, in units of tau
    LOG_PZ = -0.5 * Z ** 2 - np.log(np.exp(-0.5 * Z ** 2).sum())

    def __init__(self, tau=0.5, q=0.0, q_shared=0.0, prior_mean=0.0, prior_sd=1.5, spread=1.0, trend_sd=0.0,
                 trend_q=0.0):
        B = self.B
        self.tau = tau
        self.trend = trend_sd > 0
        d = B + self.trend
        self.x = np.zeros(d)  # mean: skill logits, then v
        self.x[:B] = prior_mean
        self.P = np.zeros((d, d))  # covariance
        self.P[:B, :B] = spread ** 2 * np.eye(B) + prior_sd ** 2 * np.ones((B, B))
        self.Q = np.zeros((d, d))  # added between puzzles
        self.Q[:B, :B] = q * np.eye(B) + q_shared * np.ones((B, B))
        if self.trend:
            self.P[B, B], self.Q[B, B] = trend_sd ** 2, trend_q
        self.F = np.eye(d)  # skill += v between puzzles
        if self.trend:
            self.F[:B, B] = 1.0

    @property
    def mu(self):
        return self.x[:self.B]

    @property
    def cov(self):
        return self.P[:self.B, :self.B]

    def _objective(self, s, o, n, mu, prec):
        """The log of prior x this puzzle's likelihood at skill `s` (z averaged out), and the pieces of its
        derivatives."""
        logit = s[None, :] + self.tau * self.Z[:, None]  # (z, bin)
        llz = (o * _log_sigmoid(logit) + (n - o) * _log_sigmoid(-logit)).sum(1) + self.LOG_PZ
        top = llz.max()
        lse = top + math.log(np.exp(llz - top).sum())
        return lse - 0.5 * (s - mu) @ prec @ (s - mu), logit, np.exp(llz - lse)

    @staticmethod
    def _curvature(logit, r, o, n, prec):
        """Minus the Hessian of the objective, or its Fisher part (always positive definite) where that isn't."""
        p = 1 / (1 + np.exp(-logit))
        g_z = o - n * p
        gbar = r @ g_z
        fisher = np.diag(r @ (n * p * (1 - p))) + prec
        hess = fisher - (g_z * r[:, None]).T @ g_z + np.outer(gbar, gbar)
        try:
            np.linalg.cholesky(hess)
        except np.linalg.LinAlgError:
            hess = fisher
        return hess, gbar

    def update(self, o, n):
        """Folds in one puzzle's optimal moves `o` out of `n` per bin, then steps to the next puzzle; returns the
        log marginal likelihood of the moves (Laplace), what a bank of filters weighs its members by."""
        B = self.B
        log_ml = 0.0
        if n.sum() > 0:
            mu, cov = self.mu.copy(), self.cov.copy()
            prec = np.linalg.inv(cov)
            s = mu.copy()
            value, logit, r = self._objective(s, o, n, mu, prec)
            for _ in range(100):
                hess, gbar = self._curvature(logit, r, o, n, prec)
                step = np.linalg.solve(hess, gbar - prec @ (s - mu))
                for _ in range(30):  # backtrack until the objective doesn't fall
                    new = self._objective(s + step, o, n, mu, prec)
                    if new[0] >= value - 1e-12:
                        break
                    step = step / 2
                s = s + step
                value, logit, r = new
                if np.abs(step).max() < 1e-7:
                    break
            hess, _ = self._curvature(logit, r, o, n, prec)
            post_cov = np.linalg.inv(hess)
            log_ml = value - 0.5 * np.linalg.slogdet(cov)[1] - 0.5 * np.linalg.slogdet(hess)[1]
            # The moves bear on skill only; carry the change to the rest of the belief (v) by linear regression
            # on skill - exact for a normal belief once skill's own posterior is normal.
            gain = self.P[:, :B] @ prec
            self.x = self.x + gain @ (s - mu)
            self.P = self.P - gain @ (cov - post_cov) @ gain.T
            self.P = 0.5 * (self.P + self.P.T)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return log_ml


class Bayesian4(Teacher):
    """The Bayesian teacher ("bayes4"), in the spirit of hierarchical Bayesian models of teaching (Chen et al.,
    2024) and of when parents take over (Shachnai et al., 2025): it holds beliefs about the student, updates them
    from what the student does, and on every turn shows the optimal move if its model gives the student less than
    `theta` (0.45) chance of solving the puzzle untaught from here, within the moves left before the cap; otherwise
    the student plays. So it steps in only as far as it thinks the student needs, hands the puzzle back, and steps
    back in if the student slides. It never looks at the student's intended move.

    Its model of the student:
    - skill: the chance that a move of its own is optimal, as a logit per distance to the goal (1, 2, ..., 10, 11
      or more - near the goal is easier, and that is where students improve first). A normal belief carried from
      puzzle to puzzle by a filter (SkillFilter: each skill drawn around the student's overall level, sd `spread`,
      the level around `prior_mean`, sd `prior_sd`, integrated out); each finished puzzle's own moves are folded
      in by a Laplace approximation. Nothing else about skill is stored. `q` > 0 widens the belief between puzzles
      (the student may have changed); 0 (no drift) predicted best offline;
    - each puzzle's difficulty for this student: a shift of all the logits, normal with sd tau. tau is learned by
      a bank of filters, one per tau in TAUS, each weighed by how well it has predicted the student's moves (log
      marginal likelihood, forgetting by `forget` per puzzle) and by a log-normal prior around `tau` (sd
      `tau_sd`). Within a puzzle the belief about the shift is updated after every move the student makes itself;
    - what a mistake looks like: the share of non-optimal moves that go a move further from the goal (rather than
      sideways), per distance (prior mean `worse`, worth `worse_strength` moves; older puzzles count less, by
      `decay` per puzzle).
    The plan and the within-puzzle belief use the best-weighted filter: its tau and its mean skill. Moves the
    teacher showed say nothing about skill.

    The chance of solving alone (planned at the start of each puzzle): the student's progress as a walk in
    distance to the goal - each move of its own one closer with its skill there, else further or sideways - that
    ends when red is out or at the move cap (`step_factor` x the optimal length, at least `min_steps`, as the
    sessions'), by backward induction over (moves made, difficulty, distance); on each turn averaged over the
    current belief about the puzzle's difficulty.

    History: the last of a line of Bayesian teachers (p4-p5; the older ones, "bayes", "stuck-bayes1" and "bayes3",
    are at tag pre-cleanup). It is bayes3 - the simplest of the stuck-Bayes ablations, each part it keeps needed
    (difficulty updated within the puzzle: without, -0.12 learning at matched spend; backward mistakes: without,
    -0.15) - with the refit of skill from the stored history after every puzzle replaced by the filter. Offline
    (experiments/skill_tracking.py, 8 unhelped students x 2160 puzzles at agency 0): log loss per move 0.6588, as
    bayes3's refit (0.6587), better calibrated, about 10x faster per puzzle.

    Meant for sessions, one teacher per student, kept across puzzles. counts, per puzzle: "shown";
    "predicted_solve" (at the start, the chance it gave the student of solving untaught); "skill_at_start" (its
    belief of the chance the student's next move is optimal there, on a typical puzzle); "puzzle_shift" (its
    belief, at the end, of how much easier - in logits - the puzzle was for the student than a typical one)."""

    KIND = "bayes4"
    DEFAULT = 0.45  # theta when the spec gives none ("bayes4")
    BINS, Z, LOG_PZ = SkillFilter.B, SkillFilter.Z, SkillFilter.LOG_PZ
    TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2)

    def __init__(self, theta=DEFAULT, forget=0.99, q=0.0, decay=0.99, step_factor=10.0, min_steps=50,
                 prior_mean=0.0, prior_sd=1.5, spread=1.0, tau=0.5, tau_sd=0.5, worse=0.35, worse_strength=10.0):
        super().__init__()
        self.theta, self.forget, self.q = float(theta), float(forget), float(q)
        self.decay, self.step_factor, self.min_steps = decay, step_factor, min_steps
        self.worse, self.worse_strength = worse, worse_strength
        self.worse_moves, self.mistakes = np.zeros(self.BINS), np.zeros(self.BINS)
        self.filters = [SkillFilter(tau=t, q=self.q, prior_mean=prior_mean, prior_sd=prior_sd, spread=spread)
                        for t in self.TAUS]
        self.tau_prior = -(np.log(self.TAUS) - math.log(tau)) ** 2 / (2 * tau_sd ** 2)
        self.tau_loglik = np.zeros(len(self.TAUS))
        self._pick()
        self._pending, self._finished = False, False
        self.name = f"{self.KIND} {self.theta:g}" + (f" forget{self.forget:g}" if self.forget != 0.99 else "") \
            + (f" q{self.q:g}" if self.q else "")

    # --- the teaching interface ---

    def start(self, state, rng):
        new = self._finished or state != self._puzzle
        super().start(state, rng)
        if new:
            self._end_puzzle()
            self._begin_puzzle(state)

    def solve_alone(self, state, step):
        """The model's chance of solving untaught from `state`, `step` moves into this attempt."""
        k, m = self.distances()[state], self._offset + step
        if k == 0:
            return 1.0
        if m >= self._cap:
            return 0.0
        return float(_softmax(self._shift) @ self._alone[m, :, min(k, self._K)])

    def wants_to_teach(self, state, step, proposal=None):
        return self.distances()[state] != 0 and self.solve_alone(state, step) < self.theta

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step):
            return None
        self.counts["shown"] += 1
        return self.optimal_move(state)

    def observe(self, state, move, next_state, taught):
        self._moves += 1
        if taught:
            return
        distances = self.distances()
        k, after = distances[state], distances[next_state]
        b = min(k, self.BINS) - 1
        good = after == k - 1
        self._optimal[b] += good
        self._own[b] += 1
        if not good:
            self._mistakes[b] += 1
            self._worse[b] += after == k + 1
        logit = self.skill[b] + math.exp(self.log_tau) * self.Z
        self._shift += _log_sigmoid(logit) if good else _log_sigmoid(-logit)

    def end_attempt(self, result):
        self._offset += len(result["moves"])
        self.counts["puzzle_shift"] = float(_softmax(self._shift) @ (math.exp(self.log_tau) * self.Z))
        if result["solved"] or result.get("surrendered"):
            self._end_puzzle()
            self._finished = True

    # --- beliefs ---

    def _begin_puzzle(self, state):
        distances = self.distances()
        self._d = distances[state]
        self._cap = max(self.min_steps, int(round(self.step_factor * self._d)))
        self._K = max(distances.values())
        self._offset = self._moves = 0
        self._optimal, self._own = np.zeros(self.BINS), np.zeros(self.BINS)
        self._worse, self._mistakes = np.zeros(self.BINS), np.zeros(self.BINS)
        self._shift = self.LOG_PZ.copy()
        self._pending, self._finished = True, False
        self._plan()
        self.counts["skill_at_start"] = float(1 / (1 + math.exp(-self.skill[min(self._d, self.BINS) - 1])))

    def _end_puzzle(self):
        """Folds the finished puzzle's evidence into the beliefs about the student."""
        if not self._pending:
            return
        self._pending = False
        self.worse_moves = self.decay * self.worse_moves + self._worse
        self.mistakes = self.decay * self.mistakes + self._mistakes
        ll = np.array([f.update(self._optimal, self._own) for f in self.filters])
        self.tau_loglik = self.forget * self.tau_loglik + ll
        self._pick()

    def _pick(self):
        best = self.filters[int(np.argmax(self.tau_prior + self.tau_loglik))]
        self.skill, self.log_tau = best.mu.copy(), math.log(best.tau)

    # --- deciding ---

    def _plan(self):
        """Backward induction: self._alone[m, difficulty, k] = the chance of solving untaught from a board k from
        the goal after m moves on the puzzle."""
        K, cap = self._K, self._cap
        k = np.arange(K + 1)
        b = np.clip(k, 1, self.BINS) - 1
        p = 1 / (1 + np.exp(-(self.skill[b][None, :] + math.exp(self.log_tau) * self.Z[:, None])))
        worse = ((self.worse_strength * self.worse + self.worse_moves) / (self.worse_strength + self.mistakes))[b]
        down, up, out = np.maximum(k - 1, 0), np.minimum(k + 1, K), k == 0
        alone = np.broadcast_to(out.astype(float), p.shape)  # at the cap: solved only if red is out
        self._alone = np.empty((cap, len(self.Z), K + 1))
        for m in range(cap - 1, -1, -1):
            after = np.where(out, 1.0, alone)
            alone = np.where(out, 1.0, p * after[:, down] + (1 - p) * (worse * after[:, up] + (1 - worse) * after))
            self._alone[m] = alone
        self.counts["predicted_solve"] = float(np.exp(self.LOG_PZ) @ alone[:, self._d])


# --- the registry -------------------------------------------------------------------------------

TEACHERS = {"random": Random, "bayes4": Bayesian4}
DEFAULT = "bayes4"  # the default teacher's kind
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "retired_teachers.json")) as _f:
    # Every retired spec in a saved run's config (models/sessions and the archive), with the name its results
    # are saved under, computed by its own code at tag pre-cleanup: runs that used it still load.
    RETIRED = json.load(_f)


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
    """The name a teacher's results are saved under - for retired specs of saved runs too (RETIRED); any other
    retired spec raises RetiredTeacher."""
    if spec in RETIRED:
        return RETIRED[spec]
    return make_teacher(spec).name
