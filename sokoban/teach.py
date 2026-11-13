"""Teachers that decide, push by push, when to show a student the optimal push
during an attempt at a level: the counterpart of scripts/teach.py.

A teacher knows the level's oracle distances (label.StateSpace, computed the
first time it has to teach on that level), so the push it shows is always
optimal: one that brings the board a push closer to solved (one of them at
random when several do). What varies between teachers is *when* they show
it. Random(p) shows it on each turn with probability p: p = 0 never teaches,
p = 1 shows every push (the student only watches). Each push shown is one
unit of the teaching budget (an intervention).

Unlike Rush Hour, a board can be past saving: on a deadlocked board (oracle
distance -1, or a box on a dead cell, which the oracle doesn't enumerate) no
push is optimal, so the teacher shows nothing and the student plays on until
it restarts. A teacher never pushes the board into a deadlock.

The interface matches scripts/teach.py, with a Level in place of a Rush Hour
start board. The student is and_or.play(level, teacher=...):
- start(level, rng): an attempt at `level` begins; `rng` (a random.Random) is
  the attempt's own randomness, shared with the student as in Rush Hour.
- advise(state, step, proposal): the push to show on board `state`, `step`
  pushes into the attempt (restarts don't reset it), or None to let the
  student play its own. `proposal()` returns the push the student would play
  here, planning a new pass if it has none under way (computed only if asked,
  so teachers that don't look leave the student's random draws untouched).
- veto(state, push): True to take back the push the student is about to
  play on its own, so it chooses again without it (Undo; only
  student.solve_intuition asks, and_or.play doesn't).
- force_restart(state, stretch): True to send the student back to the
  start now, `stretch` pushes into its current stretch (Restart; only
  student.solve_intuition asks).
- observe(state, push, next_state, taught): `push` was just played on
  `state`; `taught` says whether the teacher showed it.
- restart(state): the student gave up on a deadlocked or hopeless board and
  is back at the start `state` (no Rush Hour counterpart).
- end_attempt(result): the attempt is over (play()'s result).

`counts` (a Counter) tallies what a teacher wants reported: every teacher
counts "shown" (pushes shown) and "no_move" (turns it wanted to teach on a
board with no optimal push).
"""
import functools
import math
import random
from collections import Counter, deque

import numpy as np

from label import StateSpace
from level import Level


@functools.lru_cache(maxsize=4)
def oracle(board):
    """The oracle for a level, by its board line: shared by every teacher in
    this process, so the search runs once per level."""
    return StateSpace(Level.from_line(board))


class Teacher:
    """Base class: shows an optimal push whenever wants_to_teach says so."""
    name = "teacher"

    def __init__(self):
        self.counts = Counter()
        self.rng = random.Random()
        self.level = None

    def start(self, level, rng):
        self.rng = rng
        self.level = level

    def wants_to_teach(self, state, step, proposal):
        raise NotImplementedError

    def space(self):
        """The level's oracle (the search runs the first time it's needed)."""
        return oracle(self.level.to_line())

    def distance(self, state):
        """Optimal pushes from `state` to solved, -1 for a deadlock."""
        d = self.space().distance(state)
        return -1 if d is None else d  # not enumerated: a box on a dead cell

    def optimal_pushes(self, state):
        """Every push that brings `state` one push closer to solved (none on
        a solved or deadlocked board)."""
        d = self.distance(state)
        if d <= 0:
            return []
        return [push for push in self.level.pushes(state)
                if self.distance(self.level.apply(state, push)) == d - 1]

    def optimal_push(self, state):
        """One optimal push at random, or None if there is none."""
        options = self.optimal_pushes(state)
        return self.rng.choice(options) if options else None

    def is_optimal(self, state, push):
        d = self.distance(state)
        return d > 0 and self.distance(self.level.apply(state, push)) == d - 1

    def mistake_kind(self, state, next_state):
        """None if the push from `state` to `next_state` was optimal, else
        "deadlock" (the level can no longer be solved), "worse" (further
        from solved) or "sideways" (the same distance). Pushes from a board
        that was already deadlocked are "deadlock" too."""
        before, after = self.distance(state), self.distance(next_state)
        if after == -1:
            return "deadlock"
        if after == before - 1:
            return None
        return "worse" if after > before else "sideways"

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step, proposal):
            return None
        push = self.optimal_push(state)
        self.counts["shown" if push is not None else "no_move"] += 1
        return push

    def veto(self, state, push):
        return False

    def force_restart(self, state, stretch):
        return False

    def observe(self, state, push, next_state, taught):
        pass

    def restart(self, state):
        pass

    def end_attempt(self, result):
        pass

    def __repr__(self):
        return self.name


class Random(Teacher):
    """Shows the optimal push on each turn with probability `p`, independently.
    p = 0 and p = 1 decide without drawing from the attempt's random numbers,
    so p = 0 leaves the student's own sampling exactly as it is with no
    teacher at all."""

    def __init__(self, p):
        super().__init__()
        self.p = p
        self.name = f"random {p:g}"

    def wants_to_teach(self, state, step, proposal=None):
        if self.p <= 0:
            return False
        if self.p >= 1:
            return True
        return self.rng.random() < self.p


class Guard(Teacher):
    """Steps in only when the student is one push from a deadlock: when the
    push it is about to play would leave the level unsolvable (oracle distance
    -1, a box on a dead cell included), it shows the optimal push instead,
    moving the board toward the goal rather than into the dead end. Anywhere
    else, the student plays alone. Like Rush Hour's CorrectMistakes, it needs
    to see the push the student intends (`proposal`), an idealization.

    It acts on each push it catches with probability `q` (q = 1: every one,
    without drawing from the attempt's random numbers). A board that is
    already deadlocked has no optimal push, so it can only guard boards that
    are still solvable, and with q = 1 the student never reaches one that
    isn't. counts: "caught" (deadlocking pushes it saw coming), "shown"."""

    def __init__(self, q=1.0):
        super().__init__()
        self.q = q
        self.name = f"guard {q:g}"

    def advise(self, state, step, proposal=None):
        if proposal is None or self.distance(state) <= 0:
            return None
        push = proposal()
        if push is None or self.distance(self.level.apply(state, push)) != -1:
            return None
        self.counts["caught"] += 1
        if self.q < 1 and self.rng.random() >= self.q:
            return None
        shown = self.optimal_push(state)
        self.counts["shown" if shown is not None else "no_move"] += 1
        return shown


class Undo(Teacher):
    """Like Guard, it steps in only when the push the student is about to play
    would leave the level unsolvable, but it shows nothing: it takes the push
    back and the student chooses again from its own policy, without that push
    (and without any other it took back on this board). Every push the
    student finally plays is its own choice, so a solved attempt's labels
    describe the student's own play, not routes the teacher steered. This is
    the nearest Sokoban gets to Rush Hour, where a bad move can be undone.

    It acts with probability `q` per deadlocking push (q = 1: always, without
    drawing from the attempt's random numbers). A push taken back doesn't
    count toward the attempt's push budget. counts: "caught", "undone".
    Only student.solve_intuition asks for vetoes."""

    def __init__(self, q=1.0):
        super().__init__()
        self.q = q
        self.name = f"undo {q:g}"

    def advise(self, state, step, proposal=None):
        return None

    def veto(self, state, push):
        if self.q <= 0:  # never acts: leave the attempt's random numbers as they are with no teacher
            return False
        if self.distance(state) <= 0 or self.distance(self.level.apply(state, push)) != -1:
            return False
        self.counts["caught"] += 1
        if self.q < 1 and self.rng.random() >= self.q:
            return False
        self.counts["undone"] += 1
        return True


class Restart(Teacher):
    """Shows nothing and takes nothing back: it only sends the student back to
    the start. It does so on a board the oracle says is deadlocked (where the
    student's quick test hasn't noticed yet, so it would play on), and - when
    `k` is finite - once the stretch can no longer finish within k times the
    optimal pushes: stretch pushes + distance to solved > k * optimal. Without
    a teacher a stretch ends only at a deadlock the quick test catches, and
    only short winning stretches get labelled; this keeps every labelled win
    within k of optimal and stops pushes being wasted on lost boards.

    With `undo`, it also takes back every deadlocking push, as Undo(1), so the
    student never deadlocks and restarts only for straying. counts: "dead",
    "stray" (restarts forced for each). Only student.solve_intuition asks."""

    def __init__(self, k=float("inf"), undo=False):
        super().__init__()
        self.k, self.undo = k, undo
        self.name = ("undo+" if undo else "") + ("restart dead" if k == float("inf") else f"restart {k:g}")
        self.optimal = None

    def start(self, level, rng):
        super().start(level, rng)
        self.optimal = None

    def advise(self, state, step, proposal=None):
        return None

    def veto(self, state, push):
        if not self.undo or self.distance(state) <= 0 or self.distance(self.level.apply(state, push)) != -1:
            return False
        self.counts["undone"] += 1
        return True

    def force_restart(self, state, stretch):
        d = self.distance(state)
        if d == -1:
            self.counts["dead"] += 1
            return True
        if self.k == float("inf"):
            return False
        if self.optimal is None:
            self.optimal = self.distance(self.level.start)
        if d > 0 and stretch + d > self.k * self.optimal:
            self.counts["stray"] += 1
            return True
        return False


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Bayes(Teacher):
    """The Sokoban port of Rush Hour's simplest Bayesian teacher (scripts/teach.py
    Bayesian3): a model of the student, fitted from what it does, forecasts its
    chance of solving the level untaught from here within its push cap, and the
    teacher acts only while that chance is below `theta`. With `act` "undo" it
    acts as Undo does - taking back a push that would deadlock a solvable board,
    so the student chooses again - and with "show" as Random does, showing the
    optimal push. It never looks at the push the student intends except to see
    whether it deadlocks (undo needs that, as Undo does). With "restart" it
    shows nothing and takes nothing back: it sends the student back to the
    start whenever its plan says a fresh attempt now gives a better chance of
    solving than playing on, by more than `theta` (see _plan_restarts) - in
    this model, on a deadlocked board the quick test missed, or on one far
    enough beyond the start's distance, given the pushes left. With "undo-value"
    (Rush Hour's original Bayesian rule) it takes a deadlocking push back only
    when its plan, which counts on its own later undos, says doing so raises
    the chance of solving by more than a cost per undo (see _plan_undos); the
    cost adjusts after every level so that about `theta` of the pushes are
    taken back (here theta is a spending target, not a threshold).

    The model (refitted after every level; older levels count less, by `decay`
    per level, the last `window` kept):
    - skill: the chance that a push of the student's own is optimal, as a logit
      per distance to solved (1, 2, ..., 10, 11 or more), drawn around an
      overall level (sd `spread`) with a population prior (`prior_mean`,
      `prior_sd`); fitted MAP as in Rush Hour, each level's difficulty
      integrated out;
    - each level's difficulty for this student: a logit shift, normal with sd
      tau (fitted, prior around `tau`), updated within the level after every
      push of its own;
    - what a mistake does, per distance: deadlock the level, move it further
      from solved, or leave it as far (Dirichlet prior with means `dead`,
      `worse` and the rest, worth `mistake_strength` pushes); how far a worse
      push moves it (+1, +2, +3 or more; pushes can't be pulled back, so it can
      jump), pooled over distances;
    - how fast it notices a deadlock: its quick test fires at once on a board
      just deadlocked with chance r0 (a box pushed onto a dead cell, say), or
      after each further push on it with chance r1, and then it restarts from
      the start (Beta priors `notice`, `notice_later`, worth `notice_strength`).
    Pushes it was shown say nothing about the student. A push it took back counts
    as a deadlocking mistake of the student's own, and the push the student plays
    instead counts as another draw (a slight overcount of its pushes).

    The forecast, planned at the start of each level: the student's distance to
    solved is a walk - each push of its own one closer with its skill there,
    else a mistake of one of the three kinds; a deadlocked board waits for the
    quick test, then restarts at the start distance, the pushes spent still
    counting - ending at solved or at the push cap (`step_factor` x optimal, at
    least `min_steps`, as the sessions'). Backward induction over (pushes made,
    difficulty, distance) gives the chance of solving untaught; on each turn
    the teacher averages it over its current belief about the level's
    difficulty and acts if it is below theta.

    Meant for sessions: one teacher per student, kept across levels, one
    attempt per level (each start() begins a new level). `predicted` is the
    chance it gave the student, at the start of the current level, of solving
    it untaught. counts: "caught" and "undone" (undo), "shown" (show), "dead"
    and "far" (restarts forced on a deadlocked board, and on one too far from
    solved)."""

    BINS = 11
    Z = np.linspace(-3, 3, 13)  # quadrature points for a level's difficulty, in units of tau
    LOG_PZ = -0.5 * Z ** 2 - np.log(np.exp(-0.5 * Z ** 2).sum())
    JUMPS = 3  # a worse push: +1, +2, or +3 or more

    def __init__(self, theta=0.3, act="undo", decay=0.99, window=300, step_factor=5.0, min_steps=50,
                 prior_mean=0.0, prior_sd=1.5, spread=1.0, tau=0.5, tau_sd=0.5, dead=0.3, worse=0.3,
                 mistake_strength=10.0, jumps=(0.6, 0.25, 0.15), jump_strength=10.0, notice=0.5, notice_later=0.2,
                 notice_strength=10.0, cost=0.02, cost_rate=0.2):
        super().__init__()
        if act not in ("undo", "show", "restart", "undo-value"):
            raise ValueError(f"act: undo, show, restart or undo-value, not {act!r}")
        self.theta, self.act = float(theta), act
        self.cost, self.cost_rate = cost, cost_rate  # undo-value: the cost per undo, adjusted toward theta's spend
        self.recent_undone = self.recent_pushes = 0.0
        self.name = f"bayes-{act} {self.theta:g}"
        self.decay, self.step_factor, self.min_steps = decay, step_factor, min_steps
        self.prior_mean, self.prior_sd, self.spread = prior_mean, prior_sd, spread
        self.tau0, self.tau_sd = tau, tau_sd
        self.kind_prior = mistake_strength * np.array([dead, worse, 1 - dead - worse])
        self.jump_prior = jump_strength * np.array(jumps)
        self.notice_prior = notice_strength * np.array([[notice, 1 - notice], [notice_later, 1 - notice_later]])
        self.skill = np.full(self.BINS, float(prior_mean))
        self.overall, self.log_tau = float(prior_mean), math.log(tau)
        self.history = deque(maxlen=window)  # per level: (optimal pushes, own pushes) per bin
        self.kinds = np.zeros((self.BINS, 3))  # mistakes per bin: deadlock, worse, sideways (decayed)
        self.jumps = np.zeros(self.JUMPS)
        self.notices = np.zeros((2, 2))  # [at once, later] x [noticed, missed]
        self.predicted = None
        self._pending = False

    # --- the teaching interface ---

    def start(self, level, rng):
        self._end_level()
        super().start(level, rng)
        self._begin_level()

    def solve_alone(self, state):
        """The model's chance of solving untaught from `state`, with the pushes made so far."""
        k = self.distance(state)
        if k == 0:
            return 1.0
        if k < 0 or self._m >= self._cap:
            return 0.0
        return float(_softmax(self._shift) @ self._alone[self._m, :, min(k, self._K)])

    def restart_gain(self, state):
        """(act "restart") How much a fresh attempt now raises the chance of solving, less theta, averaged over
        the belief about the level's difficulty; positive: restart."""
        k = self.distance(state)
        if k == 0 or self._m >= self._cap:
            return -1.0
        col = self._K + 1 if k < 0 else min(k, self._K)
        return float(_softmax(self._shift) @ self._gain[self._m, :, col])

    def force_restart(self, state, stretch):
        if self.act != "restart" or self.restart_gain(state) <= 0:
            return False
        dead = self.distance(state) < 0
        self.counts["dead" if dead else "far"] += 1
        if dead and self._fresh is not None:  # the quick test checked this board and missed the deadlock
            self._notices[0 if self._fresh else 1, 1] += 1
        self._fresh = None  # a restart the teacher forced says nothing about noticing
        return True

    def advise(self, state, step, proposal=None):
        if self.act != "show" or self.distance(state) <= 0 or self.solve_alone(state) >= self.theta:
            return None
        push = self.optimal_push(state)
        self.counts["shown" if push is not None else "no_move"] += 1
        return push

    def undo_gain(self, state):
        """(act "undo-value") How much taking back a deadlocking push here raises the chance of solving, less
        the cost of an undo, averaged over the belief about the level's difficulty; positive: take it back."""
        k = self.distance(state)
        if k <= 0 or self._m >= self._cap:
            return -1.0
        return float(_softmax(self._shift) @ self._undo_gain[self._m, :, min(k, self._K)])

    def veto(self, state, push):
        if self.act not in ("undo", "undo-value"):
            return False
        k = self.distance(state)
        if k <= 0 or self.distance(self.level.apply(state, push)) != -1:
            return False
        self.counts["caught"] += 1
        if self.act == "undo" and self.solve_alone(state) >= self.theta:
            return False
        if self.act == "undo-value" and self.undo_gain(state) <= 0:
            return False
        self._undone += 1
        self.counts["undone"] += 1
        self._own_push(k, -1)  # the student's own choice, a deadlocking one, though never played
        return True

    def observe(self, state, push, next_state, taught):
        self._m += 1
        k, after = self.distance(state), self.distance(next_state)
        if k < 0:  # a push on a deadlocked board: the quick test missed it
            self._notices[0 if self._fresh else 1, 1] += 1
            self._fresh = False
            return
        if after < 0:
            self._fresh = True  # just deadlocked: the quick test checks next
        if not taught:
            self._own_push(k, after)

    def restart(self, state):
        if self._fresh is not None:
            self._notices[0 if self._fresh else 1, 0] += 1
        self._fresh = None

    # --- beliefs ---

    def _own_push(self, k, after):
        b = min(k, self.BINS) - 1
        good = after == k - 1
        self._optimal[b] += good
        self._own[b] += 1
        if not good:
            kind = 0 if after < 0 else 1 if after > k else 2
            self._kinds[b, kind] += 1
            if kind == 1:
                self._jumps[min(after - k, self.JUMPS) - 1] += 1
        logit = self.skill[b] + math.exp(self.log_tau) * self.Z
        self._shift += _log_sigmoid(logit) if good else _log_sigmoid(-logit)

    def _begin_level(self):
        space = self.space()
        self._d = self.distance(self.level.start)
        self._K = max(space.dist)
        self._cap = max(self.min_steps, math.ceil(self.step_factor * self._d))
        self._m, self._fresh = 0, None
        self._optimal, self._own = np.zeros(self.BINS), np.zeros(self.BINS)
        self._kinds, self._jumps, self._notices = np.zeros((self.BINS, 3)), np.zeros(self.JUMPS), np.zeros((2, 2))
        self._shift = self.LOG_PZ.copy()
        self._undone = 0
        self._pending = True
        self._plan()
        self.predicted = float(np.exp(self.LOG_PZ) @ self._alone[0, :, self._d])

    def _end_level(self):
        """Folds the finished level's evidence into the beliefs about the student."""
        if not self._pending:
            return
        self._pending = False
        self.history.append((self._optimal, self._own))
        self.kinds = self.decay * self.kinds + self._kinds
        self.jumps = self.decay * self.jumps + self._jumps
        self.notices = self.decay * self.notices + self._notices
        if self.act == "undo-value":  # move the cost toward spending theta of the pushes on undos
            self.recent_undone = 0.9 * self.recent_undone + self._undone
            self.recent_pushes = 0.9 * self.recent_pushes + self._m
            if self.recent_pushes > 0:
                rate = self.recent_undone / self.recent_pushes
                self.cost = min(1.0, max(1e-6, self.cost * math.exp(self.cost_rate * (rate - self.theta) / self.theta)))
        self._fit()

    def _fit(self):
        """MAP skill per distance, overall level and tau, each level's difficulty integrated out (as Rush Hour's
        Bayesian._fit)."""
        from scipy.optimize import minimize
        optimal = np.array([o for o, _ in self.history])
        own = np.array([n for _, n in self.history])
        weight = self.decay ** np.arange(len(own) - 1, -1, -1)
        B, Z = self.BINS, self.Z

        def objective(x):
            skill, overall, log_tau = x[:B], x[B], x[B + 1]
            tau = math.exp(log_tau)
            logit = skill[None, :] + tau * Z[:, None]  # (difficulty, bin)
            ll = optimal @ _log_sigmoid(logit).T + (own - optimal) @ _log_sigmoid(-logit).T + self.LOG_PZ
            top = ll.max(1, keepdims=True)
            lse = top[:, 0] + np.log(np.exp(ll - top).sum(1))
            r = weight[:, None] * np.exp(ll - lse[:, None])  # weighted posterior over each level's difficulty
            resid = r.T @ optimal - (r.T @ own) / (1 + np.exp(-logit))  # (difficulty, bin)
            value = -weight @ lse + ((skill - overall) ** 2).sum() / (2 * self.spread ** 2) \
                + (overall - self.prior_mean) ** 2 / (2 * self.prior_sd ** 2) \
                + (log_tau - math.log(self.tau0)) ** 2 / (2 * self.tau_sd ** 2)
            grad = np.empty(B + 2)
            grad[:B] = -resid.sum(0) + (skill - overall) / self.spread ** 2
            grad[B] = -(skill - overall).sum() / self.spread ** 2 + (overall - self.prior_mean) / self.prior_sd ** 2
            grad[B + 1] = -tau * (resid.sum(1) @ Z) + (log_tau - math.log(self.tau0)) / self.tau_sd ** 2
            return value, grad

        x = minimize(objective, np.r_[self.skill, self.overall, self.log_tau], jac=True, method="L-BFGS-B").x
        self.skill, self.overall, self.log_tau = x[:B], float(x[B]), float(x[B + 1])

    def mistake_shares(self):
        """(per bin: chance a mistake deadlocks, moves further, stays as far; worse jumps +1/+2/+3;
        (r0, r1): chances the quick test notices a deadlock at once, and after each further push)."""
        kinds = self.kind_prior + self.kinds
        jumps = self.jump_prior + self.jumps
        notices = self.notice_prior + self.notices
        return kinds / kinds.sum(1, keepdims=True), jumps / jumps.sum(), notices[:, 0] / notices.sum(1)

    # --- deciding ---

    def _plan(self):
        """Backward induction: self._alone[m, difficulty, k] = the chance of solving untaught from a board k
        from solved with m pushes made."""
        K, cap, d = self._K, self._cap, self._d
        k = np.arange(K + 1)
        b = np.clip(k, 1, self.BINS) - 1
        p = 1 / (1 + np.exp(-(self.skill[b][None, :] + math.exp(self.log_tau) * self.Z[:, None])))  # (z, k)
        kinds, jumps, (r0, r1) = self.mistake_shares()
        dead, worse, sideways = kinds[b, 0], kinds[b, 1], kinds[b, 2]
        down = np.maximum(k - 1, 0)
        ups = [np.minimum(k + j, K) for j in range(1, self.JUMPS + 1)]
        solved = (k == 0).astype(float)
        alone = np.empty((cap + 1, len(self.Z), K + 1))
        alone[cap] = solved
        missed = np.zeros(len(self.Z))  # on a deadlocked board the quick test missed, m pushes made: E(m)
        landed = np.zeros(len(self.Z))  # just deadlocked, m pushes made, the quick test about to check: D(m)
        for m in range(cap - 1, -1, -1):
            nxt = alone[m + 1]
            up = sum(jumps[j] * nxt[:, ups[j]] for j in range(self.JUMPS))
            play = p * nxt[:, down] + (1 - p) * (dead * landed[:, None] + worse * up + sideways * nxt)
            alone[m] = np.where(k == 0, 1.0, play)
            # E(m): one more push on the dead board, then the test checks (nothing after the cap)
            missed = r1 * nxt[:, d] + (1 - r1) * missed if m + 1 < cap else np.zeros(len(self.Z))
            landed = r0 * alone[m, :, d] + (1 - r0) * missed
        self._alone = alone
        if self.act == "restart":
            self._plan_restarts(p, dead, worse, sideways, jumps, r0, r1, down, ups, solved)
        if self.act == "undo-value":
            self._plan_undos(p, dead, worse, sideways, jumps, r0, r1, down, ups, solved)

    def _plan_undos(self, p, dead, worse, sideways, jumps, r0, r1, down, ups, solved):
        """Backward induction with the teacher free to take back a deadlocking push (at self.cost each): the
        student then plays a push from its other choices (in the model, one that doesn't deadlock, the rest of its
        push distribution renormalised). self._value[m, difficulty, k]: the chance of solving, undoing when it
        pays; self._undo_gain[m, difficulty, k]: undoing a deadlocking push on board k, m pushes made, less
        letting it stand."""
        K, cap, d, c = self._K, self._cap, self._d, self.cost
        nz = len(self.Z)
        value = np.empty((cap + 1, nz, K + 1))
        value[cap] = solved
        gain = np.empty((cap, nz, K + 1))
        missed = np.zeros(nz)
        landed = np.zeros(nz)  # value on a board just deadlocked (the quick test about to check), at m + 1
        for m in range(cap - 1, -1, -1):
            nxt = value[m + 1]
            up = sum(jumps[j] * nxt[:, ups[j]] for j in range(self.JUMPS))
            keep = p * nxt[:, down] + (1 - p) * (worse * up + sideways * nxt)  # the pushes that don't deadlock
            undo = keep / np.maximum(1 - (1 - p) * dead, 1e-12) - c
            gain[m] = undo - landed[:, None]
            value[m] = np.where(solved > 0, 1.0, keep + (1 - p) * dead * np.maximum(landed[:, None], undo))
            missed = r1 * nxt[:, d] + (1 - r1) * missed if m + 1 < cap else np.zeros(nz)
            landed = r0 * value[m, :, d] + (1 - r0) * missed
        self._value, self._undo_gain = value, gain

    def _plan_restarts(self, p, dead, worse, sideways, jumps, r0, r1, down, ups, solved):
        """Backward induction with the teacher free to restart the student on any turn (a restart costs no
        push). self._value[m, difficulty, k]: the chance of solving, restarting when it pays; self._gain[m,
        difficulty, k] = restarting now minus playing on, less theta (column K + 1: a deadlocked board the quick
        test missed)."""
        K, cap, d, c = self._K, self._cap, self._d, self.theta
        nz = len(self.Z)
        value = np.empty((cap + 1, nz, K + 1))
        value[cap] = solved
        gain = np.empty((cap, nz, K + 2))
        missed = np.zeros(nz)  # value on a deadlocked board the test missed, about to push (from the step after)
        landed = np.zeros(nz)
        for m in range(cap - 1, -1, -1):
            nxt = value[m + 1]
            up = sum(jumps[j] * nxt[:, ups[j]] for j in range(self.JUMPS))
            play = np.where(solved > 0, 1.0, p * nxt[:, down] + (1 - p) * (dead * landed[:, None] + worse * up
                                                                        + sideways * nxt))
            fresh = play[:, d] - c  # restart: back at the start, the same pushes made
            # on a deadlocked board, playing on: one more push, then the quick test checks it
            on_dead = r1 * nxt[:, d] + (1 - r1) * missed if m + 1 < cap else np.zeros(nz)
            gain[m, :, :K + 1] = fresh[:, None] - play
            gain[m, :, K + 1] = fresh - on_dead
            value[m] = np.where(solved > 0, 1.0, np.maximum(play, fresh[:, None]))
            missed = np.maximum(on_dead, fresh)
            landed = r0 * value[m, :, d] + (1 - r0) * missed
        self._value, self._gain = value, gain


TEACHERS = {"random": Random, "guard": Guard, "undo": Undo, "restart": Restart,
            "undo+restart": lambda k: Restart(k, undo=True),
            "bayes-undo": lambda theta: Bayes(theta, "undo"), "bayes-show": lambda theta: Bayes(theta, "show"),
            "bayes-restart": lambda theta: Bayes(theta, "restart"),
            "bayes-undo-value": lambda budget: Bayes(budget, "undo-value")}


def make_teacher(spec):
    """A teacher from "kind:parameter" (e.g. "random:0.1"); "none" is None."""
    if spec in (None, "", "none"):
        return None
    kind, _, arg = spec.partition(":")
    if kind not in TEACHERS:
        raise ValueError(f"unknown teacher {kind!r}; kinds: {', '.join(TEACHERS)}")
    return TEACHERS[kind](float(arg) if arg else 1.0)
