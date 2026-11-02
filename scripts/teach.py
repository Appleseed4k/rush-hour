"""Teachers that decide, move by move, when to show a student the optimal
move during a solve attempt.

A teacher knows each puzzle's breadth-first-search distances (lib.multi_bfs,
computed the first time it has to teach on that puzzle), so the move it shows
is always optimal: one that brings the board a move closer to the goal (one
of them at random when several do). What varies between teachers is *when*
they show it - the schedule an adaptive teacher would eventually learn. The
fixed schedules are Random(p): p = 0 never teaches (the student learns only
from its own attempts), p = 1 shows every move (the student only watches).
CorrectMistakes steps in only when the student is about to go wrong - which
needs to see the move the student intends, an idealization;
CorrectAfterMistake, more realistically, lets a wrong move stand and shows
the best move from where it led. Each notices every mistake but acts on it
with probability q (q = 1: every one, without drawing from the attempt's
random numbers), as Random(p) acts on each turn with probability p.
Budgeted(teacher, k) caps any teacher at k interventions per attempt. Each
move shown is one unit of the teaching budget (an intervention).

Bayesian(budget) is adaptive: it keeps beliefs about the student, updated
from every move the student makes, and shows the optimal move when that is
worth more to the student's chance of solving the puzzle than it costs -
spending about `budget` of all moves, like Random(budget).

The interface, used by rl.solve_intuition during a student's attempt:
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
- end_attempt(result): the attempt is over (rl.solve_intuition's result:
  boards, moves, which moves were taught, whether it solved). Where an
  adaptive teacher would learn from how the student did; a no-op here.

`counts` (a Counter) tallies whatever a teacher wants reported; rl sums it
over a round's attempts into expert_iteration's history.
"""
import functools
import math
import random
from collections import Counter, deque

import numpy as np

from lib import car_orientations, multi_bfs, neighbors

OPPOSITE = {"l": "r", "r": "l", "u": "d", "d": "u"}


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

    def is_optimal(self, state, move):
        """Whether `move` brings `state` one move closer to the goal."""
        distances = self.distances()
        nxt = dict(neighbors(state, self._orientations)).get(move)
        return distances.get(nxt) == distances[state] - 1

    def mistake_kind(self, state, next_state):
        """None if the move from `state` to `next_state` was optimal, else
        "worse" (a move further from the goal) or "sideways" (same distance)."""
        distances = self.distances()
        before, after = distances[state], distances.get(next_state)
        if after == before - 1:
            return None
        return "worse" if after == before + 1 else "sideways"

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


def _acts(teacher):
    """Whether `teacher` acts on a mistake it noticed: always at q >= 1 (no
    random draw), else with probability q."""
    return teacher.q >= 1 or teacher.rng.random() < teacher.q


class CorrectMistakes(Teacher):
    """Lets the student play, but steps in when the move it is about to make
    isn't optimal (doesn't bring the board a move closer to the goal) and
    shows an optimal one instead - a tutor correcting errors - with
    probability `q` for each such move. It sees the move the student intends
    before it is played, an idealization (see CorrectAfterMistake). The
    student acts on its own boards, as in DAgger; at q = 1 every mistake is
    corrected, so every attempt is solved optimally, and how much it teaches
    falls as the student improves."""

    def __init__(self, q=1.0):
        super().__init__()
        self.q = q
        self.name = "correct mistakes" if q >= 1 else f"correct mistakes {q}"

    def wants_to_teach(self, state, step, proposal=None):
        if proposal is None:
            raise ValueError("CorrectMistakes needs the student's proposed move")
        if self.is_optimal(state, proposal()) or not _acts(self):
            return False
        self.counts["corrections"] += 1
        return True


class CorrectAfterMistake(Teacher):
    """Lets every move the student makes stand. After a move that wasn't
    optimal, with probability `q`, it shows the optimal move from the board
    the mistake led to, on the student's next turn - a tutor who can only
    react to what was played; otherwise it says nothing. The student isn't
    told its move was a mistake, only shown what to do now.

    counts: mistakes noticed, as "mistakes_worse" (the mistake took the board
    a move further from the goal) or "mistakes_sideways" (same distance);
    and per correction ("corrections"), "undo_optimal" (taking the mistake
    back is an optimal move - exactly when it was worse) and "shown_undo"
    (the move shown was that undo - the teacher picks among tied optimal
    moves at random)."""

    def __init__(self, q=1.0):
        super().__init__()
        self.q = q
        self.name = "correct after mistake" if q >= 1 else f"correct after mistake {q}"

    def start(self, state, rng):
        super().start(state, rng)
        self._mistake = None

    def observe(self, state, move, next_state, taught):
        kind = None if taught else self.mistake_kind(state, next_state)
        if kind is None:
            return
        self.counts[f"mistakes_{kind}"] += 1
        if _acts(self):
            self._mistake = (move, next_state)

    def wants_to_teach(self, state, step, proposal=None):
        return self._mistake is not None and self._mistake[1] == state

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step, proposal):
            return None
        (car, direction, steps), _ = self._mistake
        self._mistake = None
        shown = self.optimal_move(state)
        undo = (car, OPPOSITE[direction], steps)
        self.counts["corrections"] += 1
        self.counts["undo_optimal"] += self.is_optimal(state, undo)
        self.counts["shown_undo"] += shown == undo
        return shown


class Budgeted(Teacher):
    """`teacher`, allowed to show at most `k` moves per attempt - after that
    it stays silent for the rest of the attempt. The wrapped teacher still
    notices everything (its counts keep tallying mistakes), but nothing it
    would show past the budget reaches the student."""

    def __init__(self, teacher, k):
        self.teacher = teacher
        self.k = k
        self.name = f"{teacher.name} budget {k}"
        self.spent = 0

    @property
    def counts(self):
        return self.teacher.counts

    @counts.setter
    def counts(self, value):
        self.teacher.counts = value

    def start(self, state, rng):
        self.teacher.start(state, rng)
        self.spent = 0

    def advise(self, state, step, proposal=None):
        if self.spent >= self.k:
            return None
        move = self.teacher.advise(state, step, proposal)
        if move is not None:
            self.spent += 1
        return move

    def observe(self, state, move, next_state, taught):
        self.teacher.observe(state, move, next_state, taught)

    def end_attempt(self, result):
        self.teacher.end_attempt(result)


@functools.lru_cache(maxsize=None)
def _human_hazard():
    """People's surrender rates by moves / optimal (human_data.surrender_hazard), read once per process."""
    import human_data
    edges, rates, _ = human_data.surrender_hazard()
    return np.array(edges, dtype=float), np.array(rates)


def _log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class Bayesian(Teacher):
    """An adaptive teacher in the spirit of hierarchical Bayesian models of
    teaching (Chen et al., 2024) and of when parents take over (Shachnai et
    al., 2025): it holds beliefs about the student, updates them from what
    the student does, and steps in when its beliefs say the student needs it.

    Beliefs, a hierarchy (fitted, MAP, after every puzzle):
    - the student's skill: the chance that a move of its own is optimal,
      as a logit per distance to the goal (1, 2, ..., 10, 11 or more - near
      the goal is easier, and that is where students improve first), each
      drawn around the student's overall level (sd `spread`), which has a
      population prior (mean `prior_mean`, sd `prior_sd`, logits);
    - each puzzle's difficulty for this student: a shift of all those
      logits, normal with sd tau (itself fitted, prior around `tau`) - some
      puzzles are much harder for it than others of the same length. Within
      a puzzle, the teacher's belief about the shift is updated after every
      move the student makes itself;
    - what a mistake looks like: the share of non-optimal moves that go a
      move further from the goal (rather than sideways), per distance (prior
      mean `worse`, worth `worse_strength` moves);
    - the student's persistence: its chance of giving up after a move, as
      people's (human_data.surrender_hazard) times a factor with a
      log-normal prior around 1 (sd `persistence_sd`), from the moves after
      which it carried on and the one after which it gave up, if it did.
    The student keeps learning, so older evidence counts less: each puzzle's
    by `decay` per puzzle since (the last `window` puzzles are kept).
    Moves the teacher showed say nothing about skill.

    Deciding (planned at the start of each puzzle): the student's progress is
    modelled as a walk in distance to the goal - each move of its own one
    closer with its skill there, else further or sideways - that ends when
    red is out (worth 1), when it gives up, or at the move cap
    (`step_factor` x the optimal length, at least `min_steps`, as the
    sessions'). Showing the optimal move brings the board one closer for a
    cost. Backward induction over (distance, moves made) gives, for every
    possible difficulty of the puzzle, how much better showing is than
    letting the student play; on each turn the teacher averages that over
    its current belief about the puzzle's difficulty (Chen et al.'s
    marginalising over what the learner knows) and shows the move if it
    comes out ahead. So it steps in where the student is far from the goal,
    struggling on this puzzle, and running out of moves or patience, and
    stays out where it would get there alone.

    The cost per move shown adjusts itself, after each puzzle, so that
    about `budget` of the moves (over the last ten or so puzzles) are shown
    - the same spending as Random(budget), placed where it should matter
    most. It never looks at the student's intended move.

    Meant for sessions, one teacher per student, kept across puzzles (in
    rl.expert_iteration's parallel attempts every puzzle gets a fresh copy,
    so it never gets to know the student). counts, per puzzle:
    "shown"; "predicted_solve" (at the start, the chance it gave the student
    of solving untaught); "skill_at_start" (its belief of the chance the
    student's next move is optimal there, on a typical puzzle);
    "puzzle_shift" (its belief, at the end, of how much easier - in logits -
    the puzzle was for the student than a typical one); "cost"."""

    BINS = 11
    Z = np.linspace(-3, 3, 13)  # quadrature points for a puzzle's difficulty, in units of tau
    LOG_PZ = -0.5 * Z ** 2 - np.log(np.exp(-0.5 * Z ** 2).sum())
    S = np.logspace(-2, 1, 31)  # persistence factors on people's surrender rates

    def __init__(self, budget=0.1, decay=0.99, window=300, step_factor=10.0, min_steps=50, prior_mean=0.0,
                 prior_sd=1.5, spread=1.0, tau=0.5, tau_sd=0.5, worse=0.35, worse_strength=10.0,
                 persistence_sd=1.0, cost=0.02, cost_rate=0.2):
        super().__init__()
        self.budget = budget
        self.name = f"bayes {budget}"
        self.decay, self.step_factor, self.min_steps = decay, step_factor, min_steps
        self.prior_mean, self.prior_sd, self.spread = prior_mean, prior_sd, spread
        self.tau0, self.tau_sd = tau, tau_sd
        self.worse, self.worse_strength = worse, worse_strength
        self.cost, self.cost_rate = cost, cost_rate
        self.skill = np.full(self.BINS, float(prior_mean))
        self.level, self.log_tau = float(prior_mean), math.log(tau)
        self.history = deque(maxlen=window)  # per puzzle: (optimal moves, moves) of the student's own, per bin
        self.worse_moves, self.mistakes = np.zeros(self.BINS), np.zeros(self.BINS)
        self.persistence_prior = -0.5 * (np.log(self.S) / persistence_sd) ** 2
        self.persistence_loglik = np.zeros(len(self.S))
        self.recent_shown = self.recent_moves = 0.0
        self._pending, self._finished = False, False

    # --- the teaching interface ---

    def start(self, state, rng):
        new = self._finished or state != self._puzzle
        super().start(state, rng)
        if new:
            self._end_puzzle()
            self._begin_puzzle(state)

    def wants_to_teach(self, state, step, proposal=None):
        k = self.distances()[state]
        m = self._offset + step
        if self.budget <= 0 or k == 0 or m >= self._cap:
            return False
        return float(_softmax(self._shift) @ self._advantage[m, :, min(k, self._K)]) > 0

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step):
            return None
        self.counts["shown"] += 1
        self._shown += 1
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
        self._observe_persistence(result)
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
        self._offset = self._shown = self._moves = 0
        self._optimal, self._own = np.zeros(self.BINS), np.zeros(self.BINS)
        self._worse, self._mistakes = np.zeros(self.BINS), np.zeros(self.BINS)
        self._shift = self.LOG_PZ.copy()
        self._pending, self._finished = True, False
        self._plan()
        self.counts["cost"] = self.cost
        self.counts["skill_at_start"] = float(1 / (1 + math.exp(-self.skill[min(self._d, self.BINS) - 1])))

    def _end_puzzle(self):
        """Folds the finished puzzle's evidence into the beliefs about the student, and adjusts the cost."""
        if not self._pending:
            return
        self._pending = False
        self.history.append((self._optimal, self._own))
        self.worse_moves = self.decay * self.worse_moves + self._worse
        self.mistakes = self.decay * self.mistakes + self._mistakes
        self.persistence_loglik *= self.decay
        self.recent_shown = 0.9 * self.recent_shown + self._shown
        self.recent_moves = 0.9 * self.recent_moves + self._moves
        if self.budget > 0 and self.recent_moves > 0:
            rate = self.recent_shown / self.recent_moves
            self.cost = min(1.0, max(1e-6, self.cost * math.exp(self.cost_rate * (rate - self.budget) / self.budget)))
        self._fit()

    def _fit(self):
        """MAP skill per distance, overall level and tau, each puzzle's difficulty integrated out."""
        from scipy.optimize import minimize
        optimal = np.array([o for o, _ in self.history])
        own = np.array([n for _, n in self.history])
        weight = self.decay ** np.arange(len(own) - 1, -1, -1)
        B, Z = self.BINS, self.Z

        def objective(x):
            skill, level, log_tau = x[:B], x[B], x[B + 1]
            tau = math.exp(log_tau)
            logit = skill[None, :] + tau * Z[:, None]  # (difficulty, bin)
            ll = optimal @ _log_sigmoid(logit).T + (own - optimal) @ _log_sigmoid(-logit).T + self.LOG_PZ
            top = ll.max(1, keepdims=True)
            lse = top[:, 0] + np.log(np.exp(ll - top).sum(1))
            r = weight[:, None] * np.exp(ll - lse[:, None])  # weighted posterior over each puzzle's difficulty
            resid = r.T @ optimal - (r.T @ own) / (1 + np.exp(-logit))  # (difficulty, bin)
            value = -weight @ lse + ((skill - level) ** 2).sum() / (2 * self.spread ** 2) \
                + (level - self.prior_mean) ** 2 / (2 * self.prior_sd ** 2) \
                + (log_tau - math.log(self.tau0)) ** 2 / (2 * self.tau_sd ** 2)
            grad = np.empty(B + 2)
            grad[:B] = -resid.sum(0) + (skill - level) / self.spread ** 2
            grad[B] = -(skill - level).sum() / self.spread ** 2 + (level - self.prior_mean) / self.prior_sd ** 2
            grad[B + 1] = -tau * (resid.sum(1) @ Z) + (log_tau - math.log(self.tau0)) / self.tau_sd ** 2
            return value, grad

        x = minimize(objective, np.r_[self.skill, self.level, self.log_tau], jac=True, method="L-BFGS-B").x
        self.skill, self.level, self.log_tau = x[:B], float(x[B]), float(x[B + 1])

    def _observe_persistence(self, result):
        """The attempt's moves after which the student carried on, and the one it gave up after, if any (there
        is no choice after the solving move)."""
        n = len(result["moves"])
        if n == 0:
            return
        hazard = np.minimum(1.0, np.outer(self.S, self._human_rate(self._offset + np.arange(1, n + 1))))
        carried_on = hazard[:, :-1] if result["solved"] or result.get("surrendered") else hazard
        self.persistence_loglik += np.log1p(-np.minimum(carried_on, 1 - 1e-9)).sum(1)
        if result.get("surrendered"):
            self.persistence_loglik += np.log(np.maximum(hazard[:, -1], 1e-12))

    def _human_rate(self, moves):
        edges, rates = _human_hazard()
        i = np.searchsorted(edges, np.asarray(moves) / self._d, side="right") - 1
        return rates[np.clip(i, 0, len(rates) - 1)]

    # --- deciding ---

    def _plan(self):
        """Backward induction: self._advantage[m, difficulty, k] = how much better showing the optimal move
        is than letting the student play, on a board k from the goal after m moves on the puzzle."""
        K, cap = self._K, self._cap
        k = np.arange(K + 1)
        b = np.clip(k, 1, self.BINS) - 1
        p = 1 / (1 + np.exp(-(self.skill[b][None, :] + math.exp(self.log_tau) * self.Z[:, None])))
        worse = ((self.worse_strength * self.worse + self.worse_moves) / (self.worse_strength + self.mistakes))[b]
        persistence = _softmax(self.persistence_prior + self.persistence_loglik)
        hazard = persistence @ np.minimum(1.0, np.outer(self.S, self._human_rate(np.arange(cap + 1))))
        down, up, out = np.maximum(k - 1, 0), np.minimum(k + 1, K), k == 0
        value = np.broadcast_to(out.astype(float), p.shape)  # at the cap: solved only if red is out
        alone = value  # the same, never teaching
        self._advantage = np.empty((cap, len(self.Z), K + 1))
        for m in range(cap - 1, -1, -1):
            after = np.where(out, 1.0, (1 - hazard[m + 1]) * value)  # value once the next move lands on k
            play = p * after[:, down] + (1 - p) * (worse * after[:, up] + (1 - worse) * after)
            show = after[:, down] - self.cost
            self._advantage[m] = show - play
            value = np.where(out, 1.0, np.maximum(show, play))
            after = np.where(out, 1.0, (1 - hazard[m + 1]) * alone)
            alone = np.where(out, 1.0, p * after[:, down] + (1 - p) * (worse * after[:, up] + (1 - worse) * after))
        self.counts["predicted_solve"] = float(np.exp(self.LOG_PZ) @ alone[:, self._d])
