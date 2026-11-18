"""Teachers that decide, move by move, when to show a student the optimal
move during a solve attempt.

Terms: "the student" is the simulated student, the policy network that plays
and learns (rl.PolicyAgent, rl.learning_session). "Model of the student" (or
student model) always means a teacher's own beliefs about that student, as in
the Bayesian teachers below - never the network itself.

A teacher knows each puzzle's breadth-first-search distances (lib.multi_bfs,
computed the first time it has to teach on that puzzle), so the move it shows
is always optimal: one that brings the board a move closer to the goal (one
of them at random when several do). What varies between teachers is *when*
they show it - the schedule an adaptive teacher would eventually learn. The
fixed schedules are Random(p): p = 0 never teaches (the student learns only
from its own attempts), p = 1 shows every move (the student only watches).
Each move shown is one unit of the teaching budget (an intervention). The
mistake-correcting teachers of the offline-teaching phase (CorrectMistakes,
CorrectAfterMistake, Budgeted) are archived in scripts/old/p2_teach/teachers.py.

Bayesian(budget) is adaptive: it keeps beliefs about the student, updated
from every move the student makes, and shows the optimal move when that is
worth more to the student's chance of solving the puzzle than it costs -
spending about `budget` of all moves, like Random(budget). Bayesian3
("bayes3") keeps that model of the student but shows the move whenever the
model gives the student under theta (0.45) chance of solving alone from
here - no budget, no cost. Bayesian4 ("bayes4"), the default teacher, is
bayes3 with the model of the student carried from puzzle to puzzle by a
filter (SkillFilter) instead of refitted from the stored history.

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


def level_shares(state, distances=None):
    """Per distance to the goal k (1 .. the farthest board): the share of all legal moves, over every board k
    from the goal, that are optimal (lead to a board k - 1 away) - how rare the right move is at that distance.
    Students who fail a long puzzle mostly stall at one distance, usually the one where this share is lowest."""
    distances = distances or multi_bfs(state)
    orientations = car_orientations(state)
    down, total = Counter(), Counter()
    for board, k in distances.items():
        if k == 0:
            continue
        for _, nxt in neighbors(board, orientations):
            total[k] += 1
            down[k] += distances[nxt] == k - 1
    return {k: down[k] / total[k] for k in sorted(total)}


class Wall(Teacher):
    """Helps only at the wall a student is stuck at: the closest it has got to the goal on this puzzle, when it
    hasn't got any closer for `patience` x the puzzle's optimal length moves of its own (counted since its last new
    closest approach). Then, once it is back at that distance, it shows the optimal move - the step through the
    wall it keeps failing to find - and steps back; anywhere else, and before then, the student plays alone.

    Why the student's progress, not the layout: in unhelped students' traced play (teach.Trace), failing students'
    closest approaches cluster per puzzle (a wall), but the distance level whose moves are least often optimal
    (level_shares) - raw or relative to the typical share at that distance - picks it out poorly (8-14% exactly),
    and students spend almost no moves at a wall (median 0) before getting through or falling back. Being stuck,
    by contrast, separates attempts that would fail from ones that would succeed: at 24-26 moves, 2 x length
    stuck moves fires on 55% of attempts that went on to fail and 19% of ones that went on to solve; stuck
    stretches grow with length, hence patience in units of it. About a third of failing attempts never come back
    to their closest approach; this teacher doesn't help them.

    counts: "shown", "stuck" (1 if it ever triggered), "best" (the closest approach at the end)."""

    def __init__(self, patience=2.0):
        super().__init__()
        self.patience = float(patience)
        self.name = f"wall {self.patience:g}"
        self._for = None

    def start(self, state, rng):
        super().start(state, rng)
        if self._for != state:  # a new puzzle (a later attempt at the same one carries on)
            self._for = state
            self._d = self.distances()[state]
            self.best, self.since = self._d, 0

    def wants_to_teach(self, state, step, proposal=None):
        return self.since >= self.patience * self._d and self.distances()[state] == self.best > 0

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step):
            return None
        self.counts["shown"] += 1
        self.counts["stuck"] = 1
        return self.optimal_move(state)

    def observe(self, state, move, next_state, taught):
        k = self.distances()[next_state]
        if k < self.best:
            self.best, self.since = k, 0
        elif not taught:
            self.since += 1
        self.counts["best"] = self.best


class Stuck(Teacher):
    """Wall's trigger, helping wherever the student is: once it has made `patience` x the puzzle's optimal length
    moves of its own without getting closer to the goal than ever before on this puzzle, it shows the optimal move
    on every turn - from wherever the student has wandered to - until the board is closer than the student's
    closest approach (led back and through the wall), then steps back and the count starts again. Reaches the
    failing attempts that never come back to their closest approach (about a third), which Wall can't.

    counts: "shown", "stuck" (1 if it ever triggered), "best"."""

    def __init__(self, patience=1.0):
        super().__init__()
        self.patience = float(patience)
        self.name = f"stuck {self.patience:g}"
        self._for = None

    def start(self, state, rng):
        super().start(state, rng)
        if self._for != state:
            self._for = state
            self._d = self.distances()[state]
            self.best, self.since = self._d, 0

    def wants_to_teach(self, state, step, proposal=None):
        return self.since >= self.patience * self._d and self.distances()[state] > 0

    def advise(self, state, step, proposal=None):
        if not self.wants_to_teach(state, step):
            return None
        self.counts["shown"] += 1
        self.counts["stuck"] = 1
        return self.optimal_move(state)

    def observe(self, state, move, next_state, taught):
        k = self.distances()[next_state]
        if k < self.best:
            self.best, self.since = k, 0
        elif not taught:
            self.since += 1
        self.counts["best"] = self.best


class Hint(Stuck):
    """Stuck's trigger, help as a hint: once the student has made `patience` x the puzzle's optimal length moves of
    its own without getting closer to the goal than ever before, it shows `k` optimal moves in a row, from wherever
    the student is, then hands the puzzle back - and stays out until the student is stuck that long again (its
    count starts over after the hint), however it does in the meantime. Help in bursts of a fixed size, with no
    model of the student: against StuckBayes (bursts as long as its model says the student needs, and back in
    whenever it slides) and take-over (StuckBayes with theta 1: every move once stuck) at matched spend.

    counts: "shown", "hints" (bursts started), "stuck" (1 if it ever triggered), "best"."""

    def __init__(self, k=3, patience=1.0):
        super().__init__(patience)
        self.k = int(k)
        self.name = f"hint {self.k}" + (f" p{self.patience:g}" if self.patience != 1 else "")

    def start(self, state, rng):
        new = self._for != state
        super().start(state, rng)
        if new:
            self.left = 0  # moves left in the current hint

    def advise(self, state, step, proposal=None):
        if self.distances()[state] == 0:
            return None
        if self.left == 0:
            if self.since < self.patience * self._d:
                return None
            self.left = self.k
            self.counts["hints"] += 1
            self.counts["stuck"] = 1
        self.left -= 1
        if self.left == 0:
            self.since = 0  # handed back: the student gets a full patience before the next hint
        self.counts["shown"] += 1
        return self.optimal_move(state)


class Trace(Teacher):
    """Never teaches: only records each attempt's path, for checking a teacher's model of the student offline
    (e.g. Bayesian's calibration) against how unhelped students really play. counts["path"]: the distance to the
    goal at the start and after each move, every attempt in a row; counts["max_distance"]: the farthest any
    board of the puzzle is from the goal."""
    name = "trace"

    def __init__(self, _=1.0):
        super().__init__()

    def wants_to_teach(self, state, step, proposal=None):
        return False

    def advise(self, state, step, proposal=None):
        return None

    def observe(self, state, move, next_state, taught):
        path = self.counts.setdefault("path", [])
        if not path:
            path.append(self.distances()[state])
            self.counts["max_distance"] = max(self.distances().values())
        path.append(self.distances()[next_state])


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
        p = 1 / (1 + np.exp(-(self.skill[b][None, :] + self._logit_extra(k)[None, :] + math.exp(self.log_tau) * self.Z[:, None])))
        worse = ((self.worse_strength * self.worse + self.worse_moves) / (self.worse_strength + self.mistakes))[b]
        persistence = _softmax(self.persistence_prior + self.persistence_loglik)
        hazard = persistence @ np.minimum(1.0, np.outer(self.S, self._human_rate(np.arange(cap + 1))))
        down, up, out = np.maximum(k - 1, 0), np.minimum(k + 1, K), k == 0
        value = np.broadcast_to(out.astype(float), p.shape)  # at the cap: solved only if red is out
        alone = value  # the same, never teaching
        solve = value if self.TRACKS_SOLVE else None  # the chance of solving, following the plan
        self._advantage = np.empty((cap, len(self.Z), K + 1))
        if self.KEEPS_ALONE:
            self._alone = np.empty((cap, len(self.Z), K + 1))
        for m in range(cap - 1, -1, -1):
            after = np.where(out, 1.0, (1 - hazard[m + 1]) * value)  # value once the next move lands on k
            play = p * after[:, down] + (1 - p) * (worse * after[:, up] + (1 - worse) * after)
            if solve is not None:
                after_solve = np.where(out, 1.0, (1 - hazard[m + 1]) * solve)
            show = self._show_value(after[:, down], after_solve[:, down] if solve is not None else None)
            self._advantage[m] = show - play
            if solve is not None:
                play_solve = p * after_solve[:, down] + (1 - p) * (worse * after_solve[:, up]
                                                                   + (1 - worse) * after_solve)
                solve = np.where(out, 1.0, np.where(show > play, after_solve[:, down], play_solve))
            value = np.where(out, 1.0, np.maximum(show, play))
            after = np.where(out, 1.0, (1 - hazard[m + 1]) * alone)
            alone = np.where(out, 1.0, p * after[:, down] + (1 - p) * (worse * after[:, up] + (1 - worse) * after))
            if self.KEEPS_ALONE:
                self._alone[m] = alone
        self.counts["predicted_solve"] = float(np.exp(self.LOG_PZ) @ alone[:, self._d])

    TRACKS_SOLVE = False  # whether _plan also follows the chance of solving, for _show_value
    KEEPS_ALONE = False  # whether _plan keeps the chance of solving untaught from every (moves made, difficulty, k)

    def _logit_extra(self, k):
        """Added to the skill logit on boards `k` (an array of distances) from the goal: none here."""
        return np.zeros(len(k))

    def _show_value(self, after_down, solve_down=None):
        """The value of showing the move, given the value of the board it leads to (and, with TRACKS_SOLVE,
        the chance of solving from there): that value, less the cost."""
        return after_down - self.cost


class LearningBayesian(Bayesian):
    """Bayesian, aiming at what the student learns rather than at solving the puzzle in front of it. The student
    learns from the puzzles it solves (their boards become examples), and a move it was shown teaches only
    `agency` of what a move of its own does (rl.agency_weights). So a solved puzzle is worth about
    1 - (1 - agency) x the share of its moves that were shown, an unsolved one nothing: help pays where it turns a
    puzzle the student would fail into one it solves, and costs a share of that solve where it would have got
    there alone.

    Planned as Bayesian's, with that value: each move shown costs c = (1 - agency) / n times the chance of then
    solving, n the moves a solved attempt at this puzzle is expected to take (the student's recent moves /
    optimal on the puzzles it solved, times this one's optimal length). Summed over the moves shown, the plan's
    value is then exactly E[solved x (1 - c x moves shown)] under the teacher's model of the student. There is
    no budget: how much it shows follows from its beliefs - most where a solve is unlikely without help - and
    falls as the student gets better. counts: as Bayesian's, with "cost" the per-move share c."""

    TRACKS_SOLVE = True

    def __init__(self, agency=0.0, ratio=2.5, **kwargs):
        super().__init__(budget=1.0, cost_rate=0.0, **kwargs)  # budget unused: nothing adjusts the cost
        self.agency = agency
        self.name = f"bayes-learn {agency:g}"
        self.ratio = ratio  # recent moves / optimal over the solved puzzles, as an average decaying by 0.9

    def _begin_puzzle(self, state):
        d = self.distances()[state]
        self.cost = (1 - self.agency) / max(1.0, self.ratio * d)
        super()._begin_puzzle(state)

    def end_attempt(self, result):
        if result["solved"]:
            self.ratio = 0.9 * self.ratio + 0.1 * (self._offset + len(result["moves"])) / max(1, self._d)
        super().end_attempt(result)

    def _show_value(self, after_down, solve_down=None):
        return after_down - self.cost * solve_down


class TwoEnds(Bayesian):
    """Bayesian's model of the student, recalibrated against unhelped students' traced play (teach.Trace): the
    chance a move of the student's own is optimal depends on how far the board is from the goal AND from the
    puzzle's start - high at both ends (the puzzles start at their farthest board, where no move can make things
    worse; near the goal the plan is in view), low in the middle - as a logit per distance to the goal plus one
    per distance below the starting distance (prior sd `start_sd`), each puzzle shifted as before. With only the
    first, a long puzzle's hard middle borrowed the easy starts of short puzzles at the same distance, and the
    teacher overrated long puzzles (24-26 moves: 0.73 predicted untaught against 0.43 solved). Both terms run
    to 16, and evidence is kept longer (`decay` 0.998 over up to `window` 1000 puzzles), since long puzzles are
    a fraction of the recent ones. On the trace: 24-26 moves 0.53 against 0.43, Brier 0.173 (Bayesian 0.200).

    A mixin: Bayesian2 and LearningBayesian2 are Bayesian and LearningBayesian with this model."""

    BINS = 16
    START_BINS = 16

    def __init__(self, *args, start_sd=1.0, decay=0.998, window=1000, **kwargs):
        super().__init__(*args, decay=decay, window=window, **kwargs)
        self.start_skill = np.zeros(self.START_BINS)
        self.start_sd = start_sd

    def _cells(self, k):
        return min(k, self.BINS) - 1, min(max(self._d - k, 0), self.START_BINS - 1)

    def _begin_puzzle(self, state):
        self._d = self.distances()[state]  # _plan (in super's _begin_puzzle) reads it through _logit_extra
        super()._begin_puzzle(state)
        self._cell_counts = np.zeros((2, self.BINS, self.START_BINS))  # optimal, own moves per cell

    def observe(self, state, move, next_state, taught):
        self._moves += 1
        if taught:
            return
        distances = self.distances()
        k, after = distances[state], distances[next_state]
        b, j = self._cells(k)
        good = after == k - 1
        self._optimal[b] += good
        self._own[b] += 1
        self._cell_counts[0, b, j] += good
        self._cell_counts[1, b, j] += 1
        if not good:
            self._mistakes[b] += 1
            self._worse[b] += after == k + 1
        logit = self.skill[b] + self.start_skill[j] + math.exp(self.log_tau) * self.Z
        self._shift += _log_sigmoid(logit) if good else _log_sigmoid(-logit)

    def _end_puzzle(self):
        if self._pending:  # the history keeps the moves per cell
            self._optimal, self._own = self._cell_counts[0].ravel(), self._cell_counts[1].ravel()
        super()._end_puzzle()

    def _fit(self):
        """As Bayesian._fit, with the distance-from-start terms: MAP, each puzzle's difficulty integrated out."""
        from scipy.optimize import minimize
        optimal = np.array([o for o, _ in self.history])
        own = np.array([n for _, n in self.history])
        weight = self.decay ** np.arange(len(own) - 1, -1, -1)
        B, J, Z = self.BINS, self.START_BINS, self.Z
        gb, sb = np.repeat(np.arange(B), J), np.tile(np.arange(J), B)

        def objective(x):
            skill, start, level, log_tau = x[:B], x[B:B + J], x[B + J], x[B + J + 1]
            tau = math.exp(log_tau)
            logit = (skill[gb] + start[sb])[None, :] + tau * Z[:, None]  # (difficulty, cell)
            ll = optimal @ _log_sigmoid(logit).T + (own - optimal) @ _log_sigmoid(-logit).T + self.LOG_PZ
            top = ll.max(1, keepdims=True)
            lse = top[:, 0] + np.log(np.exp(ll - top).sum(1))
            r = weight[:, None] * np.exp(ll - lse[:, None])
            resid = r.T @ optimal - (r.T @ own) / (1 + np.exp(-logit))
            value = -weight @ lse + ((skill - level) ** 2).sum() / (2 * self.spread ** 2) \
                + (start ** 2).sum() / (2 * self.start_sd ** 2) \
                + (level - self.prior_mean) ** 2 / (2 * self.prior_sd ** 2) \
                + (log_tau - math.log(self.tau0)) ** 2 / (2 * self.tau_sd ** 2)
            cell = resid.sum(0)
            grad = np.empty(B + J + 2)
            grad[:B] = -np.bincount(gb, cell, B) + (skill - level) / self.spread ** 2
            grad[B:B + J] = -np.bincount(sb, cell, J) + start / self.start_sd ** 2
            grad[B + J] = -(skill - level).sum() / self.spread ** 2 + (level - self.prior_mean) / self.prior_sd ** 2
            grad[B + J + 1] = -tau * (resid.sum(1) @ Z) + (log_tau - math.log(self.tau0)) / self.tau_sd ** 2
            return value, grad

        x = minimize(objective, np.r_[self.skill, self.start_skill, self.level, self.log_tau], jac=True,
                     method="L-BFGS-B").x
        self.skill, self.start_skill = x[:B], x[B:B + J]
        self.level, self.log_tau = float(x[B + J]), float(x[B + J + 1])

    def _logit_extra(self, k):
        return self.start_skill[np.clip(self._d - np.asarray(k), 0, self.START_BINS - 1)]


class Bayesian2(TwoEnds, Bayesian):
    """Bayesian with the recalibrated model of the student (TwoEnds)."""

    def __init__(self, budget=0.1, **kwargs):
        super().__init__(budget, **kwargs)
        self.name = f"bayes2 {budget}"


class LearningBayesian2(TwoEnds, LearningBayesian):
    """LearningBayesian with the recalibrated model of the student (TwoEnds)."""

    def __init__(self, agency=0.0, **kwargs):
        super().__init__(agency, **kwargs)
        self.name = f"bayes-learn2 {agency:g}"


class StuckGate:
    """Wall's "when" with Bayesian's "how much", on whichever Bayesian model of the student it is mixed into
    (StuckBayes: the calibrated one, StuckBayes1: the original): it stays out until the student is stuck -
    `patience` x the puzzle's optimal length moves of its own without a new closest approach of its own - and from
    then on, wherever the student is, shows the optimal move whenever its model of the student (this puzzle's
    difficulty updated by every move the student has made on it, being stuck included) gives it less than `theta`
    chance of solving untaught from here, with the moves and patience it has left. So it leads the student only as
    far as it thinks the student needs, and steps back in at once if the student slides back, without waiting out
    another patience - until the student gets closer than ever by a move of its own, which closes the gate again.
    No budget: the spending follows from theta and patience.

    Ablation switches, each removing one part of the model (all on by default; the name says which are off):
    - `patience` 0: no gate - the solve-chance rule alone, from the first move ("p0");
    - `update` 0: the belief about this puzzle's difficulty is never updated by the student's moves on it - the
      plan still averages over the prior ("no-update");
    - `refit` n: the model of the student (skill per distance, level, tau, how mistakes go, persistence) learns
      from the first n puzzles only, then stays fixed for the rest of the session ("frozen<n>");
    - `giveup` 0: the plan has no chance of the student giving up - only the move cap ends a puzzle
      ("no-giveup");
    - `backward` 0: every mistake is sideways - the plan never has the student move away from the goal
      ("no-backward").

    counts: as Bayesian's, plus "stuck" (1 if the gate ever opened) and "best"."""

    KEEPS_ALONE = True
    KIND = "stuck-bayes"
    PATIENCE, GIVEUP = 1.0, 1  # the defaults; the name tags only settings that differ from them

    def __init__(self, theta=0.5, patience=None, update=1, refit=math.inf, giveup=None, backward=1, **kwargs):
        patience = self.PATIENCE if patience is None else patience
        giveup = self.GIVEUP if giveup is None else giveup
        if not backward:
            kwargs.update(worse=0.0, worse_strength=1e12)  # the prior share of backward mistakes, 0, never moves
        super().__init__(budget=0.0, **kwargs)  # no budget: nothing adjusts the cost (unused here)
        self.theta, self.patience = float(theta), float(patience)
        self.update, self.refit, self.giveup, self.backward = bool(update), refit, bool(giveup), bool(backward)
        self.fitted = 0  # puzzles the model of the student has learned from
        self.name = f"{self.KIND} {self.theta:g}" + (f" p{self.patience:g}" if self.patience != self.PATIENCE else "") \
            + ("" if self.update else " no-update") + (f" frozen{self.refit:g}" if self.refit != math.inf else "") \
            + ("" if self.giveup == bool(self.GIVEUP) else (" giveup" if self.giveup else " no-giveup")) \
            + ("" if self.backward else " no-backward")

    def _begin_puzzle(self, state):
        super()._begin_puzzle(state)
        self.best, self.since = self._d, 0

    def _end_puzzle(self):
        if self._pending and self.fitted >= self.refit:
            self._pending = False  # frozen: the finished puzzle teaches the model nothing
            return
        self.fitted += self._pending
        super()._end_puzzle()

    def _observe_persistence(self, result):
        if self.fitted < self.refit:
            super()._observe_persistence(result)

    def _human_rate(self, moves):
        return super()._human_rate(moves) if self.giveup else np.zeros(np.shape(moves))

    def solve_alone(self, state, step):
        """The model's chance of solving untaught from `state`, `step` moves into this attempt."""
        k, m = self.distances()[state], self._offset + step
        if k == 0:
            return 1.0
        if m >= self._cap:
            return 0.0
        return float(_softmax(self._shift) @ self._alone[m, :, min(k, self._K)])

    def wants_to_teach(self, state, step, proposal=None):
        if self.distances()[state] == 0 or self.since < self.patience * self._d:
            return False
        self.counts["stuck"] = 1
        return self.solve_alone(state, step) < self.theta

    def observe(self, state, move, next_state, taught):
        super().observe(state, move, next_state, taught)
        if not self.update:
            self._shift = self.LOG_PZ.copy()
        k = self.distances()[next_state]
        if not taught:
            self.since = 0 if k < self.best else self.since + 1
        self.best = min(self.best, k)
        self.counts["best"] = self.best


class StuckBayes(StuckGate, Bayesian2):
    """StuckGate on the calibrated model of the student (Bayesian2: distance to the goal and from the start)."""


class StuckBayes1(StuckGate, Bayesian):
    """StuckGate on the original model of the student (Bayesian: distance to the goal only, evidence kept for
    fewer puzzles). It overrates long puzzles (TwoEnds), so it needs a higher theta for the same spend - yet the
    original Bayesian taught better than the calibrated one at matched spend: does the same hold here?"""

    KIND = "stuck-bayes1"


class Bayesian3(StuckBayes1):
    """The default teacher until bayes4 (Bayesian4, the same with a filter for the model of the student): the
    original Bayesian model of the student with a plain threshold rule. On every turn
    it shows the optimal move if its model gives the student less than `theta` (0.45) chance of solving the
    puzzle untaught from here, within the moves left before the cap; otherwise the student plays. So it steps in
    only as far as it thinks the student needs, hands the puzzle back, and steps back in if the student slides.

    It is StuckBayes1 with no gate (patience 0) and no give-up model (only the move cap ends a puzzle in its plan):
    the most parsimonious teacher of the ablations (notebooks/p5_simplest_teacher.ipynb, agency 0). Its saved
    runs there are under "stuck-bayes1 0.45 p0 no-giveup", the same teacher. What it keeps, each needed:
    - skill per distance to the goal, refitted after every puzzle (older puzzles count less);
    - this puzzle's difficulty for the student, updated by every move of its own on it (without: -0.12 learning
      at matched spend, 7/8 students worse);
    - mistakes going backward as well as sideways (without: -0.15).
    The gate, the give-up model, the calibration (Bayesian2) and Bayesian's cost-benefit rule with a spending
    target were each dropped at no loss; against that rule, on the same model, it learned 0.13 ± 0.04 more.
    StuckGate's ablation switches still work, tagged against these defaults (e.g. patience=1 adds "p1")."""

    KIND = "bayes3"
    PATIENCE, GIVEUP = 0.0, 0
    DEFAULT = 0.45  # theta when the spec gives none ("bayes3")

    def __init__(self, theta=DEFAULT, **kwargs):
        super().__init__(theta, **kwargs)


class SkillFilter:
    """A normal belief over the skill logits per distance bin, carried from puzzle to puzzle: bayes3's prior
    (skill ~ N(level, spread^2), level ~ N(prior_mean, prior_sd^2), the level integrated out), each puzzle folded in
    by a Laplace approximation with its difficulty z averaged over the grid, then `q` x I + `q_shared` x 11^T added to
    the covariance (independent and shared drift: the student may have changed). With `trend_sd` > 0 the belief
    also covers a learning rate v (logits per puzzle, every distance alike): skill moves by v between puzzles, and
    v by noise of variance `trend_q` - a Kalman filter with a local linear trend. Used by Bayesian4, and tested
    offline against bayes3's refit in scripts/skill_tracking.py."""

    B, Z, LOG_PZ = Bayesian.BINS, Bayesian.Z, Bayesian.LOG_PZ

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


class Bayesian4(Bayesian3):
    """The default teacher: bayes3 with its model of the student carried from puzzle to puzzle by a filter, rather
    than refitted after every puzzle from the stored history. The deciding rule is bayes3's (theta 0.45).

    Skill per distance is a normal belief (SkillFilter: bayes3's hierarchical prior, the overall level integrated
    out), and each finished puzzle's own moves are folded into it by a Laplace approximation; nothing else about
    skill is stored. tau (how much puzzles' difficulty varies for the student) is learned by a bank of filters, one
    per tau in TAUS, each weighed by how well it has predicted the student's moves (log marginal likelihood, the
    sums forgetting by `forget` per puzzle) and by bayes3's prior on tau. The plan and the within-puzzle belief use
    the best-weighted filter: its tau and its mean skill, as bayes3 uses its MAP fit. `q` > 0 widens every filter
    between puzzles (the student may have changed); 0 (no drift) predicted best offline. How mistakes go (forward,
    sideways or backward) is counted as in bayes3, with `decay`.

    Offline (scripts/skill_tracking.py, 8 unhelped students x 2160 puzzles at agency 0): log loss per move 0.6588,
    as bayes3's refit (0.6587), better calibrated, about 10x faster per puzzle than the refit."""

    KIND = "bayes4"
    TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2)

    def __init__(self, theta=Bayesian3.DEFAULT, forget=0.99, q=0.0, **kwargs):
        super().__init__(theta, **kwargs)
        self.forget, self.q = float(forget), float(q)
        self.history = deque(maxlen=0)  # nothing stored: the filters carry the evidence
        self.filters = [SkillFilter(tau=t, q=self.q, prior_mean=self.prior_mean, prior_sd=self.prior_sd,
                                    spread=self.spread) for t in self.TAUS]
        self.tau_prior = -(np.log(self.TAUS) - math.log(self.tau0)) ** 2 / (2 * self.tau_sd ** 2)
        self.tau_loglik = np.zeros(len(self.TAUS))
        self._pick()
        self.name += (f" forget{self.forget:g}" if self.forget != 0.99 else "") + (f" q{self.q:g}" if self.q else "")

    def _pick(self):
        best = self.filters[int(np.argmax(self.tau_prior + self.tau_loglik))]
        self.skill, self.log_tau = best.mu.copy(), math.log(best.tau)

    def _fit(self):
        ll = np.array([f.update(self._optimal, self._own) for f in self.filters])
        self.tau_loglik = self.forget * self.tau_loglik + ll
        self._pick()


class TimedBayes3(Bayesian3):
    """bayes3 whose theta changes once during the session: `theta` for its first `switch` puzzles, `late` from
    then on (1080: half of a 72-trial session at multiplier 30). A test of whether the best amount of help depends
    on time, which a single theta can't express. counts: as bayes3's, plus "theta" (this puzzle's)."""

    KIND = "bayes3-time"
    SWITCH = 1080

    def __init__(self, theta=0.9, late=0.25, switch=SWITCH, **kwargs):
        super().__init__(theta, **kwargs)
        self.early, self.late, self.switch, self.puzzles = float(theta), float(late), int(switch), 0
        self.name = f"{self.KIND} {self.early:g} then {self.late:g}" + (
            f" at {self.switch}" if self.switch != self.SWITCH else "")

    def _begin_puzzle(self, state):
        self.puzzles += 1
        self.theta = self.early if self.puzzles <= self.switch else self.late
        self.counts["theta"] = self.theta
        super()._begin_puzzle(state)


class LengthBayes3(Bayesian3):
    """bayes3 with one theta for short puzzles and another for long: `theta` below an optimal length of `split`
    moves, `long` from there (18: the 9-11 and 14-16 bins against 19-21 and 24-26). A test of whether the best
    amount of help depends on the puzzle's length, which a single theta can't express. counts: as bayes3's, plus
    "theta" (this puzzle's)."""

    KIND = "bayes3-length"
    SPLIT = 18

    def __init__(self, theta=0.25, long=0.9, split=SPLIT, **kwargs):
        super().__init__(theta, **kwargs)
        self.short, self.long, self.split = float(theta), float(long), int(split)
        self.name = f"{self.KIND} {self.short:g} short {self.long:g} long" + (
            f" at {self.split}" if self.split != self.SPLIT else "")

    def _begin_puzzle(self, state):
        self.theta = self.short if self.distances()[state] < self.split else self.long
        self.counts["theta"] = self.theta
        super()._begin_puzzle(state)


class HintBayes(StuckBayes):
    """StuckBayes as hint and hand back: the same gate (stuck for `patience` x the puzzle's optimal length moves of
    its own) and the same help (the optimal move while its model gives the student under `theta` chance of solving
    alone), but each rescue is one burst. Once the chance is back at `theta` or above - or `k` moves have been
    shown in a row - it hands the puzzle back and the gate closes: the student has to be stuck that long again
    before more help, even if it slides back at once (where StuckBayes steps straight back in). With k = inf the
    two differ only in that; a finite k also caps every burst. theta 1 with k = inf would be take-over.

    counts: as StuckBayes's, plus "hints" (bursts started)."""

    def __init__(self, theta=0.5, k=math.inf, patience=1.0, **kwargs):
        super().__init__(theta, patience, **kwargs)
        self.k = k
        self.name = f"hint-bayes {self.theta:g}" + (f" k{self.k:g}" if self.k != math.inf else "") \
            + (f" p{self.patience:g}" if self.patience != 1 else "")

    def _begin_puzzle(self, state):
        super()._begin_puzzle(state)
        self.burst = None  # moves shown in the current hint, None outside one

    def wants_to_teach(self, state, step, proposal=None):
        if self.distances()[state] == 0:
            return False
        if self.burst is None:  # the gate: stuck, and the model says the student needs help
            if self.since < self.patience * self._d:
                return False
            self.counts["stuck"] = 1
            if self.solve_alone(state, step) >= self.theta:
                return False
            self.burst = 0
            self.counts["hints"] += 1
        elif self.burst >= self.k or self.solve_alone(state, step) >= self.theta:
            self.burst, self.since = None, 0  # handed back: a full patience before the next hint
            return False
        self.burst += 1
        return True
