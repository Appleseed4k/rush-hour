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
CorrectAfterMistake, more realistically, lets a wrong move stand, points it
out, and shows the best move from where it led; FlagMistakes only points
mistakes out, without showing anything. Each notices every mistake but acts
on it with probability q (q = 1: every one, without drawing from the
attempt's random numbers), as Random(p) acts on each turn with probability
p. Budgeted(teacher, k) caps any teacher at k interventions per attempt.

Pointing out a mistake tells the student that move was wrong on that board -
nothing more: it isn't told what was right there, and a move that isn't
pointed out means nothing (it isn't approved). Each intervention - a move
shown, or a mistake pointed out - is one unit of the teaching budget; a
mistake pointed out together with the move shown right after it counts once.

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
  Returns True to point `move` out as a mistake, else False.
- end_attempt(result): the attempt is over (rl.solve_intuition's result:
  boards, moves, which moves were taught, whether it solved). Where an
  adaptive teacher would learn from how the student did; a no-op here.

`counts` (a Counter) tallies whatever a teacher wants reported; rl sums it
over a round's attempts into expert_iteration's history.
"""
import random
from collections import Counter

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
        return False

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
    optimal, with probability `q`, it points the mistake out and shows the
    optimal move from the board the mistake led to, on the student's next
    turn - a tutor who can only react to what was played; otherwise it says
    nothing. Pointing out and showing count as one intervention.

    counts: mistakes noticed, as "mistakes_worse" (the mistake took the board
    a move further from the goal) or "mistakes_sideways" (same distance);
    "pointed_out"; and per correction, "undo_optimal" (taking the mistake
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
            return False
        self.counts[f"mistakes_{kind}"] += 1
        if not _acts(self):
            return False
        self._mistake = (move, next_state)
        self.counts["pointed_out"] += 1
        return True

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


class FlagMistakes(Teacher):
    """Notices every move of the student's that wasn't optimal and points it
    out, right after it is played, with probability `q`. It never shows a
    move, so the student has to find the right one itself. Each mistake
    pointed out is one intervention; mistakes noticed but not pointed out
    cost nothing and teach nothing.

    counts: mistakes noticed, as "mistakes_worse" / "mistakes_sideways" (see
    CorrectAfterMistake), and "pointed_out"."""

    def __init__(self, q=1.0):
        super().__init__()
        self.q = q
        self.name = "flag mistakes" if q >= 1 else f"flag mistakes {q}"

    def wants_to_teach(self, state, step, proposal=None):
        return False

    def observe(self, state, move, next_state, taught):
        kind = None if taught else self.mistake_kind(state, next_state)
        if kind is None:
            return False
        self.counts[f"mistakes_{kind}"] += 1
        if not _acts(self):
            return False
        self.counts["pointed_out"] += 1
        return True


class Budgeted(Teacher):
    """`teacher`, allowed at most `k` interventions per attempt - after that it
    stays silent for the rest of the attempt. Interventions are counted as
    in the teaching budget: each move shown and each mistake pointed out is
    one, except that a mistake pointed out and the move shown right after it
    count once together. The wrapped teacher still notices everything (its
    counts keep tallying mistakes), but nothing it would say past the budget
    reaches the student."""

    def __init__(self, teacher, k):
        self.teacher = teacher
        self.k = k
        self.name = f"{teacher.name} budget {k}"
        self.spent = 0
        self._flagged_last = False

    @property
    def counts(self):
        return self.teacher.counts

    @counts.setter
    def counts(self, value):
        self.teacher.counts = value

    def start(self, state, rng):
        self.teacher.start(state, rng)
        self.spent = 0
        self._flagged_last = False

    def advise(self, state, step, proposal=None):
        paired = self._flagged_last  # a correction right after a mistake pointed out is already paid for
        self._flagged_last = False
        if not paired and self.spent >= self.k:
            return None
        move = self.teacher.advise(state, step, proposal)
        if move is not None and not paired:
            self.spent += 1
        return move

    def observe(self, state, move, next_state, taught):
        flagged = self.teacher.observe(state, move, next_state, taught)
        if not flagged or self.spent >= self.k:
            return False
        self.spent += 1
        self._flagged_last = True
        return True

    def end_attempt(self, result):
        self.teacher.end_attempt(result)
