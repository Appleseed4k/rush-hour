"""Phase 2 (archived): the teachers of the expert-iteration experiments (notebooks/old/p2_teach.ipynb), moved
out of teach.py. teach_experiment_rounds.py registers correct-after and correct-before again; the session
experiments no longer offer them (teach_experiment.RETIRED keeps their saved names).

- CorrectMistakes ("correct-before") steps in only when the student is about to go wrong - which needs to see the
  move the student intends, an idealization;
- CorrectAfterMistake ("correct-after"), more realistically, lets a wrong move stand and shows the best move from
  where it led. Each notices every mistake but acts on it with probability q (q = 1: every one, without drawing
  from the attempt's random numbers), as Random(p) acts on each turn with probability p.
- Budgeted(teacher, k) caps any teacher at k interventions per attempt.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # the shared scripts
from teach import OPPOSITE, Teacher


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
