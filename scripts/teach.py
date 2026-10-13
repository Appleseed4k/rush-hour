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
the best move from where it led.

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


class CorrectMistakes(Teacher):
    """Lets the student play, but steps in whenever the move it is about to make
    isn't optimal (doesn't bring the board a move closer to the goal) and
    shows an optimal one instead - a tutor correcting errors. It sees the
    move the student intends before it is played, an idealization (see
    CorrectAfterMistake). The student acts on its own boards, as in DAgger,
    yet every mistake is corrected, so every attempt is solved optimally;
    how much it teaches falls as the student improves."""
    name = "correct mistakes"

    def wants_to_teach(self, state, step, proposal=None):
        if proposal is None:
            raise ValueError("CorrectMistakes needs the student's proposed move")
        wrong = not self.is_optimal(state, proposal())
        self.counts["corrections"] += wrong
        return wrong


class CorrectAfterMistake(Teacher):
    """Lets every move the student makes stand. After a move that wasn't
    optimal, it shows the optimal move from the board the mistake led to, on
    the student's next turn - a tutor who can only react to what was played.

    counts per correction: "mistakes_worse" (the mistake took the board a move
    further from the goal) or "mistakes_sideways" (same distance);
    "undo_optimal" (taking the mistake back is an optimal move - exactly when
    it was worse); "shown_undo" (the move shown was that undo - the teacher
    picks among tied optimal moves at random)."""
    name = "correct after mistake"

    def start(self, state, rng):
        super().start(state, rng)
        self._mistake = None

    def observe(self, state, move, next_state, taught):
        distances = self.distances()
        if taught or distances.get(next_state) == distances[state] - 1:
            return
        self._mistake = (move, next_state)
        worse = distances.get(next_state) == distances[state] + 1
        self.counts["mistakes_worse" if worse else "mistakes_sideways"] += 1

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
