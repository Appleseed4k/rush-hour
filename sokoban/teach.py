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
import random
from collections import Counter

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


TEACHERS = {"random": Random}


def make_teacher(spec):
    """A teacher from "kind:parameter" (e.g. "random:0.1"); "none" is None."""
    if spec in (None, "", "none"):
        return None
    kind, _, arg = spec.partition(":")
    if kind not in TEACHERS:
        raise ValueError(f"unknown teacher {kind!r}; kinds: {', '.join(TEACHERS)}")
    return TEACHERS[kind](float(arg) if arg else 1.0)
