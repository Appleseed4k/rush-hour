"""Loader for the Olieslagers et al. (2023) Rush Hour experiment (OSF 23vrb).

The archive ships no puzzle layouts (data/raw_data/problems/ is empty, also on
OSF), so only what the raw event log itself reveals is available: per attempt,
the sequence of slides (piece, start/end top-left cell, timestamps). Boards
cannot be rebuilt from it; matching against data/rush_nw.txt found no puzzle
that fits, so the experiment's puzzles are not in that pool.
"""
import collections
import csv
import os

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "23vrb-osfstorage-archive")
RAW = os.path.join(ARCHIVE, "data", "raw_data", "rush_hour", "all_subjects.csv")
RED = 8  # the red car's piece id in the raw data


class Move:
    """One completed slide: piece, start/end top-left cell, and timestamps (ms)
    of the previous move's end (or the attempt start), the first touch of this
    move, and the slide's own drag start and end."""
    __slots__ = ("piece", "c0", "c1", "t_prev", "t_touch", "t_drag", "t_end")

    def __init__(self, piece, c0, c1, t_prev, t_touch, t_drag, t_end):
        self.piece, self.c0, self.c1 = piece, c0, c1
        self.t_prev, self.t_touch, self.t_drag, self.t_end = t_prev, t_touch, t_drag, t_end

    @property
    def vertical(self):
        return (self.c1 - self.c0) % 6 == 0

    @property
    def amount(self):
        """Signed cells slid along the piece's axis (+ is right/down)."""
        d = self.c1 - self.c0
        return d // 6 if d % 6 == 0 else d


def read_events(path=RAW):
    """{subject: [row, ...]} in file order, rows as dicts with int fields."""
    by_subject = collections.defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            for key in ("move", "t", "piece", "target"):
                row[key] = int(row[key])
            by_subject[row["subject"]].append(row)
    return by_subject


def keep_subject(rows):
    """pre_process.jl's filter_subjects: solved at least one puzzle, spent >= 60
    minutes or saw all 70 puzzles, and no timestamp ran backwards."""
    times = [r["t"] for r in rows if r["t"] >= 0]
    minutes = [(t - times[0]) / 60000 for t in times]
    ok_span = minutes[-1] - minutes[0] >= 60 or len({r["instance"] for r in rows}) == 70
    return ok_span and any(r["event"] == "win" for r in rows) and min(b - a for a, b in zip(minutes, minutes[1:])) >= 0


def attempts(rows):
    """Yields (instance, attempt_index, solved, [Move, ...]) per attempt.

    A move is the last drag of a move number whose piece actually changed cell
    (the number counts completed slides; earlier drags under it are no-op
    touches). An attempt starts at each 'start' event and ends at 'win'."""
    by_instance = collections.OrderedDict()
    for r in rows:
        by_instance.setdefault(r["instance"], []).append(r)
    for instance, rs in by_instance.items():
        if rs[-1]["event"] != "win" and not any(r["event"] == "win" for r in rs):
            continue  # pre_process.jl drops puzzles the subject never solved
        attempt, current, solved, drag, first_touch, t_prev = 0, None, False, None, {}, None
        for r in rs:
            ev = r["event"]
            if ev == "start":
                if current is not None:
                    yield instance, attempt, solved, current
                attempt += 1
                current, solved, drag, first_touch, t_prev = [], False, None, {}, r["t"]
            elif current is None:
                continue
            elif ev == "drag_start":
                drag = r
                first_touch.setdefault(r["move"], r["t"])
            elif ev == "drag_end" and drag is not None:
                if r["target"] != drag["target"] and r["move"] == drag["move"]:
                    current.append(Move(r["piece"], drag["target"], r["target"], t_prev,
                                        first_touch.get(r["move"], drag["t"]), drag["t"], r["t"]))
                    t_prev = r["t"]
                drag = None
            elif ev == "win":
                solved = True
                yield instance, attempt, solved, current
                current = None
        if current is not None:
            yield instance, attempt, solved, current


def load(path=RAW, filtered=True):
    """{subject: [(instance, attempt, solved, moves), ...]}"""
    out = {}
    for subject, rows in read_events(path).items():
        if filtered and not keep_subject(rows):
            continue
        out[subject] = list(attempts(rows))
    return out


def optimal_length(instance):
    """The puzzle's optimal solution length, from its name (suffix = length + 2)."""
    return int(instance.rsplit("_", 1)[1]) - 2
