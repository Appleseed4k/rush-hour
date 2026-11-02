"""Loader for the Olieslagers et al. (2023) Rush Hour experiment (OSF 23vrb).

The OSF archive ships no puzzle layouts (data/raw_data/rush_hour/problems/ is
empty there); the 70 layouts come from the authors' GitHub repository
(github.com/WeiJiMaLab/rush_hour, data/raw_data/problems/) and go in that
folder. puzzle() turns one into this repo's state format. They are not in
data/rush_nw.txt. Checked: every board is valid, and 117,130 of the 117,145
recorded slides replay as legal moves (the rest are drags off the board).

Lengths: the paper counts a puzzle solved once the red car's path to the
exit is clear; this repo (lib.find_goal_states) once the red car reaches the
edge - one move more. optimal_length() is the paper's (instance name's
suffix - 2), distance() this repo's (+1, which breadth-first search confirms
for all 70 puzzles). Compare move counts with distance(): trials() counts
the final slide to the exit as a move.
"""
import collections
import csv
import os

ARCHIVE = os.path.join(os.path.dirname(__file__), "..", "23vrb-osfstorage-archive")
RAW = os.path.join(ARCHIVE, "data", "raw_data", "rush_hour", "all_subjects.csv")
PROBLEMS = os.path.join(ARCHIVE, "data", "raw_data", "rush_hour", "problems")
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
    """The puzzle's optimal solution length as the paper counts it (until the
    red car's path is clear), from its name (suffix = length + 2)."""
    return int(instance.rsplit("_", 1)[1]) - 2


def distance(instance):
    """The puzzle's optimal solution length as this repo counts it (until the
    red car reaches the exit): the paper's + 1."""
    return optimal_length(instance) + 1


def puzzle(instance, folder=PROBLEMS):
    """The puzzle's starting board in this repo's state format (as
    lib.parse_puzzle_line gives): ("red", cells) first, then every other car
    as (name, cells) sorted by name - names are the layout's car ids."""
    import json
    with open(os.path.join(folder, instance + ".json")) as f:
        data = json.load(f)
    cars = []
    for car in data["cars"]:
        row, col = divmod(car["position"], 6)
        dr, dc = (0, 1) if car["orientation"] == "horizontal" else (1, 0)
        cells = tuple((row + dr * k, col + dc * k) for k in range(car["length"]))
        cars.append(("red" if car["id"] == "r" else car["id"], cells))
    red = [c for c in cars if c[0] == "red"]
    return tuple(red) + tuple(sorted(c for c in cars if c[0] != "red"))


def puzzles(folder=PROBLEMS):
    """{instance: starting board} for every layout in `folder`."""
    return {f[:-5]: puzzle(f[:-5], folder) for f in sorted(os.listdir(folder)) if f.endswith(".json")}


# --- per-puzzle summaries, for learning curves --------------------------------------------------

def trials(path=RAW, filtered=True):
    """One row per (participant, puzzle), in play order, as a pandas DataFrame:
    subject, instance, trial (1 = first puzzle played), optimal (the paper's
    length), distance (this repo's, one more - use it with moves),
    solved, surrendered, attempts (1 + restarts), moves (over all attempts,
    counted as pre_process.jl does: a drag_end whose move number the next
    event doesn't share), seconds. Unsolved puzzles are kept. Puzzles shown
    for an instant at the start and never played are left out."""
    import pandas as pd
    out = []
    for subject, rows in read_events(path).items():
        if filtered and not keep_subject(rows):
            continue
        by_instance = collections.OrderedDict()
        for r in rows:
            if r["event"] not in ("BONUS_FAIL", "BONUS_SUCCESS"):
                by_instance.setdefault(r["instance"], []).append(r)
        for instance, rs in by_instance.items():
            events = [r for r in rs if r["event"] in ("start", "restart", "drag_end", "win")]
            moves = sum(1 for k, r in enumerate(events)
                        if r["event"] == "drag_end" and (k + 1 == len(events) or events[k + 1]["move"] != r["move"]))
            seconds = (max(r["t"] for r in rs) - min(r["t"] for r in rs)) / 1000
            if moves == 0 and seconds == 0:
                continue
            kinds = {r["event"] for r in rs}
            out.append({"subject": subject, "instance": instance, "t0": min(r["t"] for r in rs),
                        "optimal": optimal_length(instance), "distance": distance(instance),
                        "solved": "win" in kinds,
                        "surrendered": "surrender" in kinds,
                        "attempts": 1 + sum(r["event"] == "restart" for r in rs), "moves": moves,
                        "seconds": seconds})
    df = pd.DataFrame(out)
    df["trial"] = df.groupby("subject").t0.rank(method="first").astype(int)
    return df.sort_values(["subject", "trial"]).drop(columns="t0").reset_index(drop=True)


def trend(df, y, x="trial", person="subject", puzzle="instance", per=10, iterations=100):
    """Within-person, within-puzzle trend of column `y` in `x` - how much `y`
    changes per `per` trials for the same person, comparing like puzzles
    with like - as (estimate, standard error clustered by person). Person
    and puzzle effects are removed by alternating demeaning, so comparing
    early and late trials isn't confounded by who plays more puzzles or by
    which puzzles came when. Works for simulated students too (person =
    student, puzzle = the session's puzzle index)."""
    import numpy as np
    Y, X = df[y].astype(float), df[x].astype(float)
    for _ in range(iterations):
        for g in (df[person], df[puzzle]):
            Y = Y - Y.groupby(g).transform("mean")
            X = X - X.groupby(g).transform("mean")
    b = (X * Y).sum() / (X * X).sum()
    score = (X * (Y - b * X)).groupby(df[person]).sum()
    return per * b, per * float(np.sqrt((score ** 2).sum())) / (X * X).sum()


SURRENDER_EDGES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 10)  # moves so far / optimal length


def surrender_hazard(trials_df=None, edges=SURRENDER_EDGES):
    """People's chance of giving up after a move, by how many moves they have
    made on the puzzle so far relative to its optimal length (this repo's
    distance, which counts the final slide out, as the moves do; all attempts
    at it counted): per bin [edges[i], edges[i + 1]) (the last bin open
    ended), surrenders / moves made in that bin. Puzzles solved count as
    moves at risk with no surrender. Returns (edges, rates, counts) -
    counts as (surrenders, moves) per bin. Pooled over puzzle lengths: the
    shape is similar across them, and longer puzzles' higher surrender rates
    come from the extra moves."""
    import numpy as np
    df = trials() if trials_df is None else trials_df
    bounds = list(edges) + [np.inf]
    counts = []
    for a, b in zip(bounds, bounds[1:]):
        at_risk = surrenders = 0
        for moves, d, gave_up in zip(df.moves, df.distance, df.surrendered):
            k = np.arange(1, moves + 1) / d
            at_risk += int(((k >= a) & (k < b)).sum())
            surrenders += bool(gave_up and moves > 0 and a <= moves / d < b)
        counts.append((surrenders, at_risk))
    rates = [s / n if n else 0.0 for s, n in counts]
    return list(edges), rates, counts
