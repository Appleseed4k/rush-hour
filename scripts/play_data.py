"""Loader for games recorded by the Rush Hour play page (web/).

The page saves each game to its artifact's store; its owner's Export button
saves all of them as one JSON file:

    {"version", "exportedAt", "players": [{"player", "lastSession", "lastSeen"}],
     "games": [{"player", "id", "session", "teacher", "number", "poolIndex",
                "board", "distance", "startedAt", "endedAt", "outcome",
                "moveCount", "attempts", "moves": [{"car", "fromRow", "fromCol",
                "toRow", "toCol", "t", "taught"}], "restarts": [{"t", "afterMove"}],
                "checks": [{"t", "afterMove", "distance", "p", "advantage", "shown"}],
                "beliefs": {"a", "b"}, "delay", "delayChanges": [{"t", "afterMove", "seconds"}]}]}

`board` is the puzzle's rush_nw.txt bitboard and `distance` this repo's length
(the red car reaches the edge). The page counts a puzzle solved when the red
car reaches the edge, and the move that does so is in `moves`, so moves /
distance compares with human_data.trials() directly. Times `t` are ms since
the game started. outcome is solved, gave_up, left (moved on unsolved - the
page offers no way to, so this shouldn't occur) or playing (the tab closed
mid-game, or the last save was lost).

teacher is the page's teacher: "none", or "delay:<seconds>" - after that long
without a move the page makes an optimal move for the player, recorded in
`moves` with taught: true (the key is absent on the player's own moves), or
"bayes:<seconds>[:<name>=<value>,...]" - after that long, a Bayesian teacher
decides whether to make the move: `checks` has each decision (the distance to
the goal then, its posterior mean p of the chance the player's own move is
optimal, the gain in their chance of solving it predicted from one move shown,
and whether it showed one), `beliefs` its Beta(a, b) posterior when the game
started (null for other teachers). Games saved before there was a teacher
(version 1) have no teacher key: "none".

From version 3 the Bayesian teacher is "bayes3:<seconds>[:<name>=<value>,...]"
(scripts/teach.py's Bayesian3, ported): after that long without a move it makes
the move if its model gives the player under theta chance of solving alone.
Its checks are {"t", "afterMove", "distance", "delay", "solveAlone", "shift",
"shown"} - the chance it gave, and its mean belief of how much harder (below
0) or easier the puzzle is for the player, in logits - and beliefs is
{"skill" (a logit per distance 1..11+), "level", "tau", "worse" (the share of
mistakes going backward, per distance), "puzzles"} at the game's start. Every
game records `delay`, the teacher's wait in seconds at its start (null: no
teacher), and `delayChanges`, when the owner's testing slider changed it.
"""
import json
import os

from lib import car_orientations, neighbors, parse_puzzle_line


def load(path):
    """The export's games, sorted by player then start time. `path` may be a
    list of exports - one per published page, e.g. one per teacher condition
    (a player's id is the same on every page of one owner's)."""
    games = []
    for one in [path] if isinstance(path, (str, os.PathLike)) else path:
        with open(one) as f:
            games += json.load(f)["games"]
    return sorted(games, key=lambda g: (g["player"], g["startedAt"]))


def puzzle(board):
    """The game's starting board in this repo's state format (as
    lib.parse_puzzle_line gives: ("red", cells), then the other cars by
    letter)."""
    return parse_puzzle_line(f"0 {board} 0")[1]


def replay(game):
    """The board after each move, as a list of states - the starting board
    first, a restart returning to it. Checks every move is a legal slide."""
    state = start = puzzle(game["board"])
    orientations = car_orientations(start)
    restarts = sorted(r["afterMove"] for r in game.get("restarts", []))
    states = [start]
    for k, m in enumerate(game["moves"]):
        if k in restarts:
            state = start
        name = "red" if m["car"] == "A" else m["car"]
        dr, dc = m["toRow"] - m["fromRow"], m["toCol"] - m["fromCol"]
        cells = dict(state)[name]
        assert cells[0] == (m["fromRow"], m["fromCol"]), f"game {game['id']} move {k}: {name} isn't there"
        moved = tuple((r + dr, c + dc) for r, c in cells)
        nxt = next((s for _, s in neighbors(state, orientations) if dict(s)[name] == moved), None)
        assert nxt is not None, f"game {game['id']} move {k}: illegal slide of {name}"
        state = nxt
        states.append(state)
    return states


def trials(path, finished=True):
    """One row per game, in play order, as a pandas DataFrame with
    human_data.trials()'s columns, so its trend() and surrender_hazard() work
    on it: subject (the player's id), instance (the board), trial (1 = the
    player's first game, over all their sessions), optimal (distance - 1,
    the paper's convention), distance, solved, surrendered, attempts (1 +
    restarts), moves (over all attempts, the final slide out included,
    the teacher's too), seconds; plus session, poolIndex, outcome, teacher,
    taught (moves the teacher made) and checks (times a Bayesian teacher
    decided whether to show a move). Games never touched (no
    moves and not given up) are left out. finished=True also drops games
    that were neither solved nor given up, after numbering trials, so trial
    still counts every game the player saw."""
    import pandas as pd
    out = []
    for g in load(path):
        if not g["moves"] and g["outcome"] != "gave_up":
            continue
        t0 = pd.Timestamp(g["startedAt"])
        if g.get("endedAt"):
            seconds = (pd.Timestamp(g["endedAt"]) - t0).total_seconds()
        else:
            seconds = g["moves"][-1]["t"] / 1000
        out.append({"subject": g["player"], "instance": g["board"], "t0": t0,
                    "optimal": g["distance"] - 1, "distance": g["distance"],
                    "solved": g["outcome"] == "solved", "surrendered": g["outcome"] == "gave_up",
                    "attempts": 1 + len(g.get("restarts", [])), "moves": len(g["moves"]),
                    "seconds": seconds, "session": g["session"], "poolIndex": g["poolIndex"],
                    "outcome": g["outcome"], "teacher": g.get("teacher", "none"),
                    "taught": sum(bool(m.get("taught")) for m in g["moves"]),
                    "checks": len(g.get("checks") or [])})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["trial"] = df.groupby("subject").t0.rank(method="first").astype(int)
    if finished:
        df = df[df.outcome.isin(["solved", "gave_up"])]
    return df.sort_values(["subject", "trial"]).drop(columns="t0").reset_index(drop=True)
