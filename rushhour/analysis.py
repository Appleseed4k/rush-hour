"""Helpers the notebooks share: launching session runs (sessions.py) in
the background, and reading their saved results.

A teacher's entry in a run is {"config", "sessions", "results"}: sessions[student] is one record per puzzle
played, results[student][n] the test attempts after n puzzles. Test score: moves / optimal with an unsolved
attempt counted at its move cap (lower is better); learning is the drop in that score over the session."""
import os

import numpy as np

from . import paths, run_queue as rq, sessions as se

OUT = se.OUT
NEGATABLE = {"even_batches"}  # on by default in sessions: False must be passed as --no-even-batches


def argv(settings):
    """sessions' command line (after the run name) for these settings."""
    out = []
    for k, v in settings.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else ([f"--no-{flag[2:]}"] if k in NEGATABLE else [])
        elif isinstance(v, list):
            out += [flag, *map(str, v)]
        else:
            out += [flag, str(v)]
    return out


def launch(queue, teachers, run, settings):
    """Plays `teachers` (specs) in `run` in the background (run_queue); does nothing for teachers already saved."""
    here = os.getcwd()
    os.chdir(paths.ROOT)  # the runs resolve relative paths (--init) from the repository root
    try:
        rq.launch(queue, [(run, argv(settings) + ["--teachers", *teachers])])
    finally:
        os.chdir(here)


def watch(queue):
    if os.path.exists(os.path.join(OUT, "queues", f"{queue}.json")):
        rq.watch(queue)  # refreshes every 30 s until the queue stops
    else:
        print(f"{queue} hasn't been launched")


def entry(run, spec):
    """A teacher's entry in a run, or None until it is saved."""
    if not os.path.exists(os.path.join(OUT, run, "config.json")):
        return None
    loaded = se.load_session(run, OUT)
    found = loaded["teachers"].get(se.teacher_name(spec))
    if not found or not found["sessions"] or not found["results"]:
        return None
    return {"config": loaded["config"], **found}


def measures(attempts, config):
    """(score, solve rate, moves / optimal over solved attempts) of test attempts; score: moves / optimal with an
    unsolved attempt counted at its move cap (lower is better)."""
    score = [(a["moves"] if a["solved"] else se.step_cap(a["distance"], config)) / a["distance"] for a in attempts]
    solved = [a["moves"] / a["distance"] for a in attempts if a["solved"]]
    return np.mean(score), np.mean([a["solved"] for a in attempts]), np.mean(solved) if solved else np.nan


def final(e):
    """{student: (score, solve rate, moves / optimal over solved)} at the end of the session."""
    return {s: measures(by_n[max(by_n)], e["config"]) for s, by_n in e["results"].items()}


def spend(e):
    """Share of the session's moves the teacher showed."""
    records = [x for record in e["sessions"].values() for x in record]
    return sum(x["taught"] for x in records) / sum(x["moves"] for x in records)

