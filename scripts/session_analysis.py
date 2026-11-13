"""Helpers the session notebooks share (p5_simplest_teacher, p6_hindsight): launching session_experiment runs in
the background, and reading their saved results.

A teacher's entry in a run is {"config", "sessions", "results"}: sessions[student] is one record per puzzle
played, results[student][n] the test attempts after n puzzles. Test score: moves / optimal with an unsolved
attempt counted at its move cap (lower is better); learning is the drop in that score over the session."""
import os

import numpy as np
import pandas as pd

import run_queue as rq
import session_experiment as se
import teach_experiment as te

OUT = se.OUT
HERE = os.path.dirname(os.path.abspath(__file__))
NEGATABLE = {"even_batches"}  # on by default in session_experiment: False must be passed as --no-even-batches


def argv(settings):
    """session_experiment's command line (after the run name) for these settings."""
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
    os.chdir(HERE)  # the runs resolve relative paths (--init) from scripts/
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
    score = [(a["moves"] if a["solved"] else te.step_cap(a["distance"], config)) / a["distance"] for a in attempts]
    solved = [a["moves"] / a["distance"] for a in attempts if a["solved"]]
    return np.mean(score), np.mean([a["solved"] for a in attempts]), np.mean(solved) if solved else np.nan


def curve(e):
    """[(puzzles played, (mean test score, standard error across students))] along the session."""
    points = sorted({n for by_n in e["results"].values() for n in by_n})
    return [(n, te.mean_se([measures(by_n[n], e["config"])[0] for by_n in e["results"].values() if n in by_n]))
            for n in points]


def final(e):
    """{student: (score, solve rate, moves / optimal over solved)} at the end of the session."""
    return {s: measures(by_n[max(by_n)], e["config"]) for s, by_n in e["results"].items()}


def learning(e):
    """Each student's test score gain over the session (all students start from the same network)."""
    start = measures(e["results"][0][0], e["config"])[0]
    return [start - v[0] for v in final(e).values()]


def gains(e):
    """{student: test score gain over the session}."""
    start = measures(e["results"][0][0], e["config"])[0]
    return {s: start - v[0] for s, v in final(e).items()}


def spend(e):
    """Share of the session's moves the teacher showed."""
    records = [x for record in e["sessions"].values() for x in record]
    return sum(x["taught"] for x in records) / sum(x["moves"] for x in records)


def paired(e, base):
    """End score minus `base`'s, student by student: "mean ± se (k/n better)"; below 0: e's students learned more."""
    end, b = final(e), final(base)
    d = [end[s][0] - b[s][0] for s in end if s in b]
    m, s = te.mean_se(d)
    return f"{m:+.3f} ± {s:.3f} ({sum(x < 0 for x in d)}/{len(d)})"


def teacher_table(specs, against, run):
    """Spend, solves, how help was given, learning, and the end score against each of `against` (specs), paired
    by student."""
    rows = []
    for spec in specs:
        e = entry(run, spec)
        if e is None:
            print(f"{se.teacher_name(spec)}: not saved yet")
            continue
        records = [x for record in e["sessions"].values() for x in record]
        helped = [x for x in records if x["taught"] > 0]
        m, s = te.mean_se(learning(e))
        row = {"teacher": se.teacher_name(spec),
               "shown": spend(e),
               "solved as played": np.mean([x["solved"] for x in records]),
               "helped puzzles": len(helped) / len(records),
               "shown per helped": np.mean([x["taught"] for x in helped]) if helped else np.nan,
               "helped solved": np.mean([x["solved"] for x in helped]) if helped else np.nan,
               "own moves on helped": np.mean([x["moves"] - x["taught"] for x in helped]) if helped else np.nan,
               "learning": f"{m:+.3f} ± {s:.3f}"}
        for other in against:
            base = entry(run, other)
            row[f"vs {se.teacher_name(other)}"] = paired(e, base) if base is not None and other != spec else ""
        rows.append(row)
    return pd.DataFrame(rows).round(3)
