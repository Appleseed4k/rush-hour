"""What a teacher's help does to the student's hindsight labels, at each hindsight window.

Run from scripts/:  python hindsight_teacher_labels.py [puzzles] [students] [processes]
Replays the first `puzzles` puzzles of teach_shortcut_agency1's sessions (its settings and seeds, learning as in the
run, so the teacher acts as it did there) for each teacher below, records every attempt, and labels the solved ones
with each label variant of hindsight_labels.py (own, W = 1, 3, 5, 10, inf). Every third puzzle is scored against
exact distances (multi_bfs), which is the slow part.

Per teacher and variant: labels per puzzle (over all scored puzzles, unsolved ones giving none), labels on boards
off every optimal path from the start, distance / true distance, and the share of target moves that are optimal.
Output: models/sessions/hindsight_teacher_labels.json (or env LABEL_OUT)."""
import json, os, random, sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rl
import session_experiment as se
import teach_experiment as te
from hindsight_labels import VARIANTS, labels, score
from lib import bfs, car_orientations, multi_bfs

RUN = "teach_shortcut_agency1"
TEACHERS = ["none", "random:0.3", "bayes3:0.45", "bayes3:0.9", "bayes3:0.97"]
KEEP = ["own", "walk (W=1)", "W=3", "W=5", "W=10", "W=inf (all)"]
VARIANTS = [(n, o) for n, o in VARIANTS if n in KEEP]
EVERY = 3


def play(job):
    """One student's first `n` puzzles with `spec` as in the run; returns per scored puzzle its label scores."""
    student, spec, n = job
    torch.set_num_threads(1)
    run_dir = os.path.join(se.OUT, RUN)
    config = json.load(open(os.path.join(run_dir, "config.json")))
    session, _ = se.sample_puzzles(config)
    caps = {s: te.step_cap(d, config) for d, s in session}
    puzzles = se.order(session, student, config["seed"])[:n]
    captured, original = [], rl.solve_intuition

    def capture(*a, **k):
        r = original(*a, **k)
        captured.append(r)
        return r

    rl.solve_intuition = capture
    agent = rl.PolicyAgent.load(os.path.join(run_dir, "initial.pt"), lr=config["lr"])
    record = rl.learning_session(
        agent, [s for _, _, s in puzzles], se.make_teacher(spec), max_attempts=config["max_attempts"],
        surrender=se.surrender_rule(config, [(d, s) for _, d, s in puzzles]) if config.get("surrender") else None,
        max_steps=caps.__getitem__, memory=config["memory"], explore=config["explore"],
        window=None if config["window"] < 0 else config["window"], encode=config["encode"],
        steps=config["steps"], block=config.get("block", 1), seed=config["seed"] + 7919 * (student + 1),
        shown_weight=config.get("shown_weight", 1.0), labels=config.get("labels", "own"),
        agency_step=config.get("agency_step", "off"))
    rl.solve_intuition = original
    assert len(captured) == len(puzzles)  # one attempt per puzzle
    out = []
    for k, ((_, d, state), r) in enumerate(zip(puzzles, captured)):
        if k % EVERY:
            continue
        row = {"d": d, "solved": r["solved"], "moves": len(r["moves"]), "taught": sum(r["taught"])}
        if r["solved"]:
            dist = multi_bfs(state)
            start = bfs([state], car_orientations(state))
            for name, opts in VARIANTS:
                row[name] = score(labels(r, opts), dist, start, d)
        out.append(row)
    return spec, student, out, sum(e["taught"] for e in record) / max(1, sum(e["moves"] for e in record))


def summarise(rows):
    table = []
    for name, _ in VARIANTS:
        scored = [x for r in rows if r["solved"] for x in r[name]]
        opt, ratio, off = (np.array(c) for c in zip(*scored))
        table.append({"labels": name, "labels per puzzle": len(scored) / len(rows),
                      "off-path per puzzle": off.sum() / len(rows), "distance / true": ratio.mean(),
                      "targets optimal": opt.mean(),
                      "off-path: distance / true": ratio[off].mean() if off.any() else np.nan})
    return table


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 540
    students = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    processes = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    jobs = [(s, spec, n) for spec in TEACHERS for s in range(students)]
    with Pool(processes) as pool:
        done = pool.map(play, jobs, chunksize=1)
    result = {}
    for spec in TEACHERS:
        rows = [r for sp, _, out, _ in done if sp == spec for r in out]
        shown = np.mean([sh for sp, _, _, sh in done if sp == spec])
        result[spec] = {"shown": shown, "solved": np.mean([r["solved"] for r in rows]),
                        "moves per puzzle": np.mean([r["moves"] for r in rows]), "scored": len(rows),
                        "table": summarise(rows)}
    out = os.environ.get("LABEL_OUT", os.path.join(se.OUT, "hindsight_teacher_labels.json"))
    json.dump({"puzzles": n, "students": students, "every": EVERY, "teachers": result}, open(out, "w"), indent=1,
              default=float)
    import pandas as pd
    pd.set_option("display.width", 250)
    for spec, r in result.items():
        print(f"\n{spec}: shown {r['shown']:.1%}, solved {r['solved']:.0%}, {r['moves per puzzle']:.1f} moves")
        print(pd.DataFrame(r["table"]).round(3).to_string(index=False))
