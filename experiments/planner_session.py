"""The calibrated session with the planner in the loop.

The session student (MODEL.md) plays every puzzle by intuition and learns from its solved attempts. The judgement it
learns that way doesn't help the forward + backward planner (notebooks/forward_planning.ipynb, "Judgement or depth"):
its snapshots barely lift the planner's mid-puzzle skill and undo more. Here the same session is played by the planner
itself, so the network learns from the planner's own experience:

- the calibrated session's settings (ao_people_all_m10, read from the archive's config.json): the same puzzles at
  people's lengths, the same order per student, x10 multiplier, full hindsight, learning rate, giving up (people's
  hazard x 0.3), move cap, seed, 8 students, no teacher;
- each puzzle is played by forward.solve_forward (PlannerSolver): 4 policy rollouts (up to 20 moves) and 4 AND-OR
  passes per turn, judged by the value head; the soft board memory (beta 2, half-life 1) also judging candidates
  (x 4); one shared lapse rate gamma, fixed (so what changes over the session is the network, not depth);
- labels come from the boards actually played (full hindsight, as before); imagined boards teach nothing.

Snapshots after 18, 36, 54 and 72 of people's trials go to models/sessions/<run>/no_teacher/ like a session run's,
with sessions.json (one record per puzzle). fit_memory.py `snapshots --run <run>` fits them per stage.

    python experiments/planner_session.py                      # run planner_people_all_m10
    python experiments/planner_session.py --smoke              # 1 student, 20 puzzles, nothing saved
    python experiments/planner_session.py --fit-start          # the gamma that starts where people do
    python experiments/planner_session.py --fit-forward        # forward planning alone: rollouts x gamma
    python experiments/planner_session.py --fit-flat           # flat (moves considered x tau), with/without a check
    python experiments/planner_session.py --fit-two-mode       # sharp flat forward + AND-OR plans at impasses
    python experiments/planner_session.py --gamma 0.5 --run planner_people_all_m10_g0.5

--fit-start calibrates gamma as rushhour/surrender.py calibrated the start network: the planner (start network)
plays the session's held-out test puzzles at people's lengths (calibrated_v2: 20 each of 6, 10, 13, 15 moves) 10 times
each with no giving up, up to the session's cap; giving up at the session's scale (0.3) is applied afterwards
(surrender.predict), and the result is compared with people's first 10 trials (solve rate, moves / optimal on
solved; misfit in squared standard errors). Writes models/summary/planner_start_fit.csv.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import paths, sessions as ss, student as st  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forward  # noqa: E402

ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(paths.ROOT)), "rush-hour-archive", "models", "sessions")
CALIBRATED = os.path.join(ARCHIVE, "ao_people_all_m10", "config.json")
PLANNER = {"k": 4, "depth": 20, "backward": 4, "evaluation": "value", "soft_memory": (2, 1), "candidate_memory": 4,
           "gamma": 0.03, "rollout_gamma": 0.03}


def _student(job):
    student, config, puzzles, caps, run_dir, n_puzzles, save = job
    torch.set_num_threads(1)
    agent = st.PolicyAgent.load(config["init"], lr=config["lr"])
    teacher_dir = os.path.join(run_dir, "no_teacher")
    points = ss.test_points(config, len(puzzles))
    saved, t0 = [0], time.time()

    def snapshot(n, agent):
        for p in points:
            if save and saved[0] < p <= n:
                agent.save(os.path.join(teacher_dir, f"student{student:02d}_puzzle{p:03d}.pt"))
        saved[0] = n
        if save:
            ss.write_progress(run_dir, f"no_teacher_student{student}", done=n, total=len(puzzles))
        if n % 50 == 0:
            print(f"student {student}: {n} puzzles, {time.time() - t0:.0f}s", flush=True)

    puzzles = puzzles[:n_puzzles]
    record = st.learning_session(
        agent, [s for _, _, s in puzzles], None, max_attempts=config["max_attempts"],
        surrender=ss.surrender_rule(config, [(d, s) for _, d, s in puzzles]) if config.get("surrender") else None,
        max_steps=caps.__getitem__, memory=config["memory"], explore=config["explore"],
        window=None if config["window"] < 0 else config["window"], encode=config["encode"],
        steps=config["steps"], block=config.get("block", 1), seed=config["seed"] + 7919 * (student + 1),
        on_puzzle=snapshot, labels=config.get("labels", "own"), agency_step=config.get("agency_step", "off"),
        even_batches=config.get("even_batches", False), solver=forward.PlannerSolver(**config["planner"]))
    m = ss.multiplier(config)
    for entry, (i, d, _) in zip(record, puzzles):
        entry.update(step=entry["puzzle"] + 1, trial=entry["puzzle"] // m + 1, puzzle=i, distance=d)
    return student, record


START_GRID = (0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
MIXED_START = {f"gamma {g}": {**PLANNER, "gamma": g, "rollout_gamma": g} for g in START_GRID}
# forward planning alone: k rollouts (each simulated move past the first continues with probability 1 - gamma, at most
# 20), judged by the value head with the soft memory; k = 1 is a policy sample (intuition with the soft memory)
FORWARD_K, FORWARD_GAMMA = (2, 3, 4, 8), (0.1, 0.3, 0.5, 1.0)
FORWARD_START = ({"intuition, hard memory": {"k": 1}, "intuition, soft memory": {"k": 1, "soft_memory": (2, 1)}}
                 | {f"forward k{k} gamma {g}": {"k": k, "depth": 20, "evaluation": "value", "soft_memory": (2, 1),
                                                  "candidate_memory": 4, "rollout_gamma": g}
                    for k in FORWARD_K for g in FORWARD_GAMMA})


# flat, then fast (Collins et al.): the policy's `consider` most likely moves looked at one move deep, a softmax over
# the value head's moves to go / tau; optionally one checking rollout from the chosen move (forward.solve_flat)
FLAT_CONSIDER, FLAT_TAU = (2, 3, 5), (0.25, 0.5, 1, 2, 4)
FLAT_START = ({f"flat m{m} tau {t}": {"flat": True, "consider": m, "tau": t}
               for m in FLAT_CONSIDER + (None,) for t in FLAT_TAU}
              | {f"flat+check m{m} tau {t}": {"flat": True, "consider": m, "tau": t, "check": True}
                 for m in FLAT_CONSIDER for t in FLAT_TAU})


# two modes: sharp flat forward choice, an AND-OR plan at an impasse (forward.solve_two_mode)
TWO_CONSIDER, TWO_TAU, TWO_TRIGGER, TWO_GAMMA = (3, 5), (0.25, 0.5, 1), ("loop", "stall", "either"), (0.07, 0.3)
TWO_START = {f"two-mode m{m} tau {t} {tr} gamma {g}": {"two_mode": True, "consider": m, "tau": t, "trigger": tr,
                                                         "gamma": g}
             for m in TWO_CONSIDER for t in TWO_TAU for tr in TWO_TRIGGER for g in TWO_GAMMA}


def _start_job(args):
    name, kw, puzzle, state, distance, run = args
    import random
    import zlib
    rng = random.Random(zlib.crc32(f"{name}|{puzzle}|{run}".encode()))
    random.seed(rng.random())
    trace = []
    kw = dict(kw)
    solve = (forward.solve_flat if kw.pop("flat", False) else
             forward.solve_two_mode if kw.pop("two_mode", False) else forward.solve_forward)
    out = solve(forward._W["agent"], state, max(50, round(10 * distance)), rng=rng, trace=trace, **kw)
    return {"config": name, "puzzle": puzzle, "distance": distance, "run": run,
            "moves": len(trace) if out["solved"] else None, "played": len(trace),
            "backward": out.get("backward"), "impasses": out.get("impasses")}


def fit_start(configs, out, runs=10, scale=0.3):
    """Each config (name -> solve_forward kwargs) with the start network on the session's test puzzles, no giving
    up; then giving up at `scale` applied (surrender.predict) and compared with people's first 10 trials."""
    import random
    import pandas as pd
    from rushhour import human_data as hd, surrender as sr
    (hs, hs_se), (hr, hr_se), first = sr.human_target(10)
    edges, rates, _ = hd.surrender_hazard()
    puzzles = sr.puzzle_set("test")
    with open(paths.START, "rb") as f:
        weights = f.read()
    jobs = [(c, kw, n, s, d, r) for c, kw in configs.items() for n, (d, s) in puzzles.items() for r in range(runs)]
    random.Random(0).shuffle(jobs)
    with Pool(initializer=forward._init_one, initargs=(weights,)) as pool:
        df = pd.DataFrame(pool.map(_start_job, jobs, chunksize=2))
    df.to_csv(os.path.join(paths.MODELS, "summary", out), index=False)
    print(f"people, trials 1-10: solved {hs:.1%} ± {hs_se:.1%}, moves / optimal {hr:.2f} ± {hr_se:.2f}")
    for c in configs:
        d = df[df.config == c]
        att = {n: [None if m != m else int(m) for m in x.moves] for n, x in d.groupby("puzzle")}
        ps, pr = sr.predict(att, puzzles, first, edges, rates, scale, True)
        p0, r0 = sr.predict(att, puzzles, first, edges, rates, 0.0, True)
        misfit = ((ps - hs) / hs_se) ** 2 + ((pr - hr) / hr_se) ** 2
        extra = (f"   backward moves {d.backward.sum() / d.played.sum():.1%}, impasses per 10 moves "
                 f"{10 * d.impasses.sum() / d.played.sum():.2f}" if d.impasses.notna().any() else "")
        print(f"{c}: solved {ps:.1%}, moves / optimal {pr:.2f} (misfit {misfit:.1f})   "
              f"no giving up: {p0:.1%} {r0:.2f}{extra}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="planner_people_all_m10")
    ap.add_argument("--gamma", type=float, default=PLANNER["gamma"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fit-start", action="store_true")
    ap.add_argument("--fit-forward", action="store_true")
    ap.add_argument("--fit-flat", action="store_true")
    ap.add_argument("--fit-two-mode", action="store_true")
    args = ap.parse_args()
    if args.fit_start:
        fit_start(MIXED_START, "planner_start_fit.csv")
        return
    if args.fit_forward:
        fit_start(FORWARD_START, "forward_start_fit.csv")
        return
    if args.fit_flat:
        fit_start(FLAT_START, "flat_start_fit.csv")
        return
    if args.fit_two_mode:
        fit_start(TWO_START, "two_mode_start_fit.csv")
        return
    with open(CALIBRATED) as f:
        config = json.load(f)
    config["init"] = paths.START
    config["run"] = args.run
    config["planner"] = {**PLANNER, "gamma": args.gamma, "rollout_gamma": args.gamma}
    run_dir = os.path.join(paths.SESSIONS, args.run)
    if not args.smoke:
        if os.path.exists(os.path.join(run_dir, "no_teacher", "sessions.json")):
            raise SystemExit(f"{run_dir} already has sessions.json")
        os.makedirs(os.path.join(run_dir, "no_teacher"), exist_ok=True)
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
    session, test = ss.sample_puzzles(config)
    caps = {s: ss.step_cap(d, config) for d, s in session + test}
    students = 1 if args.smoke else config["students"]
    n = 20 if args.smoke else len(session)
    jobs = [(s, config, ss.order(session, s, config["seed"]), caps, run_dir, n, not args.smoke)
            for s in range(students)]
    with Pool(students) as pool:
        records = dict(pool.map(_student, jobs))
    for s, record in sorted(records.items()):
        solved = [e for e in record if e["solved"]]
        print(f"student {s}: solved {len(solved)}/{len(record)}, moves / optimal (solved) "
              f"{sum(e['moves'] / e['distance'] for e in solved) / max(1, len(solved)):.2f}")
    if not args.smoke:
        with open(os.path.join(run_dir, "no_teacher", "sessions.json"), "w") as f:
            json.dump({str(s): records[s] for s in sorted(records)}, f)
        print("saved to", run_dir)


if __name__ == "__main__":
    main()
