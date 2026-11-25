"""The calibrated session with the planner in the loop.

The session student (MODEL.md) plays every puzzle by intuition and learns from its solved attempts. The judgement it
learns that way doesn't help the forward + backward planner (notebooks/p2.2_forward_planning.ipynb, "Judgement or depth"):
its snapshots barely lift the planner's mid-puzzle skill and undo more. Here the same session is played by the planner
itself, so the network learns from the planner's own experience:

- the calibrated session's settings (ao_people_all_m10, read from the archive's config.json): the same puzzles at
  people's lengths, the same order per student, x10 multiplier, full hindsight, learning rate, move cap, seed, 8
  students, no teacher; the start (--init, default paths.START) and giving up (people's hazard x --surrender-scale,
  default the fitted student's, spec.STUDENT) are the current ones - the runs before 2026-11-24 started from
  paths.DEST_START at 0.3;
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
    python experiments/planner_session.py --fit-two-mode       # the impasse switch: sharp flat forward + AND-OR plans at impasses
    python experiments/planner_session.py --fit-lean           # forward / two-mode students: intuition + flat + one-move AND-OR
    python experiments/planner_session.py --fit-lean-start     # ... its refitted weights on the no-dest apprentice's epochs
    python experiments/planner_session.py --fit-shape          # ... with goal in sight and a sharper intuition
    python experiments/planner_session.py --fit-human          # the start (checkpoints, planners) on people's layouts
    python experiments/planner_session.py --fit-strong         # stronger networks x higher temperatures, ditto
    python experiments/planner_session.py --lean --run lean_people_all_m10_ao     # the session played by the two-mode student
                                                               # (then: the dest-head start, --w-ao 0.1 --flat-start 0.08)
    python experiments/planner_session.py --lean --run lean_people_all_m10_noao --w-ao 0 --flat-start 0.18 \\
        --flat-end 0.28 --no-opening                           # ... and by the forward student (flat set to match the start)
    python experiments/planner_session.py --lean --run lean_people_all_m10_ao_nodest --w-ao 0.1 --flat-start 0.08
                                                               # ... from the start without a dest head (paths.START
                                                               # since 2026-11-24), with the dest-ordered weights
    python experiments/planner_session.py --lean --run lean_people_all_m10_ao_nodest_refit   # ... with the weights
                                                               # refitted to policy-ordered AND-OR (the defaults)
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
from rushhour.student import spec  # noqa: E402

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
        even_batches=config.get("even_batches", False),
        solver=(forward.LeanSolver(multiplier=ss.multiplier(config), **config["lean"]) if config.get("lean") else
                forward.PlannerSolver(**config["planner"])))
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


# the impasse switch: sharp flat forward choice, an AND-OR plan at an impasse (forward.solve_impasse_switch)
TWO_CONSIDER, TWO_TAU, TWO_TRIGGER, TWO_GAMMA = (3, 5), (0.25, 0.5, 1), ("loop", "stall", "either"), (0.07, 0.3)
TWO_START = {f"two-mode m{m} tau {t} {tr} gamma {g}": {"two_mode": True, "consider": m, "tau": t, "trigger": tr,
                                                         "gamma": g}
             for m in TWO_CONSIDER for t in TWO_TAU for tr in TWO_TRIGGER for g in TWO_GAMMA}


# the two-mode student (the forward student: w_ao 0), "lean" in the code: intuition, flat planning (w_flat) and
# one-move AND-OR deliberations (w_ao); opening: the first
# move a deliberation with a one-move follow-through (forward.solve_lean)
LEAN_FLAT, LEAN_AO = (0, 0.08, 0.18), (0, 0.1, 0.2)
LEAN_START = {f"lean flat {f} and-or {a}{' opening' if o else ''}": {"lean": True, "w_flat": f, "w_ao": a,
                                                                       "opening": o}
              for f in LEAN_FLAT for a in LEAN_AO for o in (False, True)}


# the near-goal gap: goal in sight (a search of up to D moves for red's way out) and a sharper intuition (temperature),
# with a weaker checkpoint to give the sharper policy a chance to start where people do (forward.solve_lean)
NETS = {"start": paths.START, "epoch 1": os.path.join(paths.APPRENTICE, "gamma0.07_free", "round1_epoch01.pt")}
NETS |= {f"round {r} epoch {e}": os.path.join(paths.APPRENTICE, "gamma0.07_free", f"round{r}_epoch{e:02d}.pt")
         for r, e in [(1, 3), (1, 4), (1, 5), (1, 8), (1, 11), (2, 2), (2, 5)]}
# the start on people's own layouts: phase 1's choice of checkpoint (intuition, hard memory) refitted there, and the
# planners of phase 2 with the start network
_LEAN = {"lean": True, "w_flat": 0.08, "w_ao": 0.1, "opening": True}
HUMAN_START = ({f"intuition, hard memory, {net}": {"k": 1, "net": net}
                for net in ["epoch 1", "start", "round 1 epoch 3", "round 1 epoch 4", "round 1 epoch 5",
                            "round 1 epoch 8", "round 1 epoch 11", "round 2 epoch 2", "round 2 epoch 5"]}
               | {"intuition, soft memory": {"k": 1, "soft_memory": (2, 1)},
                  "lean": dict(_LEAN), "lean, sight 2": {**_LEAN, "sight": 2},
                  "lean, sight 2, temperature 0.75": {**_LEAN, "sight": 2, "temperature": 0.75},
                  "lean, temperature 0.5": {**_LEAN, "temperature": 0.5}})  # no rollouts: implausible at people's speed
SHAPE = {f"lean {net}, temperature {t}, sight {d}": {"lean": True, "w_flat": 0.08, "w_ao": 0.1, "opening": True,
                                                       "net": net, "temperature": t, "sight": d}
         for net in NETS for t in (1.0, 0.75, 0.5) for d in (0, 2, 3)}

# the two-mode student's start with its refitted weights (2026-11-24): each early epoch of the no-dest apprentice
# (paths.START is its epoch 3, fitted by intuition), flat at its mean over trials 1-10 of the ramp
NETS |= {f"no-dest epoch {e}": os.path.join(paths.APPRENTICE, "gamma0.07_free_nodest", f"round1_epoch{e:02d}.pt")
         for e in range(1, 6)}
_REFIT = {"lean": True, "w_flat": round(sum(forward.LeanSolver().flat_at(t) for t in range(1, 11)) / 10, 3),
          "w_ao": forward.LeanSolver().w_ao, "opening": True}
LEAN_NODEST = ({f"two-mode, no-dest epoch {e}": {**_REFIT, "net": f"no-dest epoch {e}"} for e in range(1, 6)}
               | {"two-mode, dest-ordered weights, no-dest epoch 3": {**_LEAN, "net": "no-dest epoch 3"}})

# robust but inefficient: a stronger network sampled at a higher temperature (intuition, hard memory), on people's
# layouts against their trials and their first attempts
STRONG = {f"intuition, {net}, temperature {t}": {"k": 1, "net": net, "temperature": t}
          for net in ["start", "round 1 epoch 8", "round 1 epoch 11", "round 2 epoch 2", "round 2 epoch 5"]
          for t in (1.0, 1.25, 1.5, 2.0)}

def _start_job(args):
    name, kw, puzzle, state, distance, run = args
    import random
    import zlib
    rng = random.Random(zlib.crc32(f"{name}|{puzzle}|{run}".encode()))
    random.seed(rng.random())
    trace = []
    kw = dict(kw)
    forward._W["agent"] = forward._W["agents"][kw.pop("net", "start")]
    solve = (forward.solve_flat if kw.pop("flat", False) else
             forward.solve_impasse_switch if kw.pop("two_mode", False) else
             forward.solve_lean if kw.pop("lean", False) else forward.solve_forward)
    out = solve(forward._W["agent"], state, max(50, round(10 * distance)), rng=rng, trace=trace, **kw)
    return {"config": name, "puzzle": puzzle, "distance": distance, "run": run,
            "moves": len(trace) if out["solved"] else None, "played": len(trace),
            "backward": out.get("backward"), "impasses": out.get("impasses")}


def human_puzzles():
    """{layout: (distance, state)} for the 70 layouts people played (human_data.puzzles() trips on stray files)."""
    from rushhour import human_data as hd
    return {n: (hd.distance(n), hd.puzzle(n)) for n in sorted(hd.trials().instance.unique())}


def first_attempt_target(block=10):
    """People's first attempts in trials 1..block (a restart ends an attempt, as giving up does): per-person means of
    solving and of moves / optimal when solved, each (mean, se) across people."""
    import pandas as pd
    from rushhour import human_data as hd, sessions as se
    trial = hd.trials()[["subject", "instance", "trial", "distance"]]
    rows = [(s, inst, att, solved, len(moves)) for s, atts in hd.load(unsolved=True).items()
            for inst, att, solved, moves in atts]
    a = pd.DataFrame(rows, columns=["subject", "instance", "attempt", "solved", "moves"]).merge(
        trial, on=["subject", "instance"])
    a = a[a.trial <= block]
    a = a[a.attempt == a.groupby(["subject", "instance"]).attempt.transform("min")]
    won = a[a.solved]
    return (se.mean_se(list(a.groupby("subject").solved.mean())),
            se.mean_se(list((won.moves / won.distance).groupby(won.subject).mean())))


def fit_start(configs, out, runs=10, scale=0.3, layouts="test"):
    """Each config (name -> solve_forward kwargs) with the start network on the session's test puzzles, no giving
    up; then giving up at `scale` applied (surrender.predict) and compared with people's first 10 trials.
    layouts="human": on the 70 layouts people played instead, each of their first 10 trials predicted on its own
    layout (surrender.py --puzzles human); also prints the giving-up scale that fits best."""
    import numpy as np
    import random
    import pandas as pd
    from rushhour import human_data as hd, surrender as sr
    (hs, hs_se), (hr, hr_se), first = sr.human_target(10)
    (fs, fs_se), (fr, fr_se) = first_attempt_target(10)
    edges, rates, _ = hd.surrender_hazard()
    puzzles = human_puzzles() if layouts == "human" else sr.puzzle_set("test")
    by_length = layouts != "human"
    weights = {}
    for name in {kw.get("net", "start") for kw in configs.values()}:
        with open(NETS[name], "rb") as f:
            weights[name] = f.read()
    jobs = [(c, kw, n, s, d, r) for c, kw in configs.items() for n, (d, s) in puzzles.items() for r in range(runs)]
    random.Random(0).shuffle(jobs)
    with Pool(initializer=forward._init, initargs=(weights,)) as pool:
        df = pd.DataFrame(pool.map(_start_job, jobs, chunksize=2))
    df.to_csv(os.path.join(paths.MODELS, "summary", out), index=False)
    print(f"people, trials 1-10: solved {hs:.1%} ± {hs_se:.1%}, moves / optimal {hr:.2f} ± {hr_se:.2f}; "
          f"first attempts: solved {fs:.1%} ± {fs_se:.1%}, moves / optimal {fr:.2f} ± {fr_se:.2f}")
    for c in configs:
        d = df[df.config == c]
        att = {n: [None if m != m else int(m) for m in x.moves] for n, x in d.groupby("puzzle")}
        ps, pr = sr.predict(att, puzzles, first, edges, rates, scale, by_length)
        p0, r0 = sr.predict(att, puzzles, first, edges, rates, 0.0, by_length)
        misfit = ((ps - hs) / hs_se) ** 2 + ((pr - hr) / hr_se) ** 2
        fits = [(((p - hs) / hs_se) ** 2 + ((q - hr) / hr_se) ** 2, s_)
                for s_ in np.arange(0, 2.01, 0.05)
                for p, q in [sr.predict(att, puzzles, first, edges, rates, s_, by_length)]]
        best_m, best_s = min(fits)
        firsts = [(((p - fs) / fs_se) ** 2 + ((q - fr) / fr_se) ** 2, s_, p, q)
                  for s_ in np.arange(0, 2.01, 0.05)
                  for p, q in [sr.predict(att, puzzles, first, edges, rates, s_, by_length)]]
        f_m, f_s, f_p, f_q = min(firsts)
        extra = (f"   backward moves {d.backward.sum() / d.played.sum():.1%}, impasses per 10 moves "
                 f"{10 * d.impasses.sum() / d.played.sum():.2f}" if d.impasses.notna().any() else "")
        print(f"{c}: solved {ps:.1%}, moves / optimal {pr:.2f} (misfit {misfit:.1f})   "
              f"no giving up: {p0:.1%} {r0:.2f}   best scale {best_s:.2f} (misfit {best_m:.1f})   "
              f"vs first attempts: scale {f_s:.2f} -> {f_p:.1%} {f_q:.2f} (misfit {f_m:.1f}){extra}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="planner_people_all_m10")
    ap.add_argument("--gamma", type=float, default=PLANNER["gamma"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fit-start", action="store_true")
    ap.add_argument("--fit-forward", action="store_true")
    ap.add_argument("--fit-flat", action="store_true")
    ap.add_argument("--fit-two-mode", action="store_true")
    ap.add_argument("--fit-lean", action="store_true")
    ap.add_argument("--fit-shape", action="store_true")
    ap.add_argument("--fit-lean-start", action="store_true",
                    help="the two-mode student's refitted weights on the no-dest apprentice's epochs 1-5")
    ap.add_argument("--fit-human", action="store_true", help="the start on people's own 70 layouts")
    ap.add_argument("--fit-strong", action="store_true", help="stronger networks x higher temperatures, ditto")
    ap.add_argument("--lean", action="store_true",
                    help="play the session with the two-mode student (--w-ao 0: the forward student)")
    ap.add_argument("--w-ao", type=float, default=0.18)
    ap.add_argument("--flat-start", type=float, default=0.06)
    ap.add_argument("--flat-end", type=float, default=0.18)
    ap.add_argument("--no-opening", action="store_true")
    ap.add_argument("--ao-order", choices=forward.AO_ORDERS, default="policy",
                    help="with --lean: what orders AND-OR's options (forward.solve_lean's ao_order)")
    ap.add_argument("--seed", type=int, default=None, help="the session's seed (default: the calibrated config's)")
    ap.add_argument("--init", default=paths.START, help="the start network (default paths.START)")
    ap.add_argument("--surrender-scale", type=float, default=spec.STUDENT["surrender_scale"],
                    help="giving up as people's hazard x this (default: the fitted student's, with paths.START)")
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
    if args.fit_lean:
        fit_start(LEAN_START, "lean_start_fit.csv")
        return
    if args.fit_lean_start:
        fit_start(LEAN_NODEST, "lean_nodest_start_fit.csv", scale=spec.STUDENT["surrender_scale"])
        return
    if args.fit_shape:
        fit_start(SHAPE, "shape_start_fit.csv")
        return
    if args.fit_human:
        fit_start(HUMAN_START, "human_start_fit.csv", layouts="human")
        return
    if args.fit_strong:
        fit_start(STRONG, "strong_start_fit.csv", layouts="human")
        return
    with open(CALIBRATED) as f:
        config = json.load(f)
    config["init"] = args.init
    config["run"] = args.run
    if args.seed is not None:
        config["seed"] = args.seed
    config["surrender_scale"] = args.surrender_scale
    config["planner"] = {**PLANNER, "gamma": args.gamma, "rollout_gamma": args.gamma}
    if args.lean:
        config["lean"] = {"w_ao": args.w_ao, "flat_start": args.flat_start, "flat_end": args.flat_end, "ramp": 30,
                          "opening": not args.no_opening}
        if args.ao_order != "policy":
            config["lean"]["ao_order"] = args.ao_order
        del config["planner"]
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
