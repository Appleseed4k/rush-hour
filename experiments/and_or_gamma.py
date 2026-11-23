"""This repo's plain AND-OR solver (and_or.solve, no network) on the humans' 70 layouts, over a grid of lapse rates
gamma, with protect_movers on and off - for notebooks/archive/p1.3_session.ipynb's section on gamma.

Plain AND-OR is what expert iteration's first round learns from (student.expert_iteration with student "and_or", gamma
0.07 by default; phase 1). Each layout is played RUNS times per (protect_movers, gamma):
whole plans, searched depth-first, gamma the chance a subgoal is abandoned for a random move, replanning after each
pass, up to a cap of STEP_FACTOR x the layout's length (at least MIN_STEPS) with no giving up - people's giving up is
applied afterwards from the move count, as surrender.py does. A plan that would cross the cap counts as unsolved.

protect_movers off with gamma 0 is left out: with no lapses to end it, its search grows without bound on some
layouts (prb58853_16 among them; runs still searching after 5 minutes). With gamma > 0 it costs no more than on.

    python and_or_gamma.py     # about a minute on every core; writes models/sessions/and_or_gamma/sweep.csv
"""
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import and_or, human_data as hd, paths  # noqa: E402

OUT = os.path.join(paths.SESSIONS, "and_or_gamma")
GAMMAS = [0.0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15, 0.3]
RUNS, STEP_FACTOR, MIN_STEPS = 20, 10, 50


def rollout(args):
    protect, gamma, instance, run = args
    random.seed(f"{gamma}-{instance}-{run}")  # and_or draws from the random module
    state, distance = hd.puzzle(instance), hd.distance(instance)
    cap = max(MIN_STEPS, STEP_FACTOR * distance)
    played, start, solved = 0, time.process_time(), False
    while played < cap:
        attempt = and_or.solve(state, gamma=gamma, protect_movers=protect)
        if type(attempt) is list:  # red exits at the end of this plan
            played += len(attempt)
            solved = played <= cap
            break
        state, moves = attempt
        if not moves:  # no plan and no fallback move: stuck
            break
        played += len(moves)
    return protect, gamma, instance, distance, run, min(played, cap) if solved else cap, solved, \
        time.process_time() - start


def main():
    instances = sorted(hd.trials().instance.unique())
    jobs = [(p, g, i, r) for p in (True, False) for g in GAMMAS if p or g > 0 for i in instances for r in range(RUNS)]
    os.makedirs(OUT, exist_ok=True)
    with mp.Pool() as pool, open(os.path.join(OUT, "sweep.csv"), "w") as f:
        f.write("protect,gamma,instance,distance,run,moves,solved,cpu_seconds\n")
        for row in pool.imap_unordered(rollout, jobs, chunksize=8):
            f.write(",".join(map(str, row)) + "\n")
    print("saved", os.path.join(OUT, "sweep.csv"))


if __name__ == "__main__":
    main()
