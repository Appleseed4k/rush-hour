"""Network-ordered AND-OR (deliberation only) on the humans' 70 layouts, cached for notebooks/summary.ipynb.

The apprentice's dest head orders AND-OR's candidate moves and picks what a lapse plays (PolicyAgent.heuristic, with
Gumbel noise at temperature HEURISTIC_TEMPERATURE as during expert iteration); no intuitive moves are played - the
threshold > 1 end of solve_arbitrated. Each checkpoint plays each layout RUNS times per gamma, exactly as
p3_session/and_or_gamma.py plays plain AND-OR (whole plans, protect_movers off, replanning after each pass, a cap
of STEP_FACTOR x the layout's length, at least MIN_STEPS, with no giving up: a plan that would cross the cap counts
as unsolved), so people's giving up can be applied afterwards from the move count and the rows compare directly
with that sweep. The same checkpoints also play the layouts by intuition alone (fit_surrender.attempts: the policy
head at temperature 1, memory 10, same cap) for the other end.

    python and_or_network.py     # every core; writes models/summary/and_or_network/{sweep,intuition}.csv
"""
import multiprocessing as mp
import os
import random
import sys
import time

os.environ["AND_OR_PROTECT_MOVERS"] = "0"  # as the apprentice's expert (gamma0.07_free) was trained
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path += [os.path.join(HERE, ".."), os.path.join(HERE, "..", "p3_session")]
import and_or  # noqa: E402
import fit_surrender as fs  # noqa: E402
import human_data as hd  # noqa: E402
import rl  # noqa: E402

OUT = os.path.join(ROOT, "models", "summary", "and_or_network")
FOLDER = os.path.join(ROOT, "models", "apprentice", "gamma0.07_free")
CHECKPOINTS = ["round0_epoch00.pt", "round1_epoch05.pt", "round2_epoch02.pt", "round3_epoch01.pt"]  # kept epochs
GAMMAS = [0.01, 0.035, 0.07, 0.15, 0.3]
HEURISTIC_TEMPERATURE = 1.0
RUNS, STEP_FACTOR, MIN_STEPS = 20, 10, 50


def init(path):
    agent = rl.PolicyAgent.load(path)
    agent.temperature = HEURISTIC_TEMPERATURE
    rl._init_attempt_worker(*rl._agent_payload(agent))


def rollout(args):
    gamma, instance, run = args
    random.seed(f"{gamma}-{instance}-{run}")  # and_or and the Gumbel noise draw from the random module
    heuristic = rl._WORKER["agent"].heuristic
    state, distance = hd.puzzle(instance), hd.distance(instance)
    cap = max(MIN_STEPS, STEP_FACTOR * distance)
    played, start, solved = 0, time.process_time(), False
    while played < cap:
        attempt = and_or.solve(state, heuristic=heuristic, gamma=gamma)
        if type(attempt) is list:
            played += len(attempt)
            solved = played <= cap
            break
        state, moves = attempt
        if not moves:
            break
        played += len(moves)
    return gamma, instance, distance, run, min(played, cap) if solved else cap, solved, time.process_time() - start


def main():
    assert not and_or.PROTECT_MOVERS
    os.makedirs(OUT, exist_ok=True)
    instances = sorted(hd.trials().instance.unique())
    human = {i: (hd.distance(i), hd.puzzle(i)) for i in instances}  # puzzle_set("human") trips on an empty json
    sweep_path, intuition_path = os.path.join(OUT, "sweep.csv"), os.path.join(OUT, "intuition.csv")
    with open(sweep_path, "w") as f:
        f.write("checkpoint,gamma,instance,distance,run,moves,solved,cpu_seconds\n")
    with open(intuition_path, "w") as f:
        f.write("checkpoint,instance,distance,run,moves,solved\n")
    for ck in CHECKPOINTS:
        path = os.path.join(FOLDER, ck)
        t = time.time()
        runs = fs.attempts(path, {i: human[i] for i in instances}, RUNS)
        with open(intuition_path, "a") as f:
            for inst, ms in runs.items():
                d = human[inst][0]
                cap = max(MIN_STEPS, int(round(STEP_FACTOR * d)))
                for r, m in enumerate(ms):
                    f.write(f"{ck},{inst},{d},{r},{m if m is not None else cap},{m is not None}\n")
        print(f"{ck} intuition: {time.time() - t:.0f} s", flush=True)
        jobs = [(g, i, r) for g in GAMMAS for i in instances for r in range(RUNS)]
        t = time.time()
        with mp.Pool(initializer=init, initargs=(path,)) as pool, open(sweep_path, "a") as f:
            for k, row in enumerate(pool.imap_unordered(rollout, jobs, chunksize=4)):
                f.write(ck + "," + ",".join(map(str, row)) + "\n")
                if k % 1000 == 0:
                    print(f"{ck} AND-OR: {k}/{len(jobs)} ({time.time() - t:.0f} s)", flush=True)
        print(f"{ck} AND-OR done: {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
