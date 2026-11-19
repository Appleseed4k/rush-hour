"""The Olieslagers et al. (2023) AND-OR model against the 42 participants, cached for notebooks/summary.ipynb.

The model is the port of the archive's gamma_only_model (scripts/original_model.py): a per-board
distribution over the next move, from the board's AND-OR tree with lapse rate gamma. Two passes:

    python and_or_humans.py likelihood   # log-probability of every human move at each gamma in GAMMAS
    python and_or_humans.py simulate     # the model played forward on the 70 layouts at each gamma in SIM_GAMMAS

likelihood replays every attempt at every puzzle a participant solved (pre_process.jl drops the others), with
the car moved last reset at each start and restart, as the paper does; it writes moves.npz (one row per move:
subject, trial, instance, attempt, log-probability per gamma). simulate plays each layout RUNS times per gamma, up
to a cap of STEP_FACTOR x its length (at least MIN_STEPS) with no giving up, and writes simulations.csv (gamma,
instance, distance, run, moves, solved); giving up can be applied afterwards from the move count. Both run in
parallel over every core and take a few minutes.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path += [os.path.join(HERE, "..")]
import human_data as hd  # noqa: E402
import original_model as om  # noqa: E402
from and_or import legal_moves  # noqa: E402

OUT = os.path.join(HERE, "..", "..", "models", "summary", "and_or_humans")
GAMMAS = np.exp(np.linspace(np.log(0.002), np.log(0.6), 41))
SIM_GAMMAS = [0.0, 0.01, 0.02, 0.035, 0.05, 0.07, 0.1, 0.15, 0.25]
RUNS, STEP_FACTOR, MIN_STEPS = 10, 10, 50
DELTAS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}


def piece_names(instance):
    """Human piece number -> car name: the layout's numbered cars keep their number, the red car is "red"."""
    with open(os.path.join(hd.PROBLEMS, instance + ".json")) as f:
        ids = {c["id"] for c in json.load(f)["cars"]}
    return lambda k: str(k) if str(k) in ids else "red"


def human_move(state, name, c0, c1):
    """(next board, the move as and_or's (car, direction, steps)) for a human slide from cell c0 to c1."""
    cars = dict(state)
    (r0, k0), (r1, k1) = divmod(c0, 6), divmod(c1, 6)
    if cars[name][0] != (r0, k0):
        raise ValueError(f"{name} isn't at cell {c0}")
    dr, dc = r1 - r0, k1 - k0
    move = (name, ("d" if dr > 0 else "u") if dr else ("r" if dc > 0 else "l"), abs(dr) + abs(dc))
    return play(state, move), move


def play(state, move):
    name, direction, steps = move
    dr, dc = DELTAS[direction]
    return tuple((n, tuple((r + dr * steps, c + dc * steps) for r, c in cells) if n == name else cells)
                 for n, cells in state)


def solved(state):
    return max(c for _, c in dict(state)["red"]) == 5


def log_probs(state, move, prev_car, gammas=GAMMAS):
    """log P(move) at each gamma, as original_model.action_distribution computes it (the tree is built once)."""
    root_and, AND, OR, _ = om.build_tree(state, True)
    leaves = [(depth, leaf, v) for (depth, leaf), v in om.propagate(root_and, AND, OR, True).items() if v > 0]
    legal = [m for m in legal_moves(state) if m[2] <= om.MAX_AMOUNT]
    out = np.empty(len(gammas))
    for i, g in enumerate(gammas):
        probs, cycle, total = dict.fromkeys(legal, 0.0), 0.0, 0.0
        for depth, leaf, v in leaves:
            v *= (1 - g) ** depth
            total += v
            if leaf == om.CYCLE or (prev_car is not None and leaf[0] == prev_car):
                cycle += v
                continue
            m = om._to_repo_move(state, leaf)
            if m in probs:
                probs[m] += v
        stop = 1 - total
        z = sum(probs.values()) + stop
        scale = 1 + cycle / z if z > 0 else 1.0
        p = probs.get(move, 0.0) * scale + stop * scale / len(legal)
        out[i] = np.log(max(p, 1e-300))
    return out


def subject_moves(args):
    subject, attempts, trial_of = args
    rows, too_large, broken = [], 0, 0
    for instance, attempt, _, moves in attempts:
        name = piece_names(instance)
        state, prev = hd.puzzle(instance), None
        for k, m in enumerate(moves):
            try:
                nxt, move = human_move(state, name(m.piece), m.c0, m.c1)
            except ValueError:  # a slide that doesn't replay (15 of 117,145): the rest of the attempt is dropped
                broken += 1
                break
            try:
                rows.append((trial_of[instance], instance, attempt, k, log_probs(state, move, prev)))
            except om.TreeTooLarge:
                too_large += 1
            state, prev = nxt, move[0]
    return subject, rows, too_large, broken


def likelihood():
    data, trials = hd.load(), hd.trials()
    trial_of = {s: dict(zip(d.instance, d.trial)) for s, d in trials.groupby("subject")}
    jobs = [(s, data[s], trial_of[s]) for s in sorted(data)]
    cols = {k: [] for k in ("subject", "trial", "instance", "attempt", "move", "logp")}
    with mp.Pool() as pool:
        for subject, rows, too_large, broken in pool.imap_unordered(subject_moves, jobs):
            for trial, instance, attempt, k, lp in rows:
                for key, v in zip(cols, (subject, trial, instance, attempt, k, lp)):
                    cols[key].append(v)
            print(f"{subject[:14]}: {len(rows)} moves, {too_large} trees too large, {broken} attempts cut", flush=True)
    np.savez_compressed(os.path.join(OUT, "moves.npz"), gammas=GAMMAS, **{k: np.array(v) for k, v in cols.items()})


def rollout(args):
    gamma, instance, run = args
    rng = np.random.default_rng([int(gamma * 1e4), int(instance[3:].split("_")[0]), run])
    state, prev = hd.puzzle(instance), None
    distance = hd.distance(instance)
    cap = max(MIN_STEPS, STEP_FACTOR * distance)
    for t in range(cap):
        if solved(state):
            return gamma, instance, distance, run, t, True
        try:
            probs, _, _ = om.action_distribution(state, gamma, prev)
        except om.TreeTooLarge:
            legal = [m for m in legal_moves(state) if m[2] <= om.MAX_AMOUNT]
            probs = dict.fromkeys(legal, 1 / len(legal))
        moves = list(probs)
        w = np.clip([probs[m] for m in moves], 0, None)  # float error can leave -1e-17
        move = moves[rng.choice(len(moves), p=w / w.sum())]
        state, prev = play(state, move), move[0]
    return gamma, instance, distance, run, cap, solved(state)


def simulate():
    instances = sorted(hd.trials().instance.unique())
    jobs = [(g, i, r) for g in SIM_GAMMAS for i in instances for r in range(RUNS)]
    with mp.Pool() as pool, open(os.path.join(OUT, "simulations.csv"), "w") as f:
        f.write("gamma,instance,distance,run,moves,solved\n")
        for k, row in enumerate(pool.imap_unordered(rollout, jobs, chunksize=4)):
            f.write(",".join(map(str, row)) + "\n")
            if k % 500 == 0:
                print(f"{k}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"likelihood": likelihood, "simulate": simulate}[sys.argv[1]]()
