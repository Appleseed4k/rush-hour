"""Picks a starting checkpoint and surrender scale so the student begins a
session where people do.

    python fit_surrender.py                      # rounds 3-10 of the default folder
    python fit_surrender.py --rounds 5 6 7 --runs 20

The target is people's first --block trials (default 10): solve rate and
moves / optimal over solved puzzles (this repo's lengths), each a mean of
per-person means. A checkpoint plays a puzzle set `--runs` times per puzzle
with no surrender, up to the session's move cap: --puzzles human, the very
layouts people played (human_data's 70), each human trial predicted by the
student on that layout; or --puzzles test (default), the session's held-out
test puzzles from the pool (the sessions' own puzzle source - the student
finds the human layouts much harder than pool puzzles of the same length),
each human trial predicted by the student's mean over test puzzles of its
length. Surrender is a hazard on the move count alone, so these
attempts give every scale's outcome without replaying: an attempt solved in
n moves survives the rule with probability prod over its first n - 1 moves
of (1 - scale x rate); an unsolved one stays unsolved. Each human trial in
the block is then predicted by the student's expected result on the same
puzzle, averaged as people's are. The attempts are cached per checkpoint
(--cache), so a different block or grid costs nothing.

Prints, per checkpoint, the best scale (least squares against the human
means in units of their standard errors) and the fit at scales 0 and 1.
"""
import argparse
import json
import os

import numpy as np

import human_data as hd
import rl
import session_experiment as se
import teach_experiment as te

FOLDER = os.path.join(te.OUT, "fresh_p0.1", "random_0.1")


def human_target(block=10):
    """People's first-block (solve rate, its SE), (moves / optimal over solved, its SE), and trials."""
    df = hd.trials()
    first = df[df.trial <= block]
    solve = first.groupby("subject").solved.mean()
    won = first[first.solved]
    ratio = (won.moves / won.distance).groupby(won.subject).mean()
    return te.mean_se(list(solve)), te.mean_se(list(ratio)), first


def puzzle_set(which, config_run="calibrated_v2"):
    """{name: (distance, state)}: the human layouts, or a session run's test puzzles."""
    if which == "human":
        return {n: (hd.distance(n), s) for n, s in hd.puzzles().items()}
    with open(os.path.join(se.OUT, config_run, "config.json")) as f:
        config = json.load(f)
    _, test = se.sample_puzzles(config)
    return {f"test{i}": (d, s) for i, (d, s) in enumerate(test)}


def attempts(path, puzzles, runs, step_factor=10.0, min_steps=50, seed=0, processes=None):
    """{name: [moves of each solved attempt, None if unsolved]} with no surrender."""
    agent = rl.PolicyAgent.load(path)
    names = list(puzzles)
    states = [puzzles[n] for n in names]
    caps = {s: max(min_steps, int(round(step_factor * d))) for d, s in states}
    out = rl.evaluate_intuition(agent, states, n_runs=runs, temperature=1.0, memory=10, max_steps=caps.__getitem__,
                                seed=seed, processes=processes)
    return {n: [r["moves"] for r in out[i * runs:(i + 1) * runs]] for i, n in enumerate(names)}


def survival(moves, distance, edges, rates, scale):
    """Chance an attempt solved in `moves` moves isn't given up first (no check after the solving move)."""
    rule = rl.SurrenderRule(edges, rates, {0: distance}, scale)
    return float(np.prod([1 - rule(0, k) for k in range(1, moves)]))


def predict(runs, puzzles, first, edges, rates, scale, by_length):
    """The student's (solve rate, moves / optimal over solved) on the human trials in `first`: on the same
    layout, or (by_length) pooled over `puzzles` of the trial's length."""
    per_puzzle, pooled = {}, {}
    for name, moves in runs.items():
        d = puzzles[name][0]
        w = [survival(m, d, edges, rates, scale) if m is not None else 0.0 for m in moves]
        per_puzzle[name] = (sum(w) / len(w),
                            sum(wi * m / d for wi, m in zip(w, moves) if m is not None) / sum(w) if sum(w) else
                            float("nan"))
        pooled.setdefault(d, []).append((w, moves))
    for d, group in pooled.items():
        w = [wi for ws, _ in group for wi in ws]
        moves = [m for _, ms in group for m in ms]
        pooled[d] = (sum(w) / len(w), sum(wi * m / d for wi, m in zip(w, moves) if m is not None) / sum(w))
    solve, ratio = [], []
    for _, trials in first.groupby("subject"):
        ps = ([pooled[d] for d in trials.distance] if by_length else [per_puzzle[n] for n in trials.instance])
        solve.append(np.mean([p for p, _ in ps]))
        weight = sum(p for p, r in ps if r == r)
        ratio.append(sum(p * r for p, r in ps if r == r) / weight if weight else float("nan"))
    return float(np.mean(solve)), float(np.nanmean(ratio))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--folder", default=FOLDER, help="teach run folder whose kept checkpoints are the candidates")
    ap.add_argument("--rounds", type=int, nargs="+", default=list(range(3, 11)))
    ap.add_argument("--puzzles", choices=("test", "human"), default="test")
    ap.add_argument("--runs", type=int, default=10, help="attempts per puzzle")
    ap.add_argument("--block", type=int, default=10, help="human trials 1..block are the target")
    ap.add_argument("--scales", type=float, nargs=3, default=(0.0, 6.0, 0.05), metavar=("LO", "HI", "STEP"))
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--cache", default=os.path.join(te.ROOT, "models", "sessions", "surrender_fit"))
    args = ap.parse_args(argv)

    (hs, hs_se), (hr, hr_se), first = human_target(args.block)
    print(f"people, trials 1-{args.block}: solved {hs:.1%} ± {hs_se:.1%}, moves / optimal {hr:.2f} ± {hr_se:.2f}")
    edges, rates, _ = hd.surrender_hazard()
    puzzles = puzzle_set(args.puzzles)
    manifest = json.load(open(os.path.join(args.folder, "manifest.json")))
    files = {m["round"]: m["file"] for m in manifest if m["kept"]}
    os.makedirs(args.cache, exist_ok=True)
    scales = np.arange(args.scales[0], args.scales[1] + 1e-9, args.scales[2])
    for rnd in args.rounds:
        path = os.path.join(args.folder, files[rnd])
        cache = os.path.join(args.cache, f"{os.path.basename(args.folder)}_{files[rnd][:-3]}_{args.puzzles}_runs{args.runs}.json")
        if os.path.exists(cache):
            runs = json.load(open(cache))
        else:
            runs = attempts(path, puzzles, args.runs, processes=args.processes)
            json.dump(runs, open(cache, "w"))
        fits = [(s, *predict(runs, puzzles, first, edges, rates, s, args.puzzles == "test")) for s in scales]
        loss = [((ps - hs) / hs_se) ** 2 + ((pr - hr) / hr_se) ** 2 for _, ps, pr in fits]
        best = fits[int(np.argmin(loss))]
        at = {round(s, 6): (ps, pr) for s, ps, pr in fits}
        print(f"round {rnd:>2}: best scale {best[0]:.2f} -> solved {best[1]:.1%}, moves / optimal {best[2]:.2f} "
              f"(misfit {min(loss):.1f})   scale 0: {at[0.0][0]:.1%} {at[0.0][1]:.2f}   "
              f"scale 1: {at[1.0][0]:.1%} {at[1.0][1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
