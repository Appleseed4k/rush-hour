"""A prototype forward planner for the student: heuristically guided rollouts, scored by the novelty of the
features they reach, weighted by the dest head's relevance. Offline only - the session student still plays by
intuition (solve_intuition); this asks whether rollouts would help it, and whether they help more as the network
gains experience.

Every turn, on board b:
1. relevance: the dest head on b gives every feature (car, positions) - a car standing somewhere on its axis -
   r(f) = sigmoid(dest logit), its estimate of how soon an optimal route brings that car there (MODEL.md: target
   0.8^(k-1));
2. proposals: K rollouts of up to D moves, each move sampled from the policy head (softmax at `temperature` over
   legal moves, never back to one of the last `memory` boards of the attempt or to a board earlier in the rollout,
   unless every move would be);
3. evaluation: a rollout that solves the puzzle wins (the shortest); otherwise its score is the sum of r(f) over the
   features it reaches that are new - not on any board the attempt has seen (novelty="attempt"; "board": not on b,
   so no memory of the attempt), each counted once, discounted by `discount`^(t-1) at the move t it is reached;
   relevance="none" counts each new feature as 1 (novelty alone);
4. the first move of the best rollout (the first sampled, on a tie) is played.

K=1 is intuition exactly: one sampled move. Rollout boards are evaluated in batches of K.

    python experiments/forward.py smoke                # a few puzzles, to time it
    python experiments/forward.py                      # the comparison: models/summary/forward.csv
    python experiments/forward.py uniform              # the uniform-proposal control: forward_uniform.csv
    python experiments/forward.py value                # the value head as evaluator: forward_value.csv
"""
import io
import math
import os
import random
import sys
import time
from collections import deque
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, paths, sessions  # noqa: E402
from rushhour.and_or import _red_steps  # noqa: E402
from rushhour.student import PolicyAgent  # noqa: E402
from rushhour.student.labels import _destinations  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "forward.csv")
SESSION = os.path.join(paths.MODELS, "sessions", "aostart_free_m10_s2_agency1")


def evaluate_many(agent, states):
    """Fills agent._evaluate's cache for every state in `states` not yet in it, in one batch."""
    missing = list({s for s in states if s not in agent._scores})
    if not missing:
        return
    if len(agent._scores) > 50_000:
        agent._scores.clear()
    agent.net.eval()
    with torch.no_grad():
        policy, dest, _ = agent.net.heads(torch.stack([agent.encoder.encode(s) for s in missing]).to(agent.device))
    for s, p, d in zip(missing, policy.cpu(), dest.cpu()):
        agent._scores[s] = (p, d)


def values(agent, states):
    """The value head's moves to go (its output x 10) for each of `states`, batched and cached on the agent."""
    cache = agent.__dict__.setdefault("_values", {})
    missing = list({s for s in states if s not in cache})
    if missing:
        if len(cache) > 50_000:
            cache.clear()
        agent.net.eval()
        with torch.no_grad():
            v = agent.net.heads(torch.stack([agent.encoder.encode(s) for s in missing]).to(agent.device))[2]
        cache.update(zip(missing, (10 * v.cpu()).tolist()))
    return [cache[s] for s in states]


def relevance_map(agent, state):
    """{(car, positions): sigmoid(dest logit)} for every on-board destination of every car on `state`."""
    dest = agent._evaluate(state)[1]
    sd = dict(state)
    out = {}
    for car, direction, steps, moved in _destinations(state, agent.size, agent.max_slide):
        out[(car, moved)] = 1 / (1 + math.exp(-dest[agent.action_id(state, car, direction, steps)].item()))
    return out


def sample_move(agent, state, orientations, avoid, temperature, rng, uniform=False):
    """A (move, next board) sampled from the policy head (`uniform`: uniformly, the network unused), avoiding
    boards in `avoid` if it can."""
    options = list(board.neighbors(state, orientations))
    fresh = [o for o in options if o[1] not in avoid] or options
    if uniform:
        return rng.choice(fresh)
    logits = agent._evaluate(state)[0][[agent.action_id(state, *m) for m, _ in fresh]]
    if temperature > 0:
        return rng.choices(fresh, torch.softmax(logits / temperature, dim=0).tolist())[0]
    return fresh[int(logits.argmax())]


def solve_forward(agent, state, max_steps, k=8, depth=4, memory=10, temperature=1.0, novelty="attempt",
                  relevance="dest", discount=1.0, proposals="policy", evaluation="novelty", rng=None):
    """One attempt by the forward planner (module docstring). proposals="uniform": rollout moves drawn uniformly
    from the legal ones (same avoidance), so the network acts only through relevance (or value).

    evaluation: how unsolved rollouts are ranked - "novelty" (the module docstring's score), "value" (the end board's
    moves to go by the value head, fewest first; no novelty) or "novelty then value" (novelty score first, the value
    head breaking ties, as best-first width search ranks by novelty, then a goal heuristic).

    Returns {"solved", "moves", "ties"} - ties: turns on which every rollout had the same novelty score (with
    "novelty", nothing to choose between, so a plain policy sample; with "novelty then value", the value decides)."""
    rng = rng or random
    uniform = proposals == "uniform"
    orientations = board.car_orientations(state)
    recent = deque([state], maxlen=memory)
    seen = set(state)
    moves, ties = 0, 0
    while moves < max_steps and _red_steps(state) > 0:
        if k == 1:
            move, state = sample_move(agent, state, orientations, recent, temperature, rng, uniform)
        else:
            use_novelty, use_value = evaluation != "value", evaluation != "novelty"
            rel = relevance_map(agent, state) if relevance == "dest" and use_novelty else None
            old = seen if novelty == "attempt" else set(state)
            paths_ = [[state] for _ in range(k)]
            firsts = [None] * k
            done = [None] * k  # the move at which a rollout solves
            for t in range(depth):
                live = [i for i in range(k) if done[i] is None]
                if not live:
                    break
                if not uniform:
                    evaluate_many(agent, [paths_[i][-1] for i in live])
                for i in live:
                    move, nxt = sample_move(agent, paths_[i][-1], orientations, set(recent) | set(paths_[i]),
                                            temperature, rng, uniform)
                    if t == 0:
                        firsts[i] = (move, nxt)
                    paths_[i].append(nxt)
                    if _red_steps(nxt) <= 0:
                        done[i] = t + 1
            solved = [i for i in range(k) if done[i] is not None]
            if solved:
                best = min(solved, key=lambda i: done[i])
            else:
                scores = [0.0] * k if not use_novelty else []
                for path in paths_ if use_novelty else []:
                    got, score = set(), 0.0
                    for t, b in enumerate(path[1:]):
                        for f in b:
                            if f not in old and f not in got:
                                got.add(f)
                                score += (rel[f] if rel is not None else 1.0) * discount ** t
                    scores.append(score)
                ends = [-v for v in values(agent, [path[-1] for path in paths_])] if use_value else [0.0] * k
                best = max(range(k), key=lambda i: (scores[i], ends[i]))  # the first, on a full tie
                ties += use_novelty and all(s == scores[best] for s in scores)
            move, state = firsts[best]
        moves += 1
        recent.append(state)
        seen.update(state)
    return {"solved": _red_steps(state) <= 0, "moves": moves, "ties": ties}


_W = {}


def _init(weights):
    torch.set_num_threads(1)
    _W["agents"] = {name: _load_bytes(b) for name, b in weights.items()}


def _load_bytes(b):
    checkpoint = torch.load(io.BytesIO(b), weights_only=False, map_location="cpu")
    agent = PolicyAgent(size=checkpoint["size"], max_slide=checkpoint["max_slide"], hidden=checkpoint["hidden"],
                        blocks=checkpoint["blocks"])
    agent.net.load_state_dict(checkpoint["state_dict"])
    return agent


def _job(job):
    net, cond, state, distance, cap, seed = job
    t0 = time.time()
    out = solve_forward(_W["agents"][net], state, cap, rng=random.Random(seed), **cond["kw"])
    return {"network": net, "condition": cond["name"], "distance": distance, "cap": cap, **out,
            "seconds": time.time() - t0}


def puzzles(per_bin=25):
    """The session's held-out test puzzles (aostart runs, seed 0): the first `per_bin` of each bin's 50."""
    config = {"test_bins": ["9-11", "14-16", "19-21", "24-26"], "test_per_bin": 50, "seed": 0}
    test = board.sample_by_distance(board.load_puzzle_pool(paths.POOL), sessions.bins(config["test_bins"]), 50,
                                    seed=0)
    return [p for b in range(4) for p in test[b * 50: b * 50 + per_bin]]


def networks():
    nets = {"start": paths.START,
            "round 3": os.path.join(paths.APPRENTICE, "gamma0.07_free", "round3_epoch03.pt")}
    for teacher in ("no_teacher", "random_1.0"):
        for s in range(8):
            nets[f"{teacher} 720 s{s}"] = os.path.join(SESSION, teacher, f"student{s:02d}_puzzle720.pt")
    return nets


CONDITIONS = [
    {"name": "intuition", "kw": {"k": 1}},
    {"name": "K8 D4", "kw": {"k": 8, "depth": 4}},
    {"name": "K8 D4 novelty only", "kw": {"k": 8, "depth": 4, "relevance": "none"}},
    {"name": "K8 D4 no attempt memory", "kw": {"k": 8, "depth": 4, "novelty": "board"}},
    {"name": "K16 D8", "kw": {"k": 16, "depth": 8}},
]

# the value head as the evaluator: alone, and breaking novelty's ties
VALUE = [
    {"name": "K8 D4 value", "kw": {"k": 8, "depth": 4, "evaluation": "value"}},
    {"name": "K8 D4 novelty then value", "kw": {"k": 8, "depth": 4, "evaluation": "novelty then value",
                                                "relevance": "none"}},
    {"name": "K8 D4 uniform value", "kw": {"k": 8, "depth": 4, "evaluation": "value", "proposals": "uniform"}},
    {"name": "K8 D4 uniform novelty then value", "kw": {"k": 8, "depth": 4, "evaluation": "novelty then value",
                                                        "relevance": "none", "proposals": "uniform"}},
]

# the control: rollouts proposed uniformly, judged with and without the dest head (novelty only uses no network)
UNIFORM = [
    {"name": "K8 D4 uniform", "kw": {"k": 8, "depth": 4, "proposals": "uniform"}},
    {"name": "K8 D4 uniform novelty only", "kw": {"k": 8, "depth": 4, "proposals": "uniform", "relevance": "none"}},
    {"name": "intuition uniform", "kw": {"k": 1, "proposals": "uniform"}},
]


def run(nets, conds, items, runs, processes=None):
    weights = {}
    for name, path in nets.items():
        with open(path, "rb") as f:
            weights[name] = f.read()
    jobs = [(net, cond, state, d, max(50, 10 * d), 1_000_003 * r + 1009 * i + c)
            for net in nets for c, cond in enumerate(conds) for i, (d, state) in enumerate(items)
            for r in range(runs)]
    random.Random(0).shuffle(jobs)  # mix slow and fast jobs across workers
    with Pool(processes, initializer=_init, initargs=(weights,)) as pool:
        return pd.DataFrame(pool.map(_job, jobs, chunksize=1))


def summarize(df):
    df = df.assign(ratio=np.where(df.solved, df.moves, df.cap) / df.distance,
                   group=df.network.str.replace(r" s\d$", "", regex=True))
    return (df.groupby(["group", "condition"])
            .agg(solved=("solved", "mean"), ratio=("ratio", "mean"), ties=("ties", "sum"), moves=("moves", "sum"),
                 seconds=("seconds", "mean"), n=("solved", "size"))
            .assign(ties=lambda g: g.ties / g.moves).drop(columns="moves"))


if __name__ == "__main__":
    if sys.argv[1:2] == ["smoke"]:
        items = puzzles(2)
        nets = {k: v for k, v in networks().items() if k in ("start", "round 3")}
        df = run(nets, CONDITIONS, items, 1)
        print(summarize(df).round(3).to_string())
        sys.exit()
    if sys.argv[1:2] in (["uniform"], ["value"]):  # models/summary/forward_uniform.csv, forward_value.csv
        df = run(networks(), {"uniform": UNIFORM, "value": VALUE}[sys.argv[1]], puzzles(), 2)
        df.to_csv(OUT.replace(".csv", f"_{sys.argv[1]}.csv"), index=False)
        print(summarize(df).round(3).to_string())
        sys.exit()
    df = run(networks(), CONDITIONS, puzzles(), 2)
    df.to_csv(OUT, index=False)
    print(summarize(df).round(3).to_string())
    print("wrote", OUT)
