"""A human-likeness profile: move-level measures computed the same way for people and for simulated models on the
70 layouts people played (Olieslagers et al. 2023), so models can be compared with people on more than solve rate
and moves.

Everything rests on one table with a row per move (MOVE_COLUMNS), for any source:
- distance to the goal before and after (breadth-first search), so: optimal (one closer), worse, a new closest
  approach of the attempt;
- revisits: whether the board it leads to was already on the attempt's boards, and how many moves ago; whether it
  undoes the previous move; whether the moved car's new placement (car, cells) was already seen, and how long ago;
- attention: the moved car's role in the AND-OR tree of the board (Olieslagers et al.'s tree, paper_model): red,
  depth 1 (blocks red), depth 2 (blocks a blocker), depth 3+, or not in the tree; and whether it is the car moved last;
- the slide: cells moved, and whether it went as far as it could;
- people only: seconds before the move (from the previous move's end, or the attempt's start, to the first touch).

Sources: people (every attempt, solved or not; trial = the puzzle's place in the person's order), and models played
RUNS times on each layout, capped at max(50, 10 x optimal), no giving up:
- "AND-OR (paper)": Olieslagers et al.'s per-board AND-OR distribution at its best-fitting gamma (0.07 on people's
  moves, models/summary/and_or_humans);
- "AND-OR (network)": and_or.solve ordered by the start network (PolicyAgent.heuristic: its dest head when the cache
  was made, its policy head since 2026-11-24), gamma 0.07, Gumbel temperature 1 (p2.1's deliberation);
- "intuition": the start network's policy at temperature 1 with the 10-board memory (the session student);
- "forward": experiments/forward.py's planner on the start network, K8 D4, lam 16, novelty half-life 80;
- "random": uniform legal moves with the 10-board memory.

    python experiments/human_profile.py           # writes models/summary/human_profile/moves.csv.gz
    python experiments/human_profile.py smoke     # 3 layouts, 2 runs, a few people
"""
import zlib
import multiprocessing as mp
import pickle
import os
import random
import sys
from collections import deque

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import board, human_data as hd, paper_model as om, paths  # noqa: E402
from rushhour.and_or import AndNode, solve as and_or_solve  # noqa: E402
from rushhour.student import PolicyAgent  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import and_or_humans as aoh  # noqa: E402
import forward  # noqa: E402

OUT = os.path.join(paths.MODELS, "summary", "human_profile")
RUNS, STEP_FACTOR, MIN_STEPS = 10, 10, 50
GAMMA = 0.07
MODELS = ["AND-OR (paper)", "AND-OR (network)", "intuition", "forward", "random", "intuition, soft memory",
          "forward, soft memory", "forward, fitted", "value planner, soft memory", "backward 1, soft memory",
          "backward 4, soft memory", "forward 4 + backward 4, soft memory"]
SOFT = (2, 1)  # the soft board memory fitted to people's revisits (experiments/fit_memory.py): beta 2, half-life 1
ROLES = ["red", "depth 1", "depth 2", "depth 3+", "not in tree", "tree too large"]


# --- the per-move table ----------------------------------------------------------------------------------------

def car_roles(state):
    """{car: role} on `state`: its shallowest place in the board's AND-OR tree (red's blockers are depth 1)."""
    try:
        _, _, OR, _ = om.build_tree(state, True)
    except (om.TreeTooLarge, RecursionError):
        return None
    depth = {}
    for d, row in OR:  # OR node (d, (blocker, amounts...)): a subgoal at depth d - 1 below red
        depth[row[0]] = min(depth.get(row[0], 99), d - 1)
    return {name: "red" if name == "red" else ("not in tree" if name not in depth else
                                                f"depth {depth[name]}" if depth[name] < 3 else "depth 3+")
            for name, _ in state}


def rows_of(source, who, trial, instance, attempt, boards, moves, solved, dist, roles, seconds=None):
    """One dict per move of an attempt (boards: the start and every board after a move)."""
    out = []
    last_board, last_place = {boards[0]: 0}, {f: 0 for f in boards[0]}
    best = dist[boards[0]]
    orient = board.car_orientations(boards[0])
    for t, (move, b0, b1) in enumerate(zip(moves, boards, boards[1:])):
        car, direction, steps = move
        place = (car, dict(b1)[car])
        reach = max(s for (c, d, s), _ in board.neighbors(b0, orient) if c == car and d == direction)
        r = roles.get(b0)
        out.append({
            "source": source, "who": who, "trial": trial, "instance": instance, "attempt": attempt,
            "optimal length": dist[boards[0]], "solved": solved, "move": t,
            "distance": dist[b0], "after": dist[b1],
            "new best": dist[b1] < best,
            "board seen": b1 in last_board, "board age": t + 1 - last_board[b1] if b1 in last_board else np.nan,
            "undo": t >= 1 and b1 == boards[t - 1],
            "placement seen": place in last_place,
            "placement age": t + 1 - last_place[place] if place in last_place else np.nan,
            "role": "tree too large" if r is None else r[car],
            "same car": t >= 1 and moves[t - 1][0] == car,
            "steps": steps, "full slide": steps == reach,
            "seconds": np.nan if seconds is None else seconds[t],
        })
        best = min(best, dist[b1])
        last_board[b1] = t + 1
        for f in b1:
            last_place[f] = t + 1
    return out


# --- people -----------------------------------------------------------------------------------------------------

def people_attempts():
    """[(who, trial, instance, attempt, boards, moves, solved, seconds)] for every replayable attempt."""
    trial_of = {s: dict(zip(d.instance, d.trial)) for s, d in hd.trials().groupby("subject")}
    out, cut = [], 0
    for subject, attempts in hd.load(unsolved=True).items():
        for instance, attempt, solved, moves in attempts:
            name = aoh.piece_names(instance)
            state = hd.puzzle(instance)
            boards, played, seconds = [state], [], []
            try:
                for m in moves:
                    nxt, move = aoh.human_move(state, name(m.piece), m.c0, m.c1)
                    if move not in {mv for mv, _ in board.neighbors(state, board.car_orientations(state))}:
                        raise ValueError("not a legal slide")  # e.g. through another car
                    state = nxt
                    boards.append(state)
                    played.append(move)
                    seconds.append((m.t_touch - m.t_prev) / 1000)
            except ValueError:  # a slide that doesn't replay: keep the attempt up to it
                solved, cut = False, cut + 1
            if played:
                out.append((subject, trial_of[subject].get(instance), instance, attempt, boards, played,
                            solved, seconds))  # the recorded win: red's path is clear (the drag off the board isn't a slide)
    print(f"{cut} human attempts cut at a slide that doesn't replay", flush=True)
    return out


# --- models -----------------------------------------------------------------------------------------------------

_W = {}


def _init(weights):
    torch.set_num_threads(1)
    _W["agent"] = forward._load_bytes(weights) if weights is not None else None


def _record(state, play_one, cap):
    boards, moves = [state], []
    while len(moves) < cap and not aoh.solved(state):
        step = play_one(state, moves)
        if not step:
            break
        for move in step:
            state = aoh.play(state, move)
            boards.append(state)
            moves.append(move)
            if aoh.solved(state) or len(moves) >= cap:
                break
    return boards, moves, aoh.solved(state)


def simulate(job):
    """One model attempt on one layout: (model, run, instance, boards, moves, solved)."""
    model, instance, run = job
    seed = zlib.crc32(f"{model}|{instance}|{run}".encode())
    rng, nprng = random.Random(seed), np.random.default_rng(seed)
    random.seed(seed)
    start = hd.puzzle(instance)
    cap = max(MIN_STEPS, STEP_FACTOR * hd.distance(instance))
    agent = _W["agent"]
    orient = board.car_orientations(start)

    if model == "AND-OR (paper)":
        def play_one(state, moves):
            try:
                probs, _, _ = om.action_distribution(state, GAMMA, moves[-1][0] if moves else None)
            except (om.TreeTooLarge, RecursionError):
                legal = [m for m, _ in board.neighbors(state, orient) if m[2] <= om.MAX_AMOUNT]
                probs = dict.fromkeys(legal, 1 / len(legal))
            options = list(probs)
            w = np.clip([probs[m] for m in options], 0, None)
            return [options[nprng.choice(len(options), p=w / w.sum())]]
        boards, moves, solved = _record(start, play_one, cap)
    elif model == "AND-OR (network)":
        agent.temperature = 1.0

        def play_one(state, moves):
            att = and_or_solve(state, heuristic=agent.heuristic, gamma=GAMMA)
            return att if type(att) is list else att[1]
        boards, moves, solved = _record(start, play_one, cap)
    else:
        fw = {"k": 8, "depth": 4, "evaluation": "weighted", "lam": 16, "half_life": 80}
        kw = {"intuition": {"k": 1}, "random": {"k": 1, "proposals": "uniform"}, "forward": fw,
              "intuition, soft memory": {"k": 1, "soft_memory": SOFT},
              "forward, soft memory": fw | {"soft_memory": SOFT},
              # fitted to people's revisits (fit_memory.py planner): little novelty, short-lived
              "forward, fitted": fw | {"lam": 2, "half_life": 2, "soft_memory": SOFT},
              "value planner, soft memory": {"k": 8, "depth": 4, "evaluation": "value", "soft_memory": SOFT},
              # forward and backward together (forward.py INTEGRATE)
              "backward 1, soft memory": {"k": 0, "backward": 1, "evaluation": "value", "soft_memory": SOFT},
              "backward 4, soft memory": {"k": 0, "backward": 4, "evaluation": "value", "soft_memory": SOFT},
              "forward 4 + backward 4, soft memory": {"k": 4, "depth": 4, "backward": 4, "evaluation": "value",
                                                      "soft_memory": SOFT}}[model]
        trace = []
        out = forward.solve_forward(agent, start, cap, rng=rng, trace=trace, **kw)
        boards, moves, solved = [start] + [b for _, b in trace], [m for m, _ in trace], out["solved"]
    return model, run, instance, boards, moves, solved


# --- roles, in parallel over the distinct boards ----------------------------------------------------------------

def _roles_job(states):
    return [(s, car_roles(s)) for s in states]


def all_roles(states, pool):
    states = list(states)
    chunks = [states[i::256] for i in range(256)]
    roles = {}
    for part in pool.imap_unordered(_roles_job, chunks):
        roles.update(part)
    return roles


def build(instances=None, runs=RUNS, max_people=None):
    instances = instances or sorted(hd.trials().instance.unique())
    people = [a for a in people_attempts() if a[2] in instances]
    if max_people:
        keep = sorted({a[0] for a in people})[:max_people]
        people = [a for a in people if a[0] in keep]
    with open(paths.START, "rb") as f:
        weights = f.read()
    jobs = [(m, i, r) for m in MODELS for i in instances for r in range(runs)]
    random.Random(0).shuffle(jobs)
    cache = os.path.join(OUT, f"sims_{len(instances)}x{runs}.pkl")  # the simulations and roles, reused if present
    sims, roles = [], {}
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            sims, roles = pickle.load(f)
    done = {s[0] for s in sims}
    with mp.Pool(initializer=_init, initargs=(weights,)) as pool:
        sims += pool.map(simulate, [j for j in jobs if j[0] not in done], chunksize=1)
        print(f"{len(people)} human attempts, {len(sims)} model attempts", flush=True)
        boards = {b for a in people for b in a[4][:-1]} | {b for s in sims for b in s[3][:-1]}
        roles.update(all_roles(boards - set(roles), pool))
    with open(cache, "wb") as f:
        pickle.dump((sims, roles), f)
    print(f"{len(boards)} distinct boards; {sum(r is None for r in roles.values())} trees too large", flush=True)
    dists = {i: board.multi_bfs(hd.puzzle(i)) for i in instances}
    rows = []
    for who, trial, instance, attempt, bs, moves, solved, seconds in people:
        rows += rows_of("people", who, trial, instance, attempt, bs, moves, solved, dists[instance], roles, seconds)
    for model, run, instance, bs, moves, solved in sims:
        rows += rows_of(model, run, None, instance, 0, bs, moves, solved, dists[instance], roles)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] == ["smoke"]:
        df = build(sorted(hd.trials().instance.unique())[:3], runs=2, max_people=5)
        print(df.groupby("source")[["new best", "board seen", "placement seen", "same car", "full slide"]].mean()
              .round(3).to_string())
        print(df.groupby(["source", "role"]).size().unstack(fill_value=0).to_string())
        sys.exit()
    df = build()
    df.to_csv(os.path.join(OUT, "moves.csv.gz"), index=False)
    print("wrote", len(df), "moves")
