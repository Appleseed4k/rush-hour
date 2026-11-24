"""An animated page showing how the two-mode student (forward.solve_lean) chooses each move of one attempt.

Each move it draws a mode: with probability w_ao a deliberation (one AND-OR pass from the board, its first move
played), with probability w_flat flat planning (the policy's top 5 moves, a softmax over the value head's moves to
go), otherwise intuition (a policy sample); the first move of an attempt is always a deliberation and the next is the
same pass's second move (the opening). The soft board memory lowers moves back to recent boards in intuition and flat.

`explain` replays solve_lean move by move with the same random draws and records what each mode looked at: the draw
against the mode weights, intuition's probabilities, flat planning's candidates with their moves to go, AND-OR's plan.
It is checked against solve_lean itself (same seed, same moves). The page (two_mode_template.html filled in, written
to visualization/p2_two_mode/, which git ignores) steps through
the attempt: the board with the move played, the draw, the candidates, and a timeline of modes and distance to goal.

    python experiments/two_mode_animation.py                        # picks an attempt that shows every mode
    python experiments/two_mode_animation.py --layout prb10166_16 --seed 3

Defaults: the session's start network and the two-mode session's weights at its start (w_flat 0.08, w_ao 0.1, the
opening), on one of people's layouts, no giving up.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import forward
from forward import AndNode, _red_steps, and_or_solve, board, values
from rushhour import human_data as hd, paths

OUT = os.path.join(paths.ROOT, "visualization", "p2_two_mode", "two_mode_moves.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_mode_template.html")


def _move(m):
    car, direction, steps = m
    return {"car": car, "dir": direction, "steps": steps}


def explain(agent, state, max_steps, w_flat=0.08, w_ao=0.10, opening=True, consider=5, tau=1.0, gamma=0.07,
            soft_memory=(2, 1), rng=None):
    """solve_lean (sight 0, temperature 1, no giving up), recording each move's decision. Returns (frames, solved)."""
    agent.temperature = 1.0
    orientations = board.car_orientations(state)
    dist = board.multi_bfs(state)
    last_board = {state: 0}
    beta, half = soft_memory
    moves, pending, frames = 0, [], []

    def penalty(b):
        when = last_board.get(b)
        return 0.0 if when is None else beta * 0.5 ** ((moves + 1 - when) / half)

    while moves < max_steps and _red_steps(state) > 0:
        r = rng.random()
        options = list(board.neighbors(state, orientations))
        pen = [penalty(b) for _, b in options]
        frame = {"board": [[n, [list(p) for p in ps]] for n, ps in state], "distance": dist[state], "draw": r,
                 "w_ao": w_ao, "w_flat": w_flat, "plan": None, "candidates": [], "why": "draw"}
        played = None
        if pending:
            played = pending.pop(0)
            frame.update(mode="and-or", why="follow-through")
            frame["plan"] = frames[-1]["plan"]
        elif (opening and moves == 0) or r < w_ao:
            att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
            steps = (att if type(att) is list else att[1]) or []
            frame["why"] = "opening" if opening and moves == 0 else "draw"
            if steps:
                plan, b = [], state
                for s in steps:
                    b2 = AndNode(b, *s).apply(b)
                    plan.append({**_move(s), "board": [[n, [list(p) for p in ps]] for n, ps in b2]})
                    b = b2
                frame["plan"] = {"moves": plan, "complete": type(att) is list}
                nxt = AndNode(state, *steps[0]).apply(state)
                played = (steps[0], nxt)
                frame["mode"] = "and-or"
                if opening and moves == 0 and len(steps) > 1 and _red_steps(nxt) > 0:
                    pending = [(steps[1], AndNode(nxt, *steps[1]).apply(nxt))]
            else:
                frame["why"] = "and-or found nothing"
        elif r < w_ao + w_flat:
            logits = agent.policy_logits(state)[[agent.action_id(state, *m) for m, _ in options]].tolist()
            order = sorted(range(len(options)), key=lambda i: logits[i] - pen[i], reverse=True)
            pool = order[:consider]
            win = [i for i in pool if _red_steps(options[i][1]) <= 0]
            ends = values(agent, [options[i][1] for i in pool])
            if win:
                probs, pick = [1.0 if i == win[0] else 0.0 for i in pool], win[0]
            else:
                probs = torch.softmax(torch.tensor([-e / tau - pen[i] for e, i in zip(ends, pool)]), 0).tolist()
                pick = pool[rng.choices(range(len(pool)), probs)[0]]
            played = options[pick]
            frame["mode"] = "flat"
            frame["candidates"] = [{**_move(options[i][0]), "togo": e, "p": p, "penalty": pen[i],
                                    "true": dist[options[i][1]], "chosen": i == pick}
                                   for i, e, p in zip(pool, ends, probs)]
        if played is None:
            logits = agent.policy_logits(state)[[agent.action_id(state, *m) for m, _ in options]]
            probs = torch.softmax(logits - torch.tensor(pen, dtype=logits.dtype), dim=0).tolist()
            played = rng.choices(options, probs)[0]
            frame["mode"] = "intuition"
            ranked = sorted(range(len(options)), key=lambda i: -probs[i])
            frame["candidates"] = [{**_move(options[i][0]), "p": probs[i], "penalty": pen[i],
                                    "true": dist[options[i][1]], "chosen": options[i] == played} for i in ranked]
        move, state = played
        moves += 1
        last_board[state] = moves
        frame.update(move=_move(move), after=dist[state], optimal=dist[state] < frame["distance"])
        frames.append(frame)
    return frames, _red_steps(state) <= 0


def attempt(agent, state, seed, cap, **kw):
    """explain() and solve_lean on the same seed; asserts they play the same moves."""
    agent.__dict__.pop("_values", None)
    random.seed(seed)
    torch.manual_seed(seed)
    frames, solved = explain(agent, state, cap, rng=random.Random(seed), **kw)
    agent.__dict__.pop("_values", None)
    random.seed(seed)
    torch.manual_seed(seed)
    trace = []
    forward.solve_lean(agent, state, cap, rng=random.Random(seed), trace=trace, **kw)
    assert [_move(m) for m, _ in trace] == [f["move"] for f in frames], "explain() drifted from solve_lean"
    return frames, solved


def pick(agent, layouts, seeds, **kw):
    """The first (layout, seed) whose attempt solves in 1.5-3.5 x optimal and has at least two flat moves and two
    AND-OR moves after the opening - an attempt that shows every mode (chosen, not typical)."""
    for name in layouts:
        state, d = hd.puzzle(name), hd.distance(name)
        for seed in seeds:
            frames, solved = attempt(agent, state, seed, max(50, 10 * d), **kw)
            modes = [f["mode"] for f in frames[2:]]
            if solved and 1.5 * d <= len(frames) <= 3.5 * d and modes.count("flat") >= 2 and modes.count("and-or") >= 2:
                return name, seed, frames
    raise SystemExit("no attempt found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--w-flat", type=float, default=0.08)
    ap.add_argument("--w-ao", type=float, default=0.10)
    ap.add_argument("--net", default=paths.START)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    with open(args.net, "rb") as f:
        agent = forward._load_bytes(f.read())
    kw = {"w_flat": args.w_flat, "w_ao": args.w_ao, "opening": True}
    if args.layout:
        state, d = hd.puzzle(args.layout), hd.distance(args.layout)
        name, seed = args.layout, args.seed or 0
        frames, _ = attempt(agent, state, seed, max(50, 10 * d), **kw)
    else:
        layouts = [n for n in sorted(hd.trials().instance.unique()) if hd.distance(n) in (10, 13)]
        name, seed, frames = pick(agent, layouts, range(40), **kw)
    data = {"layout": name, "seed": seed, "optimal": hd.distance(name), "network": os.path.basename(args.net),
            "w_flat": args.w_flat, "w_ao": args.w_ao, "frames": frames}
    with open(TEMPLATE, encoding="utf-8") as f:
        page = f.read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    modes = [fr["mode"] for fr in frames]
    print(f"{name} (optimal {data['optimal']}), seed {seed}: {len(frames)} moves - "
          + ", ".join(f"{m} {modes.count(m)}" for m in ("intuition", "flat", "and-or")) + f" -> {args.out}")


if __name__ == "__main__":
    main()
