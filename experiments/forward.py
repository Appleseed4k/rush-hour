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
    python experiments/forward.py sweep                # novelty's weight against value, and fading memory
    python experiments/forward.py integrate            # forward rollouts and backward AND-OR plans as candidates
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
from rushhour.and_or import AndNode, _red_steps, solve as and_or_solve  # noqa: E402
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
    if len(cache) > 50_000:  # clear before working out what is missing, so every state asked for is filled
        cache.clear()
    missing = list({s for s in states if s not in cache})
    if missing:
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


def sample_move(agent, state, orientations, avoid, temperature, rng, uniform=False, penalty=None):
    """A (move, next board) sampled from the policy head (`uniform`: uniformly, the network unused), avoiding
    boards in `avoid` if it can. With `penalty` (board -> a number), nothing is avoided: each move's logit is
    lowered by the penalty of the board it leads to (a soft memory)."""
    options = list(board.neighbors(state, orientations))
    if penalty is not None:
        logits = (torch.zeros(len(options)) if uniform else
                  agent._evaluate(state)[0][[agent.action_id(state, *m) for m, _ in options]] / max(temperature, 1e-9))
        logits = logits - torch.tensor([penalty(b) for _, b in options], dtype=logits.dtype)
        return rng.choices(options, torch.softmax(logits, dim=0).tolist())[0]
    fresh = [o for o in options if o[1] not in avoid] or options
    if uniform:
        return rng.choice(fresh)
    logits = agent._evaluate(state)[0][[agent.action_id(state, *m) for m, _ in fresh]]
    if temperature > 0:
        return rng.choices(fresh, torch.softmax(logits / temperature, dim=0).tolist())[0]
    return fresh[int(logits.argmax())]


def solve_forward(agent, state, max_steps, k=8, depth=4, memory=10, temperature=1.0, novelty="attempt",
                  relevance="dest", discount=1.0, proposals="policy", evaluation="novelty", lam=1.0, half_life=None, rng=None, trace=None, soft_memory=None, backward=0, gamma=0.07, rollout_gamma=None,
                  candidate_memory=0.0, commit=None, sources=None, surrender=None):
    """One attempt by the forward planner (module docstring). proposals="uniform": rollout moves drawn uniformly
    from the legal ones (same avoidance), so the network acts only through relevance (or value).

    evaluation: how unsolved rollouts are ranked - "novelty" (the module docstring's score), "value" (the end board's
    moves to go by the value head, fewest first; no novelty) or "novelty then value" (novelty score first, the value
    head breaking ties, as best-first width search ranks by novelty, then a goal heuristic) or "weighted" (one
    compensatory score: lam x novelty - the end board's moves to go, so lam is how many moves of predicted progress
    one new feature is worth; lam -> infinity is "novelty then value", lam = 0 is "value").

    With "weighted", novelty fades: a feature last on the attempt's boards `age` moves ago counts
    1 - 0.5^(age / half_life) (1 if never seen; 0 on the current board), so a placement seen long ago detracts
    less than one seen recently. half_life=None never forgets (every seen feature counts 0, as in "novelty").

    `trace`, if a list, gets (move, board after it) for every move played.

    soft_memory=(beta, half_life) replaces the hard rule (never back to one of the last `memory` boards) with a
    fading one: a move to a board the attempt last had `age` moves ago (2 = undoing the last move) has its logit
    lowered by beta x 0.5^(age / half_life) - for the move played and inside rollouts (where the rollout's own
    boards count too, at their place in it).

    backward=N adds N candidates from backward reasoning to the k policy rollouts each turn (k may be 0): one
    and_or.solve pass each from the current board, ordered by the agent's dest head with Gumbel noise at temperature 1
    and lapse rate `gamma` - a whole plan to the goal (it competes as a solving candidate) or the moves made until a
    lapse. Candidates of both kinds are judged alike by `evaluation`; the first move of the best is played.

    rollout_gamma: rollouts lose the thread as AND-OR does - after each simulated move past the first, a rollout
    stops with this probability (a geometric length averaging 1 / rollout_gamma moves, capped at `depth`). Setting it
    to `gamma` shares one lapse rate between forward and backward reasoning.

    Backward plans and the memory (the mixed planner undoes too often - each turn's fresh AND-OR pass can move a car
    straight back):
    candidate_memory=c: the soft memory also judges the candidates - a candidate whose first move leads to a board the
    attempt had `age` moves ago is ranked as if c x beta x 0.5^(age / half_life) more moves from the goal (a solving
    one as that many moves longer).
    commit="plan": once a backward plan is played, the student keeps to it without deliberating, each later move with
    probability 1 - gamma (the shared lapse), until it ends or lapses. commit="candidate": the rest of the plan joins
    the next turn's candidates (judged alike; a whole plan to the goal still counts as solving).
    `sources`, if a list, gets "forward", "backward" or "commit" for every move played.

    `surrender` (a function of moves made, as solve_intuition's): after each move that doesn't solve the puzzle, the
    student gives up with that probability (drawn from `rng`).

    Returns {"solved", "surrendered", "moves", "ties", "backward"} - backward: turns on which a backward plan was chosen; ties: turns on which every rollout had the same novelty score (with
    "novelty", nothing to choose between, so a plain policy sample; with "novelty then value", the value decides)."""
    rng = rng or random
    uniform = proposals == "uniform"
    if backward:
        agent.temperature = 1.0  # Gumbel noise in the heuristic, as AND-OR's network ordering uses it
    orientations = board.car_orientations(state)
    recent = deque([state], maxlen=memory)
    seen = set(state)
    last_seen = {f: 0 for f in state}  # feature -> the move count when it was last on the attempt's board
    last_board = {state: 0}  # board -> the move count when the attempt was last there

    def soft(now, extra=None):
        """The soft memory's penalty for a move made after `now` moves (extra: a rollout's own boards)."""
        if soft_memory is None:
            return None
        beta, half = soft_memory

        def penalty(b):
            when = extra.get(b, last_board.get(b)) if extra else last_board.get(b)
            return 0.0 if when is None else beta * 0.5 ** ((now + 1 - when) / half)
        return penalty
    moves, ties, chose_backward = 0, 0, 0
    plan_left, plan_solves = [], False  # the rest of the backward plan last played: [(move, board)]

    def first_penalty(b):
        pen = soft(moves)
        return candidate_memory * pen(b) if candidate_memory and pen else 0.0
    while moves < max_steps and _red_steps(state) > 0:
        source = "forward"
        if commit == "plan" and plan_left and rng.random() >= gamma:
            move, state = plan_left.pop(0)
            source = "commit"
            chose_backward += 1
        elif k == 1 and not backward:
            move, state = sample_move(agent, state, orientations, recent, temperature, rng, uniform, soft(moves))
        else:
            use_novelty, use_value = evaluation != "value", evaluation != "novelty"
            rel = relevance_map(agent, state) if relevance == "dest" and use_novelty else None
            old = seen if novelty == "attempt" else set(state)
            paths_ = [[state] for _ in range(k)]
            firsts = [None] * k
            done = [None] * k  # the move at which a rollout solves
            stopped = set()  # rollouts that lost the thread (rollout_gamma)
            for t in range(depth):
                if rollout_gamma is not None and t > 0:
                    stopped |= {i for i in range(k) if done[i] is None and rng.random() < rollout_gamma}
                live = [i for i in range(k) if done[i] is None and i not in stopped]
                if not live:
                    break
                if not uniform:
                    evaluate_many(agent, [paths_[i][-1] for i in live])
                for i in live:
                    own = {b: moves + j for j, b in enumerate(paths_[i])} if soft_memory else None
                    move, nxt = sample_move(agent, paths_[i][-1], orientations, set(recent) | set(paths_[i]),
                                            temperature, rng, uniform, soft(moves + t, own))
                    if t == 0:
                        firsts[i] = (move, nxt)
                    paths_[i].append(nxt)
                    if _red_steps(nxt) <= 0:
                        done[i] = t + 1
            plans = {}  # candidate -> (its [(move, board)], whether it reaches the goal)
            remembered = None
            if commit == "candidate" and plan_left:
                remembered = len(paths_)
                plans[remembered] = (plan_left, plan_solves)
                paths_.append([state] + [b for _, b in plan_left])
                firsts.append(plan_left[0])
                done.append(len(plan_left) if plan_solves else None)
            for _ in range(backward):
                att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
                plan = att if type(att) is list else att[1]
                if not plan:
                    continue
                path = [state]
                for m in plan:
                    path.append(AndNode(path[-1], *m).apply(path[-1]))
                plans[len(paths_)] = (list(zip(plan, path[1:])), type(att) is list)
                paths_.append(path)
                firsts.append((plan[0], path[1]))
                done.append(len(plan) if type(att) is list else None)
            n = len(paths_)
            solved = [i for i in range(n) if done[i] is not None]
            if solved:
                best = min(solved, key=lambda i: done[i] + first_penalty(firsts[i][1]))
            elif evaluation == "weighted":
                scores = []
                ends = values(agent, [path[-1] for path in paths_])
                for path, end in zip(paths_, ends):
                    got, nov = set(), 0.0
                    for b in path[1:]:
                        for f in b:
                            if f in got:
                                continue
                            got.add(f)
                            if f not in last_seen:
                                nov += 1.0
                            elif half_life is not None:
                                nov += 1.0 - 0.5 ** ((moves - last_seen[f]) / half_life)
                    scores.append(lam * nov - end)
                best = int(np.argmax(scores))
            else:
                scores = [0.0] * n if not use_novelty else []
                for path in paths_ if use_novelty else []:
                    got, score = set(), 0.0
                    for t, b in enumerate(path[1:]):
                        for f in b:
                            if f not in old and f not in got:
                                got.add(f)
                                score += (rel[f] if rel is not None else 1.0) * discount ** t
                    scores.append(score)
                ends = [-v for v in values(agent, [path[-1] for path in paths_])] if use_value else [0.0] * n
                ends = [e - first_penalty(f[1]) for e, f in zip(ends, firsts)]
                best = max(range(n), key=lambda i: (scores[i], ends[i]))  # the first, on a full tie
                ties += use_novelty and all(s == scores[best] for s in scores)
            chose_backward += best >= k
            source = "commit" if best == remembered else "backward" if best >= k else "forward"
            plan_left, plan_solves = (plans[best][0][1:], plans[best][1]) if best in plans else ([], False)
            move, state = firsts[best]
        moves += 1
        if trace is not None:
            trace.append((move, state))
        if sources is not None:
            sources.append(source)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(moves):
            return {"solved": False, "surrendered": True, "moves": moves, "ties": ties, "backward": chose_backward}
        last_board[state] = moves
        recent.append(state)
        seen.update(state)
        for f in state:
            last_seen[f] = moves
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "moves": moves, "ties": ties,
            "backward": chose_backward}


def solve_flat(agent, state, max_steps, consider=3, tau=1.0, check=False, rollout_gamma=0.3, depth=20, retries=2,
               soft_memory=(2, 1), rng=None, trace=None, surrender=None, rejections=None):
    """Flat, then (optionally) fast - the novice planner of Collins et al.: each turn the student considers only the
    policy's `consider` most likely moves (consider=None: every legal move), looks one move deep at each (the value
    head's moves to go from the board it leads to) and picks one by a softmax over -moves to go / `tau` (a move that
    solves the puzzle is always taken). The soft board memory (beta x 0.5^(age / half-life) for a move back to a board
    the attempt had `age` moves ago) lowers a move's policy score when choosing what to consider, and its choice score.

    check=True adds one simulated run from the chosen move: a policy rollout (soft memory, each move past the first
    continuing with probability 1 - rollout_gamma, at most `depth`). The move is played if the rollout solves the puzzle
    or ends on a board the value head rates closer to the goal than the current one; otherwise it is rejected and
    another considered move is picked from the rest, up to `retries` rejections (then the next pick is played
    unchecked). `rejections`, if a list, gets the number of rejections on every turn.

    Returns {"solved", "surrendered", "moves", "checks"} (checks: rollouts run)."""
    rng = rng or random
    orientations = board.car_orientations(state)
    last_board = {state: 0}
    beta, half = soft_memory if soft_memory else (0.0, 1.0)

    def penalty(b, now, extra=None):
        when = extra.get(b, last_board.get(b)) if extra else last_board.get(b)
        return 0.0 if when is None else beta * 0.5 ** ((now + 1 - when) / half)

    moves, checks = 0, 0
    while moves < max_steps and _red_steps(state) > 0:
        options = list(board.neighbors(state, orientations))
        pen = [penalty(b, moves) for _, b in options]
        logits = agent._evaluate(state)[0][[agent.action_id(state, *m) for m, _ in options]].tolist()
        order = sorted(range(len(options)), key=lambda i: logits[i] - pen[i], reverse=True)
        pool = order if consider is None else order[:consider]
        ends = dict(zip(pool, values(agent, [options[i][1] for i in pool])))
        here = None
        rejected = 0
        while True:
            win = [i for i in pool if _red_steps(options[i][1]) <= 0]
            if win:
                pick = win[0]
            else:
                score = torch.tensor([-ends[i] / tau - pen[i] for i in pool])
                pick = pool[rng.choices(range(len(pool)), torch.softmax(score, 0).tolist())[0]]
            if not check or win or rejected >= retries or len(pool) == 1:
                break
            checks += 1
            path = [state, options[pick][1]]
            own = {state: moves, options[pick][1]: moves + 1}
            for t in range(1, depth):
                if _red_steps(path[-1]) <= 0 or rng.random() < rollout_gamma:
                    break
                _, nxt = sample_move(agent, path[-1], orientations, set(), 1.0, rng,
                                     penalty=lambda b, now=moves + t: penalty(b, now, own))
                own[nxt] = moves + t + 1
                path.append(nxt)
            if here is None:
                here = values(agent, [state])[0]
            if _red_steps(path[-1]) <= 0 or values(agent, [path[-1]])[0] < here:
                break
            rejected += 1
            pool = [i for i in pool if i != pick]
        if rejections is not None:
            rejections.append(rejected)
        move, state = options[pick]
        moves += 1
        if trace is not None:
            trace.append((move, state))
        last_board[state] = moves
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(moves):
            return {"solved": False, "surrendered": True, "moves": moves, "checks": checks}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "moves": moves, "checks": checks}


def _flat_pick(agent, state, options, pen, consider, tau, rng):
    """solve_flat's choice: the index (into `options`) of the move picked from the policy's `consider` most likely
    (memory penalties `pen` subtracted), by a softmax over -moves to go / tau - pen; a solving move always."""
    logits = agent._evaluate(state)[0][[agent.action_id(state, *m) for m, _ in options]].tolist()
    order = sorted(range(len(options)), key=lambda i: logits[i] - pen[i], reverse=True)
    pool = order if consider is None else order[:consider]
    win = [i for i in pool if _red_steps(options[i][1]) <= 0]
    if win:
        return win[0]
    ends = values(agent, [options[i][1] for i in pool])
    score = torch.tensor([-e / tau - pen[i] for e, i in zip(ends, pool)])
    return pool[rng.choices(range(len(pool)), torch.softmax(score, 0).tolist())[0]]


def solve_two_mode(agent, state, max_steps, consider=3, tau=0.5, trigger="loop", stall=4, window=10, gamma=0.07,
                   soft_memory=(2, 1), rng=None, trace=None, surrender=None, sources=None):
    """Forward by default, backward at an impasse.

    Forward mode: a sharp flat choice (solve_flat without the check: the policy's `consider` most likely moves, looked
    at one move deep, softmax over -moves to go / tau; the soft board memory as there). Alone it gets stuck in loops
    (a board the value head rates better than its neighbours without being on the way out).

    Impasse (`trigger`), judged after each forward move: "loop" - the move led back to a board the attempt had within
    the last `window` moves; "stall" - the value head's moves to go haven't beaten the attempt's best for `stall`
    forward moves; "either".

    Backward mode: at an impasse, one AND-OR pass from the current board (the dest head ordering its options with
    Gumbel noise; lapse rate `gamma`) - a whole plan to the goal or the moves made until it lapsed. The student follows
    it without deliberating: its first move at once, each later one with probability 1 - gamma. When it ends or lapses,
    back to forward mode (the stall count restarts from the board reached).

    `sources`, if a list, gets "forward" or "backward" per move. Returns {"solved", "surrendered", "moves",
    "impasses", "backward"} (backward: moves played from a plan)."""
    rng = rng or random
    agent.temperature = 1.0
    orientations = board.car_orientations(state)
    last_board = {state: 0}
    beta, half = soft_memory if soft_memory else (0.0, 1.0)
    moves, impasses, from_plan = 0, 0, 0

    def penalty(b):
        when = last_board.get(b)
        return 0.0 if when is None else beta * 0.5 ** ((moves + 1 - when) / half)

    best, since = values(agent, [state])[0], 0
    plan, impasse = [], False
    while moves < max_steps and _red_steps(state) > 0:
        source = "forward"
        if plan and rng.random() < gamma:
            plan = []  # lost the thread
            best, since = values(agent, [state])[0], 0
        if impasse and not plan:
            impasses += 1
            att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
            steps = (att if type(att) is list else att[1]) or []
            path = [state]
            for m in steps:
                path.append(AndNode(path[-1], *m).apply(path[-1]))
            plan = list(zip(steps, path[1:]))
        impasse = False
        if plan:
            move, state = plan.pop(0)
            source = "backward"
            from_plan += 1
            if not plan:
                best, since = values(agent, [state])[0], 0
        else:
            options = list(board.neighbors(state, orientations))
            pick = _flat_pick(agent, state, options, [penalty(b) for _, b in options], consider, tau, rng)
            move, state = options[pick]
        moves += 1
        if source == "forward" and _red_steps(state) > 0:
            prev = last_board.get(state)
            looped = prev is not None and moves - prev <= window
            v = values(agent, [state])[0]
            if v < best - 1e-9:
                best, since = v, 0
            else:
                since += 1
            stalled = since >= stall
            impasse = {"loop": looped, "stall": stalled, "either": looped or stalled}[trigger]
            if impasse:
                since = 0
        last_board[state] = moves
        if trace is not None:
            trace.append((move, state))
        if sources is not None:
            sources.append(source)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(moves):
            return {"solved": False, "surrendered": True, "moves": moves, "impasses": impasses,
                    "backward": from_plan}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "moves": moves, "impasses": impasses,
            "backward": from_plan}


def solve_lean(agent, state, max_steps, w_flat=0.08, w_ao=0.10, opening=False, consider=5, tau=1.0, gamma=0.07,
               soft_memory=(2, 1), rng=None, trace=None, surrender=None, sources=None):
    """The lean student - the generative side of the move-by-move mixture (notebooks/forward_planning.ipynb, 2c):
    each move is made by one of three modes, drawn independently per move:
    - with probability w_ao, a deliberation: one AND-OR pass from the board (the dest head ordering its options with
      Gumbel noise, lapse rate `gamma`), and its first move is played - one move per search, as in the paper's model;
      no plan is kept;
    - with probability w_flat, flat planning (_flat_pick: the policy's `consider` most likely moves, a softmax over
      -moves to go / tau);
    - otherwise intuition: a policy sample.
    The soft board memory (beta x 0.5^(age / half-life)) lowers moves back to recent boards in intuition and flat.
    opening=True: the attempt's first move is always a deliberation, and the move after it is the same pass's second
    move (people's slow first look and its one-move follow-through).

    `sources`, if a list, gets "intuition", "flat" or "and-or" per move. Returns {"solved", "surrendered", "moves",
    "backward"} (backward: AND-OR moves)."""
    rng = rng or random
    agent.temperature = 1.0
    orientations = board.car_orientations(state)
    last_board = {state: 0}
    beta, half = soft_memory if soft_memory else (0.0, 1.0)
    moves, from_ao, pending = 0, 0, []

    def penalty(b):
        when = last_board.get(b)
        return 0.0 if when is None else beta * 0.5 ** ((moves + 1 - when) / half)

    while moves < max_steps and _red_steps(state) > 0:
        r = rng.random()
        src = "intuition"
        played = None
        if pending:
            played = pending.pop(0)
            src = "and-or"
        elif (opening and moves == 0) or r < w_ao:
            att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
            steps = (att if type(att) is list else att[1]) or []
            if steps:
                nxt = AndNode(state, *steps[0]).apply(state)
                played = (steps[0], nxt)
                src = "and-or"
                if opening and moves == 0 and len(steps) > 1 and _red_steps(nxt) > 0:
                    pending = [(steps[1], AndNode(nxt, *steps[1]).apply(nxt))]
        elif r < w_ao + w_flat:
            options = list(board.neighbors(state, orientations))
            played = options[_flat_pick(agent, state, options, [penalty(b) for _, b in options], consider, tau, rng)]
            src = "flat"
        if played is None:
            played = sample_move(agent, state, orientations, set(), 1.0, rng, penalty=penalty)
            src = "intuition"
        move, state = played
        moves += 1
        from_ao += src == "and-or"
        last_board[state] = moves
        if trace is not None:
            trace.append((move, state))
        if sources is not None:
            sources.append(src)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(moves):
            return {"solved": False, "surrendered": True, "moves": moves, "backward": from_ao}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "moves": moves, "backward": from_ao}


class PlannerSolver:
    """solve_forward as learning_session's `solver`: one attempt, returned as solve_intuition's dict (the boards
    actually played, start first - the planner's imagined boards teach nothing). Plain and picklable. The value cache
    is cleared each attempt (the network changes between puzzles), and AND-OR's random module is seeded from `rng`."""

    def __init__(self, **kw):
        self.kw = kw

    def __call__(self, agent, state, cap, rng, gives_up):
        agent.__dict__.pop("_values", None)
        random.seed(rng.random())
        trace = []
        out = solve_forward(agent, state, cap, rng=rng, trace=trace, surrender=gives_up, **self.kw)
        return {"solved": out["solved"], "surrendered": out["surrendered"],
                "boards": [state] + [b for _, b in trace], "moves": [m for m, _ in trace],
                "taught": [False] * len(trace), "backward": out["backward"]}


_W = {}


def _init(weights):
    torch.set_num_threads(1)
    _W["agents"] = {name: _load_bytes(b) for name, b in weights.items()}


def _init_one(weights):
    """Pool initializer for one network: _W["agent"]."""
    torch.set_num_threads(1)
    _W["agent"] = _load_bytes(weights)


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


def networks(students=8):
    nets = {"start": paths.START,
            "round 3": os.path.join(paths.APPRENTICE, "gamma0.07_free", "round3_epoch03.pt")}
    for teacher in ("no_teacher", "random_1.0"):
        for s in range(students):
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

# one compensatory score, lam x novelty - moves to go: lam swept (never forgetting), then fading memory
LAMS = (0.5, 1, 2, 4, 8, 16, 32)
SWEEP = ([{"name": f"K8 D4 weighted lam{lam}", "kw": {"k": 8, "depth": 4, "evaluation": "weighted", "lam": lam}}
          for lam in LAMS]
         + [{"name": f"K8 D4 uniform weighted lam{lam}",
             "kw": {"k": 8, "depth": 4, "evaluation": "weighted", "lam": lam, "proposals": "uniform"}} for lam in LAMS]
         + [{"name": f"K8 D4 weighted lam{lam} half-life{h}",
             "kw": {"k": 8, "depth": 4, "evaluation": "weighted", "lam": lam, "half_life": h}}
            for lam in (4, 16) for h in (5, 20, 80)])

# forward and backward together: policy rollouts and AND-OR passes as candidates, judged by the value head, with the
# soft board memory fitted to people (experiments/fit_memory.py)
SOFT = (2, 1)
INTEGRATE = [
    {"name": "intuition, soft memory", "kw": {"k": 1, "soft_memory": SOFT}},
    {"name": "forward 8", "kw": {"k": 8, "depth": 4, "evaluation": "value", "soft_memory": SOFT}},
    {"name": "backward 1", "kw": {"k": 0, "backward": 1, "evaluation": "value", "soft_memory": SOFT}},
    {"name": "backward 4", "kw": {"k": 0, "backward": 4, "evaluation": "value", "soft_memory": SOFT}},
    {"name": "forward 4 + backward 4", "kw": {"k": 4, "depth": 4, "backward": 4, "evaluation": "value",
                                              "soft_memory": SOFT}},
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
    if sys.argv[1:2] == ["integrate"]:  # models/summary/forward_integrate.csv; 4 students per teacher
        df = run(networks(4), INTEGRATE, puzzles(), 2)
        df.to_csv(OUT.replace(".csv", "_integrate.csv"), index=False)
        print(summarize(df).round(3).to_string())
        sys.exit()
    if sys.argv[1:2] == ["sweep"]:  # models/summary/forward_sweep.csv; 4 students per teacher
        df = run(networks(4), SWEEP, puzzles(), 2)
        df.to_csv(OUT.replace(".csv", "_sweep.csv"), index=False)
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
