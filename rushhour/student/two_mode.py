"""The two-mode student: how it plays a puzzle (MODEL.md, section 4).

Each move is made by one of three modes, drawn independently per move:
- with probability w_ao, a **deliberation**: one AND-OR pass from the board (the policy network ordering its options,
  with Gumbel noise, lapse rate gamma), whose first move is played - one move per search, as in the paper's model; no
  plan is kept;
- with probability w_flat, **flat planning**: the policy's `consider` most likely moves, each judged one move ahead by
  the value network (moves to go), one picked by a softmax over -moves to go / tau;
- otherwise **intuition**: a move sampled from the policy.
A soft memory of the attempt's boards (beta x 0.5^(age / half-life)) lowers moves back to recent boards in intuition and
flat planning. With `opening`, the attempt's first move is always a deliberation, and the move after it is the same
pass's second move (people's slow first look). With `w_ao` 0 it is the forward student.

The mode weights were fitted to people's moves (experiments/mode_mixture.py): AND-OR 18%, flat planning rising from 6%
to 18% over the first 30 trials (TwoModeSolver), intuition the rest. A teacher (teachers.py) may show the optimal move
on any turn instead, as with the intuition student (solve.solve_intuition); untaught, play is unchanged.

The code began in experiments/forward.py as solve_lean / LeanSolver ("lean" in the run and table names there), which
are now these.
"""
import random

import torch

from .. import board
from ..and_or import AndNode, _red_steps, solve as and_or_solve


def values(agent, states):
    """The value head's moves to go (its output x 10) for each of `states`, batched and cached on the agent - the
    head of its separate value network (agent.value_agent) if it has one."""
    cache = agent.__dict__.setdefault("_values", {})
    if len(cache) > 50_000:  # clear before working out what is missing, so every state asked for is filled
        cache.clear()
    missing = list({s for s in states if s not in cache})
    if missing:
        judge = getattr(agent, "value_agent", None) or agent
        judge.net.eval()
        with torch.no_grad():
            v = judge.net.heads(torch.stack([judge.encoder.encode(s) for s in missing]).to(judge.device))[1]
        cache.update(zip(missing, (10 * v.cpu()).tolist()))
    return [cache[s] for s in states]


def sample_move(agent, state, orientations, avoid, temperature, rng, uniform=False, penalty=None):
    """A (move, next board) sampled from the policy head (`uniform`: uniformly, the network unused), avoiding
    boards in `avoid` if it can. With `penalty` (board -> a number), nothing is avoided: each move's logit is
    lowered by the penalty of the board it leads to (a soft memory)."""
    options = list(board.neighbors(state, orientations))
    if penalty is not None:
        logits = (torch.zeros(len(options)) if uniform else
                  agent.policy_logits(state)[[agent.action_id(state, *m) for m, _ in options]] / max(temperature, 1e-9))
        logits = logits - torch.tensor([penalty(b) for _, b in options], dtype=logits.dtype)
        return rng.choices(options, torch.softmax(logits, dim=0).tolist())[0]
    fresh = [o for o in options if o[1] not in avoid] or options
    if uniform:
        return rng.choice(fresh)
    logits = agent.policy_logits(state)[[agent.action_id(state, *m) for m, _ in fresh]]
    if temperature > 0:
        return rng.choices(fresh, torch.softmax(logits / temperature, dim=0).tolist())[0]
    return fresh[int(logits.argmax())]


def flat_pick(agent, state, options, pen, consider, tau, rng):
    """Flat planning's choice: the index (into `options`) of the move picked from the policy's `consider` most likely
    (memory penalties `pen` subtracted), by a softmax over -moves to go / tau - pen; a solving move always."""
    logits = agent.policy_logits(state)[[agent.action_id(state, *m) for m, _ in options]].tolist()
    order = sorted(range(len(options)), key=lambda i: logits[i] - pen[i], reverse=True)
    pool = order if consider is None else order[:consider]
    win = [i for i in pool if _red_steps(options[i][1]) <= 0]
    if win:
        return win[0]
    ends = values(agent, [options[i][1] for i in pool])
    score = torch.tensor([-e / tau - pen[i] for e, i in zip(ends, pool)])
    return pool[rng.choices(range(len(pool)), torch.softmax(score, 0).tolist())[0]]


def goal_in_sight(state, orientations, depth):
    """(first move, board after it) of a shortest sequence of at most `depth` moves that gets red out, or None."""
    frontier, seen = [(state, None)], {state}
    for _ in range(depth):
        nxt = []
        for b, first in frontier:
            for m, c in board.neighbors(b, orientations):
                if c in seen:
                    continue
                seen.add(c)
                f = first or (m, c)
                if _red_steps(c) <= 0:
                    return f
                nxt.append((c, f))
        frontier = nxt
    return None


AO_ORDERS = ("policy", "none")


def solve_two_mode(agent, state, max_steps, w_flat=0.08, w_ao=0.10, opening=False, consider=5, tau=1.0, gamma=0.07,
                   soft_memory=(2, 1), rng=None, trace=None, surrender=None, sources=None, sight=0, temperature=1.0,
                   ao_order="policy", teacher=None, stuck=None):
    """One attempt by the two-mode student (module docstring); with `w_ao` 0, the forward student.

    ao_order: what orders AND-OR's options (and picks a lapse's move) - "policy" (the policy network,
    PolicyAgent.heuristic) or "none" (random order, plain AND-OR). (Until 2026-11-24, "dest": the dest head.)
    sight=D ("goal in sight"): before anything else, a search of every sequence of up to D moves; if one gets red out,
    its first move is played. temperature: intuition's sampling temperature (1: as the session student).

    `teacher` (teachers.py), if given, is asked first on every turn, as in solve_intuition: if it shows a move, that
    move is played (a shown move also ends the opening's follow-through), otherwise the student's own; it may look at
    the student's move (computed once, when first asked). Untaught, the random draws are exactly those without a
    teacher.

    `stuck` ({"w_ao": a, "w_flat": f}, either optional): on a board the attempt has been on before, the modes are drawn
    with these odds instead (planning more when stuck); None: the same odds everywhere. The draws are the same either
    way, only the thresholds differ.

    `trace`, if a list, gets (move, board after it) per move; `sources`, if a list, "intuition", "flat", "and-or",
    "sight" or "shown". `surrender` (a function of moves made): after each move that doesn't solve the puzzle, the
    student gives up with that probability (drawn from `rng`). Returns {"solved", "surrendered", "moves" (a count),
    "backward" (AND-OR moves), "taught" (per move, whether the teacher showed it)}."""
    rng = rng or random
    agent.temperature = 1.0
    if ao_order not in AO_ORDERS:
        raise ValueError(f"ao_order: one of {AO_ORDERS}, not {ao_order!r}")
    heuristic = agent.heuristic if ao_order == "policy" else None
    orientations = board.car_orientations(state)
    last_board = {state: 0}
    beta, half = soft_memory if soft_memory else (0.0, 1.0)
    moves, from_ao, pending, taught = 0, 0, [], []
    returned = False  # whether the board the student is on was visited before in this attempt

    def penalty(b):
        when = last_board.get(b)
        return 0.0 if when is None else beta * 0.5 ** ((moves + 1 - when) / half)

    def choose():
        """The student's own move on this board: ((move, board after it), mode)."""
        nonlocal pending
        r = rng.random()
        p_ao, p_flat = w_ao, w_flat
        if stuck is not None and returned:
            p_ao, p_flat = stuck.get("w_ao", w_ao), stuck.get("w_flat", w_flat)
        src = "intuition"
        played = None
        seen_goal = goal_in_sight(state, orientations, sight) if sight and not pending else None
        if pending:
            played = pending.pop(0)
            src = "and-or"
        elif seen_goal is not None:
            played = seen_goal
            src = "sight"
        elif (opening and moves == 0) or r < p_ao:
            att = and_or_solve(state, heuristic=heuristic, gamma=gamma)
            steps = (att if type(att) is list else att[1]) or []
            if steps:
                nxt = AndNode(state, *steps[0]).apply(state)
                played = (steps[0], nxt)
                src = "and-or"
                if opening and moves == 0 and len(steps) > 1 and _red_steps(nxt) > 0:
                    pending = [(steps[1], AndNode(nxt, *steps[1]).apply(nxt))]
        elif r < p_ao + p_flat:
            options = list(board.neighbors(state, orientations))
            played = options[flat_pick(agent, state, options, [penalty(b) for _, b in options], consider, tau, rng)]
            src = "flat"
        if played is None:
            played = sample_move(agent, state, orientations, set(), temperature, rng, penalty=penalty)
            src = "intuition"
        return played, src

    while moves < max_steps and _red_steps(state) > 0:
        proposed = []

        def propose():
            if not proposed:
                proposed.append(choose())
            return proposed[0][0][0]

        shown = teacher.advise(state, moves, propose) if teacher is not None else None
        if shown is not None:
            played, src = (shown, dict(board.neighbors(state, orientations))[shown]), "shown"
            pending = []
        else:
            if not proposed:
                proposed.append(choose())
            played, src = proposed[0]
        move, after = played
        if teacher is not None:
            teacher.observe(state, move, after, shown is not None)
        state = after
        moves += 1
        from_ao += src == "and-or"
        taught.append(shown is not None)
        returned = state in last_board
        last_board[state] = moves
        if trace is not None:
            trace.append((move, state))
        if sources is not None:
            sources.append(src)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(moves):
            return {"solved": False, "surrendered": True, "moves": moves, "backward": from_ao, "taught": taught}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "moves": moves, "backward": from_ao,
            "taught": taught}


class TwoModeSolver:
    """solve_two_mode as learning_session's `solver`, with the flat share rising over the session: on the puzzle of
    human trial t (t = puzzles played // multiplier + 1), w_flat = flat_start + (flat_end - flat_start) x min(1, (t - 1)
    / ramp) - people's flat share rose from about 6% to 18% over their first 30 trials, and 18% of their moves fit a
    policy-ordered AND-OR pass (the mixture fit, refitted 2026-11-24 without the dest head; with dest-ordered AND-OR it
    was 8% -> 18% and 10%). One instance per student: it counts the attempts it has played (one per puzzle). The value
    cache is cleared each attempt; AND-OR's random module is seeded from `rng`."""

    def __init__(self, w_ao=0.18, flat_start=0.06, flat_end=0.18, ramp=30, multiplier=10, opening=True, **kw):
        self.w_ao, self.flat_start, self.flat_end, self.ramp = w_ao, flat_start, flat_end, ramp
        self.multiplier, self.opening, self.kw, self.calls = multiplier, opening, kw, 0

    def flat_at(self, trial):
        return self.flat_start + (self.flat_end - self.flat_start) * min(1.0, (trial - 1) / self.ramp)

    def __call__(self, agent, state, cap, rng, gives_up, teacher=None):
        trial = self.calls // self.multiplier + 1
        self.calls += 1
        agent.__dict__.pop("_values", None)
        random.seed(rng.random())
        trace = []
        out = solve_two_mode(agent, state, cap, w_flat=self.flat_at(trial), w_ao=self.w_ao, opening=self.opening,
                             rng=rng, trace=trace, surrender=gives_up, teacher=teacher, **self.kw)
        return {"solved": out["solved"], "surrendered": out["surrendered"],
                "boards": [state] + [b for _, b in trace], "moves": [m for m, _ in trace],
                "taught": out["taught"], "backward": out["backward"]}
