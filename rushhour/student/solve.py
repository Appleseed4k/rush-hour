"""How the student solves a puzzle: by intuition alone (solve_intuition, with a teacher and giving up), by
deliberation alone (solve_and_or: AND-OR ordered by the network) or arbitrating between them (solve_arbitrated),
and parallel evaluation of each over many puzzles."""
import io
import random
from collections import deque
from multiprocessing import Pool


import torch

from ..and_or import AndNode, _red_steps, solve as and_or_solve
from ..board import car_orientations, neighbors
from .network import PolicyAgent


_WORKER = {}


def _init_attempt_worker(weights, config):
    """Pool initializer: a CPU copy of the guiding agent (None = plain AND-OR)."""
    torch.set_num_threads(1)
    _WORKER.clear()
    if weights is not None:
        size, max_slide, hidden, blocks, temperature = config
        agent = PolicyAgent(size, max_slide, hidden=hidden, blocks=blocks, temperature=temperature)
        agent.net.load_state_dict(torch.load(io.BytesIO(weights), map_location="cpu"))
        _WORKER["agent"] = agent


def _cap(max_steps, state):
    """The move cap for an attempt from `state`: `max_steps` itself, or
    max_steps(state) when it's a callable (e.g. a cap scaled by the puzzle's
    optimal length)."""
    return max_steps(state) if callable(max_steps) else max_steps


def _agent_payload(agent):
    """(weights as bytes, config) for rebuilding a CPU copy of `agent` in a
    worker (see _init_attempt_worker)."""
    buffer = io.BytesIO()
    torch.save({k: v.cpu() for k, v in agent.net.state_dict().items()}, buffer)
    return buffer.getvalue(), (agent.size, agent.max_slide, agent.hidden, agent.blocks, agent.temperature)


def solve_and_or(state, max_steps=200, heuristic=None, gamma=0.0):
    """Repeated and_or.solve passes from `state` until red exits or
    `max_steps` moves are played. Returns the moves if solved, else None.

    `heuristic`, if given (a trained PolicyAgent's `.heuristic`), orders
    AND-OR's candidate moves and picks what a lapse plays - the human-like
    solver. `gamma` is the lapse probability - see and_or.solve."""
    moves = []
    while len(moves) < max_steps:
        att = and_or_solve(state, heuristic=heuristic, gamma=gamma)
        if type(att) is list:
            moves.extend(att)
            return moves
        state, step_moves = att
        if not step_moves:
            return None
        moves.extend(step_moves)
    return None


def solve_arbitrated(agent, state, max_steps=200, gamma=0.0, threshold=0.5, memory=10, temperature=1.0):
    """Solves from `state` by arbitrating, every turn, between intuition and
    deliberation - the human-like solver. Returns (moves if solved else
    None, stats).

    - Intuition: a move sampled from softmax(logits / temperature) with the
      random module - at the default 1, in proportion to the policy head's
      own probabilities; at 0, always its top legal move. Played when the
      top move's probability (softmax over legal moves, the confidence) is
      at least `threshold` and the chosen move doesn't lead back to one of
      the last `memory` boards (None = every board this attempt). One move.
    - Deliberation: otherwise, one and_or.solve pass ordered by the agent's
      heuristic, with lapse probability `gamma`; its moves are played. A
      move that would revisit a recent board counts as low confidence -
      going in circles is a reason to stop and think.

    `temperature` is intuition's noise only; the agent's own `temperature`
    is deliberation's (see PolicyAgent). threshold 0 is pure intuition
    (apart from revisits); above 1, pure deliberation. stats counts
    "intuitive" moves, "deliberations" (passes) and "deliberate" moves."""
    orientations = car_orientations(state)
    recent = deque([state], maxlen=memory)
    moves = []
    stats = {"intuitive": 0, "deliberations": 0, "deliberate": 0}
    while len(moves) < max_steps and _red_steps(state) > 0:
        options = list(neighbors(state, orientations))
        if threshold <= 1:
            logits = agent.policy_logits(state)[[agent.action_id(state, *move) for move, _ in options]]
            probs = torch.softmax(logits, dim=0)
            best = int(probs.argmax())
            choice = best
            if temperature > 0:
                choice = random.choices(range(len(options)), torch.softmax(logits / temperature, dim=0).tolist())[0]
            move, nxt = options[choice]
            if probs[best].item() >= threshold and nxt not in recent:
                state = nxt
                moves.append(move)
                recent.append(state)
                stats["intuitive"] += 1
                continue
        att = and_or_solve(state, heuristic=agent.heuristic, gamma=gamma)
        step = att if type(att) is list else att[1]
        stats["deliberations"] += 1
        if not step:
            break
        for move in step:
            state = AndNode(state, *move).apply(state)
            moves.append(move)
            recent.append(state)
        stats["deliberate"] += len(step)
    return (moves if _red_steps(state) <= 0 else None), stats


def solve_intuition(agent, state, max_steps=200, memory=10, temperature=0.0, teacher=None, rng=None,
                    surrender=None):
    """Pure intuition, no search: every turn the policy head chooses among the
    legal moves that don't return to one of the last `memory` boards (all
    legal moves if every one does) - its top move at `temperature` 0, else a
    move sampled from softmax(logits / temperature) with `rng` (default: the
    random module). With a `teacher` (see teachers.py), each turn the teacher
    may show the optimal move instead, which is played; it can look at the
    move the student would have played (the archived CorrectMistakes did).

    `surrender`, if given, is a function of the number of moves played so
    far: after each move that doesn't solve the puzzle, the student gives
    up with that probability (drawn from `rng`) - see SurrenderRule.

    Returns {"solved", "surrendered", "boards" (start first), "moves",
    "taught" (per move, whether the teacher showed it)}."""
    rng = rng or random
    orientations = car_orientations(state)
    recent = deque([state], maxlen=memory)
    boards, moves, taught = [state], [], []
    while len(moves) < max_steps and _red_steps(state) > 0:
        options = dict(neighbors(state, orientations))  # move -> board it leads to
        proposed = []

        def propose():
            """The student's own move on this board, chosen once, when first asked."""
            if not proposed:
                fresh = [m for m, nxt in options.items() if nxt not in recent] or list(options)
                logits = agent.policy_logits(state)[[agent.action_id(state, *m) for m in fresh]]
                if temperature > 0:
                    proposed.append(rng.choices(fresh, torch.softmax(logits / temperature, dim=0).tolist())[0])
                else:
                    proposed.append(fresh[int(logits.argmax())])
            return proposed[0]

        move = teacher.advise(state, len(moves), propose) if teacher is not None else None
        taught.append(move is not None)
        if move is None:
            move = propose()
        if teacher is not None:
            teacher.observe(state, move, options[move], taught[-1])
        state = options[move]
        boards.append(state)
        moves.append(move)
        recent.append(state)
        if surrender is not None and _red_steps(state) > 0 and rng.random() < surrender(len(moves)):
            return {"solved": False, "surrendered": True, "boards": boards, "moves": moves, "taught": taught}
    return {"solved": _red_steps(state) <= 0, "surrendered": False, "boards": boards, "moves": moves,
            "taught": taught}


class SurrenderRule():
    """When a student gives up on a puzzle, as people do: after each move, it
    surrenders with a probability that depends on how many moves it has made
    on the puzzle so far relative to the optimal length - `rates[i]` while
    moves / optimal is in [edges[i], edges[i + 1]) (the last rate beyond the
    last edge). The rates are fitted to the human data
    (human_data.surrender_hazard). `distances` maps each puzzle's starting
    state to its optimal length. Called as rule(state, moves): the
    probability of giving up after `moves` moves on the puzzle starting at
    `state`. `scale` multiplies every rate (capped at 1): the human shape
    with a student-specific level, fitted so the student's starting solve
    rate and efficiency match people's. A plain object, so it can be sent
    to worker processes."""

    def __init__(self, edges, rates, distances, scale=1.0):
        self.edges, self.distances, self.scale = list(edges), dict(distances), scale
        self.rates = [min(1.0, scale * r) for r in rates]

    def __call__(self, state, moves):
        ratio = moves / self.distances[state]
        i = 0
        while i + 1 < len(self.edges) and ratio >= self.edges[i + 1]:
            i += 1
        return self.rates[min(i, len(self.rates) - 1)]

    def for_puzzle(self, state, offset=0):
        """The surrender function solve_intuition takes, for an attempt at
        `state` after `offset` moves already made on it (earlier attempts)."""
        return lambda moves: self(state, offset + moves)


def _intuition_job(job):
    """One untaught solve_intuition attempt by the worker's agent."""
    state, distance, memory, max_steps, temperature, seed, *rest = job
    surrender = rest[0].for_puzzle(state) if rest and rest[0] is not None else None
    result = solve_intuition(_WORKER["agent"], state, max_steps, memory, temperature, rng=random.Random(seed),
                             surrender=surrender)
    return {"solved": result["solved"], "moves": len(result["moves"]) if result["solved"] else None,
            "distance": distance}


def evaluate_intuition(agent, puzzles, n_runs=3, temperature=1.0, memory=10, max_steps=200, seed=0,
                       processes=None, surrender=None):
    """`n_runs` untaught solve_intuition attempts on every (distance, state) in
    `puzzles`, in parallel (each worker gets a CPU copy of the agent): the
    student's behavior with no teacher, sampling its moves at `temperature`
    as it does while learning (0 = always its top move, deterministic).
    `max_steps` is a number or a callable giving each puzzle's cap from its
    state. `surrender`: a SurrenderRule, to let the student give up as
    people do (None: it never does). Returns one dict per attempt, puzzle by
    puzzle (`n_runs` each): solved, moves (None if unsolved), distance."""
    jobs = [(state, d, memory, _cap(max_steps, state), temperature, hash((seed, i, r)) & 0xFFFFFFFF, surrender)
            for i, (d, state) in enumerate(puzzles) for r in range(n_runs)]
    with Pool(processes, initializer=_init_attempt_worker, initargs=_agent_payload(agent)) as pool:
        return pool.map(_intuition_job, jobs, chunksize=2)


def _arbitrated_job(job):
    """One solve_arbitrated attempt by the worker's agent (see label_attempts' Pool setup)."""
    state, distance, gamma, threshold, memory, max_steps, temperature, seed = job
    random.seed(seed)
    moves, stats = solve_arbitrated(_WORKER["agent"], state, max_steps, gamma, threshold, memory, temperature)
    return {"solved": moves is not None, "moves": len(moves) if moves is not None else None,
            "distance": distance, **stats}


def evaluate_arbitrated(agent, puzzles, n_runs=3, gamma=0.0, threshold=0.5, memory=10, max_steps=200, seed=0,
                        processes=None, temperature=1.0):
    """solve_arbitrated `n_runs` times on every (distance, state) in `puzzles`,
    in parallel (each worker gets a CPU copy of the agent, at its
    temperature for deliberation; `temperature` is intuition's). Returns one
    dict per attempt, puzzle by puzzle: solved, moves (None if unsolved),
    distance, and the stats of solve_arbitrated."""
    jobs = [(state, d, gamma, threshold, memory, max_steps, temperature, hash((seed, i, r)) & 0xFFFFFFFF)
            for i, (d, state) in enumerate(puzzles) for r in range(n_runs)]
    with Pool(processes, initializer=_init_attempt_worker, initargs=_agent_payload(agent)) as pool:
        return pool.map(_arbitrated_job, jobs, chunksize=2)
