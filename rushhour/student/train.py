"""How the student learns: offline training and expert iteration (the apprentice's pretraining), and the
online updates of a learning session, puzzle by puzzle, with or without a teacher."""
import json
import math
import os
import random
from collections import Counter, deque

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from .network import PolicyAgent
from .solve import _cap, solve_intuition
from .labels import agency_weights, intuition_examples, label_attempts, progress_examples


def _batch(agent, examples):
    device = agent.device
    n = len(examples)
    x = torch.from_numpy(np.stack([e.x for e in examples])).float().to(device)
    distance = torch.tensor([e.distance if e.distance is not None else float("nan") for e in examples],
                            dtype=torch.float32, device=device)  # nan: no distance target
    n_actions = len(agent.action_space)
    legal = torch.zeros(n, n_actions, dtype=torch.bool)
    optimal = torch.zeros(n, n_actions, dtype=torch.bool)
    for r, e in enumerate(examples):
        legal[r, e.legal] = True
        optimal[r, e.optimal] = True
    return x, distance, legal.to(device), optimal.to(device)


def _loss(agent, examples):
    """Policy loss (probability on the target moves), plus the value loss
    where there is a distance target."""
    x, distance, legal, optimal = _batch(agent, examples)
    policy, value = agent.net.heads(x)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    loss = -logp.masked_fill(~optimal, float("-inf")).logsumexp(1).mean()
    has_distance = ~torch.isnan(distance)
    if has_distance.any():
        loss = loss + 0.5 * F.smooth_l1_loss(value[has_distance], distance[has_distance] / 10)
    return loss


def _weighted_loss(agent, examples, weights):
    """_loss with a weight per example, scaling its move and distance targets. Each term is still a mean over the
    batch's targets, so weights below 1 mean less learning, not the same learning shared out differently."""
    x, distance, legal, optimal = _batch(agent, examples)
    policy, value = agent.net.heads(x)
    w = torch.tensor(weights, dtype=torch.float32, device=x.device)
    logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
    loss = (-logp.masked_fill(~optimal, float("-inf")).logsumexp(1) * w).mean()
    has_distance = ~torch.isnan(distance)
    if has_distance.any():
        terms = F.smooth_l1_loss(value[has_distance], distance[has_distance] / 10, reduction="none")
        loss = loss + 0.5 * (terms * w[has_distance]).mean()
    return loss


@torch.no_grad()
def evaluate_agreement(agent, examples, batch_size=2048):
    """How well the agent's heads match the labels on `examples`:
    - agreement: how often the policy head's top legal move is a target
      (over examples that have one);
    - chance: the same for a uniformly random legal move;
    - distance_error: mean absolute error of the value head, in moves.
    """
    if not examples:
        return {"agreement": float("nan"), "chance": float("nan"), "distance_error": float("nan")}
    agent.net.eval()
    hits = chance = err = 0.0
    with_distance = with_target = 0
    for i in range(0, len(examples), batch_size):
        x, distance, legal, optimal = _batch(agent, examples[i:i + batch_size])
        policy, value = agent.net.heads(x)
        top = policy.masked_fill(~legal, float("-inf")).argmax(1)
        has_target = optimal.any(1)
        hits += optimal.gather(1, top[:, None])[has_target].float().sum().item()
        chance += (optimal.sum(1).float() / legal.sum(1).float())[has_target].sum().item()
        with_target += int(has_target.sum())
        has_distance = ~torch.isnan(distance)
        err += (value * 10 - distance)[has_distance].abs().sum().item()
        with_distance += int(has_distance.sum())
    agent.net.train()
    return {"agreement": hits / with_target if with_target else float("nan"),
            "chance": chance / with_target if with_target else float("nan"),
            "distance_error": err / with_distance if with_distance else float("nan")}


def _validation_score(scores):
    """Mean of a train_policy history entry's scores, ignoring missing (nan)
    ones; None if none are left (nothing to validate on)."""
    values = [v for v in scores.values() if v == v]
    return sum(values) / len(values) if values else None


def _kept_epoch(history):
    """The epoch (from 1) whose weights train_policy keeps: the first with the
    best validation score - or, for epochs with nothing to validate on,
    simply the latest."""
    best, kept = -1.0, 1
    for epoch, scores in enumerate(history, 1):
        score = _validation_score(scores)
        if score is None or score > best:
            best = best if score is None else score
            kept = epoch
    return kept


def train_policy(agent, examples, val_examples, epochs=40, patience=6, batch_size=256, on_epoch=None):
    """Supervised training on board Examples: the policy head puts
    probability on any target move, the value head predicts the distance.

    Keeps the weights with the best validation score - the policy head's
    agreement on `val_examples` (from puzzles held out of training) - and
    stops after `patience` epochs without improvement. With nothing to
    validate on (e.g. no validation attempt solved), it keeps each epoch's
    weights as it goes and doesn't stop early. Returns per-epoch
    validation scores as dicts.

    `on_epoch`, if given, is called as on_epoch(epoch, scores) after each
    epoch (from 1), on that epoch's own weights - e.g. to save checkpoints."""
    examples = list(examples)
    agent._scores = {}
    best, best_state, stale, history = -1.0, None, 0, []
    bar = tqdm(range(epochs), desc="training epochs")
    for _ in bar:
        agent.net.train()
        random.shuffle(examples)
        for i in range(0, len(examples), batch_size):
            loss = _loss(agent, examples[i:i + batch_size])
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()
        scores = {"agreement": evaluate_agreement(agent, val_examples)["agreement"]}
        history.append(scores)
        if on_epoch is not None:
            on_epoch(len(history), scores)
        score = _validation_score(scores)
        bar.set_postfix({k: f"{v:.1%}" for k, v in scores.items()})
        if score is None or score > best:
            best = best if score is None else score
            best_state, stale = {k: v.clone() for k, v in agent.net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    agent.net.load_state_dict(best_state)
    agent._scores = {}
    return history


def expert_iteration(agent, puzzles, val_puzzles, rounds=3, n_attempts=4, gamma=0.07, max_steps=200, seed=0,
                     checkpoint_dir=None, checkpoint_epochs=True, student="and_or", teacher=None, explore=1.0,
                     memory=10, resume=False, **train_kwargs):
    """Trains `agent` on the student's own attempts, by expert iteration. Each
    round labels attempts on its training puzzles and on `val_puzzles`
    (validation) with label_attempts and keeps training the same weights on
    them, so the examples improve as the agent does. `puzzles` is either one
    list of starting states, used every round, or a list of such lists, one
    per round (e.g. fresh puzzles each round). `max_steps` caps each
    attempt's moves - a number, or a callable giving a puzzle's cap from its
    starting state.

    `student` "and_or": round 1's attempts are plain AND-OR; each later
    round's come from AND-OR with its options ordered by the agent it has
    become (its policy head, at its temperature). No breadth-first search is involved.
    `student` "intuition": every round's attempts are the agent's own
    intuition (solve_intuition, sampling at temperature `explore`), with
    `teacher` (see teachers.py) - which knows the optimal moves - free to show
    them, on validation attempts too.

    Returns one dict per round: attempts made and solved on the training
    puzzles, the share and number of their moves the teacher showed
    ("taught", "taught_moves") out of all moves played ("moves"), its
    interventions - the teaching budget ("interventions") - the teacher's own counts
    ("teacher_counts", see teachers.py), boards learned from, the best validation scores (agreement with the
    round's own attempts) and the epoch they came from.

    `checkpoint_dir`, if given, gets the learning trajectory: the agent
    before training (round0_epoch00.pt) and after every epoch of every round
    (round{r}_epoch{e}.pt), plus manifest.json listing them in training
    order - round, epoch, validation scores, and "kept": whether these are
    the weights the round ended on (its best epoch), which the next round
    starts from. The manifest is rewritten after every round, along with
    history.json (this function's return value so far), so a run can be
    re-evaluated later without retraining. With `checkpoint_epochs` False,
    only the untrained agent and each round's kept weights are saved.

    `resume`: continue the run saved in `checkpoint_dir` - load its history,
    manifest and last kept weights into `agent` and train from the next
    round up to `rounds` in all (the optimizer starts fresh). Without a
    saved run there, it starts from scratch."""
    per_round = bool(puzzles) and isinstance(puzzles[0], list)
    if per_round and len(puzzles) < rounds:
        raise ValueError(f"{len(puzzles)} rounds of puzzles for {rounds} rounds")
    history, manifest = [], []

    def checkpoint(round_, epoch, scores):
        name = f"round{round_}_epoch{epoch:02d}.pt"
        agent.save(os.path.join(checkpoint_dir, name))
        manifest.append({"file": name, "round": round_, "epoch": epoch, "kept": round_ == 0, **scores})

    saved = checkpoint_dir is not None and os.path.exists(os.path.join(checkpoint_dir, "history.json"))
    if resume and saved:
        with open(os.path.join(checkpoint_dir, "history.json")) as f:
            history = json.load(f)
        with open(os.path.join(checkpoint_dir, "manifest.json")) as f:
            manifest = json.load(f)
        last = max((m for m in manifest if m["kept"]), key=lambda m: m["round"])
        agent.net.load_state_dict(PolicyAgent.load(os.path.join(checkpoint_dir, last["file"])).net.state_dict())
        agent._scores = {}
    elif checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint(0, 0, {})
    for r in range(len(history), rounds):
        round_puzzles = puzzles[r] if per_round else puzzles
        guide = agent if r > 0 or student == "intuition" else None
        settings = {"size": agent.size, "max_slide": agent.max_slide, "student": student, "teacher": teacher,
                    "explore": explore, "memory": memory}
        examples, info = label_attempts(round_puzzles, n_attempts, gamma, guide, seed + 2 * r, max_steps, **settings)
        val_examples, _ = label_attempts(val_puzzles, n_attempts, gamma, guide, seed + 2 * r + 1, max_steps,
                                         **settings)
        on_epoch = ((lambda epoch, s, round_=r + 1: checkpoint(round_, epoch, s))
                    if checkpoint_dir is not None and checkpoint_epochs else None)
        scores = train_policy(agent, examples, val_examples, on_epoch=on_epoch, **train_kwargs)
        best_epoch = _kept_epoch(scores)
        best = scores[best_epoch - 1]
        history.append({"round": r + 1, "attempts": info["attempts"], "solved": info["solved"],
                        "taught": info["taught"] / info["moves"] if info["moves"] else float("nan"),
                        "taught_moves": info["taught"], "moves": info["moves"],
                        "interventions": info["interventions"], "teacher_counts": info["teacher_counts"],
                        "boards": len(examples), "best_epoch": best_epoch, **best})
        if checkpoint_dir is not None:
            if not checkpoint_epochs:
                checkpoint(r + 1, best_epoch, best)  # train_policy has restored the kept weights
            for entry in manifest:
                if entry["round"] == r + 1 and entry["epoch"] == best_epoch:
                    entry["kept"] = True
            with open(os.path.join(checkpoint_dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=1)
            with open(os.path.join(checkpoint_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)
    return history


def online_update(agent, examples, steps=1, batch_size=256, weights=None, agency_step="off", rng=None,
                  even_batches=False):
    """Learns from `examples` the way a person learns from one puzzle: `steps`
    passes over just these examples (one gradient step per batch), no replay
    of anything earlier, no validation, nothing to rewind to. `weights`: one
    per example (see _weighted_loss); None trains them all alike (_loss).

    Adam divides each step by the recent size of the gradients, so it cancels a
    weight shared by a whole batch: with `agency_step` "off" the weights only
    set how examples count against each other, and a batch of shown moves at
    weight 0.1 learns as much as one of the student's own. The other settings
    make the weights also set how much is learned, by the batch's mean move
    weight m; its loss is divided by m,
    so only the step depends on it:
    - "scale": the step is scaled by m (a tenth of a step at m = 0.1);
    - "skip": the step is taken with probability m, else skipped (a full step
      one time in ten at m = 0.1) - the same learning on average.
    At m = 0 no step is taken. `rng` (default: the random module) shuffles and
    skips. With agent.train_on(device), the steps run on a copy of the network
    on that device, and the new weights are copied back at the end.

    Batches: by default `batch_size` examples each, the last holding what is left - which Adam steps as far
    as a full one, so a pass over 517 examples takes two steps on 256 and a third on just 5. `even_batches`
    splits each pass into the same number of batches, ceil(n / batch_size), of equal size instead (3 of
    about 172): the same number of steps, none of them on a handful of examples.

    Returns {"pool": examples, "batches": batches over all passes, "steps": steps taken}."""
    assert agency_step in ("off", "scale", "skip"), agency_step
    rng = rng or random
    stats = {"pool": len(examples), "batches": 0, "steps": 0}
    examples = list(examples) if weights is None else list(zip(examples, weights))
    costly = agency_step != "off" and weights is not None
    player, agent = agent, agent._learner()  # with train_on, train the copy on the training device
    base = [g["lr"] for g in agent.optimizer.param_groups]
    agent.net.train()
    try:
        for _ in range(steps):
            rng.shuffle(examples)
            if even_batches:
                k = max(1, math.ceil(len(examples) / batch_size))
                bounds = [round(j * len(examples) / k) for j in range(k + 1)]
            else:
                bounds = list(range(0, len(examples), batch_size)) + [len(examples)]
            for i, j in zip(bounds[:-1], bounds[1:]):
                batch = examples[i:j]
                stats["batches"] += 1
                if costly:
                    mean = float(np.mean([w for _, w in batch]))
                    if mean <= 0 or agency_step == "skip" and rng.random() >= mean:
                        continue
                    for g, lr in zip(agent.optimizer.param_groups, base):
                        g["lr"] = lr * (mean if agency_step == "scale" else 1.0)
                if weights is None:
                    loss = _loss(agent, batch)
                else:
                    loss = _weighted_loss(agent, [e for e, _ in batch], [w for _, w in batch])
                    if costly:
                        loss = loss / mean
                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()
                stats["steps"] += 1
    finally:
        for g, lr in zip(agent.optimizer.param_groups, base):
            g["lr"] = lr
    if agent is not player:
        player.net.load_state_dict({k: v.to(player.device) for k, v in agent.net.state_dict().items()})
    player._scores = {}
    return stats


LOSS_SUMS = ("n", "w", "policy", "w_policy", "value", "w_value", "n_value")


def example_losses(agent, examples, weights, chunk=512):
    """Each example's loss terms under the network as it is (eval mode, no gradient - so BatchNorm's running
    statistics, the random generators and the next update are untouched), with its loss weight (see
    _weighted_loss; None: all 1). Returns {name: array per example} for LOSS_SUMS but "n": n 1s, w (move weight),
    policy (-log probability on the target moves) and w * policy, value (smooth-L1 term, 0 without a distance) and
    w * value, n_value (1 with a distance). (Runs before the dest head's removal also logged dest sums.)"""
    weights = weights if weights is not None else [1.0] * len(examples)
    net = agent.net
    was_training = net.training
    net.eval()
    out = {k: [] for k in LOSS_SUMS}
    try:
        with torch.no_grad():
            for s in range(0, len(examples), chunk):
                ex, ws = examples[s:s + chunk], weights[s:s + chunk]
                x, distance, legal, optimal = _batch(agent, ex)
                policy, value = net.heads(x)
                logp = torch.log_softmax(policy.masked_fill(~legal, float("-inf")), dim=1)
                pol = (-logp.masked_fill(~optimal, float("-inf")).logsumexp(1)).cpu().numpy()
                has = ~torch.isnan(distance)
                val = torch.zeros(len(ex))
                if has.any():
                    val[has.cpu()] = F.smooth_l1_loss(value.reshape(-1)[has], distance[has] / 10,
                                                      reduction="none").cpu()
                val = val.numpy()
                for r, w in enumerate(ws):
                    out["n"].append(1.0), out["w"].append(w)
                    out["policy"].append(pol[r]), out["w_policy"].append(w * pol[r])
                    hv = bool(has[r])
                    out["value"].append(val[r]), out["w_value"].append(w * val[r]), out["n_value"].append(float(hv))
    finally:
        net.train(was_training)
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _log_losses(record, losses, puzzle, kind):
    """Adds to each puzzle's record entry {"loss": {kind: {sum name: value}}} from example_losses' arrays, given
    each example's puzzle index and kind."""
    for i in np.unique(puzzle):
        sums = {}
        for k in np.unique(kind[puzzle == i]):
            m = (puzzle == i) & (kind == k)
            sums[str(k)] = {name: float(f"{losses[name][m].sum():.6g}") for name in LOSS_SUMS}
        record[int(i)]["loss"] = sums


def learning_session(agent, puzzles, teacher=None, max_attempts=1, max_steps=200, memory=10, explore=1.0,
                     window=10, encode=1.0, steps=1, block=1, seed=0, on_puzzle=None,
                     surrender=None, shown_weight=1.0, judge=None, progress_weight=0.0, progress_tail=0.2,
                     labels="own", agency_step="off", even_batches=False, log_loss=False, solver=None):
    """A session in which a simulated, pure-intuition student (`agent`) learns online, a
    few puzzles at a time, under human-like limits on learning:

    - online: after each block of `block` puzzles it learns from those
      puzzles' attempts only (online_update, `steps` passes), and plays the
      next block with what it has become - nothing earlier is replayed;
    - encoding lapses: each label and move shown is learned with
      probability `encode` (see intuition_examples);
    - limited hindsight: loops are cut out of a solved attempt only when it
      returned to a board seen within the last `window` moves (None: any);
      or, with `labels` "path" / "all", shortcut labels instead: shortest
      routes through every board the attempt visited (shortcut_routes;
      `window` then limits which connections between visited boards are noticed);
    - no rewinding: no validation, no keeping an earlier, better version.

    `block` 1 learns after every puzzle. A larger block gives the network
    more data per update - a way to grant it more experience than a person
    would need for the same progress (see sessions.py's
    --multiplier).

    Each puzzle (a starting state) gets up to `max_attempts` attempts
    (solve_intuition, sampling at temperature `explore`, avoiding the last
    `memory` boards, with `teacher` if given), stopping at the first solve -
    a restart is a new attempt. `max_steps` caps each attempt: a number or a
    callable giving a puzzle's cap from its state. With `surrender` (a
    SurrenderRule), the student may also give up on a puzzle - counting the
    moves of all its attempts at it - which ends the puzzle: no restart.

    Agency (`shown_weight`, see agency_weights): what a shown move teaches
    counts this much, the student's own play 1. Progress labels (`judge`, an
    agent whose value head rates moves - see progress_examples): with
    `progress_weight` > 0 the student also learns from its own moves, solved
    or not, the clearest `progress_tail` share each way of its last 1000,
    each label weighing `progress_weight`. At the defaults (1, no judge) it
    learns as before. `agency_step` "scale" or "skip" (see online_update): weights below 1
    also shrink the update, not only how examples count against each other -
    with "off", a block where every move was shown learns at full strength.
    `even_batches` (see online_update): each update's batches of equal size, not 256 and a remainder.

    Returns one dict per puzzle, in order: puzzle (index), solved, surrendered, attempts,
    moves (all attempts), taught, interventions, examples learned
    from (progress labels included), progress_labels (by direction, with a
    judge). `on_puzzle(n, agent)`, if given, is called after each update with
    the number of puzzles played so far - e.g. to snapshot the student.

    `solver`, if given, plays each attempt instead of solve_intuition: solver(agent, state, cap, rng, gives_up)
    returning solve_intuition's dict (e.g. a planner; experiments/planner_session.py). No teacher with a solver.

    `log_loss` also records, per puzzle, what its examples gave the update: "loss" = {kind: sums of
    example_losses over its examples, measured just before the update that learns from them}, kind "own" or
    "shown" (on a board where a move was shown; "progress" for progress labels), and "update" = online_update's
    stats for its block (shared by the block's puzzles). It changes nothing else: same play, same updates."""
    rng = random.Random(seed)
    # the updates' own shuffling and skipping, apart from play; "off" keeps the random module, as before
    update_rng = random.Random(seed + 104729) if agency_step != "off" else None
    record = []
    pending, pending_w = [], []
    pending_idx, pending_kind, block_start = [], [], 0  # with log_loss: each pending example's puzzle and kind
    progress = judge is not None and progress_weight > 0
    weighted = shown_weight != 1 or progress
    recent = deque(maxlen=1000)
    for i, state in enumerate(puzzles):
        results = []
        if teacher is not None:
            teacher.counts = Counter()
        for _ in range(max_attempts):
            if teacher is not None:
                teacher.start(state, rng)
            gives_up = surrender.for_puzzle(state, sum(len(r["moves"]) for r in results)) if surrender else None
            if solver is not None:
                assert teacher is None, "a solver plays without a teacher"
                results.append(solver(agent, state, _cap(max_steps, state), rng, gives_up))
            else:
                results.append(solve_intuition(agent, state, _cap(max_steps, state), memory, explore, teacher, rng,
                                               gives_up))
            if teacher is not None:
                teacher.end_attempt(results[-1])
            if results[-1]["solved"] or results[-1]["surrendered"]:
                break
        examples = intuition_examples(results, agent.size, agent.max_slide, window, encode, rng, labels)
        if weighted:
            pending_w += agency_weights(examples, results, shown_weight, agent.size, agent.max_slide, window, labels)
        if log_loss:  # which examples sit on a board where a move was shown (agency_weights gives them shown_weight)
            if weighted and shown_weight != 1:
                ws, mark = pending_w[len(pending_w) - len(examples):], shown_weight
            else:
                ws, mark = agency_weights(examples, results, 0.0, agent.size, agent.max_slide, window, labels), 0.0
            pending_kind += ["shown" if a == mark else "own" for a in ws]
        extra, counts = progress_examples(results, judge, recent, progress_tail, agent.size, agent.max_slide,
                                          window, labels) if progress else ([], None)
        if log_loss:
            pending_kind += ["progress"] * len(extra)
            pending_idx += [i] * (len(examples) + len(extra))
        examples += extra
        pending_w += [progress_weight] * len(extra)
        pending += examples
        record.append({"puzzle": i, "solved": results[-1]["solved"], "surrendered": results[-1]["surrendered"],
                       "attempts": len(results),
                       "moves": sum(len(r["moves"]) for r in results),
                       "taught": sum(sum(r["taught"]) for r in results),
                       "interventions": sum(sum(r["taught"]) for r in results), "examples": len(examples),
                       "teacher_counts": dict(teacher.counts) if teacher is not None else {}}
                      | ({"progress_labels": counts} if progress else {}))
        if (i + 1) % block == 0 or i + 1 == len(puzzles):
            if log_loss and pending:
                _log_losses(record, example_losses(agent, pending, pending_w if weighted else None),
                            np.array(pending_idx), np.array(pending_kind))
            stats = None
            if pending:
                stats = online_update(agent, pending, steps, weights=pending_w if weighted else None,
                                      agency_step=agency_step, rng=update_rng, even_batches=even_batches)
            if log_loss:
                for entry in record[block_start:]:
                    entry["update"] = dict(stats or {"pool": 0, "batches": 0, "steps": 0})
            pending, pending_w, pending_idx, pending_kind = [], [], [], []
            block_start = len(record)
            if on_puzzle is not None:
                on_puzzle(i + 1, agent)
    return record
