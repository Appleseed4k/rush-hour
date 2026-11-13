import random

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from and_or import solve as and_or_solve
from rl import (
    Environment,
    PolicyAgent,
    car_anchor,
    evaluate_policy,
    valid_action_mask,
)

STEP_REWARD = -1.0  # reward for every move taken, win or not - see run_episode


def and_or_recommendation(state, gamma_lapse=0.0, heuristic=None):
    """The next move and_or.solve() recommends from `state` - the first move
    of whatever it finds (a full resolution or a partial-progress
    reposition). None only if `state` has no legal moves left.

    `gamma_lapse`/`heuristic` are passed straight through to and_or.solve
    (see its docstring)."""
    result = and_or_solve(state, heuristic=heuristic, gamma=gamma_lapse)
    moves = result if isinstance(result, list) else result[1]
    return moves[0] if moves else None


class ReinforceAgent(PolicyAgent):
    """A PolicyAgent trained by REINFORCE (Monte-Carlo policy gradient)
    against real environment rollouts, instead of supervised imitation of
    AND-OR traces - see ReinforceTrainer.train(). Subclasses PolicyAgent
    purely to reuse its network/action-space construction and save()/load();
    only how the weights get updated differs.

    If `and_or_guidance`, act() may defer to AND-OR's recommended move (see
    and_or_recommendation) with probability `guidance_scale * (1 -
    confidence)`, confidence being the policy's own top-action probability -
    so deferral fades out per-state as the policy grows confident there.
    `and_or_gamma` is that oracle's own lapse probability, independent of
    ReinforceTrainer's `gamma` (the return discount factor - an unrelated
    quantity that happens to share the name). `heuristic_confidence_
    threshold` gates whether a deferred step's AND-OR search is itself
    ordered by this agent's own .heuristic, vs. left policy-blind, once
    confidence clears that bar - so AND-OR only starts taking this agent's
    judgment into account once it's stopped being arbitrary noise.

    and_or_guidance/and_or_gamma/guidance_scale/heuristic_confidence_
    threshold aren't part of save()'s checkpoint - load() takes them as
    fresh arguments instead.
    """

    def __init__(self, size=(6, 6), max_slide=None, lr=1e-3, hidden=128,
                 and_or_guidance=False, and_or_gamma=0.0, guidance_scale=1.0,
                 heuristic_confidence_threshold=0.5):
        super().__init__(size, max_slide=max_slide, lr=lr, hidden=hidden)
        self.and_or_guidance = and_or_guidance
        self.and_or_gamma = and_or_gamma
        self.guidance_scale = guidance_scale
        self.heuristic_confidence_threshold = heuristic_confidence_threshold

    @classmethod
    def load(cls, path, lr=1e-3, and_or_guidance=False, and_or_gamma=0.0, guidance_scale=1.0,
              heuristic_confidence_threshold=0.5):
        """Reconstructs a ReinforceAgent via PolicyAgent.load() (same network
        weights, size, max_slide, hidden), then sets and_or_guidance/
        and_or_gamma/guidance_scale/heuristic_confidence_threshold fresh -
        see this class's docstring for why those aren't checkpointed."""
        agent = super().load(path, lr=lr)
        agent.and_or_guidance = and_or_guidance
        agent.and_or_gamma = and_or_gamma
        agent.guidance_scale = guidance_scale
        agent.heuristic_confidence_threshold = heuristic_confidence_threshold
        return agent

    def act(self, environment, mask):
        """Samples one action from the policy's masked softmax distribution,
        differentiably (unlike policy_avoiding_cycles's no_grad argmax, used
        for eval). May defer to AND-OR's recommended move per
        self.and_or_guidance - see this class's docstring. Whichever action
        is actually taken, the returned log_prob is always the *policy's
        own* log pi(action | s) - even on a deferred step, this nudges the
        policy toward or away from AND-OR's choice based on how the episode
        turns out, rather than directly imitating it.

        Returns ((car_name, direction, steps), log_prob, guided, entropy) -
        entropy is this state's action-distribution entropy regardless of
        which branch below actually picked the move, for an entropy bonus
        that discourages the policy from collapsing prematurely (see
        ReinforceTrainer.train's `entropy_coef`).
        """
        state = environment.get_state()
        logits = self.net(self.encoder.encode(state).unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)
        entropy = dist.entropy()

        if self.and_or_guidance:
            confidence = dist.probs.max().item()
            defer_prob = min(1.0, self.guidance_scale * (1 - confidence))
            if random.random() < defer_prob:
                guiding_heuristic = self.heuristic if confidence >= self.heuristic_confidence_threshold else None
                recommendation = and_or_recommendation(state, gamma_lapse=self.and_or_gamma, heuristic=guiding_heuristic)
                if recommendation is not None:
                    car_name, direction, steps = recommendation
                    idx = self.action_index.get((car_anchor(state, car_name), direction, steps))
                    if idx is not None and mask[idx]:
                        return recommendation, dist.log_prob(torch.tensor(idx)), True, entropy

        idx = dist.sample()
        cell, direction, steps = self.action_space[idx.item()]
        car_name = environment.car_at(cell)
        return (car_name, direction, steps), dist.log_prob(idx), False, entropy


def run_episode(agent, initial_cars, size, max_steps=200):
    """Plays one episode by sampling from agent.act() at every step, rather
    than the greedy rollout evaluate_policy() uses. Reward is STEP_REWARD
    per move, so minimizing steps is exactly maximizing return - no terminal
    bonus needed. Fixed rather than a parameter since ReinforceTrainer.train()
    normalizes returns per batch, making any rescaling here a no-op.

    Returns (solved, steps, log_probs, rewards, guided_count, entropies) -
    steps is None when unsolved, guided_count is how many actions deferred
    to AND-OR.
    """
    environment = Environment(size, initial_cars)
    log_probs, rewards, entropies = [], [], []
    guided_count = 0
    for step in range(max_steps):
        mask = valid_action_mask(environment, agent.action_space)
        if not mask.any():
            break
        action, log_prob, guided, entropy = agent.act(environment, mask)
        guided_count += guided
        environment.move(*action)
        log_probs.append(log_prob)
        rewards.append(STEP_REWARD)
        entropies.append(entropy)
        if environment.check_win():
            return True, step + 1, log_probs, rewards, guided_count, entropies
    return False, None, log_probs, rewards, guided_count, entropies


def discounted_returns(rewards, gamma):
    """G_t = sum_{k=t}^{T} gamma^(k-t) * r_k for every t in `rewards`,
    computed backward in one pass."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def _distance_weight(steps, optimal, max_steps, cap):
    """How much this episode's steps should amplify its own contribution to
    the pooled REINFORCE loss: steps/optimal, capped, so an episode solved
    right at its BFS-optimal length weights 1.0 (no amplification) and one
    solved further off weights proportionally higher - pushing the update
    toward the puzzles the policy is currently worst at. `steps` is
    max_steps (the worst case) when unsolved. 1.0 if `optimal` is unknown."""
    if optimal is None or optimal <= 0:
        return 1.0
    steps = steps if steps is not None else max_steps
    return min(steps / optimal, cap)


class ReinforceTrainer():
    """Trains one shared ReinforceAgent across multiple puzzles via
    Monte-Carlo policy gradient (REINFORCE) - the pure-RL counterpart to
    rl.PuzzleTrainer's behavior-cloning-from-AND-OR approach. No
    AND-OR solving is involved anywhere in this file: every training signal
    comes from actually playing a puzzle out and observing whether/how fast
    it got solved.

    `test_puzzles`, `train_distances`/`test_distances`, evaluate_all(), and
    the (iteration, train_results, test_results) shape of self.history all
    mirror PuzzleTrainer exactly, so the same plotting cells from
    rush_hour.ipynb work unchanged against a ReinforceTrainer's history.
    """

    def __init__(self, train_puzzles, test_puzzles=(), size=(6, 6), lr=1e-3, hidden=128,
                 train_distances=None, test_distances=None, gamma=0.99,
                 and_or_guidance=False, and_or_gamma=0.0, guidance_scale=1.0,
                 heuristic_confidence_threshold=0.5):
        self.train_puzzles = list(train_puzzles)
        self.test_puzzles = list(test_puzzles)
        self.train_distances = list(train_distances) if train_distances is not None else None
        self.test_distances = list(test_distances) if test_distances is not None else None
        self.size = size
        self.gamma = gamma
        self.agent = ReinforceAgent(size, lr=lr, hidden=hidden, and_or_guidance=and_or_guidance,
                                     and_or_gamma=and_or_gamma, guidance_scale=guidance_scale,
                                     heuristic_confidence_threshold=heuristic_confidence_threshold)
        self.history = []
        self.losses = []

    def evaluate_all(self, max_steps=200):
        """(train_results, test_results): each a list of (solved, steps)
        aligned to train_puzzles/test_puzzles, via evaluate_policy's greedy
        (policy_avoiding_cycles) rollout - no sampling, no gradient. Identical
        shape/semantics to PuzzleTrainer.evaluate_all()."""
        def results(puzzles):
            return [
                (solved, steps if solved else float("nan"))
                for solved, steps, _ in (evaluate_policy(self.agent, p, self.size, max_steps) for p in puzzles)
            ]
        return results(self.train_puzzles), results(self.test_puzzles)

    def train(self, n_iterations=100, max_steps=200, eval_every=10, batch_size=None,
              entropy_coef=0.01, distance_weight_cap=3.0, desc="REINFORCE iterations"):
        """Each iteration: plays one sampled episode (run_episode) from every
        puzzle in a batch (all of train_puzzles, or `batch_size` random
        puzzles), pools every step's discounted return (discounted_returns)
        across the batch, and normalizes returns (subtract mean, divide by
        std) as a variance-reducing baseline. Each episode's normalized
        returns are then scaled by its own _distance_weight (steps/optimal,
        via train_distances - a no-op 1.0 everywhere if train_distances
        wasn't given), so episodes solved further from optimal pull the
        update harder than ones already near it. One optimizer step is
        taken on the pooled -log_prob * weighted_return loss, minus an
        `entropy_coef` bonus on the batch's mean action-distribution entropy
        (see ReinforceAgent.act) to keep the policy from collapsing onto a
        single move too early and getting stuck short of optimal.

        `eval_every` controls how often self.history gets a checkpoint (see
        PuzzleTrainer.train - evaluate_all() rolls every puzzle out to
        max_steps, so evaluating every iteration dominates runtime past a
        handful of puzzles).

        Returns (policy_loss, train_results, test_results) from the final
        iteration and evaluate_all() against the final weights.
        """
        self.history = []
        self.losses = []
        # postfix shows this iteration's own noisy sampled rollout, not the
        # greedy evaluate_all() checkpoints in self.history below
        with tqdm(range(n_iterations), desc=desc) as pbar:
            for iteration in pbar:
                indices = range(len(self.train_puzzles))
                if batch_size is not None and batch_size < len(self.train_puzzles):
                    indices = random.sample(indices, batch_size)

                all_log_probs, all_returns, all_weights, all_entropies = [], [], [], []
                solved_count = 0
                guided_steps = 0
                total_steps = 0
                for idx in indices:
                    optimal = self.train_distances[idx] if self.train_distances is not None else None
                    solved, steps, log_probs, rewards, guided_count, entropies = run_episode(
                        self.agent, self.train_puzzles[idx], self.size, max_steps=max_steps)
                    solved_count += solved
                    guided_steps += guided_count
                    total_steps += len(log_probs)
                    if not log_probs:
                        continue
                    weight = _distance_weight(steps, optimal, max_steps, distance_weight_cap)
                    all_log_probs.extend(log_probs)
                    all_returns.extend(discounted_returns(rewards, self.gamma))
                    all_weights.extend([weight] * len(log_probs))
                    all_entropies.extend(entropies)

                postfix = {"sampled": f"{solved_count}/{len(indices)}"}
                if self.agent.and_or_guidance:
                    postfix["guided"] = f"{guided_steps / total_steps:.0%}" if total_steps else "n/a"

                if not all_log_probs:
                    tqdm.write(f"Warning: iteration {iteration} produced no episode steps - skipping this update")
                    pbar.set_postfix(**postfix, loss="n/a")
                else:
                    returns_t = torch.tensor(all_returns, dtype=torch.float32)
                    if returns_t.numel() > 1 and returns_t.std() > 1e-8:
                        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
                    returns_t = returns_t * torch.tensor(all_weights, dtype=torch.float32)
                    log_probs_t = torch.stack(all_log_probs)
                    entropy_t = torch.stack(all_entropies).mean()
                    policy_loss = -(log_probs_t * returns_t).sum() / len(indices)
                    loss = policy_loss - entropy_coef * entropy_t

                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    self.agent.optimizer.step()
                    self.losses.append(loss.item())
                    postfix["entropy"] = f"{entropy_t.item():.2f}"
                    pbar.set_postfix(**postfix, loss=f"{loss.item():.3f}")

                if iteration % eval_every == 0 or iteration == n_iterations - 1:
                    train_results, test_results = self.evaluate_all(max_steps)
                    self.history.append((iteration, train_results, test_results))

        train_results, test_results = self.evaluate_all(max_steps)
        policy_loss = self.losses[-1] if self.losses else float("nan")
        return policy_loss, train_results, test_results


def finetune_curve(agent, puzzle, size=(6, 6), n_iterations=30, max_steps=200,
                    eval_every=1, gamma=0.99, desc="REINFORCE fine-tune"):
    """Fine-tunes `agent` in place on a single puzzle via ReinforceTrainer -
    mirrors rl.finetune_curve's shape/purpose (same (epoch, solved, steps)
    curve, reusable by rl.average_curves), via policy-gradient updates
    instead of ExIt's AND-OR-guided supervised training.

    `desc` overrides the progress bar's label, for distinguishing bars when
    fine-tuning several puzzles/agents in a loop.
    """
    trainer = ReinforceTrainer([puzzle], [], size, gamma=gamma)
    trainer.agent = agent
    trainer.train(n_iterations=n_iterations, max_steps=max_steps, eval_every=eval_every, desc=desc)
    return [(iteration, train_results[0][0], train_results[0][1]) for iteration, train_results, _ in trainer.history]
