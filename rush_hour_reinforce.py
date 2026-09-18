import random

import torch
from torch.distributions import Categorical
from tqdm import tqdm

from rush_hour_and_or import solve as and_or_solve
from rush_hour_rl import (
    Environment,
    PolicyAgent,
    car_anchor,
    evaluate_policy,
    valid_action_mask,
)

STEP_REWARD = -1.0  # reward for every move taken, win or not - see run_episode


def and_or_recommendation(state, gamma_lapse=0.0, heuristic=None):
    """The single next move AND-OR's own subgoal-decomposition search
    (rush_hour_and_or.solve) recommends from `state` - the first move of
    whatever it finds, whether that's a full resolution all the way to the
    goal or just a partial-progress reposition it fell back to. None only in
    the (practically unreachable, since a car can almost always at least
    slide back the way it came) case where `state` has no legal moves left
    at all.

    `gamma_lapse`, passed straight through to rush_hour_and_or.solve as its
    `gamma` - the probability an OrNode abandons its subgoal for a random
    legal move instead (see rush_hour_and_or.py). 0.0 (the default) never
    lapses, always the search's best-effort recommendation.

    `heuristic`, also passed straight through (typically the guiding agent's
    own .heuristic method - see PolicyAgent.heuristic), replaces AND-OR's
    plain randomized search with a best-first one ordered by the heuristic's
    scores - including, notably, what a gamma_lapse falls back to: the
    heuristic's own top-ranked legal move instead of a uniformly random one.
    None (the default) is AND-OR's policy-blind search throughout.
    """
    result = and_or_solve(state, heuristic=heuristic, gamma=gamma_lapse)
    moves = result if isinstance(result, list) else result[1]
    return moves[0] if moves else None


class ReinforceAgent(PolicyAgent):
    """A PolicyAgent trained by REINFORCE (Monte-Carlo policy gradient)
    against real environment rollouts, instead of supervised imitation of
    AND-OR traces - see ReinforceTrainer.train(). Subclasses PolicyAgent
    purely to reuse its network/action-space construction and save()/load()
    (both key actions by board-cell anchor rather than car name, so one
    agent generalizes across puzzles regardless of how many cars they have -
    see car_anchor in rush_hour_rl.py); only how the weights get updated
    differs.

    `and_or_guidance`, if True, is the one place AND-OR enters this file -
    not as a teacher whose demonstrations get imitated (that's ExIt/
    PuzzleTrainer's approach, with its own self-play/regression-guard
    complexity), but as a per-step oracle act() can optionally defer to - see
    act()'s docstring. `and_or_gamma` is that oracle's own lapse probability
    (see and_or_recommendation) - independent of ReinforceTrainer's `gamma`
    (the REINFORCE return discount factor, an unrelated quantity that just
    happens to share the name).

    `guidance_scale` caps how much act() can lean on AND-OR independent of
    the policy's own confidence: the deferral chance is
    `guidance_scale * (1 - confidence)` rather than plain `1 - confidence`.
    1.0 (the default) is uncapped - full reliance early on, same as before
    this was added. Lower values are a computationally cheap stand-in for
    harder puzzles (where AND-OR's own plain search would itself be less
    reliable, giving less of a crutch) when what you actually want to study
    is how fast the policy gradient improves on its own, not how well a
    human-plausible teacher-assisted solver performs.

    `heuristic_confidence_threshold` gates a *second*, independent use of
    confidence: whether a deferred step's search is itself ordered by this
    agent's own .heuristic (see and_or_recommendation) or left heuristic-free
    (uniformly random, AND-OR's policy-blind search). Passing the heuristic
    through unconditionally sounds like it should only sharpen the gamma-lapse
    fallback, but order_candidates/_order_blockers in rush_hour_and_or.py use
    it to order *every* decision point in the search, not just the lapse -
    turning what used to be "explore a fresh random valid path every call"
    into "always commit to whatever the current network prefers". Early on
    that's a self-referential, low-diversity signal (an untrained student
    teaching itself from its own noise, and doing so most heavily exactly
    when guidance_scale*(1-confidence) is highest - i.e. when confidence, and
    therefore heuristic quality, is lowest). Gating on confidence again here
    means AND-OR only starts taking this agent's own judgment into account
    once that judgment has become non-arbitrary at the current state -
    mirroring a novice deferring to an independent teacher's unbiased
    demonstration before trusting their own instincts even for the edge
    cases. 0.5 is an arbitrary but reasonable "better than a coin flip" bar;
    1.0 disables this entirely (heuristic never used, matching AND-OR's
    original policy-blind behavior even when guided).

    None of and_or_guidance/and_or_gamma/guidance_scale/
    heuristic_confidence_threshold are part of the checkpoint save() writes
    (nothing PolicyAgent-specific is, beyond the network weights) - load()
    takes them as fresh arguments instead, same as a freshly constructed
    agent would.
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
        """Reconstructs a ReinforceAgent from a save() checkpoint via
        PolicyAgent.load() (same network weights, size, max_slide, hidden),
        then sets and_or_guidance/and_or_gamma/guidance_scale/
        heuristic_confidence_threshold on it - see this class's docstring for
        why those aren't part of the checkpoint itself."""
        agent = super().load(path, lr=lr)
        agent.and_or_guidance = and_or_guidance
        agent.and_or_gamma = and_or_gamma
        agent.guidance_scale = guidance_scale
        agent.heuristic_confidence_threshold = heuristic_confidence_threshold
        return agent

    def act(self, environment, mask):
        """Samples one action from the policy's (masked) softmax distribution.
        Unlike policy_avoiding_cycles (which ranks legal actions by logit
        under torch.no_grad(), for evaluation), this keeps the sample
        differentiable so its log-probability can be used in a REINFORCE
        update.

        If self.and_or_guidance, first rolls a
        (guidance_scale * (1 - confidence)) chance of deferring to AND-OR's
        own recommended move instead (see and_or_recommendation) - confidence
        being the policy's own top-action probability at this state. Early in
        training the policy is close to uniform over legal actions
        (confidence low everywhere), so this defers to AND-OR almost always
        (or up to guidance_scale's cap, if set below 1.0); as training makes
        the policy genuinely confident at a given state, it increasingly
        trusts its own choice there instead - self-annealing per state and
        per how far training has progressed, with no separate schedule or
        distance-to-goal threshold to tune. The recommendation's search is
        ordered by this agent's own .heuristic (see and_or_recommendation)
        only once confidence clears heuristic_confidence_threshold - below
        that, AND-OR stays policy-blind (uniformly random) so a low-confidence
        deferral gets an independent demonstration instead of a reflection of
        this agent's own not-yet-trustworthy judgment (see this class's
        docstring for why). Whichever
        action is actually taken, the log-probability used for the REINFORCE
        update is always the *policy's own* log pi(action | s) - even on a
        deferred step, this nudges the policy toward or away from AND-OR's
        choice based on how the episode turns out, rather than directly
        imitating it. That's a mild approximation (the step wasn't actually
        sampled from this policy, so it's not textbook-exact on-policy
        REINFORCE) traded for a dense source of sensible early moves
        undirected exploration alone would rarely find - see run_episode's
        cold-start problem without it.

        Returns ((car_name, direction, steps), log_prob, guided).
        """
        state = environment.get_state()
        logits = self.net(self.encoder.encode(state).unsqueeze(0)).squeeze(0)
        logits = logits.masked_fill(~mask, float("-inf"))
        dist = Categorical(logits=logits)

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
                        return recommendation, dist.log_prob(torch.tensor(idx)), True

        idx = dist.sample()
        cell, direction, steps = self.action_space[idx.item()]
        car_name = environment.car_at(cell)
        return (car_name, direction, steps), dist.log_prob(idx), False


def run_episode(agent, initial_cars, size, max_steps=200):
    """Plays one episode by sampling from agent.act() at every step, rather
    than the greedy eval rollout evaluate_policy() uses. Reward is
    STEP_REWARD per move taken - a plain shortest-path objective, since
    minimizing total steps is exactly maximizing this return - so no
    separate terminal bonus is needed: a puzzle solved in fewer moves always
    scores a higher (less negative) return than one solved in more, and an
    unsolved puzzle accumulates the worst possible return over max_steps.
    Fixed rather than a parameter: ReinforceTrainer.train() normalizes
    returns across each batch before using them, which is invariant to a
    uniform positive rescaling of every reward - so a tunable magnitude here
    would train identically for any value, and being a no-op parameter is
    more misleading than not having it at all.

    Returns (solved, steps, log_probs, rewards, guided_count) - steps is None
    when unsolved (matching evaluate_policy's convention), guided_count is
    how many of this episode's actions came from agent.act() deferring to
    AND-OR (always 0 when agent.and_or_guidance is False).
    """
    environment = Environment(size, initial_cars)
    log_probs, rewards = [], []
    guided_count = 0
    for step in range(max_steps):
        mask = valid_action_mask(environment, agent.action_space)
        if not mask.any():
            break
        action, log_prob, guided = agent.act(environment, mask)
        guided_count += guided
        environment.move(*action)
        log_probs.append(log_prob)
        rewards.append(STEP_REWARD)
        if environment.check_win():
            return True, step + 1, log_probs, rewards, guided_count
    return False, None, log_probs, rewards, guided_count


def discounted_returns(rewards, gamma):
    """G_t = sum_{k=t}^{T} gamma^(k-t) * r_k for every t in `rewards`,
    computed backward in one pass."""
    returns = [0.0] * len(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


class ReinforceTrainer():
    """Trains one shared ReinforceAgent across multiple puzzles via
    Monte-Carlo policy gradient (REINFORCE) - the pure-RL counterpart to
    rush_hour_rl.PuzzleTrainer's behavior-cloning-from-AND-OR approach. No
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
              desc="REINFORCE iterations"):
        """Each iteration: plays one sampled episode (run_episode) from every
        puzzle in a batch (all of train_puzzles, or `batch_size` puzzles
        chosen at random without replacement when the pool is too large to
        roll out in full every iteration), computes each episode's
        discounted return per step (discounted_returns), pools every step
        from every episode in the batch together, and normalizes returns
        across that whole pool (subtract mean, divide by std) as a simple
        variance-reducing baseline - vanilla per-episode REINFORCE with no
        baseline at all is usually too high-variance to learn much in a
        reasonable number of iterations. One optimizer step is taken on the
        pooled -log_prob * normalized_return loss per iteration.

        `eval_every` controls how often self.history gets a checkpoint (see
        PuzzleTrainer.train's docstring for the same tradeoff: evaluate_all()
        rolls every train + test puzzle out to max_steps, so evaluating every
        iteration dominates runtime past a handful of puzzles).

        Returns (policy_loss, train_results, test_results) - the final
        iteration's mean loss (nan if every iteration in the run produced no
        episode steps) and evaluate_all()'s result against the final weights.
        """
        self.history = []
        self.losses = []
        # Postfix reports each iteration's own sampled rollouts (solved/loss)
        # - the exploration policy's noisy, in-the-moment solve rate, not the
        # greedy evaluate_all() checkpoints in self.history/train_results
        # below, which only run every eval_every iterations and are what
        # actually measures the agent's current (deterministic) competence.
        with tqdm(range(n_iterations), desc=desc) as pbar:
            for iteration in pbar:
                pool = self.train_puzzles
                if batch_size is not None and batch_size < len(pool):
                    pool = random.sample(pool, batch_size)

                all_log_probs, all_returns = [], []
                solved_count = 0
                guided_steps = 0
                total_steps = 0
                for puzzle in pool:
                    solved, _steps, log_probs, rewards, guided_count = run_episode(
                        self.agent, puzzle, self.size, max_steps=max_steps)
                    solved_count += solved
                    guided_steps += guided_count
                    total_steps += len(log_probs)
                    if not log_probs:
                        continue
                    all_log_probs.extend(log_probs)
                    all_returns.extend(discounted_returns(rewards, self.gamma))

                postfix = {"sampled": f"{solved_count}/{len(pool)}"}
                if self.agent.and_or_guidance:
                    postfix["guided"] = f"{guided_steps / total_steps:.0%}" if total_steps else "n/a"

                if not all_log_probs:
                    tqdm.write(f"Warning: iteration {iteration} produced no episode steps - skipping this update")
                    pbar.set_postfix(**postfix, loss="n/a")
                else:
                    returns_t = torch.tensor(all_returns, dtype=torch.float32)
                    if returns_t.numel() > 1 and returns_t.std() > 1e-8:
                        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
                    log_probs_t = torch.stack(all_log_probs)
                    loss = -(log_probs_t * returns_t).sum() / len(pool)

                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    self.agent.optimizer.step()
                    self.losses.append(loss.item())
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
    mirrors rush_hour_rl.finetune_curve's shape and purpose exactly (see that
    docstring), but through Monte-Carlo policy-gradient updates instead of
    ExIt's heuristic-guided AND-OR search + supervised training. Returns the
    same (epoch, solved, steps) curve shape, so rush_hour_rl.average_curves
    can be reused unchanged on its output.

    `desc`, if given, overrides the progress bar's label - useful in a loop
    fine-tuning several puzzles/agents in a row (e.g. rush_hour_reinforce.ipynb's
    pretrained-vs-scratch cell) so each call's bar is distinguishable at a
    glance instead of all reading the same generic "REINFORCE fine-tune".
    """
    trainer = ReinforceTrainer([puzzle], [], size, gamma=gamma)
    trainer.agent = agent
    trainer.train(n_iterations=n_iterations, max_steps=max_steps, eval_every=eval_every, desc=desc)
    return [(iteration, train_results[0][0], train_results[0][1]) for iteration, train_results, _ in trainer.history]
