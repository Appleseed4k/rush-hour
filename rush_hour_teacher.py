import random
from collections import defaultdict

from rush_hour_lib import hint


class Teacher():
    """Decides, state by state, whether a training session should be steered by
    a hint instead of letting the agent's own policy choose. Hints are pulled
    automatically from the puzzle's BFS-optimal solution.

    `policy` decides *whether* to hint at a given state. "random" flips a
    `hint_prob`-weighted coin. "q" learns the decision with tabular
    Q-learning: the state is just distance-to-goal, the action is
    hint-or-not, and the reward is the student's own per-step reward,
    bootstrapped one step at a time exactly like standard tabular Q-learning
    — a hint's credit comes from what happens immediately after it, with any
    longer-run effect propagating backward through the chain of updates
    rather than a hand-widened reward window.

    Once `max_hints` hints have been given in a session, `exhausted` becomes
    True, at which point the calling training loop is expected to hand off to
    a generic solve() call for the remainder of the session. Call
    `new_session()` before each teaching session to reset the hint budget
    while keeping any learned Q-table intact across sessions.

    A "q"-policy teacher starts out `training`: each session's outcome updates
    its Q-table and decays its epsilon. Call `freeze()` once you're done
    training it (e.g. after a warm-up run over several student generations)
    to stop both, so it can be reused as a fixed, fully-evaluated policy
    across a separate group of students without drifting mid-comparison.
    """
    def __init__(self, cars, solution, policy="random", max_hints=1,
                 hint_prob=0.5, hint_reward=0.1, alpha=0.1, gamma=0.95, epsilon=1.0,
                 epsilon_decay=0.99):
        self.solution = solution
        self.policy = policy
        self.max_hints = max_hints
        self.hint_prob = hint_prob
        self.hint_reward = hint_reward
        self.hints_used = 0

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.qtable = defaultdict(lambda: {True: 0.0, False: 0.0})
        self._pending = None
        self.training = True

    @property
    def exhausted(self):
        return self.hints_used >= self.max_hints

    def freeze(self, epsilon=0.0):
        """Stops further Q-table updates and epsilon decay, fixing the "q" policy
        as-is (fully greedy by default) so it can be evaluated across a fresh
        group of students without continuing to learn mid-comparison.
        """
        self.training = False
        self.epsilon = epsilon

    def new_session(self):
        """Resets the per-session hint budget; keeps the learned Q-table (if any)."""
        self.hints_used = 0
        self._pending = None
        if self.policy == "q" and self.training:
            self.epsilon *= self.epsilon_decay

    def _context(self, state):
        return self.solution[state]

    def advise(self, state):
        """Returns an advised (car, direction, steps) action, or None to defer
        to the agent."""
        if self.exhausted:
            return None
        if self.policy == "q":
            action = self._advise_q(state)
        else:
            action = self._advise_automatic(state)
        if action is not None:
            self.hints_used += 1
        return action

    def observe(self, next_state, reward):
        """Bootstrapped Q-update for the "q" policy's last decision. No-op for
        "random" policy (only "q" has anything to learn), and for a frozen
        teacher (see `freeze()`).
        """
        if self._pending is None or not self.training:
            return
        ctx0, decision = self._pending
        self._pending = None
        ctx1 = self._context(next_state)
        q1 = self.qtable[ctx1]
        rpe = reward + self.gamma * max(q1.values()) - self.qtable[ctx0][decision]
        self.qtable[ctx0][decision] += self.alpha * rpe

    def reward(self, advised, steps, solved, penalty):
        """Reward for the action just taken. Random-policy hints get a flat
        bonus; everything else (agent moves and "q"-policy hints) uses the
        normal shaped reward, so the "q" policy learns from the same signal
        the student does.
        """
        if advised and self.policy == "random":
            return self.hint_reward
        return (1 - steps * penalty) if solved else 0

    def _advise_automatic(self, state):
        if random.random() >= self.hint_prob:
            return None
        return hint(state, self.solution)

    def _advise_q(self, state):
        ctx = self._context(state)
        q = self.qtable[ctx]
        if random.random() < self.epsilon:
            decision = random.choice([True, False])
        elif q[True] == q[False]:
            # Untrained/tied contexts fall back to the same hint_prob-weighted
            # coin flip as the "random" policy, instead of silently favoring
            # True via dict insertion order.
            decision = random.random() < self.hint_prob
        else:
            decision = max(q, key=q.get)
        action = hint(state, self.solution) if decision else None
        decision = action is not None
        self._pending = (ctx, decision)
        return action
