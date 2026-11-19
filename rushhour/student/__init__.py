"""The simulated student: a policy network trained as AND-OR's apprentice, which then learns puzzle by puzzle in a
session. Everything a teacher or an analysis needs from it is re-exported here.

- network: PolicyAgent (the network, its moves and its ordering of AND-OR's options);
- solve: solving by intuition, by deliberation or both, and evaluating that over many puzzles;
- labels: what an attempt teaches (examples, hindsight shortcuts, a teacher's shown moves);
- train: expert iteration (pretraining) and the online learning of a session.
"""
from .labels import LABELS, agency_weights, intuition_examples, label_attempts, progress_examples, shortcut_routes
from .network import PolicyAgent
from .solve import (SurrenderRule, evaluate_arbitrated, evaluate_intuition, solve_and_or, solve_arbitrated,
                    solve_intuition)
from .train import expert_iteration, learning_session, online_update, train_policy

__all__ = ["LABELS", "PolicyAgent", "SurrenderRule", "agency_weights", "evaluate_arbitrated", "evaluate_intuition", "expert_iteration", "intuition_examples", "label_attempts", "learning_session", "online_update", "progress_examples", "shortcut_routes", "solve_and_or", "solve_arbitrated", "solve_intuition", "train_policy"]
