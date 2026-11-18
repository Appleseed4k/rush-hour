# Handoff: defining the teacher as a POMDP

Summary of the discussion (2026-11-13) that set up `notebooks/p7_pomdp.ipynb`. It covers the formal definition only. Results are in p7.

**Terms.**
- **Simulated student:** the policy network that plays and learns, w.
- **Student model:** a teacher's beliefs about that student, such as bayes3's skill per distance or p7's learning model. It never means the network itself.
- **τ is overloaded.** Here it is the simulated student's softmax temperature (`explore`). In bayes3 it is how much puzzle difficulty varies: a puzzle's shift is τ·z, with z ~ N(0, 1). On slides, give them different letters.

## The simulated student: one MDP per puzzle, plus a learning rule

- **Per puzzle i:** start board s₀, optimal length d₀, move cap H = step_factor · d₀.
  - **State:** (board, moves made, the last `memory` boards it avoids). The move count is in the state because giving up depends on it.
  - **Actions:** legal moves. Giving up is a fixed hazard (`SurrenderRule`), modelled as stochastic termination rather than a chosen action.
  - **Transitions:** deterministic board moves.
  - **Reward:** 1 when red exits, finite horizon H.
- **Policy:** π_w(a | s) ∝ exp(logit_w(s, a) / τ), masked by memory. τ is `explore`, and w is the network.
- **Learning rule:** w_{k+1} = U_α(w_k, D_k) after each block (`online_update`).
  - D_k: shortcut labels through the visited boards (`labels="all"`).
  - α is agency (`shown_weight`). A shown move's examples weigh α. With `agency_step="skip"`, a batch's step is taken with probability equal to its mean weight.
- **Student = (𝓜ᵢ, π_w, U_α).** It doesn't solve its MDPs; it improves its policy across them.

## The teacher: a POMDP over the student's knowledge

| element | definition |
|---|---|
| Hidden state | the student's knowledge k (in principle k = w), plus board, move count and puzzle index |
| Actions | show the optimal move, or let the student play |
| Transitions | show: the optimal move is played. Play: a ~ π_k. At block ends k' = U_α(k, D), so learning is a state transition |
| Observations | the student's own moves and the outcome (solved, gave up, capped). Shown moves carry no information about skill |
| Reward | **terminal held-out test gain**: R = E[score_test(k_T)] − score_test(k_0) |
| Belief | a posterior over k |

**Two timescales:** moves within a puzzle (the student's problem) and puzzles across the session (the teacher's learning problem).

## Decisions made

1. **The reward is always terminal learning on held-out tests.** The test puzzles never appear in the session, so the teacher gets no credit for solves it causes by showing moves.
   - **No separate help cost.** Help costs only through U_α: at α < 1 a shown move teaches less, shown-heavy batches may take no step, and showing displaces own play. That cost is the finding, so it must not be built into the reward.
   - A per-puzzle reward (solves minus a cost per shown move) was rejected as a goal. With no cost, showing every move is optimal, so there is no dilemma. It only describes what bayes3 optimizes.
   - **Shaping, if planning needs a denser reward:** Φ(b) = expected test score under the belief, and pay Φ(b_{t+1}) − Φ(b_t) each step (Ng, Harada & Russell 1999). The steps sum to the terminal reward, so the optimal policy is the same.
   - **A finite horizon makes the best policy depend on time.** For example, help near the end has little time to turn into learning. A fixed θ can't express that.
   - Other costs, such as quitting, belong in the dynamics (states where learning stops), not in the reward.
2. **bayes3 is not a POMDP solution.** It is a Bayesian student model plus a hand-set threshold rule: knowledge-tracing style, like Corbett & Anderson's 1995 mastery threshold.
   - **Belief:** a skill logit per distance to the goal, refit after each puzzle with decay; a difficulty shift τ·z for the current puzzle, updated after each move; and the direction of mistakes. The student model is kept for the whole session; only the plan is rebuilt per puzzle.
   - **Rule:** at the start of each puzzle, backward induction over (distance, moves made) gives the chance of solving alone, for each z on a 13-point grid. This runs in a simplified version of the puzzle: a walk in distance to the goal, not the full board MDP. It shows a move when E_belief[V_alone] < θ.
   - **Why it falls short:** it targets this puzzle's solve, not test learning. It plans as if skill never changes, ignoring U_α. It ignores its own later interventions. θ isn't derived.
   - **What the θ sweeps are:** direct policy search over a one-parameter family of belief policies, scored on the right objective by simulation.
3. **Agency stays a parameter of the environment, not of the teacher.** Weighting examples by their loss doesn't remove it (tested in p7).

## The plan for a real POMDP teacher

1. **Fit a learning model** to stand in for U_α, since planning through the network is too slow (a session takes about 25 min). The state is a few numbers, such as test score per length bin. It is driven by own and shown material per interval, with agency dependence. It must be able to show both sides of the dilemma.
2. **Validate it:** fit on some conditions and predict the others. It must reproduce the dilemma: where more help starts to cost at each agency, and random 100% against random 30%.
3. **Assemble the POMDP:**
   - observations from bayes3's move model;
   - reward = expected terminal test score (shaped as above);
   - belief = a particle filter over the knowledge state.
4. **Solve with coarse actions:** a θ per puzzle or per block from {0, 0.25, 0.45, 0.75, 0.9, 1}, with bayes3's rule inside the puzzle. Solve by dynamic programming or fitted value iteration over (belief summary, puzzles left). POMCP move by move can come later. This can do three things bayes3 can't: vary help over time, vary it by distance, and value playing for what it reveals about the student.
5. **Test on the simulated student:** add it as a teacher in `teach.py`, run it with `session_experiment.py` on the same 8 students, and compare it with bayes3 θ 0.45 at each agency, paired by student. Both outcomes are results: if it beats bayes3, adapting help matters; if it ties, a fixed θ is near optimal.

Reference template: Rafferty, Brunskill, Griffiths & Shafto (2016), "Faster Teaching via POMDP Planning".

## Where it stands (details in p7)

- **Steps 1–2 done.** The model:
  - state = test score per length bin (9-11, 14-16, 19-21, 24-26);
  - material M = own examples + c·α^γ·shown examples, with c ≈ 2.5, plus a little transfer between bins;
  - each bin's score approaches a floor exponentially.
- **Validation:** each held-out teacher is predicted at r ≈ 0.94. Every dilemma signature has the right sign, but heavy targeted help's cost at low agency comes out too small.
- **Loss weighting doesn't explain c.** Examples were weighted by their logged loss, and c stays at 2.6.
- **Fit the model along whole trajectories (`fit_trajectory`), not interval by interval.** The per-interval fit treated noisy test scores as the state. The trajectory fit raises held-out r to 0.96 and predicts 20 fresh runs at r 0.98. It still under-prices heavy targeted help at low agency, by about a third.
- **A θ that changes doesn't beat a constant one.** θ was switched between 0.25 and 0.9 by session half or by puzzle length, at agency 0.1 and 0.5, and compared with constant θ 0.45 on the same students.
  - Every schedule loses or ties: from +0.01 for more help late to about −0.15 for more help on long puzzles. The noise between two runs of θ 0.45 is about ±0.05.
  - The trajectory fit, trained on p6 alone, predicts every schedule's loss. It ranks them well at agency 0.1 (rank correlation 0.8), less well at 0.5 (0.6), where it puts the costs 2–3× too low.
  - So along time and puzzle length, bayes3 at θ 0.45 is close to the best on the simulated student. The POMDP's remaining value is a model-based benchmark for that claim, plus whatever playing to learn about the student is worth.
- **An upper bound instead of a POMDP.** A play model gives the own and shown examples per bin for each teacher (none, bayes3 θ 0.25 to 0.97, random 30% and 100%), as a line in the model state. With it, the learning model becomes a simulator. Rolled forward from the first test, it reproduces the 45 constant-teacher conditions at r 0.97.
  - Every student starts from the same network, so in the model the state is known exactly. The best plan is then an upper bound on any POMDP teacher in the model.
  - The plans searched change the teacher per interval and per length bin. At agency 0 to 0.7, constant θ 0.45 is within 0.003 of the best plan, and within 0.009 across 50 resamples of students.
  - The reason: θ 0.45 keeps the student's own play highest. A little help turns failures into solves, while more help displaces own moves. At low agency, a shown example is worth too little to make up for that.
  - Only at agency 0.9–1 does the model find room (+0.02 to +0.04), in random help. Runs on the simulated student contradict it there: θ 0.75 wins and random help doesn't.
- **Next:** a POMDP solved on this model would reproduce bayes3 θ 0.45 where help has a cost, so step 3 is on hold. It's worth revisiting only with a different learning model, new kinds of actions, or a better model of high agency.
- **Carrying skill across puzzles (`scripts/skill_tracking.py`, p7's last section).** A running summary predicts the student's moves at least as well as bayes3's refit. It needs a normal over the 11 skills, plus τ learned over a grid of 8 values.
  - Without a learning rate, it ties the refit. With τ fixed, it is 0.0007 worse.
  - With a learning rate added to the belief, it beats the refit by 0.0003 per move, on 8/8 students. The learning rate is directional, so the filter expects skill to rise.
  - A random-walk drift and skill uncertainty don't help.
  - So a POMDP belief over skill can be a few numbers, and the learning rate gives the planner a prediction that skill will rise. This was tested on unhelped students at agency 0 only.
