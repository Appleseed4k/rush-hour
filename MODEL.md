# The simulated student: model card

This is the student the project's teaching results rest on. It lists every choice you would have to defend: the
parameter values, and also the architecture and the way each idea is implemented in code. For each choice it gives
the value, where the value came from, and the code that implements it. The settings are in
[`rushhour/student/spec.py`](rushhour/student/spec.py) (`STUDENT`, `AGENCY`, `PROTOCOL`).

- To run this student, use `python -m rushhour.sessions my_run --fitted --shown-weight 0.1 --teachers none bayes4`.
  Add `--student intuition` for phase 1's student on the same start.
- `spec.differences(config)` checks a saved run against the spec (`intuition=True`: against phase 1's student).

Without `--fitted`, the command line's defaults describe a plain, uncalibrated student.

**Since 2026-11-19 the student starts from AND-OR's apprentice.** Before that, every run started
from a network trained on its own play with a random teacher, at giving-up scale 0.4 and multiplier 30. Those
values are `spec.PREVIOUS`; ["The previous start"](#the-previous-start) covers what changed. The phase 1 summary (`notebooks/p1_summary.ipynb`) shows the apprentice's runs, and keeps the
previous start's teacher comparisons, which have no counterpart on the apprentice, marked as such.

**Since 2026-11-24 the network has no dest head, and the student starts from an apprentice trained without one**
(round 1 epoch 3, giving-up scale 0.35). The runs of the week between started from the same apprentice trained with
a dest head as well (round 1 epoch 2, scale 0.3: `spec.DEST`); ["The dest-head start"](#the-dest-head-start) covers
what changed and what was kept.

**Later on 2026-11-24 the policy and value heads were split into two networks** of the same shape: a policy network
trained on the move targets only, and a value network trained on the distances only (`paths.START`). The one-network
start of that day is `spec.SHARED` / `paths.SHARED_START`; ["The shared-trunk start"](#the-shared-trunk-start) covers
the change. **Each network now has 2 lane blocks, not 4** (giving-up scale 0.40): 2, 4 and 6 fit people alike, and 2
is the smallest and fastest. The 4-block start is `spec.FOUR_BLOCK` / `paths.FOUR_BLOCK_START`.

**Since then the student plays as the two-mode student** (`spec.TWO_MODE`): each move is a single AND-OR
deliberation, flat planning or intuition, with weights fitted to people's moves (§4). Phase 1's student played by
intuition alone (`spec.INTUITION`, `--student intuition`), and every teaching result so far is from it.

**Kind:**
- **parameter**: a number or switch, in `spec.py`;
- **design**: a modelling or architectural choice that has no switch;
- **implementation**: how an idea is realized in code, where a reasonable alternative would behave differently.

**Origin:**
- **people**: taken from people's data directly;
- **fitted**: chosen to match people;
- **chosen**: set by hand;
- **inherited**: from the paper or an earlier phase.

Where the code records why a choice was made, the notes under each table say so. Where they don't, no reason was
recorded.

People's data is Olieslagers et al. 2023: 42 participants, 70 layouts of optimal length 6, 10, 13 or 15 (counting
the red car's exit), 60 minutes each.

## In one paragraph

The student is two neural networks: a **policy network** that proposes moves and a **value network** that judges
how far a board is from the goal. They play Rush Hour in the modes a move-by-move fit found in people's play. Most
moves are **intuition**: a move sampled from the policy, with no search. Some are planned, either by **flat
planning** (the policy's five likeliest moves, each judged one move ahead by the value network) or by a single
**AND-OR deliberation** (one pass of the paper's model of people's planning, its first move played). Before a
session both networks imitate the AND-OR model. That imitation is stopped at the point where the student solves
puzzles as well and as efficiently as people do in their first trials.

During a session it plays one puzzle at a time, and may give up as people do. After every 10 puzzles, which stand
for one human trial, it learns from those puzzles alone, with no replay. It learns only from **solved** attempts:
every board it visited is labelled with the shortest route to the goal through the boards it saw. A move a teacher
shows also teaches, at a weight called agency (ΔA), the one parameter the teaching results sweep. Planning gets
better only as the two networks do; how often it plans is fixed (AND-OR 18% of moves) or scheduled (flat planning
6% → 18% over the first 30 trials, as people's rose).

## 1. Board, moves and the oracle

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Rules: 6×6 board, cars of 2 or 3 cells slide any distance along their lane, red exits on the right | design | - | inherited | [`board.neighbors`](rushhour/board.py:22) |
| A move is one car sliding any number of cells; lengths count red's final slide out, so the paper's 5/9/12/14 are 6/10/13/15 here | design | - | chosen | `human_data.distance` |
| Every board's exact distance to the goal, by breadth-first search over the whole puzzle: the oracle behind teachers, tests and the move cap | implementation | - | - | [`board.multi_bfs`](rushhour/board.py:58) |
| Puzzles are drawn from the pool of all puzzles, by length bin, not from people's 70 layouts | design | `data/rush_nw.txt` | chosen | [`board.sample_by_distance`](rushhour/board.py:160) |

`surrender.py` notes that the student finds people's layouts much harder than pool puzzles of the same length.

## 2. Network

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Input: 7 binary channels per cell (empty, red, horizontal car, vertical car, car start, car end, truck), plus 2 coordinate channels. No car identity | design | - | chosen | [`StateEncoder`](rushhour/student/network.py:28) |
| Action space keyed by cell: (a car's top-left cell, direction, 1-5 cells). Each cell's output predicts the moves of the car anchored there, so one network plays every puzzle and a pattern learned in one place applies everywhere | design | 720 actions | chosen | [`PolicyAgent`](rushhour/student/network.py:113), `car_anchor` |
| Trunk: a 3×3 stem, then residual lane blocks, each convolving whole rows and whole columns plus a 3×3 window, with BatchNorm | design | 96 channels, 2 blocks (592k weights per network) | chosen; 2, 4 and 6 blocks fit people alike (2026-11-24), and 2 is the smallest and fastest | [`LaneBlock`](rushhour/student/network.py:66), [`PolicyNetwork`](rushhour/student/network.py:81) |
| Two networks of that shape, each read through one head. The **policy network**'s policy head: a score per move, read at its car's anchor cell; a softmax over the legal ones. The **value network**'s value head: moves to go in tens, from the board's mean features. Each network's other head is never trained or read | design | - | chosen (2026-11-24: as good as one shared trunk, and each network learns one thing) | [`PolicyNetwork.heads`](rushhour/student/network.py:81), [`PolicyAgent.value_agent`](rushhour/student/network.py:113) |
| Both networks play: the policy network proposes every move (intuition's sample, flat planning's candidates, AND-OR's order); the value network judges flat planning's candidates. Phase 1's intuition student read only the policy network | design | - | chosen | §4 |
| The policy head also orders AND-OR's options when AND-OR is run with the network (the two-mode student's deliberations; pretraining's later rounds): a move not legal yet is scored by its slot, which is trained on boards where it is legal | design | - | chosen (2026-11-24; it orders AND-OR as well as the dest head did) | [`PolicyAgent.heuristic`](rushhour/student/network.py:233) |

Notes:
- **Row and column convolutions:** what blocks a car, or red's path to the exit, is often several cells away (`LaneBlock`).
- **Two blocks are enough.** After one block a cell has seen its own row and column and its neighbours'; after two,
  the whole board. More blocks would let one pass follow longer chains of blockers, but 4 and 6 blocks don't change the
  start, the learning, the skill by distance on people's layouts or how often deep-chain moves are made
  (`experiments/network_depth.py`; `notebooks/p2.5_value_network.ipynb`, section 6). Two blocks halve the weights and
  play a session in about 60% of the time.
- **Start and end channels:** without them, two cars parked end to end look like one longer car.
- **No dest head.** Until 2026-11-24 a third head, dest, scored every car's every position by how soon a solution
  brings it there (target 0.8^(k-1) at k moves), and ordered AND-OR's options; in pretraining it also learned which
  option AND-OR took at each subgoal. It was removed for simplicity: ordering AND-OR by the policy head plays as well,
  and an apprentice trained without dest targets learns at the same pace and, in the two-mode session, gains
  as much (`p2_summary.ipynb`, section 7). Old checkpoints load with it dropped.
- **Two networks, not one trunk.** Until later on 2026-11-24 both heads shared one trunk, trained on the sum of the
  move and distance losses. Training the policy without the distance loss learns as fast, and gives a policy that
  fits people as well; the value network alone estimates distance as well as the shared head did. Two networks make
  each learn one thing: the policy network *choosing*, the value network *judging*.
- **Value** was meant for progress labels, which are off. The two-mode student reads it for flat planning, which is
  why it is kept.

## 3. Starting knowledge: AND-OR's apprentice

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Starting networks | parameter | `models/apprentice/separate_b2/round1_epoch03.pt`: the policy network `gamma0.07_free_novalue_b2/round1_epoch03.pt` and the value network `gamma0.07_free_valueonly_b2/round1_epoch03.pt` (2 lane blocks each) | fitted | `STUDENT["init"]`, `paths.START` |
| **The expert is the AND-OR model.** It searches backward from the goal: an OR node picks which car to move, an AND node picks how, and each blocking car becomes a new subgoal | design | - | inherited (Olieslagers et al. 2023) | [`and_or.solve`](rushhour/and_or.py:364) |
| ...with lapses: at each subgoal, with probability γ, it plays one random legal move instead of planning | parameter | γ = 0.07, the paper's | inherited | [`OrNode.solve`](rushhour/and_or.py:178), [`lapse_move`](rushhour/and_or.py:89) |
| ...and this repo's variant of it. A completed plan is played out to its end and the search restarts from the new board; the paper's model plays one move per search | design | - | chosen | [`solve_and_or`](rushhour/student/solve.py:46), `attempt_puzzle` |
| ...with `protect_movers` off: a subgoal may move a car higher up the chain, as in the paper's model | parameter | off | chosen (fits people better than on; p1.3) | `and_or.PROTECT_MOVERS` |
| What the apprentice learns: plain AND-OR plays 1,000 pool puzzles of 1-30 moves (200 per bin of 1-8, 9-12, 13-16, 17-20 and 21-30), 4 attempts each. Its solved attempts, loops cut, label the boards (policy and value targets as in §6): the policy network learns the moves, the value network the distances, each from the same examples | design | 2,966 solved attempts: 68,039 boards | chosen (phase 1's settings) | [`attempt_puzzle`](rushhour/student/labels.py:233), [`erase_loops`](rushhour/student/labels.py:42), [`experiments/apprentice.py`](experiments/apprentice.py) |
| Training: Adam at learning rate 1e-3, batches of 256; validated on 100 held-out puzzles | implementation | - | chosen | [`train_policy`](rushhour/student/train.py:113) |
| **Stopped at epoch 3 of round 1**, the checkpoint that matches people's first 10 trials. That is 3 passes over plain AND-OR's attempts: imitation learning, with no expert-iteration rounds. Round 1's validation would also have kept epoch 3 | parameter | round 1, epoch 3 | fitted | [`surrender.py`](rushhour/surrender.py) |
| The fit, together with the giving-up scale (§5): every saved epoch of the policy network plays, by intuition, pool puzzles at people's lengths (20 each of 6, 10, 13, 15 moves; 10 runs each), and people's giving up is applied at each scale. The fit is to people's solve rate (87.4% ± 2.7%) and moves / optimal (2.76 ± 0.17) in trials 1-10. The value network is taken at the same epoch | implementation | best: 87.7% and 2.70, misfit 0.14 at scale 0.40 (with 4 blocks: 87.3% and 2.72, misfit 0.06 at 0.35) | fitted | `experiments/network_depth.py startfit` (`rushhour.surrender`'s fit) |
| The session starts with a fresh Adam state and a learning rate 10× lower than pretraining's | implementation | - | chosen | [`PolicyAgent.load`](rushhour/student/network.py:196) |

So the student's starting knowledge is **a partial copy of the paper's model of human planning**: the AND-OR
model's moves and how far it still had to go, learned to the point where it plays like people at the start - not the
order AND-OR tried its options in, which in plain AND-OR is random. In the session AND-OR returns as one of the
student's modes, ordered by the student's own policy network (§4).

The start was fitted by intuition, not by the two-mode student, and was not refitted for it: played by the
two-mode student, it makes 2.81 and 2.74 moves / optimal in a session's trials 1-9 (two seeds), against people's
2.77 (`lean_people_all_m10_ao_b2*`; `models/summary/network_depth/summary.csv`).

Later rounds (AND-OR ordered by the network itself) and later epochs are better than people. The phase 1 summary's
apprentice curve shows that round 1 passes people's level within a few epochs.

## 4. Playing a puzzle

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Each move is made by one of three modes, drawn independently per move: nothing on the board or in the attempt so far triggers planning | design | - | fitted: people's moves fit a mixture of the three better than any one (below) | [`solve_two_mode`](rushhour/student/two_mode.py:99) |
| **Deliberation**: one AND-OR pass from the board, its options ordered by the policy network (with Gumbel noise), lapsing at γ. Its first move is played and no plan is kept: one move per search, as in the paper's model | design + parameter | 18% of moves; γ 0.07 | fitted (the share); inherited (γ) | `w_ao`; [`and_or.solve`](rushhour/and_or.py:364), `PolicyAgent.heuristic` |
| **Flat planning**: the policy's 5 likeliest moves, each judged one move ahead by the value network, one picked by a softmax over −(moves to go) at τ 1; a move that solves the puzzle is always picked | design + parameter | 6% of moves in trial 1, rising linearly to 18% by trial 30; 5 moves, τ 1 | fitted | `flat_start`, `flat_end`, `ramp`; [`flat_pick`](rushhour/student/two_mode.py:64), [`TwoModeSolver.flat_at`](rushhour/student/two_mode.py:209) |
| **Intuition**, otherwise: a move sampled from the policy's softmax over legal moves | design + parameter | temperature 1 | chosen | [`sample_move`](rushhour/student/two_mode.py:45) |
| The opening: an attempt's first move is always a deliberation, and its second is the same pass's second move | design | - | fitted: people's slow first look (2.9 s) and the move after it fit AND-OR | `opening` |
| Soft memory: in intuition and flat planning, a move back to a board the attempt was on `age` moves ago loses β × 0.5^(age / half-life) in its logit | implementation + parameter | β 2, half-life 1 move | fitted to people's revisits and undos | `soft_memory`; `penalty` in `solve_two_mode` |
| One attempt per puzzle, no restarts (people could restart) | parameter | 1 | chosen | `max_attempts`; [`learning_session`](rushhour/student/train.py:370) |
| Move cap; reaching it ends the attempt unsolved | parameter | max(50, 10 × optimal) | chosen | [`sessions.step_cap`](rushhour/sessions.py:540) |
| A teacher acts before the student moves: if it shows a move, that move is played instead (it also ends the opening's second move). The student's own choice is computed only if the teacher asks for it, and untaught play draws the same random numbers as without a teacher | implementation | - | chosen | `propose` in `solve_two_mode` |
| A shown move is an optimal move, chosen at random when several are | implementation | - | chosen | [`Teacher.optimal_move`](rushhour/teachers.py:86) |
| Each student gets its own random order of the session's puzzles | implementation | - | chosen | [`sessions.order`](rushhour/sessions.py:225) |

The fits (`notebooks/p2.2_forward_planning.ipynb`; `p2_summary.ipynb`, section 7):
- **The mode weights** come from a move-by-move mixture over every move people made 2+ moves from the goal
  (`experiments/mode_mixture.py` → `models/summary/mode_mixture/`). Each move's probability is a mix of intuition's,
  flat planning's and AND-OR's (sampled: 100 passes per board) under the same network, and the weights are fitted per
  stage of the session. AND-OR's share is 18%, pooled (in the first fit it didn't grow with practice); flat planning's grows from 6% to 18% when
  each stage is scored with a network that has learned as long. Refitted 2026-11-24 with AND-OR ordered by the policy
  (before, by the dest head: 10%, and flat 8% → 18%; `models/summary/mode_mixture_dest/`).
- **What the AND-OR share means.** Policy-ordered AND-OR's likeliest first move is the policy's own top move 67% of
  the time, so part of "deliberation" is a sharper version of intuition. AND-OR fits best near the goal (26-34% of
  moves 2-3 from it) and on slow moves; the student uses one share everywhere.
- **The soft memory** is the best of the memories tried on people's revisits and undos (`experiments/fit_memory.py`
  → `models/summary/memory_fit.csv`). With it the student revisits boards about as often as people do. The fit saw
  only solved attempts of 1.5-3 × optimal, but in longer attempts too people return to earlier boards about as often
  (17-24% of moves at 3-8 × optimal, against 15-22%; the hard memory 3-6%). It costs the student about 10 points of
  solve rate on the teaching runs' tests against the hard memory, with the same network (p2.6, section 4).

Phase 1's student (`--student intuition`, [`solve_intuition`](rushhour/student/solve.py:119)) made every move by
intuition, at temperature `explore` 1, with a hard memory: moves back to one of the last 10 boards (`memory`) were
removed from the choice, unless every move would be one. That memory fits people's revisits worst of those tried.

## 5. Giving up

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| A hazard: after each move that doesn't solve the puzzle, it gives up with a probability set by moves made / optimal length. Nothing else matters: not the board, its progress, or its recent success | design | - | people (shape) | [`SurrenderRule`](rushhour/student/solve.py:170), [line 164](rushhour/student/solve.py:164) |
| People's rates, pooled over lengths and attempts | people | 0.16-0.93% per move over 9 bins (edges 0, 0.5, 1, 1.5, 2, 3, 4, 6, 10) | people | [`human_data.surrender_hazard`](rushhour/human_data.py:213), saved per run as `surrender_table` |
| Scaled to the student's level | parameter | × 0.40 | fitted (§3) | `surrender_scale` |
| Giving up ends the puzzle unsolved, so the student learns nothing of its own from it (§6) | design | - | chosen | - |

## 6. What a puzzle teaches

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Only solved attempts produce labels from the student's own play | design | - | chosen | [`intuition_examples`](rushhour/student/labels.py:180), [line 202](rushhour/student/labels.py:202) |
| Hindsight over a graph of visited boards: two boards are joined when one legal move connects them (moves are reversible). The shortest route to the goal in that graph is computed by breadth-first search from the goal | implementation | - | chosen | [`shortcut_routes`](rushhour/student/labels.py:66) |
| Full hindsight: **every** visited board that can reach the goal is labelled (`labels "all"`), and any two boards are connected however far apart in time (`window -1`) | parameter | `all`, -1 | fitted, with the multiplier: the best of own, W=3, W=10 and full at each multiplier (below), though not by a wide margin | [`solved_routes`](rushhour/student/labels.py:129) |
| Each labelled board gets, from its shortest route(s): the policy target, every first move tied for shortest, and the value target, moves left on the route ÷ 10. A board seen in several routes takes the one with the fewest moves left | implementation | - | chosen | [`attempt_examples`](rushhour/student/labels.py:144) |
| A shown move teaches that move on that board, solved or not, with no value target. It is skipped if a solved route already labels the board | design | - | chosen | [`_shown_example`](rushhour/student/labels.py:171) |
| Everything is remembered (no encoding lapses), and nothing is learned from failed moves (no progress labels) | parameter | `encode 1`, `progress_weight 0` | chosen | `intuition_examples`, `progress_examples` |

Two things follow from these choices:
- **The value target is an upper bound.** A route through visited boards is only as short as the boards the student
  happened to see, so its length can exceed the true distance.
- **A wandering attempt leaves more labels than a direct one.** Every visited board is labelled, so a puzzle the
  student wandered through and solved teaches more than a short, direct solve (§8).

Note that pretraining labelled AND-OR's attempts with loops cut, not with full hindsight.

## 7. How it learns

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Online, in blocks: the student plays a block of puzzles with fixed weights, then trains on that block's examples only. No replay, no validation, no keeping an earlier, better version | design | block = 10 puzzles (one human trial) | follows §9 | [`learning_session`](rushhour/student/train.py:370), update at [line 483](rushhour/student/train.py:483) |
| Each update: 3 passes over the block's examples, reshuffled each pass, one Adam step per batch | parameter + implementation | `steps 3` | chosen | [`online_update`](rushhour/student/train.py:250) |
| Equal batches: each pass is split into ceil(n / 256) batches of equal size, not 256s and a remainder | implementation | `even_batches` | chosen | `online_update` |
| Adam at a constant learning rate; its state (moment estimates) carries over across the session's updates | parameter + implementation | `lr 1e-4` | chosen | [`PolicyAgent`](rushhour/student/network.py:148) |
| Loss: the policy network, cross-entropy on the set of target moves; the value network, smooth-L1 on the distance ÷ 10; each a mean over the batch, each network with its own Adam and its own passes over the block | design | - | chosen | [`_loss`](rushhour/student/train.py:35), [`_weighted_loss`](rushhour/student/train.py:48) |
| BatchNorm uses each update's batch statistics and moves its running averages. Play uses those running averages | implementation | PyTorch default momentum 0.1 | chosen | `net.train()` in `online_update`, `net.eval()` in [`policy_logits`](rushhour/student/network.py:220) |

Notes:
- **The last batch, without equal batches:** Adam steps a batch of 5 leftover examples as far as a full one (`online_update`).
- **BatchNorm:** with 3 passes per trial, the running statistics are pulled toward each trial's mix of boards. This
  is a side effect of online learning that nothing controls.

## 8. Agency (ΔA): what a shown move teaches

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Examples on a board where a move was shown weigh ΔA. The student's own play weighs 1 (the generation effect). (With the dest head, dest targets a shown move set on earlier boards weighed ΔA too) | design + parameter | swept 0-1 | **not fitted** | [`agency_weights`](rushhour/student/labels.py:315) |
| ΔA also sets how much is learned: a batch's update is taken with probability equal to its mean weight, otherwise skipped, and its loss is divided by that mean | implementation | `agency_step "skip"` | chosen | [`online_update`, line 298](rushhour/student/train.py:298) |

Notes:
- **Why skip instead of weighting:** Adam divides each step by the recent size of the gradients, so it cancels a
  weight shared by a whole batch. A block where every move was shown would then learn at full strength
  (`online_update`). The alternative, `scale`, shrinks the step instead.
- **Help has a second cost that ΔA doesn't control.** A shown move keeps the student on the optimal path, so a fully
  shown puzzle leaves fewer boards to label (17 examples against 29-39, measured on the previous start; p1.6).

## 9. Time scale: the multiplier

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Each human trial stands for m simulated puzzles, because networks learn far less per example than people | parameter | m = 10, so a 72-trial session is 720 puzzles | fitted with full hindsight from this start (below) | [`sessions.multiplier`](rushhour/sessions.py:220) |
| Implemented as m **different** puzzles, all learned from in one update per trial, not as one puzzle replayed or extra passes | implementation | block = m ([line 184](rushhour/sessions.py:184)) | chosen | [`sessions.sample_puzzles`](rushhour/sessions.py:204) |

## 10. Randomness and reproducibility

| Choice | Kind | Code |
|---|---|---|
| Each student's play, labels and teacher share one random stream, seeded by the student. So once a teacher draws a random number (random p, or a tie between optimal moves), the student's later draws differ from the no-teacher run. Comparisons between teachers pair students by seed and puzzle order, not by identical randomness | implementation | [`sessions._student`](rushhour/sessions.py:340) (`seed + 7919 × (student + 1)`), `teacher.start(state, rng)` |
| With `agency_step` on, the update's shuffling and skipping have their own stream, apart from play | implementation | `update_rng` in `learning_session` |
| Python's hash seed is fixed (`PYTHONHASHSEED=0`, set by the command line). Set and dict order of boards decides the order of examples, and so the batches | implementation | [`sessions.py`](rushhour/sessions.py:71) |
| AND-OR draws its lapses and Gumbel noise from Python's `random` module, which the two-mode student seeds from its own stream at each attempt (and each test attempt from its seed) | implementation | [`TwoModeSolver.__call__`](rushhour/student/two_mode.py:212), `sessions._test_job` |

## What is fitted to people, and how well

1. **Giving up, by moves / optimal**: people's hazard, used as measured (§5).
2. **Starting level**: the checkpoint (round 1, epoch 3, 2 lane blocks) and the giving-up scale (0.40), fitted to
   people's first 10 trials. The misfit is 0.14, in squared standard errors over 2 statistics (§3).
3. **Learning rate over trials 1-30**: the multiplier and the hindsight, together, fitted with the dest-head start
   (`spec.DEST`), by the intuition student, and kept for this one without refitting. The two-mode student at this
   start, at ×10 with full hindsight, gains 0.51 and 0.50 moves / optimal from trials 1-9 to 46+ (two seeds; people
   0.60, from 2.77 to 2.17; `lean_people_all_m10_ao_b2*`). The fit uses three statistics, each compared with
   people's:
   - **short:** within-learner change in log moves / optimal per 10 trials, on 6- and 10-move puzzles;
   - **long:** the same on 13- and 15-move puzzles;
   - **start:** moves / optimal in trials 1-10.

   Misfit is the sum of the three z², so about 3 is a good fit. Each cell is 8 students with no teacher at
   people's lengths (runs `ao_people_{own,w3,w10,all}_m*`; `models/summary/hindsight_fit.csv`, written by
   `experiments/phase2_analysis.py`):

   | start, hindsight | ×5 | ×10 | ×20 | ×30 | ×60 | ×120 |
   |---|---|---|---|---|---|---|
   | AND-OR apprentice, own labels | 14.1 | 28.7 | 9.0 | | | |
   | AND-OR apprentice, W=3 | 9.7 | 10.0 | 4.6 | | | |
   | AND-OR apprentice, W=10 | 4.3 | 10.9 | 5.5 | | | |
   | **AND-OR apprentice, full** | 15.3 | **2.8** | 6.6 | 13.4 | 28.6 | 45.6 |
   | previous start, full | | 8.8 | 12.2 | 13.2 | 31.2 | 42.4 |

   At ×10 with full hindsight the z are −1.4 (short), +1.0 (long) and +0.3 (start): the best cell of any start,
   hindsight or multiplier tried.

   **What the grid supports:**
   - **Own labels are ruled out.** At no multiplier do long puzzles improve as people's do (z long +2.6 to +5.1).
   - **Short windows learn long puzzles too slowly.** Long puzzles are the main misfit of W=3 and W=10 at every
     multiplier except W=10 × 20, where short puzzles improve too fast instead.
   - **Full hindsight is the best fit, not the only one.** W=10 × 5 (4.3) and W=3 × 20 (4.6) are within what 8
     students can tell apart. The grid's noise is visible in its rows: misfit is not monotone in the multiplier
     (W=10: 4.3, 10.9, 5.5).
4. **How it plays**: the mode weights (AND-OR 18%, flat planning 6% → 18% over 30 trials), flat planning's 5 moves
   and τ, and the soft memory, fitted to people's moves and revisits (§4).

**Everything else is chosen, not fitted:**
- the architecture;
- the expert's variant (plans played out, `protect_movers` off) and pretraining's settings;
- intuition's temperature, the move cap, single attempts;
- that modes are drawn independently per move, and that flat planning's share follows a schedule;
- learning only from success, the loss, the optimizer and its settings;
- agency.

γ is the paper's value, not refitted here: p1.3's sweep fits people best near 0.01 with `protect_movers` off. But
apprentices trained at 0.01 and 0.07 showed no difference on pool puzzles.

## Which saved runs use this student

**The teaching runs:** `twomode_m10[_s1|_s2]_agency{0,0.1,0.3,0.5,0.7,0.9,1}` (2026-11-25,
`experiments/replication_queue.py`): phase 1's ×10 agency sweep with this student, seeds 0-2 (p2.6). Beside them,
`intuition_m10[_s1|_s2]_agency*` are the same sweep with phase 1's intuition student on this start
(`spec.differences(config, intuition=True)`). Its sessions before these have no teacher and were run by
`experiments/planner_session.py --lean` at people's lengths (config key `lean`, not `student`): at this start
`lean_people_all_m10_ao_b2` and `_b2_seed1`; at the earlier starts the runs named in the sections below.

`spec.differences(config, dest=True)` over the session runs in `models/sessions` (2026-11-19): every run below
started from the dest-head start and played by intuition.

| Runs | Matches? | How they differ |
|---|---|---|
| `aostart_free_m10_agency{0,0.1,0.3,0.5,0.7,0.9,1}` (none, bayes4, random 30%, random 100%; no teacher is in `agency1`; bayes4 at θ 0.25, 0.75 and 0.9 too in `agency{0,0.3,1}`) | **exactly** | only ΔA, the swept parameter |
| `aostart_free_m10_s{1,2}_agency*` | student yes | seeds 1 and 2 (two more sets of 8 students), the same teachers |
| `ao_people_all_m{5,10,20,30,60,120}` | the calibration | multiplier as named; sessions and tests at people's lengths, tests with giving up |
| `ao_people_{own,w3,w10}_m{5,10,20}` | the calibration's alternatives | the hindsight and multiplier as named |
| `aostart_free_agency*`, `aolambda_free_s1_agency*` | this start at ×30 | multiplier 30 (before the multiplier was refitted); seed 1 for `aolambda` |
| `aostart_protected_agency{0.1,1}` | another start | phase 1's apprentice (`protect_movers` on), scale 0.25 |
| `teach_shortcut_agency*`, `people_*`, `pilot_*`, `fitted_teachers` | the previous start | see below |
| `calibrated_v2` | - | holds only the test puzzles `surrender.py` fits on |

The teaching sweep runs on **pool puzzles of 9-26 moves**, longer than any puzzle people played. The calibration
runs use people's lengths.

**Teaching effects at ×10 are pooled over seeds 0-2, 24 students per condition** (`models/summary/ao10_agency_seeds.csv`).
Seed 0 alone overstated every gain over no teacher, because its no-teacher students ended unusually badly. Their
end score was 5.23, against 4.89 and 4.68 for seeds 1 and 2, while its teachers ended like the other seeds'.
Comparisons between two teachers are stable across seeds.

| ΔA | bayes4 vs none | random 30% vs none | random 100% vs none | bayes4 vs random 100% |
|---|---|---|---|---|
| 0 | −0.20 ± 0.06 | | | |
| 0.1 | −0.27 ± 0.06 | +0.25 ± 0.05 | +0.75 ± 0.05 | −1.02 ± 0.04 |
| 0.3 | −0.25 ± 0.06 | −0.02 ± 0.05 | +0.16 ± 0.06 | −0.41 ± 0.06 |
| 0.5 | −0.26 ± 0.06 | −0.16 ± 0.06 | −0.14 ± 0.04 | −0.13 ± 0.05 |
| 0.7 | −0.29 ± 0.08 | −0.23 ± 0.05 | −0.36 ± 0.04 | +0.08 ± 0.06 |
| 0.9 | −0.30 ± 0.08 | −0.36 ± 0.06 | −0.55 ± 0.05 | +0.25 ± 0.04 |
| 1 | −0.27 ± 0.07 | −0.36 ± 0.06 | −0.61 ± 0.05 | +0.34 ± 0.04 |

The values are end-score differences, paired by student (lower is better), as mean ± se.
- **bayes4** helps by about a quarter of a move per optimal move at every ΔA. It shows 5% of moves.
- **Random help** turns from harmful to helpful as ΔA rises. Showing every move overtakes bayes4 between ΔA 0.5
  and 0.7.

**bayes4's θ** (seed 0 only; `models/summary/bayes4_theta.csv`), paired against θ 0.45:
- **0.25** shows 4% of moves and does as well.
- **0.75** shows 10%. It is a little worse at ΔA 0 (+0.12 ± 0.10) and a little better at ΔA 1 (−0.08 ± 0.06).
- **0.9** shows 31-38%. It is clearly worse at ΔA 0 (+0.79 ± 0.11) and 0.3 (+0.32 ± 0.05).

So 0.45 stays the default. At high agency, more help does slightly better, as random 100% does. The same check with
the two-mode student (`models/summary/replication_theta.csv`, 2026-11-25) comes out the same way: 0.25 and 0.45 alike,
0.75 worse at ΔA 0 and better at 1, 0.9 worse at ΔA 0-0.3 (p2.6, section 5).

## The 4-block start

For the rest of 2026-11-24 the two networks had 4 lane blocks each (`models/apprentice/separate/round1_epoch03.pt`,
giving-up scale 0.35; `spec.FOUR_BLOCK`, `paths.FOUR_BLOCK_START`; sessions `lean_people_all_m10_ao_separate*`).
2, 4 and 6 blocks, each pretrained on the same data and fitted to people's start the same way (all at epoch 3), gave
two-mode sessions alike at two seeds each: the share of optimal moves 7-9 from the goal on people's layouts (33-36%
for every depth), the total skill misfit (1.9-2.5), the session's gain (0.42-0.62, within seed noise) and how often
deep-chain rule moves are made (22% → 24-27%). So the smallest was kept.

## The shared-trunk start

For part of 2026-11-24 the student started from `models/apprentice/gamma0.07_free_nodest/round1_epoch03.pt`: one
network with both heads on a shared trunk, trained on the sum of the two losses (`spec.SHARED`,
`paths.SHARED_START`; the two-mode sessions `lean_people_all_m10_ao_nodest*`). Splitting it into a policy network
and a value network (`experiments/separate_value.py`; `notebooks/p2.5_value_network.ipynb`):
- the policy trained without the distance loss learns as fast and matches people's start at the same epoch and scale;
- either value network gives the two-mode student the same play;
- the two-mode sessions gain as much, follow the rules the same way, and fit people's own layouts as well (a small
  edge for the two networks after the start, about the size of the replay noise; checked with a second set of
  initial weights for both policies).

## The dest-head start

From 2026-11-19 to 2026-11-24 every session started from `models/apprentice/gamma0.07_free/round1_epoch02.pt` at
giving-up scale 0.3 (`spec.DEST`, `paths.DEST_START`; `spec.differences(config, dest=True)` matches the teaching
sweep `aostart_free_m10_agency*` exactly).

**What it was:** the same apprentice, on the same data, with a third head (dest) trained as well: dest targets on every
board, and the option AND-OR took at each of its 8,266 subgoals as a ranking target. It matched people's start after
2 epochs (86.7% solved, 2.82 moves / optimal, misfit 0.2).

**Why it was replaced:** for simplicity and explainability. Phase 2 ordered AND-OR by the dest head in its
deliberations, but ordering by the policy head plays as well (the two-mode student's AND-OR moves optimal 44.3%
against 43.4%; AND-OR on every move solves 68% against 58%). Round 1 retrained without dest targets
(`gamma0.07_free_nodest`, same seed and data) learns at the same pace, matches people's start one epoch later, and
its two-mode session gains as much as the dest-head one (0.58 against 0.60 moves / optimal; people 0.60).

**What was kept:** the multiplier (×10) and full hindsight (§9 and "What is fitted"), bayes4's θ 0.45 and every saved
result were fitted or run with the dest-head start, and are kept as they were.

## The previous start

Until 2026-11-19 every session started from `models/students/round6_epoch01.pt`, at giving-up scale 0.4 and
multiplier 30 (`spec.PREVIOUS`; `spec.differences(config, previous=True)` matches the old sweep
`teach_shortcut_agency*_even` exactly).

**What it was:** expert iteration with the network as its own expert. Over 10 rounds it played 400 fresh pool puzzles
of 1-20 moves, 2 attempts each, by intuition, with a random teacher showing 10% of moves. It trained on its own
solved attempts with loops cut, plus the shown moves (Adam at 1e-3, up to 10 epochs a round, patience 3). Round 6
was the round that matched people (the p1.2 run `teach/fresh_p0.1`, archived).

**Why it was replaced:** it came out of p1.2's question (teaching during pretraining) and was kept when p1.3 calibrated
to people. So its starting knowledge came from its own play and an oracle, not from a model of human planning.
Its multiplier was calibrated with own labels and never refitted for full hindsight: full × 30 has a misfit of 13.2,
against the apprentice's 2.8 at ×10.

## What is not part of the student

- **AND-OR's whole-puzzle plans.** Pretraining's AND-OR plays each plan to its end (§3); the student deliberates
  one move at a time and keeps no plan. Plan following was looked for in people's moves after a slow move and not
  found, beyond the opening's second move.
- **The teacher's model of the student** (`teachers.Bayesian4`: skill per distance, puzzle difficulty) is the
  teacher's belief, not this network.
- **The session protocol** (`spec.PROTOCOL`) is the experimental design: 8 students, sessions of 72 trials (18 per
  length bin, each trial 10 puzzles), tests of 50 puzzles per bin every 18 trials, untaught, with no giving up.

## Premises worth challenging

These are the places where the student is a choice rather than a fit.

**How it thinks**
1. **Modes drawn, not chosen.** Which mode makes a move is a draw with fitted weights. Nothing on the board, in the
   attempt or in a failure makes the student plan (people's moves after a loop are, if anything, less AND-OR-like).
   Flat planning's rising share is a schedule, not something the student learns.
2. **Planning learns nothing of its own.** Deliberation and flat planning improve only as the two networks do. Nothing
   learns which mode to use, and the networks learn from a solved attempt's hindsight labels whichever mode made its
   moves.
3. **One network for every puzzle, keyed by cell.** What it learns transfers by board pattern. Whether people
   transfer that way is untested.

**Where it starts**

4. **An imitation stopped early.** People's starting level is matched by stopping training at epoch 3, not by
   anything in how the network learns, and the stop was fitted by intuition rather than by the two-mode student
   (§3). The expert is this repo's AND-OR variant (plans played out) at the paper's γ. Pretraining's learning rate is
   also 10× the session's.
5. **Without people's near-goal knowledge.** People make 82-85% optimal moves 2-3 from the goal from their first
   trials; the student starts at 69-73% and doesn't catch up. Networks trained further by expert iteration have that
   knowledge (and represent the rules' concepts); this imitation doesn't (`notebooks/p2.4_rules.ipynb`, sections 3
   and 6).

**What it learns from**

6. **Learning only from success.** Unsolved and abandoned attempts teach nothing of the student's own.
7. **Labels from perfect hindsight:** exact shortest routes through every board seen. This is also what makes
   wandering pay (§6, §8). It fits people best, but W=10 and W=3 come close at other multipliers. Only own labels
   are ruled out.

**How it learns**

8. **The multiplier.** 10 network puzzles for each human trial still patches the network's data hunger, though it now
   fits people's learning rate (§9). It was fitted with the intuition student.
9. **Online learning by Adam with BatchNorm.** Its forgetting and drift are those of the optimizer, not anything
   fitted (§7).
10. **Agency as a weight on shown moves, implemented by skipping updates.** It is unfitted, and the dilemma depends
    on it together with the exploration cost (§8).

**What it is tested on**

11. **The puzzle source.** Pool puzzles of 9-26 moves for teaching, against people's 70 layouts of 6-15.

**What it doesn't reproduce.** These are known, and accepted for the teaching work rather than fixed:

12. **Mid-puzzle skill.** People's share of optimal moves 7-9 from the goal rises from 48% to 59% over the session;
    the student's stays at 33-36%, with 2, 4 or 6 lane blocks alike, and even expert iteration's rounds 2-3 reach only
    37%. The gap is lookahead, not the network's capacity (`notebooks/p2.5_value_network.ipynb`, section 6). On the
    teaching runs' tests (up to 26 moves, 10 × optimal, no giving up) it shows as looping: in failed attempts the
    student keeps returning to the same few boards, and more so as its policy sharpens. People loop as much but find
    a way out (`notebooks/p2.6_teaching_replication.ipynb`, section 4).
13. **Its level on people's own layouts.** Trained and fitted on pool puzzles of people's lengths, it finds people's
    70 layouts harder than people do (3.4-3.5 moves / optimal in trials 1-9, against people's 2.3).
14. **Giving up and restarting.** Giving up is people's hazard scaled to fit (§5), not a decision; the student never
    restarts a puzzle, where people could.
15. **Individual differences.** Students differ only by seed and puzzle order: the spread of their session gains is a
    tenth of people's (SD 0.07-0.17 against 0.6-1.4).
