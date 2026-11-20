# The simulated student: model card

This is the student the project's teaching results rest on. It lists every choice you would have to defend: the
parameter values, and also the architecture and the way each idea is implemented in code. For each choice it gives
the value, where the value came from, and the code that implements it. The settings are in
[`rushhour/student/spec.py`](rushhour/student/spec.py) (`STUDENT`, `AGENCY`, `PROTOCOL`).

- To run this student, use `python -m rushhour.sessions my_run --fitted --shown-weight 0.1 --teachers none bayes4`.
- `spec.differences(config)` checks a saved run against the spec.

Without `--fitted`, the command line's defaults describe a plain, uncalibrated student.

**Since 2026-11-19 the student starts from AND-OR's apprentice.** Before that, every run in the summary started
from a network trained on its own play with a random teacher, at giving-up scale 0.4 and multiplier 30. Those
values are `spec.PREVIOUS`; ["The previous start"](#the-previous-start) covers what changed.

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

The student is a neural network that plays Rush Hour by **intuition only**: on every board it samples a move from
its policy, with no search. Before a session it imitates the AND-OR planning model, which the paper fitted to
people's moves. That imitation is stopped at the point where the network solves puzzles as well and as efficiently
as people do in their first trials.

During a session it plays one puzzle at a time, and may give up as people do. After every 10 puzzles, which stand
for one human trial, it learns from those puzzles alone, with no replay. It learns only from **solved** attempts:
every board it visited is labelled with the shortest route to the goal through the boards it saw. A move a teacher
shows also teaches, at a weight called agency (ΔA), the one parameter the teaching results sweep.

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
| Input: 7 binary channels per cell (empty, red, horizontal car, vertical car, car start, car end, truck), plus 2 coordinate channels. No car identity | design | - | chosen | [`StateEncoder`](rushhour/student/network.py:32) |
| Action space keyed by cell: (a car's top-left cell, direction, 1-5 cells). Each cell's output predicts the moves of the car anchored there, so one network plays every puzzle and a pattern learned in one place applies everywhere | design | 720 actions | chosen | [`PolicyAgent`](rushhour/student/network.py:122), `car_anchor` |
| Trunk: a 3×3 stem, then 4 residual blocks, each convolving whole rows and whole columns plus a 3×3 window, with BatchNorm | design | 96 channels, 4 blocks | chosen | [`LaneBlock`](rushhour/student/network.py:70), [`PolicyNetwork`](rushhour/student/network.py:85) |
| Three heads on the shared trunk. **Policy**: a softmax over legal moves. **Dest**: per car and cell, how soon an optimal solution brings the car there. **Value**: moves to go in tens, from the board's mean features | design | - | chosen | [`PolicyNetwork.heads`](rushhour/student/network.py:85) |
| Only the policy head plays. Dest and value are trained in every update but read by nothing in a session | design | - | chosen | §4, §7 |

Notes:
- **Row and column convolutions:** what blocks a car, or red's path to the exit, is often several cells away (`LaneBlock`).
- **Start and end channels:** without them, two cars parked end to end look like one longer car.
- **Dest** exists for deliberation: it orders AND-OR's options (p1, p8). In pretraining it also learns AND-OR's
  subgoal choices (§3).
- **Value** was meant for progress labels, which are off. Both heads still shape the trunk the policy reads.

## 3. Starting knowledge: AND-OR's apprentice

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Starting network | parameter | `models/apprentice/gamma0.07_free/round1_epoch02.pt` | fitted | `STUDENT["init"]`, `paths.START` |
| **The expert is the AND-OR model.** It searches backward from the goal: an OR node picks which car to move, an AND node picks how, and each blocking car becomes a new subgoal | design | - | inherited (Olieslagers et al. 2023) | [`and_or.solve`](rushhour/and_or.py:364) |
| ...with lapses: at each subgoal, with probability γ, it plays one random legal move instead of planning | parameter | γ = 0.07, the paper's | inherited | [`OrNode.solve`](rushhour/and_or.py:178), [`lapse_move`](rushhour/and_or.py:89) |
| ...and this repo's variant of it. A completed plan is played out to its end and the search restarts from the new board; the paper's model plays one move per search | design | - | chosen | [`solve_and_or`](rushhour/student/solve.py:46), `attempt_puzzle` |
| ...with `protect_movers` off: a subgoal may move a car higher up the chain, as in the paper's model | parameter | off | chosen (fits people better than on; p3) | `and_or.PROTECT_MOVERS` |
| What the apprentice learns: plain AND-OR plays 1,000 pool puzzles of 1-30 moves (200 per bin of 1-8, 9-12, 13-16, 17-20 and 21-30), 4 attempts each. Its solved attempts, loops cut, label the boards (policy, value and dest targets as in §6). Every subgoal of a completed plan, with the option AND-OR took, trains the dest head's ordering | design | 2,966 solved attempts: 68,039 boards, 8,266 subgoals | chosen (phase 1's settings) | [`attempt_puzzle`](rushhour/student/labels.py:363), [`erase_loops`](rushhour/student/labels.py:162), [`experiments/apprentice.py`](experiments/apprentice.py) |
| Training: Adam at learning rate 1e-3, batches of 256, board and subgoal batches interleaved; validated on 100 held-out puzzles | implementation | - | chosen | [`train_policy`](rushhour/student/train.py:177) |
| **Stopped at epoch 2 of round 1**, the checkpoint that matches people's first 10 trials. That is 2 passes over plain AND-OR's attempts: imitation learning, with no expert-iteration rounds. Round 1's validation would have kept epoch 5 | parameter | round 1, epoch 2 | fitted | [`surrender.py`](rushhour/surrender.py) |
| The fit, together with the giving-up scale (§5): every saved epoch plays pool puzzles at people's lengths (20 each of 6, 10, 13, 15 moves; 10 runs each), and people's giving up is applied at each scale. The fit is to people's solve rate (87.4% ± 2.7%) and moves / optimal (2.76 ± 0.17) in trials 1-10 | implementation | best: 86.7% and 2.82, misfit 0.2 (next best: epoch 3 at 0.7) | fitted | `python -m rushhour.surrender` |
| The session starts with a fresh Adam state and a learning rate 10× lower than pretraining's | implementation | - | chosen | [`PolicyAgent.load`](rushhour/student/network.py:205) |

So the student's starting knowledge is **a partial copy of the paper's model of human planning**: the AND-OR
model's moves, learned to the point where it plays like people at the start. In the session, AND-OR is never called.

Later rounds (AND-OR ordered by the network itself) and later epochs are better than people. The summary's
apprentice curve shows that round 1 passes people's level within a few epochs.

## 4. Playing a puzzle

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Intuition only: each move is sampled from the policy's softmax over legal moves | design + parameter | temperature 1 (`explore`) | chosen | [`solve_intuition`](rushhour/student/solve.py:119), [line 149](rushhour/student/solve.py:149) |
| Short memory: moves back to one of the recent boards are removed from the choice, unless every move would be one | implementation + parameter | 10 boards | chosen | [line 146](rushhour/student/solve.py:146) |
| One attempt per puzzle, no restarts (people could restart) | parameter | 1 | chosen | `max_attempts`; [`learning_session`](rushhour/student/train.py:457) |
| Move cap; reaching it ends the attempt unsolved | parameter | max(50, 10 × optimal) | chosen | [`sessions.step_cap`](rushhour/sessions.py:540) |
| A teacher acts before the student moves: if it shows a move, that move is played instead. The student's own choice is computed only if the teacher asks for it | implementation | - | chosen | `propose` in `solve_intuition` |
| A shown move is an optimal move, chosen at random when several are | implementation | - | chosen | [`Teacher.optimal_move`](rushhour/teachers.py:86) |
| Each student gets its own random order of the session's puzzles | implementation | - | chosen | [`sessions.order`](rushhour/sessions.py:225) |

## 5. Giving up

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| A hazard: after each move that doesn't solve the puzzle, it gives up with a probability set by moves made / optimal length. Nothing else matters: not the board, its progress, or its recent success | design | - | people (shape) | [`SurrenderRule`](rushhour/student/solve.py:170), [line 164](rushhour/student/solve.py:164) |
| People's rates, pooled over lengths and attempts | people | 0.16-0.93% per move over 9 bins (edges 0, 0.5, 1, 1.5, 2, 3, 4, 6, 10) | people | [`human_data.surrender_hazard`](rushhour/human_data.py:213), saved per run as `surrender_table` |
| Scaled to the student's level | parameter | × 0.3 | fitted (§3) | `surrender_scale` |
| Giving up ends the puzzle unsolved, so the student learns nothing of its own from it (§6) | design | - | chosen | - |

## 6. What a puzzle teaches

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Only solved attempts produce labels from the student's own play | design | - | chosen | [`intuition_examples`](rushhour/student/labels.py:310), [line 332](rushhour/student/labels.py:332) |
| Hindsight over a graph of visited boards: two boards are joined when one legal move connects them (moves are reversible). The shortest route to the goal in that graph is computed by breadth-first search from the goal | implementation | - | chosen | [`shortcut_routes`](rushhour/student/labels.py:186) |
| Full hindsight: **every** visited board that can reach the goal is labelled (`labels "all"`), and any two boards are connected however far apart in time (`window -1`) | parameter | `all`, -1 | fitted, with the multiplier: the best of own, W=3, W=10 and full at each multiplier (below), though not by a wide margin | [`solved_routes`](rushhour/student/labels.py:249) |
| Each labelled board gets, from its shortest route(s): the policy target, every first move tied for shortest; the value target, moves left on the route ÷ 10; and dest targets, 0.8^(k-1) for a car reaching a cell k moves later. A board seen in several routes takes the one with the fewest moves left | implementation | dest decay 0.8 | chosen | [`attempt_examples`](rushhour/student/labels.py:264), [`DEST_DECAY`](rushhour/student/network.py:21) |
| A shown move teaches that move on that board, solved or not, with no value or dest target. It is skipped if a solved route already labels the board | design | - | chosen | [`_shown_example`](rushhour/student/labels.py:300) |
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
| Online, in blocks: the student plays a block of puzzles with fixed weights, then trains on that block's examples only. No replay, no validation, no keeping an earlier, better version | design | block = 10 puzzles (one human trial) | follows §9 | [`learning_session`](rushhour/student/train.py:457), update at [line 563](rushhour/student/train.py:563) |
| Each update: 3 passes over the block's examples, reshuffled each pass, one Adam step per batch | parameter + implementation | `steps 3` | chosen | [`online_update`](rushhour/student/train.py:329) |
| Equal batches: each pass is split into ceil(n / 256) batches of equal size, not 256s and a remainder | implementation | `even_batches` | chosen | `online_update` |
| Adam at a constant learning rate; its state (moment estimates) carries over across the session's updates | parameter + implementation | `lr 1e-4` | chosen | [`PolicyAgent`](rushhour/student/network.py:157) |
| Loss: policy cross-entropy on the set of target moves + dest binary cross-entropy + 0.5 × value smooth-L1, each a mean over the batch | design | weights 1, 1, 0.5 | chosen | [`_loss`](rushhour/student/train.py:40), [`_weighted_loss`](rushhour/student/train.py:55) |
| BatchNorm uses each update's batch statistics and moves its running averages. Play uses those running averages | implementation | PyTorch default momentum 0.1 | chosen | `net.train()` in `online_update`, `net.eval()` in [`_evaluate`](rushhour/student/network.py:222) |

Notes:
- **The last batch, without equal batches:** Adam steps a batch of 5 leftover examples as far as a full one (`online_update`).
- **BatchNorm:** with 3 passes per trial, the running statistics are pulled toward each trial's mix of boards. This
  is a side effect of online learning that nothing controls.

## 8. Agency (ΔA): what a shown move teaches

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Examples on a board where a move was shown weigh ΔA, and so do dest targets that a shown move set on earlier boards. The student's own play weighs 1 (the generation effect) | design + parameter | swept 0-1 | **not fitted** | [`agency_weights`](rushhour/student/labels.py:455) |
| ΔA also sets how much is learned: a batch's update is taken with probability equal to its mean weight, otherwise skipped, and its loss is divided by that mean | implementation | `agency_step "skip"` | chosen | [`online_update`, line 377](rushhour/student/train.py:377) |

Notes:
- **Why skip instead of weighting:** Adam divides each step by the recent size of the gradients, so it cancels a
  weight shared by a whole batch. A block where every move was shown would then learn at full strength
  (`online_update`). The alternative, `scale`, shrinks the step instead.
- **Help has a second cost that ΔA doesn't control.** A shown move keeps the student on the optimal path, so a fully
  shown puzzle leaves fewer boards to label (17 examples against 29-39, measured on the previous start; p6).

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

## What is fitted to people, and how well

1. **Giving up, by moves / optimal**: people's hazard, used as measured (§5).
2. **Starting level**: the checkpoint (round 1, epoch 2) and the giving-up scale (0.3), fitted to people's first
   10 trials. The misfit is 0.2, in squared standard errors over 2 statistics (§3).
3. **Learning rate over trials 1-30**: the multiplier and the hindsight, together. The fit uses three statistics,
   each compared with people's:
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

**Everything else is chosen, not fitted:**
- the architecture;
- the expert's variant (plans played out, `protect_movers` off) and pretraining's settings;
- temperature, memory, the move cap, single attempts;
- learning only from success, the loss, the optimizer and its settings;
- agency.

γ is the paper's value, not refitted here: p3's sweep fits people best near 0.01 with `protect_movers` off. But
apprentices trained at 0.01 and 0.07 showed no difference on pool puzzles.

## Which saved runs use this student

`spec.differences` over the session runs in `models/sessions` (2026-11-19):

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

So 0.45 stays the default. At high agency, more help does slightly better, as random 100% does.

## The previous start

Until 2026-11-19 every session started from `models/students/round6_epoch01.pt`, at giving-up scale 0.4 and
multiplier 30 (`spec.PREVIOUS`; `spec.differences(config, previous=True)` matches the old sweep
`teach_shortcut_agency*_even` exactly).

**What it was:** expert iteration with the network as its own expert. Over 10 rounds it played 400 fresh pool puzzles
of 1-20 moves, 2 attempts each, by intuition, with a random teacher showing 10% of moves. It trained on its own
solved attempts with loops cut, plus the shown moves (Adam at 1e-3, up to 10 epochs a round, patience 3). Round 6
was the round that matched people (the p2 run `teach/fresh_p0.1`, archived).

**Why it was replaced:** it came out of p2's question (teaching during pretraining) and was kept when p3 calibrated
to people. So its starting knowledge came from its own play and an oracle, not from a model of human planning.
Its multiplier was calibrated with own labels and never refitted for full hindsight: full × 30 has a misfit of 13.2,
against the apprentice's 2.8 at ×10.

## What is not part of the student

- **AND-OR and γ** are the student's teacher during pretraining only (§3). In a session it plays by intuition, and
  AND-OR is never called. Deliberation, AND-OR ordered by the network, is p8's comparison, not part of this student.
- **The teacher's model of the student** (`teachers.Bayesian4`: skill per distance, puzzle difficulty) is the
  teacher's belief, not this network.
- **The session protocol** (`spec.PROTOCOL`) is the experimental design: 8 students, sessions of 72 trials (18 per
  length bin, each trial 10 puzzles), tests of 50 puzzles per bin every 18 trials, untaught, with no giving up.

## Premises worth challenging

These are the places where the student is a choice rather than a fit.

**How it thinks**
1. **No deliberation.** People plan, and AND-OR fits their moves. The student starts as a copy of AND-OR's moves but
   never plans itself. In p8, network-ordered AND-OR matches people's efficiency.
2. **One network for every puzzle, keyed by cell.** What it learns transfers by board pattern. Whether people
   transfer that way is untested.
3. **Heads trained but unused.** Dest and value shape the shared features without ever acting.

**Where it starts**

4. **An imitation stopped early.** People's starting level is matched by stopping training at epoch 2, not by
   anything in how the network learns. The expert is this repo's AND-OR variant (plans played out) at the paper's γ.
   Pretraining's learning rate is also 10× the session's.

**What it learns from**

5. **Learning only from success.** Unsolved and abandoned attempts teach nothing of the student's own.
6. **Labels from perfect hindsight:** exact shortest routes through every board seen. This is also what makes
   wandering pay (§6, §8). It fits people best, but W=10 and W=3 come close at other multipliers. Only own labels
   are ruled out.

**How it learns**

7. **The multiplier.** 10 network puzzles for each human trial still patches the network's data hunger, though it now
   fits people's learning rate (§9).
8. **Online learning by Adam with BatchNorm.** Its forgetting and drift are those of the optimizer, not anything
   fitted (§7).
9. **Agency as a weight on shown moves, implemented by skipping updates.** It is unfitted, and the dilemma depends
   on it together with the exploration cost (§8).

**What it is tested on**

10. **The puzzle source.** Pool puzzles of 9-26 moves for teaching, against people's 70 layouts of 6-15.
