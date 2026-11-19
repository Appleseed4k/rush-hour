# The simulated student: model card

This is the student the teaching results rest on. It lists every choice you would have to defend: the parameter
values, and also the architecture and the way each idea is implemented in code. For each choice it gives the value,
where the value came from, and the code that implements it. The settings are in
[`rushhour/student/spec.py`](rushhour/student/spec.py) (`STUDENT`, `AGENCY`, `PROTOCOL`).

- To run this student, use `python -m rushhour.sessions my_run --fitted --shown-weight 0.1 --teachers none bayes4`.
- `spec.differences(config)` checks a saved run against the spec.

Without `--fitted`, the command line's defaults describe a plain, uncalibrated student.

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
its policy, with no search. Before a session it is pretrained to people's starting level. During a session it plays
one puzzle at a time, and may give up as people do. After every 30 puzzles, which stand for one human trial, it
learns from those puzzles alone, with no replay. It learns only from **solved** attempts: every board it visited is
labelled with the shortest route to the goal through the boards it saw. A move a teacher shows also teaches, at a
weight called agency (ΔA), the one parameter the teaching results sweep.

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
- **Dest** exists for deliberation: it orders AND-OR's options (p1, p8).
- **Value** was meant for progress labels, which are off. Both heads still shape the trunk the policy reads.

## 3. Starting knowledge (pretraining)

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Starting network | parameter | `models/students/round6_epoch01.pt` | fitted | `STUDENT["init"]`, `paths.START` |
| Pretrained by expert iteration with **the network as its own expert**: each round it plays fresh puzzles by intuition and trains on its own solved attempts with loops cut out (self-imitation) | design | `student="intuition"` | chosen | [`expert_iteration`](rushhour/student/train.py:232) |
| ...with a random teacher showing 10% of moves (optimal ones, from the oracle), which also become labels | design | random 10% | chosen | the p2 run `teach/fresh_p0.1` (archived) |
| Each round: 400 new pool puzzles of 1-20 moves (100 per bin of 1-8, 9-12, 13-16 and 17-20), 2 attempts each | parameter | 10 rounds | chosen | [`label_attempts`](rushhour/student/labels.py:411) |
| Each round trains to convergence: Adam at learning rate 1e-3, batches of 256, up to 10 epochs. Training stops after 3 epochs without improvement in validation agreement (the policy's top move against held-out attempts' labels), and keeps the best epoch | implementation | - | chosen | [`train_policy`](rushhour/student/train.py:177) |
| Which round, fitted together with the giving-up scale (§5): the pair that matches people's first 10 trials (solve rate and moves / optimal) on pool test puzzles of people's lengths | parameter | round 6, scale 0.4 | fitted | [`surrender.py`](rushhour/surrender.py:99) |
| The session starts with a fresh Adam state and a learning rate 10× lower than pretraining's | implementation | - | chosen | [`PolicyAgent.load`](rushhour/student/network.py:205) |

**This starting network is not AND-OR's apprentice.** The summary's narrative presents expert iteration with
AND-OR as the expert, and that apprentice exists (`models/apprentice/gamma0.07_free`). But the sessions start from a
network that imitated its own play and an oracle teacher's moves. So the AND-OR model and its fitted γ play no part
in the student the teaching results use.

The AND-OR apprentice (`round1_epoch02.pt`, scale 0.30) is the candidate start of the newest runs (`aostart_*`,
archived). See `paths.py`.

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
| Scaled to the student's level | parameter | × 0.4 | fitted (§3) | `surrender_scale` |
| Giving up ends the puzzle unsolved, so the student learns nothing of its own from it (§6) | design | - | chosen | - |

## 6. What a puzzle teaches

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Only solved attempts produce labels from the student's own play | design | - | chosen | [`intuition_examples`](rushhour/student/labels.py:310), [line 332](rushhour/student/labels.py:332) |
| Hindsight over a graph of visited boards: two boards are joined when one legal move connects them (moves are reversible). The shortest route to the goal in that graph is computed by breadth-first search from the goal | implementation | - | chosen | [`shortcut_routes`](rushhour/student/labels.py:186) |
| Full hindsight: **every** visited board that can reach the goal is labelled (`labels "all"`), and any two boards are connected however far apart in time (`window -1`) | parameter | `all`, -1 | chosen among equal fits (§9) | [`solved_routes`](rushhour/student/labels.py:249) |
| Each labelled board gets, from its shortest route(s): the policy target, every first move tied for shortest; the value target, moves left on the route ÷ 10; and dest targets, 0.8^(k-1) for a car reaching a cell k moves later. A board seen in several routes takes the one with the fewest moves left | implementation | dest decay 0.8 | chosen | [`attempt_examples`](rushhour/student/labels.py:264), [`DEST_DECAY`](rushhour/student/network.py:21) |
| A shown move teaches that move on that board, solved or not, with no value or dest target. It is skipped if a solved route already labels the board | design | - | chosen | [`_shown_example`](rushhour/student/labels.py:300) |
| Everything is remembered (no encoding lapses), and nothing is learned from failed moves (no progress labels) | parameter | `encode 1`, `progress_weight 0` | chosen | `intuition_examples`, `progress_examples` |

Two things follow from these choices:
- **The value target is an upper bound.** A route through visited boards is only as short as the boards the student
  happened to see, so its length can exceed the true distance.
- **A wandering attempt leaves more labels than a direct one.** Every visited board is labelled, so a puzzle the
  student wandered through and solved teaches more than a short, direct solve (§8).

## 7. How it learns

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Online, in blocks: the student plays a block of puzzles with fixed weights, then trains on that block's examples only. No replay, no validation, no keeping an earlier, better version | design | block = 30 puzzles (one human trial) | follows §9 | [`learning_session`](rushhour/student/train.py:457), update at [line 563](rushhour/student/train.py:563) |
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
  shown puzzle leaves fewer boards to label (17 examples against 29-39; p6). That is why the assistance dilemma
  survives at ΔA = 1.

## 9. Time scale: the multiplier

| Choice | Kind | Value | Origin | Code |
|---|---|---|---|---|
| Each human trial stands for m simulated puzzles, because networks learn far less per example than people | parameter | m = 30, so a 72-trial session is 2,160 puzzles | fitted in p3 with own labels (loops cut); not refitted for full hindsight | [`sessions.multiplier`](rushhour/sessions.py:220) |
| Implemented as m **different** puzzles, all learned from in one update per trial, not as one puzzle replayed or extra passes | implementation | block = m ([line 184](rushhour/sessions.py:184)) | chosen | [`sessions.sample_puzzles`](rushhour/sessions.py:204) |

## 10. Randomness and reproducibility

| Choice | Kind | Code |
|---|---|---|
| Each student's play, labels and teacher share one random stream, seeded by the student. So once a teacher draws a random number (random p, or a tie between optimal moves), the student's later draws differ from the no-teacher run. Comparisons between teachers pair students by seed and puzzle order, not by identical randomness | implementation | [`sessions._student`](rushhour/sessions.py:340) (`seed + 7919 × (student + 1)`), `teacher.start(state, rng)` |
| With `agency_step` on, the update's shuffling and skipping have their own stream, apart from play | implementation | `update_rng` in `learning_session` |
| Python's hash seed is fixed (`PYTHONHASHSEED=0`, set by the command line). Set and dict order of boards decides the order of examples, and so the batches | implementation | [`sessions.py`](rushhour/sessions.py:71) |

## What is fitted to people, and how well

1. **Giving up, by moves / optimal**: people's hazard, used as measured (§5).
2. **Starting level**: the start network and the giving-up scale (0.4), fitted to people's first 10 trials (§3).
3. **Learning rate over trials 1-30**: hindsight × multiplier. The table is the misfit to people over three
   statistics, so lower is better and about 3 would be a good fit. It covers within-learner change in log moves /
   optimal per 10 trials on 6+10-move and on 13+15-move puzzles, plus moves / optimal in trials 1-10. Each cell
   is 8 students with no teacher, from the `people_*` runs, as in `summary.ipynb`.

   | hindsight | ×10 | ×20 | ×30 | ×60 | ×120 |
   |---|---|---|---|---|---|
   | own (loops cut) | 21.4 | 16.7 | 14.2 | 12.0 | **6.8** |
   | W=3 | 13.4 | 14.6 | **9.7** | 20.8 | 34.9 |
   | W=10 | 14.9 | **11.6** | 14.0 | 26.6 | 39.9 |
   | full | **8.8** | 12.2 | 13.2 | 31.2 | 42.4 |

   Every window fits about equally well at some multiplier: the longer the window, the smaller the multiplier it
   needs. So people's curves don't pin the window (p6). With 10-move puzzles alone as "short", p6 found misfits of
   3.7-5.9, with full hindsight best at ×10-20.

   The spec's pair, **full × 30, is not the best fit for full hindsight**. ×30 came from p3's calibration with own
   labels, and the summary's "fitted student" figure shows full × 20. Settling this is the first thing to defend or
   change.

**Everything else is chosen, not fitted:**
- the architecture;
- the pretraining curriculum and its oracle teacher;
- temperature, memory, the move cap, single attempts;
- learning only from success, the labels, the loss, the optimizer and its settings;
- agency.

People's plateau after trial 30 isn't matched by any setting (p6).

## Which saved runs use this student

`spec.differences` over the 34 session runs kept in `models/sessions` (2026-11-19; `and_or_gamma` is not a
session run):

| Runs | Matches? | How they differ |
|---|---|---|
| `teach_shortcut_agency{0.0,0.1,0.3,0.5,0.7,0.9,1}_even` (the summary's agency × teacher sweep; no teacher is in `0.0_even`) | **exactly** | only ΔA, the swept parameter |
| `teach_shortcut_agency{0,0.1,0.3}`, `teach_shortcut_agency0_confirm`, `pilot_all_agency0.1_skip` | student yes | uneven batches (before `even_batches`); the confirm run has 24 students and seed 1; the pilot has 6 puzzles per bin and tests every 12 trials |
| `people_*` (20 runs: the fit above) | calibration variants | window and multiplier as named; uneven batches; sessions and tests at people's lengths (6, 10, 13, 15), tests with giving up |
| `fitted_teachers` (the first teacher comparison) | earlier student | own labels instead of full hindsight; uneven batches; people's lengths |
| `calibrated_v2` | - | holds only the test puzzles `surrender.py` fits on |

The teaching sweep runs on **pool puzzles of 9-26 moves**, longer than any puzzle people played. The calibration
runs use people's lengths.

## What is not part of the student

- **AND-OR and γ** (`and_or.py`, `paper_model.py`) are the model of people's planning. They are used for the
  apprentice and for deliberation (p1, p8), not by the sessions' student (§3).
- **The teacher's model of the student** (`teachers.Bayesian4`: skill per distance, puzzle difficulty) is the
  teacher's belief, not this network.
- **The session protocol** (`spec.PROTOCOL`) is the experimental design: 8 students, sessions of 72 trials (18 per
  length bin, each trial 30 puzzles), tests of 50 puzzles per bin every 18 trials, untaught, with no giving up.

## Premises worth challenging

These are the places where the student is a choice rather than a fit.

**How it thinks**
1. **No deliberation.** People plan, and AND-OR fits their moves. In p8, network-ordered AND-OR matches people's
   efficiency, yet the student is pure intuition.
2. **One network for every puzzle, keyed by cell.** What it learns transfers by board pattern. Whether people
   transfer that way is untested.
3. **Heads trained but unused.** Dest and value shape the shared features without ever acting.

**Where it starts**

4. **The starting network's origin.** It was self-trained with an oracle teacher, not trained on a model of human
   planning (§3). Its learning rate also drops 10× at the session's start.

**What it learns from**

5. **Learning only from success.** Unsolved and abandoned attempts teach nothing of the student's own.
6. **Labels from perfect hindsight:** exact shortest routes through every board seen. This is also what makes
   wandering pay (§6, §8).

**How it learns**

7. **The multiplier as a data-efficiency patch.** 30 network puzzles for each human trial, and inconsistent with
   the hindsight fit (§9).
8. **Online learning by Adam with BatchNorm.** Its forgetting and drift are those of the optimizer, not anything
   fitted (§7).
9. **Agency as a weight on shown moves, implemented by skipping updates.** It is unfitted, and the dilemma depends
   on it together with the exploration cost (§8).

**What it is tested on**

10. **The puzzle source.** Pool puzzles of 9-26 moves for teaching, against people's 70 layouts of 6-15.
