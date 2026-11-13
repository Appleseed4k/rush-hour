# Sokoban

A port of the Rush Hour planning models to Sokoban. So far: the environment,
a full state-space labeller (the counterpart of `scripts/lib.py`'s
`multi_bfs`), labelled level sets, and an AND-OR backward planner (the
counterpart of `scripts/and_or.py`) with an evaluation against optimal
solutions.

| File | What it is |
|---|---|
| `level.py` | `Level`: parsing (XSB), push-level moves, player reachability, dead cells, rendering |
| `label.py` | `StateSpace` (every reachable position, push edges, distance to goal) and the CLI that writes dataset files |
| `and_or.py` | The AND-OR planner: `solve_pass()` (one pass) and `play()` (passes until solved, restarting on deadlock) |
| `evaluate.py` | Planner variants against optimal push counts, binned by distance |
| `teach.py` | Teachers that show the student the optimal push (the counterpart of `scripts/teach.py`): the `Teacher` interface, `Random(p)`, `Guard(q)` (shows the optimal push when the next push would deadlock) `Undo(q)` (takes that push back; the student chooses again), `Restart(k)` (only forces restarts: on an oracle deadlock, and past k x optimal) and `Bayes(theta, act)` (a fitted model of the student; undoes, shows or restarts by its forecast of the chance of solving: a threshold, the planned value of a restart, or of an undo against a spending target) |
| `teach_experiment.py` | Teachers against the AND-OR student: solve rate, pushes and restarts against pushes shown |
| `student.py` | A student that learns online (policy network over pushes, shortcut or own labels): the counterpart of `scripts/rl.py`'s intuition student and `learning_session` |
| `sessions.py` | Learning sessions with teachers, tested on held-out Boxoban levels: the counterpart of `scripts/session_experiment.py` (`pretrain`, `run`, `summary`) |
| `visualize.py` | Frame-by-frame PNGs, a trajectory plot and a GIF of the solver playing a level, written to `visualization/<level>/` (gitignored) |
| `test_level.py` | Checks `StateSpace` against a naive 0-1 BFS solver, plus solution replay, distance consistency, deadlock-test soundness and planner replay |
| `data/*.txt` | Labelled levels (format below) |
| `data/raw/` | Downloaded level sets (gitignored, see "Getting the raw levels") |

## Conventions

- **A move is a push.** The player walks for free to the cell behind a box and
  pushes it one cell. Distances count pushes, not player steps. A Rush Hour
  slide counts as one move, and this is the closest counterpart.
- **States** are `(boxes, player)`: a bitmask of box cells (row-major,
  `i * width + j`) and the player's cell. `Level.key()` merges states that
  differ only in where the player stands inside the same reachable region.
- **Pushes cannot be undone**, so unlike Rush Hour some states are
  deadlocks. `StateSpace.dist` is `-1` for those. A box on a dead cell
  (`Level.dead`: a box there can never reach a goal, even with no other boxes)
  is always a deadlock. By default the labeller doesn't enumerate such states,
  which leaves every distance unchanged.

## Dataset format

One level per line, sorted by decreasing distance like `data/rush_nw.txt`:

```
<pushes> <board> <n_states> <n_solvable> <source>
```

- `pushes`: optimal pushes from the start (`-1` = unsolvable).
- `board`: the XSB rows joined with `/`, with floor written as `-`. Read it
  back with `Level.from_line(board)`.
- `n_states`: positions reachable from the start, not counting boxes on dead
  cells.
- `n_solvable`: how many of those can still be solved.
- `source`: `<file>:<level name>`.

```python
from level import Level
from label import StateSpace

pushes, board, *_ = open("data/microban1.txt").readline().split(" ", 4)
level = Level.from_line(board)
space = StateSpace(level)          # oracle distances for every reachable position
space.solution()                   # an optimal list of (box_cell, direction) pushes
```

## Getting the raw levels

From this folder:

```bash
git clone --depth 1 https://github.com/google-deepmind/boxoban-levels.git data/raw/boxoban-levels
```

```bash
mkdir -p data/raw/microban && curl -L https://raw.githubusercontent.com/nikvoronin/funsokoban/master/levels/sb_microban_1.xsb -o data/raw/microban/microban1.xsb
```

- **Boxoban** (Guez et al., 2018): 10x10, always 4 boxes. The sets are
  unfiltered (900k train / 100k valid / 1k test), medium (450k / 50k) and
  hard (3,332).
- **Microban I** (David W. Skinner, 2000): 155 hand-made levels with 1 to many
  boxes. Free to distribute with credit.

## Labelling

```bash
python label.py data/raw/microban/microban1.xsb -o data/microban1.txt --processes 12
```

```bash
python label.py "data/raw/boxoban-levels/hard/*.txt" -o data/boxoban_hard.txt --processes 12
```

```bash
python test_level.py --microban 40 --boxoban 20
```

Levels whose state space is larger than `--max-states` (default 5M) are skipped
and reported. Microban I is missing levels 93, 144, 145, 146 and 153 (over 5M
states each). Boxes already on goals and sealed off from the player (the
decoration in 155 "The Dungeon") are treated as walls.

## The AND-OR planner (`and_or.py`)

It follows the same goal-regression scheme as `scripts/and_or.py`. The
module docstring has the details.

- **Root**: an AND over deliveries. The planner picks a (loose box, goal)
  pair, delivers it, and recurses. Pair choices are backtracked, so a bad fill
  order is undone during planning rather than played.
- **Deliver**: moves one box along a shortest empty-board route, as straight
  segments, with each segment a `PushNode`.
- **PushNode (AND)**: push box *b* *k* cells in direction *d*. Its blockers
  are
  - *path blockers*: boxes on the swept cells;
  - *access blockers* (`access=True`): boxes on the walk to the stand cell
    *b − d* that crosses the fewest boxes.

  Blockers are resolved one at a time in every order, as in Rush Hour.
- **OrNode (OR)**: a box must leave a set of cells. The candidates are its
  straight pushes that end on a live cell. With probability γ the whole pass
  lapses into one random push.
- **Protection**: boxes higher up the chain are always protected. Boxes
  already on goals are protected only with `protect_placed=True`. Without it, a
  loose box may also aim at an occupied goal, and the box already there
  becomes a path blocker.
- **Deadlocks**: candidates never end on dead cells. With `freeze=True`, a
  push that freezes a box off a goal is rejected. That test is sound: no
  solvable state is ever flagged, checked against the oracle in
  `test_level.py`. Lapses and random fallbacks can still deadlock the board,
  so `play()` restarts the level when `Level.deadlocked` fires, when no push is
  legal, or after `patience` (20) passes in a row with no plan.
- **Playing a pass**: `play()` plays a pass's pushes one at a time and stops
  as soon as the level is solved, even partway through a pass. It used to
  apply whole passes, which could push a box back off a solved board. That
  changed one of 60 check runs, Microban 18 at γ = 0.05, from 43 pushes to
  39.
- **Failed deliveries are remembered within a pass.** `deliver` backtracks
  over every way of cutting a route into straight segments. On a long route
  that can't be completed, that used to mean tens of thousands of PushNodes,
  none of them counted against the OrNode budget. Microban 155, a 175-push
  route, took 2.6 s per pass and over an hour per experiment. A delivery that
  fails from a given state fails in any search order, so each one is now
  tried once per pass. That pass now takes milliseconds, and the slowest
  sampled attempts are about 2.5x faster. Plans are the same, but the
  planner makes fewer random draws, so seeded runs (including the table
  below, made before this change) now come out slightly differently.
- **`max_visits`** caps one `play()` attempt's total search (OrNodes plus
  PushNodes). It stops unsolved once the cap is reached, so no attempt can
  run for hours. `teach_experiment.py` uses 300,000 nodes, about 4 s.
  Attempts that solved used at most about 5,400.
- **When a pass finds no full plan**, it delivers any one box it can
  ("partial"). Failing that, it plays one random push ("random").

### First results (`python evaluate.py ... --n 200 --gammas 0,0.05`)

Settings: 200 sampled levels per set (all 150 for Microban), one seed,
`max_pushes=400`. **plan** is the share of levels where a single lapse-free
pass returns a whole solution. **solved** is the share `play()` solves within
400 pushes.

| set | variant | plan (γ=0) | solved (γ=0) | solved (γ=0.05) | plan length / optimal |
|---|---|---|---|---|---|
| Boxoban unfiltered test | path only | 0.81 | 0.88 | 0.85 | 1.72 |
| | + access | 0.81 | 0.94 | 0.74 | 1.77 |
| | + access + freeze | 0.86 | 0.94 | 0.88 | 1.86 |
| | + access + freeze + protect | 0.90 | 0.94 | 0.94 | 1.29 |
| Boxoban hard | path only | 0.14 | 0.31 | 0.18 | 1.14 |
| | + access | 0.23 | 0.45 | 0.18 | 1.27 |
| | + access + freeze | 0.29 | 0.49 | 0.24 | 1.33 |
| | + access + freeze + protect | 0.22 | 0.40 | 0.38 | 1.16 |
| Microban I | path only | 0.07 | 0.29 | 0.24 | 1.94 |
| | + access | 0.18 | 0.36 | 0.29 | 1.63 |
| | + access + freeze | 0.17 | 0.38 | 0.30 | 1.98 |
| | + access + freeze + protect | 0.13 | 0.31 | 0.32 | 1.14 |

What these show so far:

- **Access blockers matter.** They raise the single-pass plan rate on
  Boxoban hard from 0.14 to 0.23, and on Microban from 0.07 to 0.18. Freeze
  pruning takes Boxoban hard further, to 0.29.
- **γ isn't on the Rush Hour scale.** A Sokoban pass visits far more OrNodes,
  most of them dead ends, so at γ = 0.05 nearly every pass on hard levels
  lapses: lapses are 92% of passes. γ needs recalibrating for this game,
  for example against human data or by matching per-pass lapse rates.
- **Protecting placed boxes** trades a little lapse-free planning power for
  robustness: with lapses, it keeps progress from being undone.
- **Structural gaps.** These are the gaps found by reading levels that never
  get a plan (Microban 1, 8, 13, 14 and 19 get none in 5 of 5 tries).
  1. *Out-of-order preconditions*: a later segment's access blocker can only
     be cleared before an earlier segment is pushed. This is a
     Sussman-anomaly-style interaction between subgoals (Microban 1).
  2. *Step aside and return*: the mover has to move out of the way and come
     back. Protecting movers forbids this, just as `PROTECT_MOVERS` does in
     Rush Hour (Microban 14).
  3. *Non-shortest routes*: a box has to detour off its shortest empty-board
     route, or be parked somewhere first (likely in Microban 19, where a box
     goes into the side pocket and back out).

  A fourth gap, goal shuffles (a loose box can only reach an occupied goal),
  was fixed: loose boxes may now aim at occupied goals (Microban 2).

  On Microban at γ = 0, 76% of passes find nothing and fall back to a random
  push. That's why the restart counts are high: about 40 restarts per run.

## Teaching (`teach.py`, `teach_experiment.py`)

`teach.py` mirrors `scripts/teach.py`. A teacher knows the level's oracle
distances and, when it chooses to, shows an optimal push (one that brings
the board a push closer to solved). The interface is the same:
`start(level, rng)`, `advise(state, step, proposal)`, `observe(...)` and
`end_attempt(result)`. It adds `restart(state)`, because a Sokoban student
restarts. `Random(p)` shows the push on each turn with probability p.

- **The student** is `and_or.play(level, teacher=...)`. Each turn the teacher
  may show a push instead of the student's. After it does, the student drops
  the rest of its pass and plans again from the new board.
- **Unlike Rush Hour, a board can be past saving.** On a deadlocked board no
  push is optimal, so the teacher shows nothing until the student restarts.
- **The student doesn't learn yet** (there's no policy network), so the
  experiment measures only the help within an attempt, not learning across
  rounds as `scripts/teach_experiment.py` does.

```bash
python teach_experiment.py data/microban1.txt data/boxoban_hard.txt --n 150 --seeds 3 --processes 9
```

This took 12 minutes. Every teacher plays the same 148 Microban and 150
Boxoban hard levels with the same 3 seeds; Microban 93, 144, 145, 146 and 153
are over the 1M-state oracle limit. Below, **solved** is the share of
attempts solved (± about 0.02) and **shown** is pushes shown per attempt.

| set | γ | teacher | solved | shown | share of pushes shown | restarts |
|---|---|---|---|---|---|---|
| Boxoban hard | 0 | none | 0.36 | 0 | 0 | 25.4 |
| | | random 0.05 | 0.53 | 5.3 | 0.02 | 21.0 |
| | | random 0.1 | 0.62 | 9.2 | 0.05 | 17.8 |
| | | random 0.2 | 0.73 | 17.0 | 0.10 | 14.4 |
| | | random 0.5 | 0.95 | 25.1 | 0.32 | 5.0 |
| | | random 1 | 1.00 | 17.4 | 1.00 | 0 |
| | 0.05 | none | 0.43 | 0 | 0 | 29.4 |
| | | random 0.1 | 0.54 | 12.3 | 0.05 | 25.6 |
| | | random 0.5 | 0.92 | 30.2 | 0.33 | 6.5 |
| Microban I | 0 | none | 0.31 | 0 | 0 | 48.7 |
| | | random 0.05 | 0.40 | 8.3 | 0.03 | 40.4 |
| | | random 0.1 | 0.42 | 16.3 | 0.06 | 36.7 |
| | | random 0.2 | 0.50 | 29.8 | 0.13 | 28.9 |
| | | random 0.5 | 0.69 | 58.8 | 0.37 | 13.1 |
| | 0.05 | none | 0.31 | 0 | 0 | 53.1 |
| | | random 0.1 | 0.41 | 17.7 | 0.06 | 43.3 |
| | | random 0.5 | 0.68 | 69.8 | 0.38 | 15.9 |

The run prints the full table, with every teacher and solve rates by optimal
length. Per-attempt rows go to `data/eval_teach.csv` (gitignored); reprint
the table from it with `python teach_experiment.py --summarize`.

- **A few shown pushes go a long way on Boxoban hard.** Showing about 5 pushes
  per attempt, 2% of those played, raises the solve rate from 0.36 to 0.53.
  Showing a third of the pushes solves 95%.
- **Microban gains less per push shown.** Its long levels (40+ optimal pushes)
  stay mostly unsolved even at p = 0.5. The student's structural gaps
  (above) aren't fixed by one push here and there.
- **Lapses help the untaught student on Boxoban hard** (0.43 at γ = 0.05
  against 0.36 at γ = 0). With γ = 0 the student repeats the same failing plan;
  a lapse varies it. With a teacher, γ = 0 is better, because a shown push
  already provides that variety.
- **Caveat: the search cap.** On Microban at γ = 0, 6% of attempts hit
  `--max-visits` (300,000 nodes) for every teacher with p < 1. On three levels
  (78, 116 and 126) every uncapped attempt solved, so some capped attempts
  would probably have solved too: the γ = 0 Microban solve rates are slightly
  low. No other setting hit the cap.

## A student that learns (`student.py`, `sessions.py`)

This is the counterpart of Rush Hour's session experiments (`scripts/rl.py`'s intuition student and
`learning_session`, and `scripts/session_experiment.py`). The analysis is in
**`notebooks/s1_teaching.ipynb`**, with figures in `notebooks/figures/s1_teaching/`.

- **Student**: a policy network over pushes. Its input is walls, goals, boxes and the reachable region; it outputs a
  push per cell and direction, plus a value head. It samples at temperature 1 and restarts when its quick deadlock
  test fires.
- **Learning**: it learns online from each block of 25 levels, using shortcut labels (`--labels all`) or its own
  winning path (`own`), plus the pushes it was shown.
- **Sessions**: 8 students per teacher play 2,000 Boxoban medium levels (8–25 pushes). Every 500 levels they are
  tested untaught on 200 held-out levels. A test attempt scores pushes / optimal, an unsolved one counted at its cap.

```bash
python sessions.py pretrain --teacher random:0.5
```

```bash
python sessions.py run shortcut_all --init models/pretrain/puzzle05400.pt --labels all
```

```bash
python sessions.py summary shortcut_all
```

Results (start: held-out solve rate 0.13). **Learning** is the start score minus the end score.

| teacher | interventions per push | learning, shortcut labels | learning, own labels | shortcut + negative labels | shortcut, batches of 64 | batches of 64, optimal labels only |
|---|---|---|---|---|---|---|
| none | 0% | **+0.34** | **+0.34** | −0.08 | +0.33 | **+0.45** |
| random 0.1 | 6% | +0.10 | +0.07 | −0.01 | +0.13 | |
| random 0.3 | 19% | +0.08 | +0.09 | +0.03 | +0.15 | |
| random 1 | 100% | +0.23 | +0.23 | | +0.29 | |
| guard 0.3 | 3.7% | +0.15 | | +0.00 | +0.21 | |
| guard 1 | 18% | +0.22 | | | +0.31 | |
| undo 0.3 | 3.9% | **+0.35** | | | **+0.40** | **+0.46** |
| undo 1 | 41% | +0.25 | | | **+0.39** | **+0.45** |
| restart dead | 12 restarts/level | +0.34 | | | | |
| restart 1.5 | 13 restarts/level | +0.34 | | | | |
| undo + restart 1.5 | | +0.29 | | | | |
| bayes-undo 0.1 | 18% | | | | +0.26 | +0.36 |
| bayes-undo 0.5 | 31% | | | | +0.33 | |
| restart dead (batches of 64) | 12 restarts/level | | | | +0.37 | |
| bayes-restart 0 | 13 restarts/level | | | | **+0.42** | |
| bayes-undo-value 0.04 | 3.9% | | | | +0.38 | |
| bayes-undo-value 0.18 | 18% | | | | +0.36 | |

- **Help costs learning at every level, and partial help costs most** (with the default batches of up to 256). No
  teacher beats every teacher, and full help beats both partial teachers, for 8 of 8 students in both runs. That makes
  learning against help a U, not Rush Hour's inverted U.
- **A deadlock guard** (`guard:q`, `teach.Guard`) steps in only when the student's next push would make the level
  unsolvable, showing the optimal push instead (with probability q). It is the most efficient help so far: guard 1
  ties full help at a fifth of the pushes shown and beats random 0.3 at the same spend (8/8). It still learns less
  than no teacher (8/8).
- **The label setting doesn't matter.** Pushes can't be undone, and the student restarts on deadlocks, so the boards it
  visits rarely join up into shortcuts. Shortcut labels add about 3 examples per solved level over the winning path.
- **Taking the push back removes the cost of help** (`undo:q`, `teach.Undo`). The undo teacher catches the same
  pushes as the guard but shows nothing: it takes the push back and the student chooses again. Undo 0.3 steps in as
  often as guard 0.3 and raises session solving as much, but learns +0.35 against +0.15 (8/8), tying no teacher
  (6/8, −0.01 ± 0.02). So the guard's cost was the shown push steering the labelled route. But undo 0.3 solves 41%
  more session levels than no teacher and learns no more: on these levels the student doesn't seem short of examples.
  Undo 1 (+0.25) takes back 14 pushes per level, and with no restarts its labels span the whole wandering route.
- **Forced restarts tie no teacher** (`restart:k`, `teach.Restart`; `undo+restart:k`). Restarting on oracle deadlocks
  lifts session solving from 0.22 to 0.30 and learns +0.34, as does also restarting past 1.5 x optimal. Adding the
  stray restart to undo 1 shortens its labelled wins and lifts it from +0.25 to +0.29 (7/8), still below no teacher.
- **A fixed step budget, partly.** Each update is 3 passes in batches of up to 256, so blocks of 100 or 250 examples
  get the same 3 Adam steps, and measured mid-session they move the weights the same distance. Equal batches
  (`--even-batches 1`, Rush Hour's fix; run `shortcut_even`) keep the step count and change little (guard 0.3
  +0.16, undo 1 +0.28). Batches of 64 (`--batch-size 64 --even-batches 1`, run `shortcut_b64`) make steps grow with
  examples: no teacher gets twice the steps (6.4 per block) and learns the same (+0.33), while teachers with bigger
  blocks get 3-3.5 times the steps and all learn more, most of all undo 1 (+0.25 to +0.39, 8/8) and guard 1 (+0.22 to
  +0.31, 8/8). Undo 0.3 (+0.40, 7/8) and undo 1 (+0.39, 6/8) are the first teachers above no teacher (end score
  −0.07 ± 0.03 and −0.06 ± 0.03). Shown pushes still cost: random 0.1 (+0.13), random 0.3 (+0.15) and guard 0.3
  (+0.21) stay below no teacher (8/8).
- **A Bayesian undo teacher** (`bayes-undo:θ`, `teach.Bayes`, the port of Rush Hour's `Bayesian3`) fits a model of
  the student and takes back deadlocking pushes only while it forecasts under θ chance of solving untaught. It does
  worse than undoing at random (batches of 64): θ 0.1 (18% of pushes) learns +0.26, below no teacher and both undo
  teachers; θ 0.5 (31%) learns +0.33, tying no teacher and below undo 1 (39%, +0.39). It sends help to the hardest
  levels and the ends of losing attempts. Its students end worse than no teacher's on the easiest held-out levels
  (0.33 against 0.39) and no better on the hardest. Its rescued wins are long, but that isn't the cause: shortcut
  labels keep 87% of a long win's labels optimal, and undo 1's wins are as long. `bayes-show:θ` shows the optimal push instead (not yet run).
- **Only optimal labels** (`--optimal-labels 1`, run `shortcut_b64_opt`; `student.optimal_only`). Keeping only the
  shortcut labels the oracle calls optimal drops about 7% of labelled boards and lifts every teacher tested (no teacher
  +0.33 to +0.45, 8/8). The undo teachers' edge over no teacher disappears (all about +0.45), and bayes-undo 0.1
  stays about 0.1 below, so label quality doesn't explain it. A lead: help shifts the examples toward the hardest
  levels, and the easiest levels slip on held-out tests.
- **Rebalancing by difficulty** (`--balance 0.30 0.28 0.23 0.18`, run `shortcut_b64_bal`) weights each update so
  every difficulty bin keeps no teacher's share of the learning. It lifts bayes-undo 0.1 from +0.26 to +0.31 (7/8),
  mostly on the easiest levels, and barely moves the others; bayes-undo 0.1 still trails no teacher by 0.06.
- **Other rules on the Bayesian model work.** `bayes-restart:0` sends the student back to the start whenever its
  plan says a fresh attempt raises the chance of solving (deadlocked boards the quick test missed, boards too far
  from solved). It learns +0.42 with batches of 64, the best teacher there, above no teacher for 8/8 students and
  above plain restart on deadlock (+0.37, 6/8). `bayes-undo-value:b` (Rush Hour's original Bayesian rule) undoes a
  deadlocking push only when that raises the chance of solving by more than a cost tuned to spend b of the pushes:
  at 18% it learns +0.36 against the threshold rule's +0.26 (8/8), and at 3.9% +0.38, tying undo 0.3. The
  threshold rule's flaw was acting on attempts already lost.
- **Negative labels for dead ends hurt** (`--negative-weight 1`, run `shortcut_neg`; `student.fatal_pushes`). The
  push before each detected deadlock is labelled "not this push" by unlikelihood. Learning drops in every condition,
  and no teacher becomes the worst. 72% of blamed pushes start from a board that was already deadlocked (the quick
  test noticed late), and negatives outnumber positive examples about 2 to 1. The policy stays flat instead of
  sharpening, so the student solves fewer session levels and gets fewer positive labels. Untested: oracle blame (the
  push that actually created the deadlock) and a smaller weight.
- **Why partial help hurts is open.** The notebook lists the candidates. `--shown-unsolved 0` tests one of them:
  that pushes shown in failed attempts pull the student off what works for it.

Settings that mattered:

- **lr and BatchNorm.** At lr 1e-3 an untaught student got worse. Sessions use 1e-4, with BatchNorm frozen during
  updates (`--freeze-norm 1`).
- **Pretraining** with a teacher at p = 0.2 never got going.
- **Not comparable across starts.** The runs share one starting network, so the tight standard errors say nothing
  about variation across starts.

## Visualizing a solve (`visualize.py`)

Like `scripts/lib.py`'s `visualize()`, it writes `move_000.png`,
`move_001.png`, ... (the board after each push), a `trajectory.png` (oracle
distance per push, with restarts and pass boundaries marked), and a
`solve.gif`. In each frame:

- The arrow is the push just played, coloured by its role in the AND-OR tree:
  blue for a delivery, orange for clearing a blocker (the caption gives its
  depth), red for a lapse, purple for a random fallback. Green is a push the
  teacher showed (`--teacher random:0.2`); those pushes are also marked on
  the trajectory plot.
- The box being delivered has a blue outline, and its goal a dashed blue ring.
- Boxes are lettered so they can be followed from frame to frame.

```bash
python visualize.py
```

```bash
python visualize.py --dataset data/microban1.txt --gamma 0.05 --want-restart --min-pushes 8 --max-pushes 20 --min-depth 1
```

```bash
python visualize.py --source hard/000:12
```

The first command picks a Boxoban hard level the solver finishes without a
restart and with a blocker chain at least two deep (hard/002:228: optimal 20,
one 36-push plan). The second shows lapses and restarts (Microban 119). The
third renders a given level. Push roles come from `solve_pass()`
(`Pass.roles`, see `and_or.push_roles`).
