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

## Visualizing a solve (`visualize.py`)

Like `scripts/lib.py`'s `visualize()`, it writes `move_000.png`,
`move_001.png`, ... (the board after each push), a `trajectory.png` (oracle
distance per push, with restarts and pass boundaries marked), and a
`solve.gif`. In each frame:

- The arrow is the push just played, coloured by its role in the AND-OR tree:
  blue for a delivery, orange for clearing a blocker (the caption gives its
  depth), red for a lapse, purple for a random fallback.
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
