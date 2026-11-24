# Rush Hour: teaching a simulated student

Can a teacher's help make a learner better on their own, and when does more help start to hurt (the
*assistance dilemma*)? This repo builds a simulated Rush Hour student that solves and learns roughly as people
do, fits it to people's play (Olieslagers et al. 2023, 42 participants), and compares teachers that decide move
by move whether to show the optimal move.

**Start with the summaries**: [`notebooks/p1_summary.ipynb`](notebooks/p1_summary.ipynb) (phase 1: a student fitted to
people, and teachers for it) and [`notebooks/p2_summary.ipynb`](notebooks/p2_summary.ipynb) (phase 2: how the student
plans, and how human-like it is).
**[`MODEL.md`](MODEL.md)** is the simulated student's model card: every assumption, its value, where the value
came from, and the code behind it.

## Layout

```
rushhour/              the library
  board.py             puzzles, moves, breadth-first search, the puzzle pool
  and_or.py            the AND-OR planner (backward subgoal search with lapses γ), as this repo solves with it
  paper_model.py       the paper's model as published (a move per tree), for fitting γ to people's moves
  student/             the simulated student
    network.py         PolicyAgent: a policy network (its moves are intuition; its scores also order
                       AND-OR's options) and a separate value network (moves to go, for flat planning)
    solve.py           solving by intuition, by deliberation, or arbitrating; evaluation over many puzzles
    labels.py          what an attempt teaches: own solved play, hindsight shortcuts, shown moves
    train.py           expert iteration (pretraining) and online learning in a session
    spec.py            the fitted student's settings (MODEL.md), and a check of a saved run against them
  teachers.py          random and bayes4 (the default: bayes3's rule on a filtered model of the student), the
                       teacher-spec registry, and the names of retired teachers so their runs still load
  sessions.py          simulated sessions: run (CLI), save, load
  run_queue.py         queues of session runs in the background
  human_data.py        the paper's participants: moves, trials, giving-up hazard
  play_data.py         games recorded on the play page (web/)
  surrender.py         fitting people's giving up and the starting checkpoint
  analysis.py          measures the notebooks share
  paths.py             where data and results live
experiments/           scripts that write the caches the notebooks read (models/summary, models/sessions)
notebooks/
  p1_summary.ipynb     phase 1's narrative and its figures: the student fitted to people, teachers, agency
  p1.7_skill_filter.html  the offline test behind bayes4: carrying skill across puzzles by a filter
                       (experiments/skill_tracking.py)
  archive/             p1.3-p1.7 lab notebooks, frozen with their outputs (see archive/README.md)
  p2_summary.ipynb     phase 2's narrative: from AND-OR and forward planners to the forward and two-mode students
  p2.1_deliberation.ipynb  network-ordered AND-OR (deliberation only) against people
  p2.2_forward_planning.ipynb  forward planning, forward + backward, planners in the session, calibration to
                       people's start, people's move times, the mixture of modes, the forward and two-mode students and how their learning compares with people's
  p2.3_human_likeness.ipynb  people against the models move by move: revisits, car roles, slides, timing;
                       the soft memory fitted to people's revisits
  p2.4_rules.ipynb     interpretable move rules over AND-OR's blocking chain (after Butler et al. 2017): what
                       the student and people learn, in words (experiments/rule_mining.py)
  p2.5_value_network.ipynb  one network or two: the policy and value as separate networks
                       (experiments/separate_value.py)
web/                   the playable Rush Hour page with teachers, for collecting human play
visualization/          figures by phase; p2_two_mode/two_mode_moves.html steps through how the two-mode student
                       picks each move of one attempt (experiments/two_mode_animation.py)
data/rush_nw.txt       every puzzle with its optimal length (the analysis pool)
23vrb-osfstorage-archive/  the paper's data (not in git)
models/                results and networks (not in git; see MODELS.md)
```

## Progress so far

Phase 1: a student fitted to people, and teachers for it (`p1_summary`).

| Step | What | Where |
|---|---|---|
| p1.1 | Port of the AND-OR model; a policy network trained as its apprentice by expert iteration | p1_summary; tag `pre-cleanup` |
| p1.2 | Teachers during expert iteration (offline teaching) | tag `pre-cleanup` |
| p1.3 | Within-session learning fitted to people: giving up, starting checkpoint, AND-OR's γ | p1_summary, `archive/p1.3` |
| p1.4 | The assistance dilemma: agency ΔA (how much a shown move teaches) and random-help sweeps | p1_summary, `archive/p1.4` |
| p1.5 | The simplest good teacher: stuck-Bayes ablations down to `bayes3` (help when the chance of solving alone < θ = 0.45) | p1_summary, `archive/p1.5` |
| p1.6 | Hindsight labels (window × multiplier fit to people); agency sweep with equal batches | p1_summary, `archive/p1.6` |
| p1.7 | Toward a POMDP teacher: a fitted learning model; schedules and an upper bound put bayes3 near optimal in-model; a skill filter as good as bayes3's refit gives bayes4, the default teacher | `archive/p1.7`, `p1.7_skill_filter.html` |
| - | The student written down (`MODEL.md`, `student/spec.py`); its start switched to AND-OR's apprentice, which fits people's start (misfit 0.2) and learning rate (2.8 at ×10); hindsight, seeds and θ checked, and the summary moved to it | `MODEL.md`, p1_summary |

Phase 2: how the student plans, and how human-like it is (`p2_summary`).

| Step | What | Where |
|---|---|---|
| p2.1 | Deliberation vs people: network-ordered AND-OR matches people's efficiency, solves fewer long puzzles | `p2.1_deliberation` |
| p2.2 | Forward planning (novelty halves moves / optimal; relevance adds nothing); forward + backward together; no planner that deliberates every move starts where people do; people's moves are mostly intuition, with flat planning growing with practice and AND-OR on slow moves and near the goal; the two-mode student (intuition, flat planning and single AND-OR moves), which starts where people do and gains as much as they do, and the forward student (without AND-OR), which stalls | `p2.2_forward_planning` |
| p2.3 | People against the models move by move; the hard 10-board memory replaced by a soft one fitted to people's revisits | `p2.3_human_likeness` |
| p2.5 | One network or two: the value loss doesn't help the policy learn, and a separate value network judges as well; the student is now a policy network and a value network, which learn as one network did | `p2.5_value_network` |
| p2.4 | Rules in words: a short list of "clear a car so the blocking chain shrinks" rules is near-sound close to the goal; people and the student come to follow it equally, except people's near-goal "open red's path" | `p2.4_rules` |
| - | The dest head removed: AND-OR ordered by the policy head plays as well, and an apprentice trained without dest targets starts where people do and learns as the dest-head one did; it is now the start of every session (tag `pre-dest-removal` has the code before) | p2_summary section 7, `MODEL.md` |

Earlier code (p1-p2 scripts, retired teachers including bayes, stuck-bayes1 and bayes3, the sokoban port before it moved to its own repo) is at tag
`pre-cleanup`.

## Running things

Set up once (see `requirements.txt` for the CUDA build of torch on Windows):

    python -m venv .venv && .venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip

Everything runs from the repository root:

    python -m rushhour.sessions my_run --fitted --shown-weight 0.1 --teachers none bayes4   # the fitted student
    python -m rushhour.run_queue watch my_queue
    python experiments/and_or_network.py

Notebooks add the repository root to `sys.path` themselves. Session runs fix Python's hash seed (`PYTHONHASHSEED=0`,
set automatically when run from the command line), so a run with the same settings replays exactly.
