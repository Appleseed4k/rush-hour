# Rush Hour: teaching a simulated student

Can a teacher's help make a learner better on their own, and when does more help start to hurt (the
*assistance dilemma*)? This repo builds a simulated Rush Hour student that solves and learns roughly as people
do, fits it to people's play (Olieslagers et al. 2023, 42 participants), and compares teachers that decide move
by move whether to show the optimal move.

**Start with [`notebooks/summary.ipynb`](notebooks/summary.ipynb)**: the project's progress in one place.
**[`MODEL.md`](MODEL.md)** is the simulated student's model card: every assumption, its value, where the value
came from, and the code behind it.

## Layout

```
rushhour/              the library
  board.py             puzzles, moves, breadth-first search, the puzzle pool
  and_or.py            the AND-OR planner (backward subgoal search with lapses γ), as this repo solves with it
  paper_model.py       the paper's model as published (a move per tree), for fitting γ to people's moves
  student/             the simulated student
    network.py         PolicyAgent: policy / dest / value heads; its moves are intuition, its ordering
                       of AND-OR's options is the heuristic deliberation uses
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
  summary.ipynb        the narrative and its figures
  p8_deliberation.ipynb  network-ordered AND-OR (deliberation only) against people
  forward_planning.ipynb  a forward planner for the student: novelty and relevance rollouts, the value head
  skill_filter.html    the offline test behind bayes4: carrying skill across puzzles by a filter
                       (experiments/skill_tracking.py)
  archive/             p3-p7 lab notebooks, frozen with their outputs (see archive/README.md)
web/                   the playable Rush Hour page with teachers, for collecting human play
data/rush_nw.txt       every puzzle with its optimal length (the analysis pool)
23vrb-osfstorage-archive/  the paper's data (not in git)
models/                results and networks (not in git; see MODELS.md)
```

## Progress so far

| Phase | What | Where |
|---|---|---|
| p1 | Port of the AND-OR model; a policy network trained as its apprentice by expert iteration | summary; tag `pre-cleanup` |
| p2 | Teachers during expert iteration (offline teaching) | tag `pre-cleanup` |
| p3 | Within-session learning fitted to people: giving up, starting checkpoint, AND-OR's γ | summary, `archive/p3` |
| p4 | The assistance dilemma: agency ΔA (how much a shown move teaches) and random-help sweeps | summary, `archive/p4` |
| p5 | The simplest good teacher: stuck-Bayes ablations down to `bayes3` (help when the chance of solving alone < θ = 0.45) | summary, `archive/p5` |
| p6 | Hindsight labels (window × multiplier fit to people); agency sweep with equal batches | summary, `archive/p6` |
| p7 | Toward a POMDP teacher: a fitted learning model; schedules and an upper bound put bayes3 near optimal in-model; a skill filter as good as bayes3's refit gives bayes4, the default teacher | `archive/p7`, `skill_filter.html` |
| p8 | Deliberation vs people: network-ordered AND-OR matches people's efficiency, solves fewer long puzzles | `p8_deliberation` |
| - | The student written down (`MODEL.md`, `student/spec.py`); its start switched to AND-OR's apprentice, which fits people's start (misfit 0.2) and learning rate (2.8 at ×10); hindsight, seeds and θ checked (phase 2), and the summary moved to it | `MODEL.md`, summary |
| - | Forward planning: rollouts judged by attempt-level novelty halve moves/optimal for every network; the dest head's relevance adds nothing, the value head helps only by breaking novelty's ties, and experience enters through the policy's proposals | `forward_planning` |

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
