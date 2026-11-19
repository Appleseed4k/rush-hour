# What is in `models/` (not in git)

`models/` holds only what the live notebooks (`summary.ipynb`, `p8_deliberation.ipynb`) read, the networks needed
to regenerate it, and the starting students: about 1 GB. Everything else from earlier work was moved, unchanged and
at the same relative paths, to the archive folder **`C:\dev\rush-hour-archive\`** on the Windows PC (about 50 GB,
2026-11-19). Nothing was deleted.

## Kept

| Path | What | Written by | Read by |
|---|---|---|---|
| `sessions/<run>/` (28 runs) | Session results: `config.json`, per teacher `sessions.json` (play) and `results.json` (tests). The students' snapshots (`*.pt`) are archived. | `python -m rushhour.sessions` | summary |
| `sessions/and_or_gamma/` | Plain AND-OR on the 70 human layouts over γ | `experiments/and_or_gamma.py` | summary, p8 |
| `sessions/calibrated_v2/` | Only its `config.json`: it defines `surrender.puzzle_set("test")` | - | summary |
| `summary/and_or_humans/` | The paper model's move likelihoods and simulations | `experiments/and_or_humans.py` | summary |
| `summary/apprentice_curve/` | Each apprentice checkpoint on the test puzzles | `experiments/apprentice_curve.py` | summary |
| `summary/and_or_network/` | Network-ordered AND-OR and intuition on the human layouts | `experiments/and_or_network.py` | p8 |
| `apprentice/gamma0.07_free/` | The AND-OR apprentice (γ 0.07, protect_movers off): every epoch's checkpoint | `experiments/apprentice.py` | summary (manifest), the curve and p8 scripts |
| `students/` | Candidate starting students, one per round of their pretraining run (was `teach/fresh_p0.1/random_0.1/`). `round6_epoch01.pt` is `paths.START`, the start of every session the summary shows | the p2 teaching run | `rushhour.surrender`, new sessions |

The 28 session runs: `fitted_teachers`, `pilot_all_agency0.1_skip`, `teach_shortcut_agency{0,0.1,0.3}`,
`teach_shortcut_agency{0.0,0.1,0.3,0.5,0.7,0.9,1}_even`, `teach_shortcut_agency0_confirm`, and
`people_{all,own,w3,w10}{,_m10,_m20,_m60,_m120}`.

## Archived (`C:\dev\rush-hour-archive\models\`)

- `sessions/`: the 97 other runs (read only by the frozen p3-p7 notebooks: the AND-OR start runs `aostart_*`,
  `ao10`/`ablation_spend`, `loss_log_*`, `schedule_*`, hindsight, multiplier and learning-rate runs, ...), the run
  queues, the `surrender_fit` cache, and every student snapshot (`*.pt`) of the kept runs.
- `teach/`: the p2 offline-teaching runs; `checkpoints/`: phase 1's apprentice (protect_movers on);
  `apprentice/gamma0.01_free`; the p1 networks (`policy_agent*.pt`, `reinforce_agent.pt`).
- `summary/`: caches nothing live reads (`ao10_agency.csv`, `aoconsistent_*.csv`, the protected apprentice curve).
- `../logs/p3_session/`: the p3 run logs that were in `scripts/p3_session/logs`.

To rerun a frozen notebook, check out tag `pre-cleanup` and move what it reads back into `models/`.

## Another machine

The Mac keeps its own `models/` with the old layout. After pulling, before running anything:

1. move `models/teach/fresh_p0.1/random_0.1` to `models/students`;
2. optionally, archive the rest the same way. Every path the code reads is in `rushhour/paths.py`.
