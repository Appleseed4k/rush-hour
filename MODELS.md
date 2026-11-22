# What is in `models/` (not in git)

`models/` holds only what the live notebooks (`summary.ipynb`, `p8_deliberation.ipynb`, `forward_planning.ipynb`, `human_likeness.ipynb`) read, the networks needed
to regenerate it, and the starting students: about 1 GB. Everything else from earlier work was moved, unchanged and
at the same relative paths, to the archive folder **`C:\dev\rush-hour-archive\`** on the Windows PC (about 50 GB,
2026-11-19). Nothing was deleted.

## Kept

| Path | What | Written by | Read by |
|---|---|---|---|
| `sessions/<run>/` (33 runs) | Session results: `config.json`, per teacher `sessions.json` (play) and `results.json` (tests). The students' snapshots (`*.pt`) are archived. | `python -m rushhour.sessions` | summary |
| `sessions/ao*/` (29 runs restored from the archive 2026-11-19, 20 more from phase 2) | The AND-OR apprentice start: `ao_people_{all,own,w3,w10}_m*` (its hindsight and multiplier fit), `aostart_free_m10_*` (the agency sweep of the current spec, with bayes4's θ), `aostart_free_m10_s{1,2}_*` (seeds 1 and 2), `aostart_free_*` and `aolambda_free_s1_*` (×30), `aostart_protected_*` (phase 1's apprentice). JSON only for the restored ones; their snapshots stay in the archive | `python -m rushhour.sessions`; `experiments/phase2_queue.py` | summary, MODEL.md, `experiments/phase2_analysis.py` |
| `sessions/surrender_fit/` | `surrender.py`'s cache of each checkpoint's attempts. The apprentice's were copied from `summary/apprentice_curve/` (the same attempts: same puzzles, runs and seed) | `python -m rushhour.surrender` | `rushhour.surrender` |
| `sessions/and_or_gamma/` | Plain AND-OR on the 70 human layouts over γ | `experiments/and_or_gamma.py` | summary, p8 |
| `sessions/teach_shortcut_agency0/trace/` | The unhelped students' paths, and `skill_tracking.json`, the sweep behind bayes4 and `notebooks/skill_filter.html` (bayes3's refit included) | `python -m rushhour.sessions`; `experiments/skill_tracking.py` (now writes `summary/skill_tracking.json`) | `experiments/skill_tracking.py` |
| `sessions/calibrated_v2/` | Only its `config.json`: it defines `surrender.puzzle_set("test")` | - | summary |
| `summary/and_or_humans/` | The paper model's move likelihoods and simulations | `experiments/and_or_humans.py` | summary |
| `summary/apprentice_curve/` | Each apprentice checkpoint on the test puzzles | `experiments/apprentice_curve.py` | summary |
| `summary/aoconsistent_calibration.csv`, `aoconsistent_agency.csv`, `ao10_agency.csv` | Tables from the apprentice-start runs: the multiplier fit, and the agency sweeps at ×30 and ×10 (restored from the archive) | the p3 notebook's last section (archived) | MODEL.md |
| `summary/hindsight_fit.csv`, `ao10_agency_seeds.csv`, `bayes4_theta.csv` | Phase 2's tables: the fit of each hindsight and multiplier, the ×10 agency sweep pooled over seeds 0-2, and bayes4's θ | `experiments/phase2_analysis.py` | MODEL.md |
| `summary/dest_quality.csv`, `dest_quality_session.csv`, `forward.csv`, `forward_uniform.csv`, `forward_value.csv`, `forward_sweep.csv`, `forward_integrate.csv` | The dest head's relevance map against BFS truth (every apprentice epoch; a session's snapshots), and the forward planner's attempts (novelty x relevance rollouts, its ablations, the uniform-proposal control, the value head as evaluator, and the novelty-weight and fading-memory sweep) | `experiments/dest_quality.py`, `experiments/forward.py` | `forward_planning.ipynb` |
| `summary/human_profile/` | `moves.csv.gz`: one row per move, for people (every attempt) and for simulated models on the 70 human layouts (paper and network AND-OR, intuition, the forward planner, random), with distance, revisit, car-role, slide and timing measures; `sims_70x10.pkl` caches the simulations and AND-OR roles | `experiments/human_profile.py` | `human_likeness.ipynb` |
| `summary/memory_fit.csv`, `memory_fit_planner.csv`, `gamma_fit.csv` | The soft board memory fitted to people's revisits (one row per attempt of the start network's intuition on the 70 human layouts, over a grid of (beta, half-life), with the ages of its revisits), the planner's novelty (lam, novelty half-life) fitted to board and car-placement returns, and the mixed forward + backward planner over a grid of forward and backward lapse rates (shared gamma) | `experiments/fit_memory.py` (`planner`, `gamma`) | `human_likeness.ipynb`, `forward_planning.ipynb` |
| `summary/and_or_network/` | Network-ordered AND-OR and intuition on the human layouts | `experiments/and_or_network.py` | p8 |
| `apprentice/gamma0.07_free/` | The AND-OR apprentice (γ 0.07, protect_movers off): every epoch's checkpoint. `round1_epoch02.pt` is `paths.START`, the start of every new session | `experiments/apprentice.py` | sessions, `rushhour.surrender`, summary (manifest), the curve and p8 scripts |
| `students/` | The previous start's candidates, one per round of their pretraining run (was `teach/fresh_p0.1/random_0.1/`). `round6_epoch01.pt` is `paths.PREVIOUS_START`, the start of the runs the summary marks *previous start* | the p2 teaching run | `rushhour.surrender --folder models/students --epochs kept` |

The 33 session runs: `fitted_teachers`, `pilot_all_agency0.1_skip`, `teach_shortcut_agency{0,0.1,0.3}`,
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
2. copy the `ao*` session runs (JSON only) and the `summary/ao*.csv` tables from the Windows PC, which ran them (or rerun `experiments/phase2_analysis.py` for phase 2's tables);
3. optionally, archive the rest the same way. Every path the code reads is in `rushhour/paths.py`.
