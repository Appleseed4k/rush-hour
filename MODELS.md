# What is in `models/` (not in git)

`models/` holds only what the live notebooks (`p1_summary.ipynb`, `p2_summary.ipynb`, `p2.1_deliberation.ipynb`, `p2.2_forward_planning.ipynb`, `p2.3_human_likeness.ipynb`) read, the networks needed
to regenerate it, and the starting students: about 1 GB. Everything else from earlier work was moved, unchanged and
at the same relative paths, to the archive folder **`C:\dev\rush-hour-archive\`** on the Windows PC (about 50 GB,
2026-11-19). Nothing was deleted.

## Kept

| Path | What | Written by | Read by |
|---|---|---|---|
| `sessions/<run>/` (33 runs) | Session results: `config.json`, per teacher `sessions.json` (play) and `results.json` (tests). The students' snapshots (`*.pt`) are archived. | `python -m rushhour.sessions` | p1_summary |
| `sessions/ao*/` (29 runs restored from the archive 2026-11-19, 20 more from phase 2) | The AND-OR apprentice start: `ao_people_{all,own,w3,w10}_m*` (its hindsight and multiplier fit), `aostart_free_m10_*` (the agency sweep of the current spec, with bayes4's θ), `aostart_free_m10_s{1,2}_*` (seeds 1 and 2), `aostart_free_*` and `aolambda_free_s1_*` (×30), `aostart_protected_*` (phase 1's apprentice). JSON only for the restored ones; their snapshots stay in the archive | `python -m rushhour.sessions`; `experiments/phase2_queue.py` | p1_summary, MODEL.md, `experiments/phase2_analysis.py` |
| `sessions/surrender_fit/` | `surrender.py`'s cache of each checkpoint's attempts. The apprentice's were copied from `summary/apprentice_curve/` (the same attempts: same puzzles, runs and seed) | `python -m rushhour.surrender` | `rushhour.surrender` |
| `sessions/and_or_gamma/` | Plain AND-OR on the 70 human layouts over γ | `experiments/and_or_gamma.py` | p1_summary, p2.1 |
| `sessions/teach_shortcut_agency0/trace/` | The unhelped students' paths, and `skill_tracking.json`, the sweep behind bayes4 and `notebooks/p1.7_skill_filter.html` (bayes3's refit included) | `python -m rushhour.sessions`; `experiments/skill_tracking.py` (now writes `summary/skill_tracking.json`) | `experiments/skill_tracking.py` |
| `sessions/calibrated_v2/` | Only its `config.json`: it defines `surrender.puzzle_set("test")` | - | p1_summary |
| `summary/and_or_humans/` | The paper model's move likelihoods and simulations | `experiments/and_or_humans.py` | p1_summary |
| `summary/apprentice_curve/` | Each apprentice checkpoint on the test puzzles | `experiments/apprentice_curve.py` | p1_summary |
| `summary/aoconsistent_calibration.csv`, `aoconsistent_agency.csv`, `ao10_agency.csv` | Tables from the apprentice-start runs: the multiplier fit, and the agency sweeps at ×30 and ×10 (restored from the archive) | the p1.3 notebook's last section (archived) | MODEL.md |
| `summary/hindsight_fit.csv`, `ao10_agency_seeds.csv`, `bayes4_theta.csv` | Phase 2's tables: the fit of each hindsight and multiplier, the ×10 agency sweep pooled over seeds 0-2, and bayes4's θ | `experiments/phase2_analysis.py` | MODEL.md |
| `summary/dest_quality.csv`, `dest_quality_session.csv`, `forward.csv`, `forward_uniform.csv`, `forward_value.csv`, `forward_sweep.csv`, `forward_integrate.csv` | The dest head's relevance map against BFS truth (every apprentice epoch; a session's snapshots), and the forward planner's attempts (novelty x relevance rollouts, its ablations, the uniform-proposal control, the value head as evaluator, and the novelty-weight and fading-memory sweep) | `experiments/dest_quality.py`, `experiments/forward.py` | `p2.2_forward_planning.ipynb` |
| `summary/human_profile/` | `moves.csv.gz`: one row per move, for people (every attempt) and for simulated models on the 70 human layouts (paper and network AND-OR, intuition, the forward planner, random), with distance, revisit, car-role, slide and timing measures; `sims_70x10.pkl` caches the simulations and AND-OR roles | `experiments/human_profile.py` | `p2.3_human_likeness.ipynb` |
| `summary/memory_fit.csv`, `memory_fit_planner.csv`, `gamma_fit.csv`, `undo_fit.csv`, `gamma_learning.csv`, `gamma_snapshots.csv`, `lean_profile.csv` | The soft board memory fitted to people's revisits (one row per attempt of the start network's intuition on the 70 human layouts, over a grid of (beta, half-life), with the ages of its revisits), the planner's novelty (lam, novelty half-life) fitted to board and car-placement returns, and the mixed forward + backward planner over a grid of forward and backward lapse rates (shared gamma), fixes for the mixed planner's backward undos (the memory judging candidates; committing to plans), one shared gamma fitted per stage of people's session, and the calibrated session's snapshots (archive: ao_people_all_m10) by intuition and in the planner per stage, and the two-mode student (with AND-OR) and forward student (without) | `experiments/fit_memory.py` (`planner`, `gamma`, `undo`, `learning`, `snapshots`, `lean`) | `p2.3_human_likeness.ipynb`, `p2.2_forward_planning.ipynb` |
| `sessions/planner_people_all_m10/` | The calibrated session (ao_people_all_m10's settings) played by the forward + backward planner (best of 8, γ 0.03): `config.json` (with the planner's settings), `no_teacher/sessions.json` and the 8 students' snapshots after 18, 36, 54, 72 trials (kept here, about 150 MB) | `experiments/planner_session.py` | `fit_memory.py snapshots --run`, `p2.2_forward_planning.ipynb` |
| `sessions/lean_people_all_m10_ao/`, `lean_people_all_m10_noao/` | The calibrated session played by the two-mode student (`_ao`: intuition, flat planning 8% to 18% over 30 trials, one-move AND-OR 10%, the opening rule) and the forward student (`_noao`: flat 18% to 28%, no AND-OR); the code calls both the lean solver. `config.json` (with the student's settings), `no_teacher/sessions.json`, the 8 students' snapshots after 18, 36, 54, 72 trials | `experiments/planner_session.py --lean` | `fit_memory.py lean-snapshots`, `p2.2_forward_planning.ipynb` |
| `summary/lean_snapshots_lean_people_all_m10_ao.csv`, `lean_snapshots_lean_people_all_m10_noao.csv` | Each of these sessions' snapshots playing the 70 human layouts with their stage's mode weights | `experiments/fit_memory.py lean-snapshots --run` | `p2.2_forward_planning.ipynb` |
| `summary/gamma_snapshots_planner_people_all_m10.csv` | That session's snapshots by intuition and in the planner per stage of people's session | `experiments/fit_memory.py snapshots --run planner_people_all_m10` | `p2.2_forward_planning.ipynb` |
| `summary/planner_start_fit.csv`, `forward_start_fit.csv`, `flat_start_fit.csv`, `two_mode_start_fit.csv`, `lean_start_fit.csv` | Each planner with the start network on the session's test puzzles (no giving up; one row per attempt), for calibrating to people's first 10 trials: best of 8 over γ; forward rollouts (k × γ) and intuition; flat (moves considered × τ) with and without a checking rollout; the impasse switch (flat forward, AND-OR plans at impasses); the forward and two-mode students (intuition, flat and one-move AND-OR, drawn per move; `lean` in the code) | `experiments/planner_session.py` (`--fit-start`, `--fit-forward`, `--fit-flat`, `--fit-two-mode`, `--fit-lean`) | `p2.2_forward_planning.ipynb` |
| `summary/shape_start_fit.csv`, `shape_profile.csv`, `rollout_shape_profile.csv` | The near-goal gap: the two-mode student with goal in sight (a search of up to D moves for red's way out) and a sharper intuition (temperature), calibrated to people's start and profiled by distance on the human layouts; rollouts near the goal | `experiments/planner_session.py --fit-shape`, `experiments/fit_memory.py shape` | p2.2 |
| `summary/human_start_fit.csv`, `strong_start_fit.csv`, `strong_shape_profile.csv` | The start recalibrated on people's own 70 layouts against their first 10 trials and their first attempts (a restart ends an attempt): checkpoints of the apprentice by intuition, temperatures, the two-mode student's variants; and the strongest candidates' skill by distance (round 2 epoch 5 by intuition fits) | `experiments/planner_session.py --fit-human`, `--fit-strong` | p2.2, p2_summary section 7 |
| `summary/pool_shape_profile.csv` | The same networks on the session's pool test puzzles, to compare with their profile on people's layouts (pool puzzles are easier 7+ moves from the goal) | `experiments/session_fit.py pool` | p2_summary section 7 |
| `sessions/r2e5_people_m5/` ... `r2e5_hardest_m20/` (bins people, harder, hardest x m 5, 10, 20), `summary/session_fit.csv` | Sessions from round 2 epoch 5 (giving up at 0.35) trained on pool puzzles of people's lengths, 9-26 or 14-31 moves, x5/x10/x20; their snapshots playing people's layouts by intuition against each stage of people (little learning transfers) | `experiments/session_fit.py run`, `fit` | p2_summary section 7 |
| `summary/mode_mixture/` | People's moves scored under intuition, flat planning and AND-OR (start network, or each stage's network from the calibrated session): `moves.csv.gz` (per move), `fit.csv` / `people.csv` (the two-way share of flat planning per stage, fast / slow, per person), `andor_counts.npz` (sampled AND-OR first moves per board), `fit3.csv` / `people3.csv` (the three-way weights per stage, distance and speed, per person), `loops.csv` (after a loop) and `slow.csv` (around slow moves) | `experiments/mode_mixture.py` (and `andor`, `loops`, `slow`) | `p2.2_forward_planning.ipynb` |
| `summary/and_or_network/` | Network-ordered AND-OR and intuition on the human layouts | `experiments/and_or_network.py` | p2.1 |
| `apprentice/gamma0.07_free/` | The AND-OR apprentice (γ 0.07, protect_movers off): every epoch's checkpoint. `round1_epoch02.pt` is `paths.START`, the start of every new session | `experiments/apprentice.py` | sessions, `rushhour.surrender`, p1_summary (manifest), the curve and p2.1 scripts |
| `students/` | The previous start's candidates, one per round of their pretraining run (was `teach/fresh_p0.1/random_0.1/`). `round6_epoch01.pt` is `paths.PREVIOUS_START`, the start of the runs the phase 1 summary marks *previous start* | the p1.2 teaching run | `rushhour.surrender --folder models/students --epochs kept` |

The 33 session runs: `fitted_teachers`, `pilot_all_agency0.1_skip`, `teach_shortcut_agency{0,0.1,0.3}`,
`teach_shortcut_agency{0.0,0.1,0.3,0.5,0.7,0.9,1}_even`, `teach_shortcut_agency0_confirm`, and
`people_{all,own,w3,w10}{,_m10,_m20,_m60,_m120}`.

## Archived (`C:\dev\rush-hour-archive\models\`)

- `sessions/`: the 97 other runs (read only by the frozen p1.3-p1.7 notebooks: the AND-OR start runs `aostart_*`,
  `ao10`/`ablation_spend`, `loss_log_*`, `schedule_*`, hindsight, multiplier and learning-rate runs, ...), the run
  queues, the `surrender_fit` cache, and every student snapshot (`*.pt`) of the kept runs.
- `teach/`: the p1.2 offline-teaching runs; `checkpoints/`: phase 1's apprentice (protect_movers on);
  `apprentice/gamma0.01_free`; the p1.1 networks (`policy_agent*.pt`, `reinforce_agent.pt`).
- `summary/`: caches nothing live reads (`ao10_agency.csv`, `aoconsistent_*.csv`, the protected apprentice curve).
- `../logs/p3_session/`: the p1.3 run logs that were in `scripts/p3_session/logs`.

To rerun a frozen notebook, check out tag `pre-cleanup` and move what it reads back into `models/`.

## Another machine

The Mac keeps its own `models/` with the old layout. After pulling, before running anything:

1. move `models/teach/fresh_p0.1/random_0.1` to `models/students`;
2. copy the `ao*` session runs (JSON only) and the `summary/ao*.csv` tables from the Windows PC, which ran them (or rerun `experiments/phase2_analysis.py` for phase 2's tables);
3. optionally, archive the rest the same way. Every path the code reads is in `rushhour/paths.py`.
