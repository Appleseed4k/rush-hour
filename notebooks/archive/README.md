# Archived lab notebooks (frozen)

p1.3–p1.7 are the lab notebooks behind `../p1_summary.ipynb`, kept read-only with their outputs as they were last run.
They were p3–p7 until 2026-11-22 (renamed to the p1.x / p2.x scheme); their own text still uses the old names.
They are not maintained: the code they import was reorganized after them, and most of the session data they read
was moved out of `models/` (see `../../MODELS.md`).

To rerun one, check out the tag it was frozen at and restore its data:

    git checkout pre-cleanup    # the repo as it was, sokoban included

| Notebook | What it covers |
|---|---|
| `p1.3_session.ipynb` | Within-session learning: fitting the simulated student to people's session (surrender, starting checkpoint, AND-OR's lapse rate γ) and the first teachers |
| `p1.4_dilemma.ipynb` | The assistance dilemma: agency ΔA, random help sweeps, calibrating the Bayesian teacher's model of the student |
| `p1.5_simplest_teacher.ipynb` | The stuck-Bayes teacher (threshold θ on the chance of solving alone), its ablations and calibration |
| `p1.6_hindsight.ipynb` | Hindsight labels (window, multiplier) fitted to people's learning rate; bayes3 across agency |
| `p1.7_pomdp.ipynb` | Toward a POMDP teacher: a small learning model fitted to the sessions (`p1.7_pomdp_definition_handoff.md`: its definition) |
