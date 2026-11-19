# Archived lab notebooks (frozen)

p3–p7 are the lab notebooks behind `../summary.ipynb`, kept read-only with their outputs as they were last run.
They are not maintained: the code they import was reorganized after them, and most of the session data they read
was moved out of `models/` (see `../../MODELS.md`).

To rerun one, check out the tag it was frozen at and restore its data:

    git checkout pre-cleanup    # the repo as it was, sokoban included

| Notebook | What it covers |
|---|---|
| `p3_session.ipynb` | Within-session learning: fitting the simulated student to people's session (surrender, starting checkpoint, AND-OR's lapse rate γ) and the first teachers |
| `p4_dilemma.ipynb` | The assistance dilemma: agency ΔA, random help sweeps, calibrating the Bayesian teacher's model of the student |
| `p5_simplest_teacher.ipynb` | The stuck-Bayes teacher (threshold θ on the chance of solving alone), its ablations and calibration |
| `p6_hindsight.ipynb` | Hindsight labels (window, multiplier) fitted to people's learning rate; bayes3 across agency |
| `p7_pomdp.ipynb` | Toward a POMDP teacher: a small learning model fitted to the sessions (`p7_pomdp_definition_handoff.md`: its definition) |
| `skill_filter.html` | Export of the offline test of how a teacher carries its skill estimate across puzzles (p4/p5) |
