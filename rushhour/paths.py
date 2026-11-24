"""Where the data and results live, relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOL = os.path.join(DATA, "rush_nw.txt")  # every 6x6 puzzle with its optimal length (the analysis pool)
HUMANS = os.path.join(ROOT, "23vrb-osfstorage-archive")  # Olieslagers et al. (2023): people's play and the layouts
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(MODELS, "sessions")  # session runs (sessions.py), one folder per run
APPRENTICE = os.path.join(MODELS, "apprentice")  # expert-iteration runs (experiments/apprentice.py)
# The student every session starts from: AND-OR's apprentice (gamma 0.07, protect_movers off) after 3 epochs of its
# first round, the epoch that matches people's start (with giving-up scale 0.40; surrender.py), as two networks with 2
# lane blocks each - a policy network trained on AND-OR's moves only (gamma0.07_free_novalue_b2) and a value network
# trained on its distances only (gamma0.07_free_valueonly_b2), saved together (PolicyAgent.value_agent;
# experiments/network_depth.py build).
START = os.path.join(APPRENTICE, "separate_b2", "round1_epoch03.pt")
# The same with 4 lane blocks, the start of 2026-11-24 before that (giving-up scale 0.35): experiments/separate_value.py.
FOUR_BLOCK_START = os.path.join(APPRENTICE, "separate", "round1_epoch03.pt")
# The start of 2026-11-24 before that: one network with both heads on a shared trunk, trained on both losses, no dest
# head (giving-up scale 0.35).
SHARED_START = os.path.join(APPRENTICE, "gamma0.07_free_nodest", "round1_epoch03.pt")
# The start from 2026-11-19 to 2026-11-24, which the runs of that week used (phase 1's aostart_* teaching runs, phase
# 2's sessions): the same apprentice trained with a dest head as well, after 2 epochs (giving-up scale 0.30).
DEST_START = os.path.join(APPRENTICE, "gamma0.07_free", "round1_epoch02.pt")
# The start before 2026-11-19, which every older run in the summary used: a network trained on its own solved attempts
# with a random 10% teacher (the p1.2 run fresh_p0.1/random_0.1, one checkpoint per round; giving-up scale 0.40).
STUDENTS = os.path.join(MODELS, "students")
PREVIOUS_START = os.path.join(STUDENTS, "round6_epoch01.pt")
