"""Where the data and results live, relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOL = os.path.join(DATA, "rush_nw.txt")  # every 6x6 puzzle with its optimal length (the analysis pool)
HUMANS = os.path.join(ROOT, "23vrb-osfstorage-archive")  # Olieslagers et al. (2023): people's play and the layouts
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(MODELS, "sessions")  # session runs (sessions.py), one folder per run
APPRENTICE = os.path.join(MODELS, "apprentice")  # expert-iteration runs (experiments/apprentice.py)
# The student every session starts from: AND-OR's apprentice (gamma 0.07, protect_movers off) after 2 epochs of its
# first round, the epoch that matches people's start (with giving-up scale 0.30; surrender.py --epochs all).
START = os.path.join(APPRENTICE, "gamma0.07_free", "round1_epoch02.pt")
# The start before 2026-11-19, which every older run in the summary used: a network trained on its own solved attempts
# with a random 10% teacher (the p2 run fresh_p0.1/random_0.1, one checkpoint per round; giving-up scale 0.40).
STUDENTS = os.path.join(MODELS, "students")
PREVIOUS_START = os.path.join(STUDENTS, "round6_epoch01.pt")
