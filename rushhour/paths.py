"""Where the data and results live, relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOL = os.path.join(DATA, "rush_nw.txt")  # every 6x6 puzzle with its optimal length (the analysis pool)
HUMANS = os.path.join(ROOT, "23vrb-osfstorage-archive")  # Olieslagers et al. (2023): people's play and the layouts
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(MODELS, "sessions")  # session runs (sessions.py), one folder per run
APPRENTICE = os.path.join(MODELS, "apprentice")  # expert-iteration runs (experiments/apprentice.py)
STUDENTS = os.path.join(MODELS, "students")  # candidate starting students, one per round of their pretraining run
START = os.path.join(STUDENTS, "round6_epoch01.pt")  # the student every session starts from (surrender.py picked it)
# The newer candidate (from 2026-11-17, runs aostart_* in the archive): the AND-OR apprentice itself,
# os.path.join(APPRENTICE, "gamma0.07_free", "round1_epoch02.pt"), with giving-up scale 0.30.
