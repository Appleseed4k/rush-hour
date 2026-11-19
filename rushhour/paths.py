"""Where the data and results live, relative to the repository root."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
POOL = os.path.join(DATA, "rush_nw.txt")  # every 6x6 puzzle with its optimal length (the analysis pool)
HUMANS = os.path.join(ROOT, "23vrb-osfstorage-archive")  # Olieslagers et al. (2023): people's play and the layouts
MODELS = os.path.join(ROOT, "models")
SESSIONS = os.path.join(MODELS, "sessions")  # session runs (sessions.py), one folder per run
APPRENTICE = os.path.join(MODELS, "apprentice")  # expert-iteration runs (experiments/apprentice.py)
STUDENTS = os.path.join(MODELS, "teach", "fresh_p0.1", "random_0.1")  # candidate starting students (surrender.py)
START = os.path.join(STUDENTS, "round6_epoch01.pt")  # the student every session starts from
