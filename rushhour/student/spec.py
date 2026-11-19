"""The simulated student the project's results rest on, written down as settings (MODEL.md explains each one: what
it assumes, where its value came from, and the code that implements it).

The code's own defaults (sessions.parse_args, learning_session) describe a plain, uncalibrated student. The student
that the summary's teaching results use is those defaults plus the settings in STUDENT, which until now lived only
in the saved runs' config.json. Use it as

    python -m rushhour.sessions my_run --fitted --teachers none bayes4 --shown-weight 0.1

(--fitted takes STUDENT and PROTOCOL as defaults; any flag given still wins), and check a saved run against it
with differences(config).

Each entry says where its value came from:
- fitted: chosen to match people (Olieslagers et al. 2023) - how, in MODEL.md;
- people: taken from people's data directly;
- chosen: set by hand, not fitted (some were compared with alternatives first, which MODEL.md notes);
"""
import os

from .. import paths

STUDENT = {
    # starting knowledge
    "init": os.path.relpath(paths.START, paths.ROOT).replace(os.sep, "/"),  # fitted, with surrender_scale
    # playing
    "explore": 1.0,  # chosen: its own moves sampled from the policy at temperature 1
    "memory": 10,  # chosen: never moves back to one of its last 10 boards, if it can avoid it
    "max_attempts": 1,  # chosen: one attempt per puzzle, no restarts
    "step_factor": 10.0, "min_steps": 50, "max_steps": None,  # chosen: the move cap, max(50, 10 x optimal)
    "surrender": True,  # people: gives up as people do (the hazard by moves / optimal, saved as surrender_table)...
    "surrender_scale": 0.4,  # fitted: ...at 0.4 x people's rates, together with init
    # what it learns from
    "labels": "all", "window": -1,  # chosen among fits: full hindsight (shortcut labels through every board seen)
    "encode": 1.0,  # chosen: no encoding lapses
    "progress_weight": 0.0,  # chosen: learns nothing from unsolved attempts (only from moves shown in them)
    # how it learns
    "multiplier": 30, "block": 30,  # fitted (with own labels, before shortcut labels): 30 puzzles per human trial
    "lr": 1e-4, "steps": 3,  # chosen: Adam at 1e-4, 3 passes over each block's examples
    "even_batches": True,  # chosen: equal batches, none of a handful of examples
    "agency_step": "skip",  # chosen: how agency acts (with shown_weight < 1, an update is skipped, not scaled)
}

# Agency (Delta A): how much a move a teacher showed teaches, the student's own play 1. Not fitted to people: the
# teaching results are a sweep over it (teach_shortcut_agency*_even).
AGENCY = "shown_weight"
AGENCIES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)

# The session and its tests (an experimental design, not the student): the teaching runs' sessions of 18 puzzles
# per length bin (x multiplier) and held-out tests every 18 human trials, all untaught and with no giving up.
PROTOCOL = {
    "students": 8,
    "session_bins": ["9-11", "14-16", "19-21", "24-26"], "per_bin": 18,
    "test_bins": ["9-11", "14-16", "19-21", "24-26"], "test_per_bin": 50,
    "test_every": 18, "test_runs": 3, "test_temperature": 1.0, "test_surrender": False,
    "seed": 0,
}


def _same(key, value):
    if key == "init" and value is not None:  # compare a checkpoint by its file name: saved runs keep the old path
        return os.path.basename(value.replace("\\", "/"))
    return value


def differences(config, protocol=True):
    """{setting: (spec value, the run's value)} where a saved run's config differs from STUDENT (and PROTOCOL);
    settings a run saved before they existed count as the code's default then (sessions.ADDED_SETTINGS)."""
    from ..sessions import ADDED_SETTINGS  # sessions imports this package
    spec = STUDENT | (PROTOCOL if protocol else {})
    out = {}
    for key, want in spec.items():
        have = config.get(key, ADDED_SETTINGS.get(key))
        if _same(key, have) != _same(key, want):
            out[key] = (want, have)
    return out
