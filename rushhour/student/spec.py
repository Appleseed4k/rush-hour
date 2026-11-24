"""The simulated student the project's results rest on, written down as settings (MODEL.md explains each one: what
it assumes, where its value came from, and the code that implements it).

The code's own defaults (sessions.parse_args, learning_session) describe a plain, uncalibrated student. The student
the project uses is those defaults plus the settings in STUDENT. Since 2026-11-19 it starts from AND-OR's apprentice,
since 2026-11-24 from one trained without a dest head, and since later that day as two networks, a policy network and
a separate value network (SHARED below: the one-network start of that day; DEST: the start and giving-up scale of the
week's runs before; PREVIOUS: the start, giving-up scale and multiplier the older runs used). Use it as

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
    # starting knowledge: AND-OR's apprentice as a policy network and a value network, stopped where it matches
    # people's start
    "init": os.path.relpath(paths.START, paths.ROOT).replace(os.sep, "/"),  # fitted, with surrender_scale
    # playing
    "explore": 1.0,  # chosen: its own moves sampled from the policy at temperature 1
    "memory": 10,  # chosen: never moves back to one of its last 10 boards, if it can avoid it
    "max_attempts": 1,  # chosen: one attempt per puzzle, no restarts
    "step_factor": 10.0, "min_steps": 50, "max_steps": None,  # chosen: the move cap, max(50, 10 x optimal)
    "surrender": True,  # people: gives up as people do (the hazard by moves / optimal, saved as surrender_table)...
    "surrender_scale": 0.40,  # fitted: ...at 0.40 x people's rates, together with init
    # what it learns from
    "labels": "all", "window": -1,  # fitted, with multiplier: full hindsight (shortcut labels through every board
    # seen) - the best of own, W=3, W=10 and full at x5-x20, though W=10 x5 and W=3 x20 come close
    "encode": 1.0,  # chosen: no encoding lapses
    "progress_weight": 0.0,  # chosen: learns nothing from unsolved attempts (only from moves shown in them)
    # how it learns
    "multiplier": 10, "block": 10,  # fitted, with the labels: 10 puzzles per human trial
    "lr": 1e-4, "steps": 3,  # chosen: Adam at 1e-4, 3 passes over each block's examples
    "even_batches": True,  # chosen: equal batches, none of a handful of examples
    "agency_step": "skip",  # chosen: how agency acts (with shown_weight < 1, an update is skipped, not scaled)
}

# Agency (Delta A): how much a move a teacher showed teaches, the student's own play 1. Not fitted to people: the
# teaching results are a sweep over it (aostart_free_m10_agency*).
AGENCY = "shown_weight"
AGENCIES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)

# The session and its tests (an experimental design, not the student): the teaching runs' sessions of 18 trials per
# length bin (each trial `multiplier` puzzles) and held-out tests every 18 human trials, untaught, with no giving up.
PROTOCOL = {
    "students": 8,
    "session_bins": ["9-11", "14-16", "19-21", "24-26"], "per_bin": 18,
    "test_bins": ["9-11", "14-16", "19-21", "24-26"], "test_per_bin": 50,
    "test_every": 18, "test_runs": 3, "test_temperature": 1.0, "test_surrender": False,
    "seed": 0,
}

# What the runs of 2026-11-19 to 2026-11-24 (aostart_*, ao_*, lean_*, planner_*) had instead: the apprentice trained
# with a dest head too, stopped after 2 epochs, and giving up fitted with it. The multiplier and labels were fitted
# with this start and are kept as they were.
DEST = {"init": os.path.relpath(paths.DEST_START, paths.ROOT).replace(os.sep, "/"), "surrender_scale": 0.3}

# The one-network start of 2026-11-24 (no dest head; policy and value heads on a shared trunk), before the separate
# networks: lean_people_all_m10_ao_nodest*, lean_people_all_m10_ao_nodest_refit*.
SHARED = {"init": os.path.relpath(paths.SHARED_START, paths.ROOT).replace(os.sep, "/"), "surrender_scale": 0.35}

# The two networks with 4 lane blocks each (2026-11-24, before the switch to 2): lean_people_all_m10_ao_separate*.
FOUR_BLOCK = {"init": os.path.relpath(paths.FOUR_BLOCK_START, paths.ROOT).replace(os.sep, "/"), "surrender_scale": 0.35}

# What the older runs (teach_shortcut_agency*, people_*, fitted_teachers) had instead: a network trained on its own
# solved attempts with a random 10% teacher, and the multiplier calibrated for it with own labels.
PREVIOUS = {"init": os.path.relpath(paths.PREVIOUS_START, paths.ROOT).replace(os.sep, "/"), "surrender_scale": 0.4,
            "multiplier": 30, "block": 30}


def _same(key, value):
    if key == "init" and value is not None:  # compare a checkpoint by its folder and file name: saved runs keep the
        return "/".join(value.replace("\\", "/").split("/")[-2:])  # old absolute path
    return value


def differences(config, protocol=True, previous=False, dest=False, shared=False, four_blocks=False):
    """{setting: (spec value, the run's value)} where a saved run's config differs from STUDENT (and PROTOCOL; with
    `dest`, the dest-head start's STUDENT | DEST; with `shared`, the one-network start's STUDENT | SHARED; with
    `four_blocks`, the 4-block start's STUDENT | FOUR_BLOCK; with `previous`, the older start's STUDENT | PREVIOUS);
    settings a
    run saved before they existed count as the code's default then (sessions.ADDED_SETTINGS)."""
    from ..sessions import ADDED_SETTINGS  # sessions imports this package
    spec = (STUDENT | (DEST if dest else {}) | (SHARED if shared else {}) | (FOUR_BLOCK if four_blocks else {})
            | (PREVIOUS if previous else {})
            | (PROTOCOL if protocol else {}))
    out = {}
    for key, want in spec.items():
        have = config.get(key, ADDED_SETTINGS.get(key))
        if _same(key, have) != _same(key, want):
            out[key] = (want, have)
    return out
