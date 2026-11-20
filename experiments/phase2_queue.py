"""Phase 2 of the switch to AND-OR's apprentice as the start (MODEL.md): the runs the existing ones can't support,
as one run_queue queue, "phase2". Every run is the fitted student (sessions --fitted) with only what's named here
changed.

1. Hindsight from this start (people's lengths, no teacher, as the ao_people_all_m* fit): own labels, W=3 and W=10
   at x5, x10 and x20, and full hindsight at x5 - is full x10 one fit among several, as on the previous start?
2. Seeds 1 and 2 for the whole x10 agency sweep, the teachers of aostart_free_m10_agency* (seed 0): 24 students per
   condition. Seed 1's four saved runs get the teachers they lack; the rest are new.
3. bayes4's theta at agencies 0, 0.3 and 1 (seed 0), added to the saved aostart_free_m10_agency* runs: 0.45 was tuned
   for bayes3 on the previous start.

    python experiments/phase2_queue.py            # checks the runs against the saved ones it extends, then launches
    python experiments/phase2_queue.py --check    # only checks
    python -m rushhour.run_queue watch phase2     # progress
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import run_queue as rq, sessions as se  # noqa: E402
from rushhour.student import spec  # noqa: E402

PEOPLE = ["--session-bins", "6-6", "10-10", "13-13", "15-15", "--test-bins", "6-6", "10-10", "13-13", "15-15",
          "--test-per-bin", "20", "--test-surrender", "--agency-step", "off"]  # the ao_people_all_m* protocol
HINDSIGHT = {"own": ["--labels", "own", "--window", "-1"], "w3": ["--labels", "all", "--window", "3"],
             "w10": ["--labels", "all", "--window", "10"], "all": []}
AGENCY_TEACHERS = {0.0: ["bayes4"], 1.0: ["none", "bayes4", "random:0.3", "random:1.0"]}  # else bayes4 and randoms
THETAS = ["bayes4:0.25", "bayes4:0.75", "bayes4:0.9"]


def name(a):
    return f"{a:g}"


def runs():
    out = []
    for h, flags in HINDSIGHT.items():
        for m in (5, 10, 20):
            if h == "all" and m != 5:
                continue  # saved: ao_people_all_m10, _m20
            out.append((f"ao_people_{h}_m{m}", ["--fitted", *PEOPLE, *flags, "--multiplier", str(m), "--teachers",
                                                "none"]))
    for seed in (1, 2):
        for a in spec.AGENCIES:
            teachers = AGENCY_TEACHERS.get(a, ["bayes4", "random:0.3", "random:1.0"])
            out.append((f"aostart_free_m10_s{seed}_agency{name(a)}",
                        ["--fitted", "--seed", str(seed), "--shown-weight", str(a), "--teachers", *teachers]))
    for a in (0.0, 0.3, 1.0):
        teachers = AGENCY_TEACHERS.get(a, ["bayes4", "random:0.3", "random:1.0"]) + THETAS
        out.append((f"aostart_free_m10_agency{name(a)}", ["--fitted", "--shown-weight", str(a), "--teachers",
                                                          *teachers]))
    return out


def check(queue):
    """Each run's settings against its saved run (sessions.check_config refuses a mismatch), and the new
    hindsight runs against ao_people_all_m10, which they must equal apart from what they vary."""
    ok = True
    ref = json.load(open(os.path.join(se.OUT, "ao_people_all_m10", "config.json")))
    for run, argv in queue:
        config = vars(se.parse_args([run, *argv]))
        path = os.path.join(se.OUT, run, "config.json")
        if os.path.exists(path):
            saved, what = json.load(open(path)), "saved"
            varies = set()
        elif run.startswith("ao_people"):
            saved, what = ref, "ao_people_all_m10"
            varies = {"run", "labels", "window", "multiplier", "block"}
        else:
            print(f"{run:<34} new   teachers {config['teachers']}  vs spec: "
                  f"{ {k: v[1] for k, v in spec.differences(config).items()} }")
            continue
        if config.get("even_batches") is None:
            config["even_batches"] = saved.get("even_batches", se.ADDED_SETTINGS["even_batches"])
        differs = sorted(k for k in set(saved) | set(config) if k not in se.SAME_RUN_IGNORES | varies
                         and se._same_key(k, saved.get(k, se.ADDED_SETTINGS.get(k)))
                         != se._same_key(k, config.get(k, se.ADDED_SETTINGS.get(k))))
        adds = [t for t in config["teachers"] if t not in saved["teachers"]] if what == "saved" else config["teachers"]
        print(f"{run:<34} {'extends' if what == 'saved' else 'new'} ({what}): adds {adds}"
              + (f"  DIFFERS: {differs}" if differs else ""))
        ok &= not differs
    return ok


if __name__ == "__main__":
    queue = runs()
    good = check(queue)
    print(f"{len(queue)} runs, {sum(len(vars(se.parse_args([r, *a]))['teachers']) for r, a in queue)} teacher runs "
          f"(saved ones are skipped)")
    if not good:
        raise SystemExit("settings differ from the saved runs: not launched")
    if "--check" not in sys.argv:
        rq.launch("phase2", queue)
