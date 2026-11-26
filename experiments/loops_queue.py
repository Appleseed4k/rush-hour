"""Stage 1 of the salience and loop-aware bayes4 test (p2.6, sections 11-): the teaching runs, with the comparisons
fixed in advance.

Student: the two-mode student of that day (the soft memory, beta 2 and half-life 1; spec.SOFT) with salient memory
(solve_two_mode's salience (0.5, never fades), fitted to people's
repeat visits by experiments/fit_salience.py; checked against people at people's lengths in
lean_people_all_m10_ao_b2_salience*), otherwise spec.STUDENT. Contrast: the soft-memory two-mode student's runs of the
replication (twomode_m10*), to which loop-aware bayes4 is added.

Teachers: none, bayes4, loop-aware bayes4 (bayes4:0.45,loops=1) and random at loop-aware bayes4's spend on this
student (random:0.1; experiments/bayes4_loops.py: 9.6% of moves). Agencies 0.1, 0.5 and 1; seeds 0-2 (24 students).

Comparisons (paired by student; primary outcome the end score on the test's 9-16-move puzzles, secondary all):
1. bayes4 against no teacher (salient student);
2. loop-aware bayes4 against bayes4 (salient student, and soft: the 2 x 2 of memory x teacher);
3. loop-aware bayes4 against random at the same spend (salient student).

    python experiments/loops_queue.py           # checks the settings, then launches queues loops_a and loops_b
    python experiments/loops_queue.py --check
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import run_queue as rq, sessions as se  # noqa: E402
from rushhour.student import spec  # noqa: E402

SALIENCE = [0.5, None]
SALIENT = spec.TWO_MODE_SOFT | {"soft_memory": [2, 1], "salience": SALIENCE}  # the soft memory then, plus salience
SOFT = spec.TWO_MODE_SOFT | {"soft_memory": [2, 1], "salience": None}
AGENCIES = (0.1, 0.5, 1.0)
LOOPS = "bayes4:0.45,loops=1"
MATCHED = "random:0.1"
PROCESSES = 8


def seed_part(seed):
    return "" if seed == 0 else f"_s{seed}"


def salient_runs(seeds):
    out = []
    for seed in seeds:
        for a in AGENCIES:
            teachers = (["none"] if a == 1.0 else []) + ["bayes4", LOOPS, MATCHED]
            out.append((f"salient_m10{seed_part(seed)}_agency{a:g}",
                        ["--fitted", "--two-mode", json.dumps(SALIENT), "--seed", str(seed), "--shown-weight", str(a),
                         "--processes", str(PROCESSES), "--teachers", *teachers]))
    return out


def soft_runs():
    """The replication's soft-memory runs at these agencies, with loop-aware bayes4 added (saved teachers are
    skipped)."""
    out = []
    for seed in (0, 1, 2):
        for a in AGENCIES:
            teachers = (["none", "bayes4", "random:0.3", "random:1.0"] if a == 1.0
                        else ["bayes4", "random:0.3", "random:1.0"])
            out.append((f"twomode_m10{seed_part(seed)}_agency{a:g}",
                        ["--fitted", "--two-mode", json.dumps(SOFT), "--seed", str(seed), "--shown-weight", str(a),
                         "--processes", str(PROCESSES), "--teachers", *teachers, LOOPS]))
    return out


def check(queue, salient):
    ok = True
    for run, argv in queue:
        config = vars(se.parse_args([run, *argv]))
        differs = {k: v for k, v in spec.differences(config, soft=True).items() if k not in ("seed", spec.AGENCY)}
        if salient:
            if se._same_key("two_mode", differs.pop("two_mode", (None, config["two_mode"]))[1])                     != se._same_key("two_mode", SALIENT):
                ok = False
                print(f"{run}: not the salient student")
        if differs:
            print(f"{run}: differs from the spec in {differs}")
            ok = False
    return ok


if __name__ == "__main__":
    queues = {"loops_a": (salient_runs((0, 1)), True),
              "loops_b": (salient_runs((2,)), True)}
    soft = soft_runs()
    good = all(check(q, s) for q, s in queues.values()) and check(soft, False)
    for name, (q, _) in queues.items():
        print(name, len(q), "runs,", sum(len(vars(se.parse_args([r, *a]))["teachers"]) for r, a in q), "teacher runs")
    print("soft (in loops_b):", len(soft), "runs, 9 new teacher runs")
    if not good:
        raise SystemExit("settings differ: not launched")
    if "--check" not in sys.argv:
        rq.launch("loops_a", queues["loops_a"][0])
        rq.launch("loops_b", queues["loops_b"][0] + soft)
