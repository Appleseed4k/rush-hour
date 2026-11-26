"""The teaching confirmation on the student as it stands after the feature tests (spec.STUDENT: the joint memory, and
the flat schedule and opening as experiments/feature_tests.py's four-seed rule left them): stage 1's design
(experiments/loops_queue.py) played again on it, to check that stage 1's conclusions carry over to the final student.

Teachers: no teacher (at agency 1 only), bayes4, loop-aware bayes4 (bayes4:0.45,loops=1) and random 10% (loop-aware
bayes4's spend in stage 1). Agencies 0.1, 0.5 and 1; seeds 0-2 (24 students); 30 teacher runs in two queues of 8
processes each.

Comparisons, fixed as in stage 1 (paired by student; primary outcome the end score on the end test's 9-16-move
puzzles, secondary all):
1. bayes4 against no teacher;
2. loop-aware bayes4 against bayes4;
3. loop-aware bayes4 against random 10%.

Runs: models/sessions/<PREFIX>_m10[_s<seed>]_agency<a>/, PREFIX naming the student ("joint", plus "_flat12" for a
constant 12% flat share, "_noopen" without the opening).

    python experiments/final_queue.py           # checks the runs against the spec, then launches final_a and final_b
    python experiments/final_queue.py --check
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import run_queue as rq, sessions as se  # noqa: E402
from rushhour.student import spec  # noqa: E402

AGENCIES = (0.1, 0.5, 1.0)
LOOPS = "bayes4:0.45,loops=1"
MATCHED = "random:0.1"
PROCESSES = 8


def prefix():
    t = spec.TWO_MODE
    return ("joint" + ("_flat12" if t["flat_start"] == t["flat_end"] == 0.12 else "")
            + ("" if t["opening"] else "_noopen"))


def name(seed, a):
    return f"{prefix()}_m10{'' if seed == 0 else f'_s{seed}'}_agency{a:g}"


def runs(seed, agencies=AGENCIES):
    return [(name(seed, a), ["--fitted", "--seed", str(seed), "--shown-weight", str(a), "--processes", str(PROCESSES),
                             "--teachers", *(["none"] if a == 1.0 else []), "bayes4", LOOPS, MATCHED])
            for a in agencies]


def check(queue):
    ok = True
    for run, argv in queue:
        differs = {k: v for k, v in spec.differences(vars(se.parse_args([run, *argv]))).items()
                   if k not in ("seed", spec.AGENCY)}
        if differs:
            print(f"{run}: differs from the spec in {differs}")
            ok = False
    return ok


QUEUES = {"final_a": lambda: runs(0) + runs(2, (0.1, 0.5)), "final_b": lambda: runs(1) + runs(2, (1.0,))}

if __name__ == "__main__":
    queues = {n: f() for n, f in QUEUES.items()}
    good = all(check(q) for q in queues.values())
    for n, q in queues.items():
        print(n, len(q), "runs,", sum(len(vars(se.parse_args([r, *a]))["teachers"]) for r, a in q), "teacher runs:",
              ", ".join(r for r, _ in q))
    if not good:
        raise SystemExit("settings differ from the spec: not launched")
    if "--check" not in sys.argv:
        for n, q in queues.items():
            rq.launch(n, q)
