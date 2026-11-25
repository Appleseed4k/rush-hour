"""Does phase 1's teaching result hold for the two-mode student? Phase 1's x10 agency sweep (aostart_free_m10_agency*,
seeds 0-2: MODEL.md, "Which saved runs use this student") played again with today's fitted student (spec.STUDENT:
the two-mode student on the 2-block, two-network start) and, alongside, phase 1's intuition student on the same start
(--student intuition), which tells the change of start apart from the change of play.

The same teachers per agency as phase 1's sweep (experiments/phase2_queue.py): bayes4 alone at agency 0; no teacher,
bayes4, random 30% and 100% at agency 1; bayes4 and the two randoms in between. Two queues, one per student, run side
by side with 8 processes each (the 8 students of a run in parallel), seed 0 first, then 1 and 2.

    python experiments/replication_queue.py            # checks the runs against the spec, then launches both queues
    python experiments/replication_queue.py --check    # only checks
    python -m rushhour.run_queue watch replicate_two_mode      # progress (or replicate_intuition)

Runs: models/sessions/{twomode,intuition}_m10[_s<seed>]_agency<a>/ (seed 0 has no _s0, as phase 1's runs).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rushhour import run_queue as rq, sessions as se  # noqa: E402
from rushhour.student import spec  # noqa: E402

STUDENTS = {"twomode": ([], {}), "intuition": (["--student", "intuition"], {"intuition": True})}
AGENCY_TEACHERS = {0.0: ["bayes4"], 1.0: ["none", "bayes4", "random:0.3", "random:1.0"]}  # else bayes4 and randoms
PROCESSES = 8


def name(student, seed, a):
    return f"{student}_m10{'' if seed == 0 else f'_s{seed}'}_agency{a:g}"


def runs(student):
    flags = STUDENTS[student][0]
    out = []
    for seed in (0, 1, 2):
        for a in sorted(spec.AGENCIES, key=lambda a: (a != 1.0, a)):  # agency 1 first: it has the no-teacher run
            teachers = AGENCY_TEACHERS.get(a, ["bayes4", "random:0.3", "random:1.0"])
            out.append((name(student, seed, a), ["--fitted", *flags, "--seed", str(seed), "--shown-weight", str(a),
                                                 "--processes", str(PROCESSES), "--teachers", *teachers]))
    return out


def check(student, queue):
    """Every run is the spec's student (or phase 1's, on the same start) apart from seed and agency."""
    ok = True
    for run, argv in queue:
        config = vars(se.parse_args([run, *argv]))
        differs = {k: v[1] for k, v in spec.differences(config, **STUDENTS[student][1]).items()
                   if k not in ("seed", spec.AGENCY)}
        if differs:
            print(f"{run}: differs from the spec in {differs}")
            ok = False
    print(f"{student}: {len(queue)} runs, {sum(len(vars(se.parse_args([r, *a]))['teachers']) for r, a in queue)} "
          f"teacher runs" + ("" if ok else " - SETTINGS DIFFER"))
    return ok


if __name__ == "__main__":
    queues = {s: runs(s) for s in STUDENTS}
    good = all([check(s, q) for s, q in queues.items()])
    if not good:
        raise SystemExit("settings differ from the spec: not launched")
    if "--check" not in sys.argv:
        for s, q in queues.items():
            rq.launch(f"replicate_{'two_mode' if s == 'twomode' else s}", q)
