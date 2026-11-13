"""Simulated learners in Sokoban with and without a teacher, tested on held-out
levels: the counterpart of scripts/session_experiment.py, on Boxoban.

    python sessions.py pretrain                                  # a starting network (models/pretrain/)
    python sessions.py run shortcut --init models/pretrain/puzzle03000.pt --teachers none random:0.1 random:0.3 random:1
    python sessions.py summary shortcut

Every student of a run starts from the same network (--init) and plays the
same session levels in its own random order, one attempt each (student.py:
pushes sampled from its network, restarts on a detected deadlock, the
teacher free to show optimal pushes), learning online after every --block
levels from that block's attempts alone (--steps passes). Student s plays
the same order with the same random numbers under every teacher, so teachers
compare student by student.

Updates keep BatchNorm on the starting network's statistics (--freeze-norm):
with blocks of a few dozen examples, letting them drift made an untaught
student worse, as did --lr 1e-3 (a 1,000-level probe from the same start:
held-out solve rate 0.12 -> 0.06 at 1e-3 with drifting statistics, 0.20 at
1e-4, 0.225 at 1e-4 frozen).

Tests: every --test-every levels, each student's current network plays the
held-out test levels untaught (--test-runs attempts each, the same random
numbers for everyone). A test attempt scores pushes / optimal, an unsolved
one counted at its cap (--step-factor x optimal, at least --min-steps), as
the Rush Hour notebook's score; lower is better. A student's *learning* is
its start score minus its end score.

Levels come from data/boxoban_medium_valid.txt (labelled by label.py, see
README). It is split once, by a fixed shuffle, into a pretraining part (the
first 10,000) and an experiment part, from which each run draws its test
levels, then its session levels.

Saved under models/sessions/<run>/ (gitignored): config.json, initial.pt,
initial_test.json, and per teacher a folder with student<s>.json (its
session record and test attempts) and snapshot student<s>_puzzle<n>.pt
files. A run started again with the same name resumes, skipping finished
students (new teachers are added).
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(HERE, "data", "boxoban_medium_valid.txt")
OUT = os.path.join(HERE, "models", "sessions")
PRETRAIN_OUT = os.path.join(HERE, "models", "pretrain")
N_PRETRAIN = 10_000
# Settings added after runs were saved, and the value those runs had.
ADDED_SETTINGS = {"shown_unsolved": 1, "negative_weight": 0.0, "batch_size": 256, "even_batches": 0,
                  "optimal_labels": 0, "balance": [], "dead": 0, "critic_beta": 0.0}


def _threads():
    import torch
    torch.set_num_threads(1)


# --- levels ----------------------------------------------------------------------------------------

def load_pool(path=POOL):
    """(pretraining part, experiment part): lists of (optimal, board), split by a fixed shuffle."""
    rows = []
    with open(path) as f:
        for line in f:
            pushes, board, _ = line.split(" ", 2)
            if int(pushes) > 0:
                rows.append((int(pushes), board))
    random.Random(12345).shuffle(rows)
    return rows[:N_PRETRAIN], rows[N_PRETRAIN:]


def bins(specs):
    return [tuple(map(int, s.split("-"))) for s in specs]


def sample_by_bins(rows, bin_specs, per_bin, seed, exclude=()):
    """`per_bin` (optimal, board) rows from each optimal-length bin, at random."""
    exclude = set(exclude)
    rng = random.Random(seed)
    out = []
    for lo, hi in bins(bin_specs):
        candidates = [r for r in rows if lo <= r[0] <= hi and r[1] not in exclude]
        if len(candidates) < per_bin:
            raise SystemExit(f"only {len(candidates)} levels in bin {lo}-{hi}, need {per_bin}")
        out += rng.sample(candidates, per_bin)
    return out


def cap(optimal, config):
    return max(config["min_steps"], math.ceil(config["step_factor"] * optimal))


def levels_of(config):
    """(session rows, test rows) for a run."""
    _, rows = load_pool()
    test = sample_by_bins(rows, config["test_bins"], config["test_per_bin"], config["seed"])
    session = sample_by_bins(rows, config["session_bins"], config["per_bin"], config["seed"] + 1,
                             exclude=[b for _, b in test])
    return session, test


def test_points(config, n):
    return sorted(set(range(0, n + 1, config["test_every"])) | {n})


# --- running -----------------------------------------------------------------------------------------

def folder(spec):
    return "none" if spec == "none" else spec.replace(":", "_")


def _test(agent, test_rows, config):
    from level import Level
    from student import test_attempts
    levels = [(opt, Level.from_line(board)) for opt, board in test_rows]
    return test_attempts(agent, levels, [cap(opt, config) for opt, _ in test_rows], config["test_runs"],
                         config["test_temperature"], config["memory"], config["seed"])


def _initial_job(job):
    run_dir, config, test_rows = job
    _threads()
    from student import Agent
    agent = Agent.load(os.path.join(run_dir, "initial.pt"))
    _student_settings(agent, config)
    return _test(agent, test_rows, config)


def _balance_of(config, optimal):
    """learning_session's `balance` from --balance (target shares per session bin), or None."""
    shares = config.get("balance") or []
    if not shares:
        return None
    bins = [tuple(map(int, b.split("-"))) for b in config["session_bins"]]
    if len(shares) != len(bins):
        raise SystemExit(f"--balance: one share per session bin ({len(bins)})")
    target = {i: s / sum(shares) for i, s in enumerate(shares)}
    return (lambda lv: next(i for i, (lo, hi) in enumerate(bins) if lo <= optimal[id(lv)] <= hi)), target


def _student_settings(agent, config):
    """The student's own settings (it uses them in tests too)."""
    agent.dead = bool(config.get("dead", 0))
    agent.critic_beta = float(config.get("critic_beta", 0.0))


def _student_job(job):
    """One student's session under one teacher, tested at the test points."""
    run_dir, config, spec, s, session_rows, test_rows = job
    _threads()
    from level import Level
    from student import Agent, learning_session
    from teach import make_teacher
    agent = Agent.load(os.path.join(run_dir, "initial.pt"), lr=config["lr"])
    agent.freeze_norm = bool(config.get("freeze_norm", 0))
    _student_settings(agent, config)
    order = list(session_rows)
    random.Random(config["seed"] + 1000 + s).shuffle(order)
    levels = [Level.from_line(board) for _, board in order]
    optimal = {id(lv): opt for lv, (opt, _) in zip(levels, order)}
    points = set(test_points(config, len(levels))) - {0}
    tests = {}
    teacher_dir = os.path.join(run_dir, folder(spec))
    t0 = time.perf_counter()

    def on_puzzle(n, agent):
        for p in sorted(points):
            if p <= n and str(p) not in tests:
                agent.save(os.path.join(teacher_dir, f"student{s:02d}_puzzle{p:05d}.pt"))
                tests[str(p)] = _test(agent, test_rows, config)

    record = learning_session(agent, levels, make_teacher(spec), lambda lv: cap(optimal[id(lv)], config),
                              config["memory"], config["explore"], config["labels"],
                              None if config["window"] < 0 else config["window"], config["steps"],
                              config["block"], config["seed"] + 7919 * (s + 1), on_puzzle,
                              shown_unsolved=bool(config.get("shown_unsolved", 1)),
                              negative_weight=config.get("negative_weight", 0.0),
                              batch_size=config.get("batch_size", 256),
                              even_batches=bool(config.get("even_batches", 0)),
                              optimal_labels=bool(config.get("optimal_labels", 0)),
                              balance=_balance_of(config, optimal))
    for entry, (opt, _) in zip(record, order):
        entry["optimal"] = opt
    out = {"record": record, "tests": tests, "seconds": round(time.perf_counter() - t0)}
    with open(os.path.join(teacher_dir, f"student{s:02d}.json"), "w") as f:
        json.dump(out, f)
    return spec, s, out["seconds"]


def run(args):
    config = {k: v for k, v in vars(args).items() if k not in ("command", "processes", "func")}
    run_dir = os.path.join(args.out, args.run)
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        differs = [k for k in config if k not in ("teachers", "out", "init")
                   and saved.get(k, ADDED_SETTINGS.get(k)) != config[k]]
        if differs:
            raise SystemExit(f"{run_dir} exists with other settings ({', '.join(differs)}); use another name")
        saved["teachers"] += [t for t in config["teachers"] if t not in saved["teachers"]]
        config = saved
    else:
        if not config["init"]:
            raise SystemExit("--init: a starting network is required (see `python sessions.py pretrain`)")
        os.makedirs(run_dir)
        from student import Agent
        Agent.load(config["init"]).save(os.path.join(run_dir, "initial.pt"))
    with open(path, "w") as f:
        json.dump(config, f, indent=1)
    session, test = levels_of(config)
    with Pool(args.processes) as pool:
        if not os.path.exists(os.path.join(run_dir, "initial_test.json")):
            jobs = [(run_dir, config, test)]
            initial = pool.apply_async(_initial_job, jobs)
        else:
            initial = None
        jobs = []
        for spec in config["teachers"]:
            os.makedirs(os.path.join(run_dir, folder(spec)), exist_ok=True)
            for s in range(config["students"]):
                if not os.path.exists(os.path.join(run_dir, folder(spec), f"student{s:02d}.json")):
                    jobs.append((run_dir, config, spec, s, session, test))
        print(f"{config['run']}: {len(jobs)} student sessions to play, {len(session)} levels each, "
              f"{len(test)} test levels", file=sys.stderr, flush=True)
        t0 = time.perf_counter()
        for n, (spec, s, seconds) in enumerate(pool.imap_unordered(_student_job, jobs), 1):
            print(f"  {n}/{len(jobs)}  {time.perf_counter() - t0:.0f}s  {spec} student {s} ({seconds}s)",
                  file=sys.stderr, flush=True)
        if initial is not None:
            with open(os.path.join(run_dir, "initial_test.json"), "w") as f:
                json.dump(initial.get(), f)
    summary(args.run, args.out)


# --- pretraining -------------------------------------------------------------------------------------

def pretrain(args):
    """One long learning session from an untrained network, with a teacher,
    on the pretraining levels, tested every --test-every levels on held-out
    pretraining levels; every tested network is saved as puzzle<n>.pt."""
    _threads()
    from level import Level
    from student import Agent, learning_session, score
    from teach import make_teacher
    os.makedirs(args.out, exist_ok=True)
    rows, _ = load_pool()
    config = {"min_steps": args.min_steps, "step_factor": args.step_factor, "test_runs": 1,
              "test_temperature": 1.0, "memory": 10, "seed": args.seed}
    test_rows = sample_by_bins(rows, args.bins, 25, args.seed)
    rest = [r for r in rows if r not in test_rows and any(lo <= r[0] <= hi for lo, hi in bins(args.bins))]
    order = random.Random(args.seed).sample(rest, min(args.puzzles, len(rest)))
    levels = [Level.from_line(board) for _, board in order]
    optimal = {id(lv): opt for lv, (opt, _) in zip(levels, order)}
    agent = Agent(args.hidden, args.blocks, lr=args.lr)
    torch_seed(args.seed)
    log = []
    t0 = time.perf_counter()

    def on_puzzle(n, agent):
        if n % args.test_every == 0 or n == len(levels):
            sc, solved = score(_test(agent, test_rows, config))
            agent.save(os.path.join(args.out, f"puzzle{n:05d}.pt"))
            window = record_so_far[-args.test_every:]
            log.append({"puzzles": n, "test_score": sc, "test_solved": solved,
                        "solved_as_played": statistics.fmean(r["solved"] for r in window) if window else None})
            print(f"{n:6d} puzzles  {time.perf_counter() - t0:5.0f}s  test score {sc:.2f}  solved {solved:.2f}"
                  f"  (as played {log[-1]['solved_as_played']})", flush=True)
            with open(os.path.join(args.out, "log.json"), "w") as f:
                json.dump({"args": vars(args) | {"func": None}, "log": log}, f, indent=1)

    record_so_far = []
    learning_session(agent, levels, make_teacher(args.teacher), lambda lv: cap(optimal[id(lv)], config),
                     labels=args.labels, steps=args.steps, block=args.block, seed=args.seed, on_puzzle=on_puzzle,
                     replay=args.replay, updates=args.updates,
                     on_attempt=lambda i, result: record_so_far.append({"solved": result["solved"]}))


def torch_seed(seed):
    import torch
    torch.manual_seed(seed)


# --- reading runs back -------------------------------------------------------------------------------

def mean_se(xs):
    xs = list(xs)
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def load_run(name, out=OUT):
    """{"config", "initial": test attempts, "teachers": {spec: {student: {"record", "tests"}}}}."""
    run_dir = os.path.join(out, name)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    initial = None
    if os.path.exists(os.path.join(run_dir, "initial_test.json")):
        with open(os.path.join(run_dir, "initial_test.json")) as f:
            initial = json.load(f)
    teachers = {}
    for spec in config["teachers"]:
        d = os.path.join(run_dir, folder(spec))
        students = {}
        for s in range(config["students"]):
            p = os.path.join(d, f"student{s:02d}.json")
            if os.path.exists(p):
                with open(p) as f:
                    students[s] = json.load(f)
        teachers[spec] = students
    return {"config": config, "initial": initial, "teachers": teachers}


def summary(name, out=OUT):
    """Per teacher: moves shown, solved as played, test score along the
    session, learning (start - end score), and end score against every other
    teacher student by student."""
    from student import score
    run = load_run(name, out)
    if run["initial"] is None:
        print("no initial test yet")
        return
    start, start_solved = score(run["initial"])
    print(f"\n{name}: labels {run['config']['labels']}, window {run['config']['window']}, "
          f"{run['config']['students']} students; start score {start:.2f} (solved {start_solved:.2f})")
    ends = {}
    rows = []
    for spec, students in run["teachers"].items():
        if not students:
            continue
        records = [e for st in students.values() for e in st["record"]]
        # help: pushes shown, pushes taken back (undo) and restarts forced (restart), per push played
        shown = (sum(e["taught"] + e.get("undone", 0) + e.get("forced", 0) for e in records)
                 / max(1, sum(e["pushes"] for e in records)))
        points = sorted({int(p) for st in students.values() for p in st["tests"]})
        curve = []
        for p in points:
            per = [score(st["tests"][str(p)]) for st in students.values() if str(p) in st["tests"]]
            curve.append((p, mean_se(x[0] for x in per), mean_se(x[1] for x in per)))
        end = {s: score(st["tests"][str(points[-1])]) for s, st in students.items()}
        ends[spec] = end
        rows.append((spec, len(students), shown, statistics.fmean(e["solved"] for e in records),
                     mean_se(start - e[0] for e in end.values()), mean_se(e[1] for e in end.values()), curve))
    print(f"{'teacher':14s} {'n':>3s} {'help':>6s} {'solved as played':>17s} {'learning':>15s} {'end solved':>15s}")
    for spec, n, shown, played, (lm, ls), (sm, ss), _ in rows:
        print(f"{spec:14s} {n:3d} {shown:6.3f} {played:17.2f} {lm:+8.2f} +- {ls:4.2f} {sm:8.2f} +- {ss:4.2f}")
    print("\ntest score along the session (mean over students; start", f"{start:.2f})")
    for spec, *_, curve in rows:
        print(f"  {spec:14s} " + "  ".join(f"{p}: {m:.2f}" for p, (m, _), _ in curve))
    specs = [r[0] for r in rows]
    if len(specs) > 1:
        print("\nend score, row minus column, paired by student (below 0: the row's students learned more; "
              "count = students better in the row)")
        print(" " * 14 + "".join(f"{c:>22s}" for c in specs))
        for a in specs:
            cells = []
            for b in specs:
                shared = [s for s in ends[a] if s in ends[b]]
                if a == b or not shared:
                    cells.append(f"{'':>22s}")
                    continue
                d = [ends[a][s][0] - ends[b][s][0] for s in shared]
                m, se = mean_se(d)
                cells.append(f"{m:+7.2f} +- {se:4.2f} ({sum(x < 0 for x in d)}/{len(d)})".rjust(22))
            print(f"{a:14s}" + "".join(cells))


# --- command line --------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="play and test a run's teachers")
    r.add_argument("run")
    r.add_argument("--teachers", nargs="+", default=["none", "random:0.1", "random:0.3", "random:1"])
    r.add_argument("--students", type=int, default=8)
    r.add_argument("--init", default=None, help="the starting network (a .pt from pretrain)")
    r.add_argument("--session-bins", nargs="+", default=["8-11", "12-15", "16-19", "20-25"])
    r.add_argument("--per-bin", type=int, default=500, help="session levels per bin")
    r.add_argument("--block", type=int, default=25, help="levels per update (a divisor of --test-every)")
    r.add_argument("--steps", type=int, default=3, help="passes over each update's examples")
    r.add_argument("--batch-size", type=int, default=256,
                   help="examples per Adam step (smaller: more steps when a block has more examples)")
    r.add_argument("--even-batches", type=int, default=0,
                   help="1: split each pass into ceil(n / batch size) equal batches, with no small remainder batch")
    r.add_argument("--balance", type=float, nargs="*", default=[],
                   help="target share of the learning per session bin (e.g. 0.30 0.28 0.23 0.18); examples are "
                        "weighted toward it (student.learning_session's balance)")
    r.add_argument("--dead", type=int, default=0,
                   help="1: the student learns which boards are dead from its detected deadlocks (a dead head) and "
                        "skips pushes onto boards that look dead (student.dead_examples)")
    r.add_argument("--critic-beta", type=float, default=0.0,
                   help="> 0: weight a win's labels by the student's own value head's view of each push, "
                        "min(1, exp(advantage / beta)) (student.critic_weights)")
    r.add_argument("--optimal-labels", type=int, default=0,
                   help="1: keep only the labels whose push the oracle calls optimal (student.optimal_only)")
    r.add_argument("--lr", type=float, default=1e-4,
                   help="1e-4 as in Rush Hour; at 1e-3 an untaught student gets worse over a session")
    r.add_argument("--freeze-norm", type=int, default=1,
                   help="1: BatchNorm keeps the starting network's statistics during updates (student.Agent.train_mode)")
    r.add_argument("--labels", choices=["own", "all"], default="all")
    r.add_argument("--negative-weight", type=float, default=0.0,
                   help="> 0: negative labels for pushes played into a detected deadlock, this much each "
                        "(student.fatal_pushes)")
    r.add_argument("--shown-unsolved", type=int, default=1,
                   help="1: pushes the teacher showed teach in every attempt; 0: only in solved ones")
    r.add_argument("--window", type=int, default=-1, help="hindsight window in pushes (-1: none)")
    r.add_argument("--memory", type=int, default=10)
    r.add_argument("--explore", type=float, default=1.0)
    r.add_argument("--step-factor", type=float, default=5.0)
    r.add_argument("--min-steps", type=int, default=50)
    r.add_argument("--test-every", type=int, default=500)
    r.add_argument("--test-bins", nargs="+", default=None, help="default: the session bins")
    r.add_argument("--test-per-bin", type=int, default=50)
    r.add_argument("--test-runs", type=int, default=3)
    r.add_argument("--test-temperature", type=float, default=1.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--processes", type=int, default=None)
    r.add_argument("--out", default=OUT)
    r.set_defaults(func=run)

    p = sub.add_parser("pretrain", help="train a starting network")
    p.add_argument("--teacher", default="random:0.2")
    p.add_argument("--puzzles", type=int, default=6000)
    p.add_argument("--bins", nargs="+", default=["8-11", "12-15", "16-19", "20-25"])
    p.add_argument("--labels", choices=["own", "all"], default="all")
    p.add_argument("--block", type=int, default=30)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--replay", type=int, default=None,
                   help="replay buffer size in examples (default: none, online updates as in sessions)")
    p.add_argument("--updates", type=int, default=40, help="with --replay: gradient steps per block")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--blocks", type=int, default=3)
    p.add_argument("--step-factor", type=float, default=5.0)
    p.add_argument("--min-steps", type=int, default=50)
    p.add_argument("--test-every", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=PRETRAIN_OUT)
    p.set_defaults(func=pretrain)

    s = sub.add_parser("summary", help="summarize a saved run")
    s.add_argument("run")
    s.add_argument("--out", default=OUT)
    s.set_defaults(func=lambda a: summary(a.run, a.out))

    args = parser.parse_args(argv)
    if args.command == "run":
        args.test_bins = args.test_bins or args.session_bins
        if args.init:
            args.init = os.path.abspath(args.init)
    args.func(args)


if __name__ == "__main__":
    main()
