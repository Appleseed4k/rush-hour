"""Simulated participants: students that learn online during one session,
under human-like limits on learning (rl.learning_session), with and without
a teacher.

Like the human experiment (human_data.py), every student gets the same set
of puzzles - by default 18 at each of the human puzzles' optimal lengths (6,
10, 13, 15, counted to the red car's exit) - in its own random order, one puzzle at a time. After each
puzzle it learns from that puzzle alone, once: no replay of earlier
puzzles, no validation, no rewinding. Each move label, move shown and
mistake pointed out is learned only with probability --encode, and loops
are cut out of a solved attempt only if they return to a board from the
last --window moves. All students of a run start from the same network
(--init: a saved checkpoint, e.g. one pretrained to human starting
performance; default: untrained).

--surrender lets the students give up on a puzzle as people do: after each
move, with a probability fitted to the human data by how many moves they
have made relative to the optimal length (human_data.surrender_hazard; the
fitted rates are saved in config.json). Giving up ends the puzzle, unsolved.
--test-surrender does the same in the held-out tests. --surrender-scale
multiplies the rates (fit_surrender.py picks a checkpoint and scale that
match people's starting level).

--multiplier m relaxes how much data the network gets: each human trial
stands for m simulated puzzles (fresh ones - the session has m times as
many), learned from together in one update by default (--block). Networks
learn far less from each example than people, so m is a free parameter to
calibrate: the m (with --lr) at which the students' learning curve over
human trials matches the humans'.

Two measures: the session itself (each puzzle's outcome as it is played,
with the teacher's help if any - the analogue of the human learning curve)
and tests along the way (every --test-every puzzles, each student's
current network on held-out puzzles, untaught; the student doesn't learn
from them).

Saved under <out>/<run>/: config.json, initial.pt, and per teacher a folder
with sessions.json (per student, one record per puzzle), snapshots
student{s}_puzzle{n}.pt (n = simulated puzzles played) and results.json (test attempts); progress/ says how
far a run under way has got (progress(); run_queue.py turns it into a progress bar). Running a
saved run again with the same settings adds the teachers it doesn't have yet (and finishes any that
stopped before saving their sessions), leaving the others as they are. load_session()
reads a run back; session_curve() and test_curve() summarize it, with
standard errors across students.

Examples (from scripts/):
    python session_experiment.py base --init ../models/teach/fresh_p0.1/random_0.1/round1_epoch06.pt
    python session_experiment.py lapses --init ... --encode 0.3 --teachers none correct-after:0.1
    python session_experiment.py base --test-only --test-runs 5
    python session_experiment.py base --teachers bayes:0.1 --init ...   # base's settings: adds a teacher
"""
import argparse
import json
import os
import random
import shutil
import statistics
import time

# One BLAS thread per process: the students run in parallel workers (one per core), and numpy's OpenBLAS otherwise
# starts a thread per core in every one of them - a Bayesian teacher's refit after each puzzle then ran ~30x slower.
# Set before numpy is first imported (spawned workers inherit the environment).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import torch
from multiprocessing import Pool
from tqdm import tqdm

import rl
import teach_experiment as te

OUT = os.path.join(te.ROOT, "models", "sessions")
TEST_SETTINGS = ("test_runs", "test_temperature", "memory", "step_factor", "min_steps", "max_steps", "seed",
                 "test_surrender", "surrender_scale")
# Settings a saved run may be re-run with: it then adds the teachers it doesn't have yet (and finishes any
# that stopped early), skipping the ones already played. (flag_penalty: retired with the flag teacher; runs
# saved before still have it.)
SAME_RUN_IGNORES = {"teachers", "processes", "out", "test_only", "overwrite", "surrender_table", "flag_penalty"}
# Settings added after runs were saved, and what those runs did: a saved run without one counts as having its default.
ADDED_SETTINGS = {"shown_weight": 1.0, "progress_weight": 0.0, "progress_tail": 0.2, "labels": "own",
                  "agency_step": "off", "train_device": "cpu"}


# --- configuration -----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name: results go to <out>/<run>/")
    ap.add_argument("--teachers", nargs="+", default=["none", "random:0.1", "correct-after:0.1"],
                    help=f"none, or kind:probability; kinds: {', '.join(te.TEACHERS)}")
    ap.add_argument("--students", type=int, default=10, help="simulated participants per teacher")
    ap.add_argument("--session-bins", nargs="+", default=["6-6", "10-10", "13-13", "15-15"],
                    help="optimal-length bins the session's puzzles are drawn from, evenly (default: the human "
                         "puzzles' lengths, counted as this repo does - see human_data.distance)")
    ap.add_argument("--per-bin", type=int, default=18, help="session puzzles per bin")
    ap.add_argument("--max-attempts", type=int, default=1, help="attempts per puzzle (restarts), until solved")
    ap.add_argument("--explore", type=float, default=1.0, help="the student's sampling temperature")
    ap.add_argument("--memory", type=int, default=10, help="recent boards the student avoids returning to")
    ap.add_argument("--window", type=int, default=10,
                    help="hindsight: loops are cut only if they return to a board from the last WINDOW moves "
                         "(-1: any board)")
    ap.add_argument("--encode", type=float, default=1.0,
                    help="probability that each label, move shown or mistake pointed out is learned")
    ap.add_argument("--steps", type=int, default=1, help="passes over each update's examples")
    ap.add_argument("--multiplier", type=int, default=1,
                    help="simulated puzzles per human trial: the network gets this many times a person's "
                         "experience (the session has per-bin x multiplier puzzles per bin)")
    ap.add_argument("--block", type=int, default=None,
                    help="puzzles per update (default: the multiplier - one update per human trial)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--shown-weight", type=float, default=1.0,
                    help="agency: how much what a shown move teaches counts, the student's own play 1 "
                         "(rl.agency_weights; 0: nothing learned from being shown)")
    ap.add_argument("--agency-step", choices=["off", "scale", "skip"], default="off",
                    help="how agency shrinks the update (rl.online_update): off = it doesn't (Adam cancels a weight "
                         "shared by the whole batch); scale = each step scaled by the batch's mean weight; skip = "
                         "each step taken with that probability")
    ap.add_argument("--train-device", default="cpu",
                    help="where the students' updates run (rl.PolicyAgent.train_on), e.g. mps: all the workers "
                         "share the GPU; play and tests stay on the CPU. The GPU's arithmetic differs slightly, "
                         "so compare runs made on the same device")
    ap.add_argument("--progress-weight", type=float, default=0.0,
                    help="progress labels: the student also learns from its own moves by its starting network's "
                         "sense of progress, each label weighing this (rl.progress_examples; 0: off)")
    ap.add_argument("--progress-tail", type=float, default=0.2,
                    help="share of its recent moves labelled each way (steps forward, steps back)")
    ap.add_argument("--labels", choices=rl.LABELS, default="own",
                    help="what a solved attempt teaches: own = the path played, loops cut out (--window); path / all = "
                         "shortcut labels, shortest routes through the boards it visited (rl.shortcut_routes)")
    ap.add_argument("--surrender", action="store_true",
                    help="students may give up on a puzzle as people do (a rule fitted to the human data, "
                         "human_data.surrender_hazard); the move cap still applies")
    ap.add_argument("--test-surrender", action="store_true", help="... in the held-out tests too")
    ap.add_argument("--surrender-scale", type=float, default=1.0,
                    help="multiplies the surrender rates (training and tests)")
    ap.add_argument("--init", default=None, help="checkpoint every student starts from (default: untrained)")
    ap.add_argument("--step-factor", type=float, default=10.0,
                    help="an attempt may take step-factor x the puzzle's optimal length in moves ...")
    ap.add_argument("--min-steps", type=int, default=50, help="... but at least this many")
    ap.add_argument("--max-steps", type=int, default=None, help="a fixed move cap instead")
    ap.add_argument("--test-every", type=int, default=18, help="test the students every this many human trials")
    ap.add_argument("--test-bins", nargs="+", default=None, help="test bins (default: the session bins)")
    ap.add_argument("--test-per-bin", type=int, default=20)
    ap.add_argument("--test-runs", type=int, default=3, help="test attempts per puzzle")
    ap.add_argument("--test-temperature", type=float, default=1.0, help="0 = always the top move")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--processes", type=int, default=None, help="students run in parallel (default: all cores)")
    ap.add_argument("--out", default=OUT, help="where runs are saved")
    ap.add_argument("--test-only", action="store_true", help="only (re)test a saved run")
    ap.add_argument("--overwrite", action="store_true", help="delete the saved run, if any, and start over")
    args = ap.parse_args(argv)
    args.test_bins = args.test_bins or args.session_bins
    args.block = args.block or args.multiplier
    if args.init is not None and not args.test_only:
        args.init = os.path.abspath(args.init)  # relative to where the command was run
        if not os.path.exists(args.init):
            ap.error(f"--init: no checkpoint at {args.init}")
    return args


def make_teacher(spec):
    return None if spec == "none" else te.make_teacher(spec)


def teacher_name(spec):
    return "no teacher" if spec == "none" else te.teacher_name(spec)


def folder(spec):
    return teacher_name(spec).replace(" ", "_")


def sample_puzzles(config):
    """(session puzzles, test puzzles) as (distance, state) lists - test first,
    the session's excluding them."""
    pool = rl.load_puzzle_pool(te.POOL)
    test = rl.sample_by_distance(pool, te.bins(config["test_bins"]), config["test_per_bin"], seed=config["seed"])
    session = rl.sample_by_distance(pool, te.bins(config["session_bins"]), config["per_bin"] * multiplier(config),
                                    seed=config["seed"] + 1, exclude=[s for _, s in test])
    return session, test


def surrender_rule(config, puzzles):
    """The run's SurrenderRule (its fitted rates are saved in config.json) for (distance, state) `puzzles`."""
    edges, rates = config["surrender_table"]
    return rl.SurrenderRule(edges, rates, {s: d for d, s in puzzles}, config.get("surrender_scale", 1.0))


def multiplier(config):
    """Simulated puzzles per human trial (1 for runs saved before there was a multiplier)."""
    return config.get("multiplier", 1)


def order(session, student, seed):
    """Student `student`'s own random order of the session's puzzles, as
    (index in the session, distance, state)."""
    puzzles = [(i, d, s) for i, (d, s) in enumerate(session)]
    random.Random(seed + 1000 + student).shuffle(puzzles)
    return puzzles


def test_points(config, n_puzzles):
    """Puzzle counts after which the students are tested (0 = before the session): every --test-every
    human trials."""
    return sorted(set(range(0, n_puzzles + 1, config["test_every"] * multiplier(config))) | {0, n_puzzles})


def _same_key(setting, value):
    """A setting's value as check_config compares it: --init by its path inside the repo, so a run saved on one
    machine (an absolute path there) can be added to on another."""
    if setting != "init" or value is None:
        return value
    path, repo = value.replace("\\", "/"), "/" + os.path.basename(os.path.abspath(te.ROOT)) + "/"
    return path.split(repo, 1)[1] if repo in path else path


def check_config(run_dir, config):
    """Saves config.json for a new run; for --test-only, returns the saved
    config with the test settings updated."""
    path = os.path.join(run_dir, "config.json")
    if config["overwrite"] and os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    if config["test_only"]:
        if not os.path.exists(path):
            raise SystemExit(f"no saved run in {run_dir}")
        with open(path) as f:
            saved = json.load(f)
        saved.update({k: config[k] for k in ("test_runs", "test_temperature", "test_surrender", "processes",
                                             "test_only")})
        with open(path, "w") as f:
            json.dump(saved, f, indent=1)
        return saved
    if os.path.exists(path) and os.path.exists(os.path.join(run_dir, "initial.pt")):
        with open(path) as f:
            saved = json.load(f)
        differs = sorted(k for k in set(saved) | set(config) if k not in SAME_RUN_IGNORES
                         and _same_key(k, saved.get(k, ADDED_SETTINGS.get(k)))
                         != _same_key(k, config.get(k, ADDED_SETTINGS.get(k))))
        if differs:
            raise SystemExit(f"{run_dir} exists with other settings ({', '.join(differs)}): use another run name, "
                             f"--test-only or --overwrite")
        saved["teachers"] += [t for t in config["teachers"] if t not in saved["teachers"]]
        saved.update({k: config[k] for k in ("processes", "test_only")})
        with open(path, "w") as f:
            json.dump(saved, f, indent=1)
        return saved
    # (a config.json without initial.pt is a run that stopped before any student started: start it afresh)
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=1)
    return config


# --- running -----------------------------------------------------------------------------------

def write_progress(run_dir, name, **values):
    """Records how far a run has got in <run_dir>/progress/<name>.json (with the time), for progress() to read
    while the run goes on: per teacher, "<folder>_sessions" (start), "<folder>_student<s>" (puzzles done, of
    total) and "<folder>_tests" (attempts done, of total)."""
    d = os.path.join(run_dir, "progress")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(values | {"time": time.time()}, f)
    os.replace(path + ".tmp", path)


def progress(run, teachers, out=OUT):
    """How far `run` has got with each of `teachers` (specs): {spec: (share of its sessions played, share of its
    test attempts done)} - 1 for what is saved, from the progress files while a step is under way, else 0."""
    run_dir = os.path.join(out, run)

    def read(name):
        try:
            with open(os.path.join(run_dir, "progress", f"{name}.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    shares = {}
    for spec in teachers:
        teacher_dir = os.path.join(run_dir, folder(spec))
        if os.path.exists(os.path.join(teacher_dir, "sessions.json")):
            played = 1.0
        else:
            start = read(f"{folder(spec)}_sessions")
            students = [read(f"{folder(spec)}_student{s}") for s in range(start["students"])] if start else []
            played = sum(p["done"] for p in students if p) / (start["students"] * start["puzzles"]) if start else 0.0
        tests = read(f"{folder(spec)}_tests")
        if tests is not None and tests["done"] < tests["total"]:
            tested = tests["done"] / tests["total"]
        else:
            tested = float(played == 1.0 and os.path.exists(os.path.join(teacher_dir, "results.json")))
        shares[spec] = (played, tested)
    return shares


def _student(job):
    """One simulated participant's session (in a worker, on the CPU): returns
    its per-puzzle record, saving snapshots at the test points."""
    student, spec, config, puzzles, caps, run_dir = job
    torch.set_num_threads(1)
    agent = rl.PolicyAgent.load(os.path.join(run_dir, "initial.pt"), lr=config["lr"])
    if config.get("train_device", "cpu") != "cpu":
        agent.train_on(config["train_device"])
    progress_weight = config.get("progress_weight", 0.0)
    judge = rl.PolicyAgent.load(os.path.join(run_dir, "initial.pt")) if progress_weight > 0 else None
    teacher_dir = os.path.join(run_dir, folder(spec))
    points = test_points(config, len(puzzles))
    saved = [0]

    def snapshot(n, agent):
        """After an update at `n` puzzles played: the snapshots for test points reached since the last one."""
        for p in points:
            if saved[0] < p <= n:
                agent.save(os.path.join(teacher_dir, f"student{student:02d}_puzzle{p:03d}.pt"))
        saved[0] = n
        write_progress(run_dir, f"{folder(spec)}_student{student}", done=n, total=len(puzzles))

    record = rl.learning_session(
        agent, [s for _, _, s in puzzles], make_teacher(spec), max_attempts=config["max_attempts"],
        surrender=surrender_rule(config, [(d, s) for _, d, s in puzzles]) if config.get("surrender") else None,
        max_steps=caps.__getitem__, memory=config["memory"], explore=config["explore"],
        window=None if config["window"] < 0 else config["window"], encode=config["encode"],
        steps=config["steps"], block=config.get("block", 1),
        seed=config["seed"] + 7919 * (student + 1),
        on_puzzle=snapshot, shown_weight=config.get("shown_weight", 1.0), judge=judge,
        progress_weight=progress_weight, progress_tail=config.get("progress_tail", 0.2),
        labels=config.get("labels", "own"), agency_step=config.get("agency_step", "off"))
    m = multiplier(config)
    for entry, (i, d, _) in zip(record, puzzles):  # trial: the human trial this puzzle counts toward
        entry.update(step=entry["puzzle"] + 1, trial=entry["puzzle"] // m + 1, puzzle=i, distance=d)
    return student, record


def run_sessions(spec, config, session, caps, run_dir):
    teacher_dir = os.path.join(run_dir, folder(spec))
    os.makedirs(teacher_dir, exist_ok=True)
    jobs = [(s, spec, config, order(session, s, config["seed"]), caps, run_dir) for s in range(config["students"])]
    write_progress(run_dir, f"{folder(spec)}_sessions", students=config["students"], puzzles=len(session))
    for s in range(config["students"]):
        write_progress(run_dir, f"{folder(spec)}_student{s}", done=0, total=len(session))
    sessions = {}
    with Pool(config["processes"]) as pool:
        for student, record in tqdm(pool.imap_unordered(_student, jobs), total=len(jobs),
                                    desc=f"sessions ({teacher_name(spec)})"):
            sessions[student] = record
    with open(os.path.join(teacher_dir, "sessions.json"), "w") as f:
        json.dump({str(s): sessions[s] for s in sorted(sessions)}, f)
    return sessions


def test(spec, config, test_puzzles, caps, run_dir, n_puzzles):
    """Tests the initial network and every student's snapshots not yet tested
    under the current test settings; saves results.json ({"settings",
    "students": {student: {puzzles played: attempts}}}, puzzles played 0 =
    the initial network, shared by all)."""
    teacher_dir = os.path.join(run_dir, folder(spec))
    path = os.path.join(teacher_dir, "results.json")
    settings = {k: config[k] for k in TEST_SETTINGS if k in config}  # (runs saved before a setting existed lack it)
    results = {"settings": settings, "students": {}}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved["settings"] == settings:
            results = saved
    todo = [(s, n) for s in range(config["students"]) for n in test_points(config, n_puzzles)
            if str(n) not in results["students"].get(str(s), {})]
    files = sorted({_snapshot(run_dir, spec, s, n) for s, n in todo})
    n_runs = config["test_runs"]
    # The same attempts as rl.evaluate_intuition, for every snapshot in one pool.
    rule = surrender_rule(config, test_puzzles) if config.get("test_surrender") else None
    jobs = [(file, state, d, config["memory"], caps[state], config["test_temperature"],
             hash((config["seed"], i, r)) & 0xFFFFFFFF, rule)
            for file in files for i, (d, state) in enumerate(test_puzzles) for r in range(n_runs)]
    per_file = len(test_puzzles) * n_runs
    write_progress(run_dir, f"{folder(spec)}_tests", done=0, total=len(jobs))
    attempts = []
    with Pool(config["processes"]) as pool:
        for attempt in tqdm(pool.imap(_test_job, jobs, chunksize=8), total=len(jobs),
                            desc=f"testing ({teacher_name(spec)})"):
            attempts.append(attempt)
            if len(attempts) % 200 == 0:
                write_progress(run_dir, f"{folder(spec)}_tests", done=len(attempts), total=len(jobs))
    by_file = {file: attempts[k * per_file:(k + 1) * per_file] for k, file in enumerate(files)}
    for s, n in todo:
        results["students"].setdefault(str(s), {})[str(n)] = by_file[_snapshot(run_dir, spec, s, n)]
    with open(path, "w") as f:
        json.dump(results, f)
    write_progress(run_dir, f"{folder(spec)}_tests", done=len(jobs), total=len(jobs))
    return results


def _snapshot(run_dir, spec, student, n):
    """Checkpoint of `student` after `n` puzzles (0: the shared initial network)."""
    if n == 0:
        return os.path.join(run_dir, "initial.pt")
    return os.path.join(run_dir, folder(spec), f"student{student:02d}_puzzle{n:03d}.pt")


_AGENT = {}


def _test_job(job):
    """One untaught test attempt by a snapshot (loaded once per worker while
    consecutive jobs share it) - as rl.evaluate_intuition's attempts."""
    file, state, distance, memory, cap, temperature, seed, *rest = job
    rule = rest[0] if rest else None
    if file not in _AGENT:
        torch.set_num_threads(1)
        _AGENT.clear()
        _AGENT[file] = rl.PolicyAgent.load(file)
    result = rl.solve_intuition(_AGENT[file], state, cap, memory, temperature, rng=random.Random(seed),
                                surrender=rule.for_puzzle(state) if rule is not None else None)
    return {"solved": result["solved"], "moves": len(result["moves"]) if result["solved"] else None,
            "distance": distance}


def main(argv=None):
    config = vars(parse_args(argv))
    run_dir = os.path.join(config["out"], config["run"])
    config = check_config(run_dir, config)
    session, test_puzzles = sample_puzzles(config)
    caps = {s: te.step_cap(d, config) for d, s in session + test_puzzles}
    if (config.get("surrender") or config.get("test_surrender")) and "surrender_table" not in config:
        import human_data
        edges, rates, _ = human_data.surrender_hazard()
        config["surrender_table"] = [edges, rates]
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=1)
    if not config["test_only"] and not os.path.exists(os.path.join(run_dir, "initial.pt")):
        random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        agent = rl.PolicyAgent.load(config["init"]) if config["init"] else rl.PolicyAgent()
        agent.save(os.path.join(run_dir, "initial.pt"))
    for spec in config["teachers"]:
        print(f"=== {config['run']}: {teacher_name(spec)} ===", flush=True)
        if not config["test_only"] and not os.path.exists(os.path.join(run_dir, folder(spec), "sessions.json")):
            run_sessions(spec, config, session, caps, run_dir)
        test(spec, config, test_puzzles, caps, run_dir, len(session))
    print(f"saved to {run_dir}")


# --- reading runs back ---------------------------------------------------------------------------

def load_session(run, out=OUT):
    """{"config", "teachers": {name: {"sessions": {student: record}, "results": {student: {puzzles played:
    attempts}}}}} for a saved run, teachers in the order they were given."""
    run_dir = os.path.join(out, run)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    teachers = {}
    for spec in config["teachers"]:
        teacher_dir = os.path.join(run_dir, folder(spec))
        entry = {"sessions": {}, "results": {}}
        path = os.path.join(teacher_dir, "sessions.json")
        if os.path.exists(path):
            with open(path) as f:
                entry["sessions"] = {int(s): record for s, record in json.load(f).items()}
        path = os.path.join(teacher_dir, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                entry["results"] = {int(s): {int(n): runs for n, runs in by_n.items()}
                                    for s, by_n in json.load(f)["students"].items()}
        teachers[teacher_name(spec)] = entry
    return {"config": config, "teachers": teachers}


def session_curve(sessions, block=10):
    """The session as it was played, in blocks of `block` human trials (with
    a --multiplier, each trial is several simulated puzzles): per block,
    (first trial, solve rate, moves / optimal over solved puzzles,
    interventions per puzzle), each a (mean, standard error) across
    students."""
    n = max(e["trial"] for record in sessions.values() for e in record)
    out = []
    for start in range(1, n + 1, block):
        chunks = [[e for e in record if start <= e["trial"] < start + block] for record in sessions.values()]
        chunks = [c for c in chunks if c]
        solve = [statistics.mean(float(e["solved"]) for e in c) for c in chunks]
        ratio = [statistics.mean(e["moves"] / e["distance"] for e in c if e["solved"])
                 for c in chunks if any(e["solved"] for e in c)]
        help_ = [statistics.mean(e["interventions"] for e in c) for c in chunks]
        out.append((start, te.mean_se(solve), te.mean_se(ratio), te.mean_se(help_)))
    return out


def test_curve(results, n_runs, distance_bins=()):
    """Tests along the session: per number of puzzles played, the solve rate
    and moves / optimal (and solve rate per distance bin), each a (mean,
    standard error) across students of the student's own average over the
    test puzzles."""
    points = sorted({n for by_n in results.values() for n in by_n})
    out = []
    for n in points:
        summaries = [te.summarize(by_n[n], n_runs, distance_bins) for by_n in results.values() if n in by_n]
        row = {key: te.mean_se([s[key][0] for s in summaries]) for key in summaries[0]}
        out.append((n, row))
    return out


if __name__ == "__main__":
    main()
