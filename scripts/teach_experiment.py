"""Teaching experiments, one configuration per run, from the command line.

A pure-intuition student learns by expert iteration (rl.expert_iteration)
with each teacher in turn (teach.py), all starting from the same initial
weights; after training, every round's kept network is tested without a
teacher, behaving as it does while learning (rl.evaluate_intuition). By
default every round brings fresh training puzzles - none repeat an earlier
round's - like new homework; --reuse-puzzles uses one set throughout.

Everything is saved under <out>/<run>/: config.json (this run's settings),
and per teacher a folder with the kept checkpoints, manifest.json,
history.json (rl.expert_iteration's per-round record) and results.json (test
attempts per round). load_run() reads a run back; summarize() turns test
attempts into solve rate and moves / optimal with standard errors.

Examples (from scripts/):
    python teach_experiment.py fresh_p0.1 --teachers random:0.1 correct-after:0.1 flag:0.1
    python teach_experiment.py fresh_p0.1 --rounds 10 --resume      # continue to 10 rounds
    python teach_experiment.py hard --train-bins 1-8 9-12 13-16 17-20 21-30 --test-bins 21-30 31-40
    python teach_experiment.py sparse25 --train-per-bin 25
    python teach_experiment.py fresh_p0.1 --test-only --test-runs 5  # re-test with other settings
"""
import argparse
import json
import os
import random
import shutil
import statistics

import torch
from tqdm import tqdm

import rl
import teach

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
POOL = os.path.join(ROOT, "data", "rush_nw.txt")
OUT = os.path.join(ROOT, "models", "teach")
TEACHERS = {"random": teach.Random, "correct-after": teach.CorrectAfterMistake, "flag": teach.FlagMistakes,
            "correct-before": teach.CorrectMistakes}
# Settings a resumed run may change; anything else must match the saved run.
RESUMABLE = {"rounds", "device", "resume", "test_only", "test_runs", "test_temperature", "out", "overwrite"}
# Settings that change what testing measures; results are recomputed when they change.
TEST_SETTINGS = ("test_runs", "test_temperature", "memory", "step_factor", "min_steps", "max_steps", "seed")


# --- configuration -----------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="run name: results go to <out>/<run>/")
    ap.add_argument("--teachers", nargs="+", default=["random:0.1", "correct-after:0.1", "flag:0.1"],
                    help=f"kind:probability for each teacher; kinds: {', '.join(TEACHERS)} (probability 1 if left out)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--train-bins", nargs="+", default=["1-8", "9-12", "13-16", "17-20"],
                    help="optimal-length bins training puzzles are drawn from, evenly")
    ap.add_argument("--train-per-bin", type=int, default=100, help="training puzzles per bin per round")
    ap.add_argument("--reuse-puzzles", action="store_true",
                    help="use the same training puzzles every round instead of fresh ones")
    ap.add_argument("--val-per-bin", type=int, default=10, help="validation puzzles per training bin")
    ap.add_argument("--test-bins", nargs="+", default=None, help="test bins (default: the training bins)")
    ap.add_argument("--test-per-bin", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=2, help="attempts per training and validation puzzle per round")
    ap.add_argument("--explore", type=float, default=1.0, help="the student's sampling temperature while learning")
    ap.add_argument("--memory", type=int, default=10, help="recent boards the student avoids returning to")
    ap.add_argument("--step-factor", type=float, default=10.0,
                    help="an attempt may take step-factor x the puzzle's optimal length in moves ...")
    ap.add_argument("--min-steps", type=int, default=50, help="... but at least this many")
    ap.add_argument("--max-steps", type=int, default=None, help="a fixed move cap instead")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--flag-penalty", type=float, default=1.0)
    ap.add_argument("--test-temperature", type=float, default=1.0, help="0 = always the top move")
    ap.add_argument("--test-runs", type=int, default=3, help="test attempts per puzzle")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available()
                    else ("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", default=OUT, help="where runs are saved")
    ap.add_argument("--resume", action="store_true", help="continue the saved run to --rounds rounds")
    ap.add_argument("--test-only", action="store_true", help="only test rounds that haven't been tested yet")
    ap.add_argument("--overwrite", action="store_true", help="delete the saved run, if any, and start over")
    args = ap.parse_args(argv)
    args.test_bins = args.test_bins or args.train_bins
    return args


def make_teacher(spec):
    kind, _, p = spec.partition(":")
    return TEACHERS[kind](float(p) if p else 1.0)


def bins(specs):
    return [tuple(int(x) for x in spec.split("-")) for spec in specs]


def folder(teacher):
    return teacher.name.replace(" ", "_")


def step_cap(distance, config):
    if config["max_steps"] is not None:
        return config["max_steps"]
    return max(config["min_steps"], int(round(config["step_factor"] * distance)))


def sample_puzzles(config):
    """(per-round training puzzles, validation, test) as (distance, state)
    lists. Test first, then validation, then each round's training puzzles,
    each excluding everything drawn before - so round r's puzzles don't
    depend on how many rounds there are."""
    pool = rl.load_puzzle_pool(POOL)
    seed = config["seed"]
    test = rl.sample_by_distance(pool, bins(config["test_bins"]), config["test_per_bin"], seed=seed)
    val = rl.sample_by_distance(pool, bins(config["train_bins"]), config["val_per_bin"], seed=seed + 1,
                                exclude=[s for _, s in test])
    used = {s for _, s in test + val}
    rounds = []
    for r in range(1 if config["reuse_puzzles"] else config["rounds"]):
        batch = rl.sample_by_distance(pool, bins(config["train_bins"]), config["train_per_bin"],
                                      seed=seed + 100 + r, exclude=used)
        used |= {s for _, s in batch}
        rounds.append(batch)
    if config["reuse_puzzles"]:
        rounds *= config["rounds"]
    return rounds, val, test


# --- running -----------------------------------------------------------------------------------

def check_config(run_dir, config):
    """Saves config.json for a new run; for an existing one, makes sure only
    resumable settings changed (and returns the saved config with those
    updated)."""
    path = os.path.join(run_dir, "config.json")
    if config["overwrite"] and os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    if not os.path.exists(path):
        if config["resume"] or config["test_only"]:
            raise SystemExit(f"no saved run in {run_dir}")
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config, f, indent=1)
        return config
    if not (config["resume"] or config["test_only"]):
        raise SystemExit(f"{run_dir} exists: use --resume, --test-only or --overwrite")
    with open(path) as f:
        saved = json.load(f)
    changed = {k for k in config if k not in RESUMABLE and saved.get(k) != config[k]}
    if config["test_only"]:
        changed = set()  # testing reuses the saved run's settings, except the test settings
    if changed:
        raise SystemExit(f"settings differ from the saved run: {sorted(changed)} (only {sorted(RESUMABLE)} may change)")
    merged = {**saved, **{k: config[k] for k in RESUMABLE if k in config}}
    if config["test_only"]:
        merged["rounds"] = saved["rounds"]
    with open(path, "w") as f:
        json.dump(merged, f, indent=1)
    return merged


def train(teacher, config, rounds, val, caps, run_dir):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    agent = rl.PolicyAgent(device=config["device"])
    return rl.expert_iteration(
        agent, [[s for _, s in batch] for batch in rounds], [s for _, s in val], rounds=config["rounds"],
        n_attempts=config["attempts"], max_steps=caps.__getitem__, seed=config["seed"],
        checkpoint_dir=os.path.join(run_dir, folder(teacher)), checkpoint_epochs=False, student="intuition",
        teacher=teacher, explore=config["explore"], memory=config["memory"], resume=config["resume"],
        epochs=config["epochs"], patience=config["patience"], flag_penalty=config["flag_penalty"])


def test(teacher, config, test_puzzles, caps, run_dir):
    """Tests every kept checkpoint not yet tested under the current test
    settings; saves results.json ({"settings", "rounds": {round: attempts}})."""
    teacher_dir = os.path.join(run_dir, folder(teacher))
    path = os.path.join(teacher_dir, "results.json")
    settings = {k: config[k] for k in TEST_SETTINGS}
    results = {"settings": settings, "rounds": {}}
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved["settings"] == settings:
            results = saved
    with open(os.path.join(teacher_dir, "manifest.json")) as f:
        kept = sorted((m for m in json.load(f) if m["kept"]), key=lambda m: m["round"])
    for m in tqdm([m for m in kept if str(m["round"]) not in results["rounds"]], desc=f"testing {teacher.name}"):
        agent = rl.PolicyAgent.load(os.path.join(teacher_dir, m["file"]))
        results["rounds"][str(m["round"])] = rl.evaluate_intuition(
            agent, test_puzzles, n_runs=config["test_runs"], temperature=config["test_temperature"],
            memory=config["memory"], max_steps=caps.__getitem__, seed=config["seed"])
        with open(path, "w") as f:
            json.dump(results, f)
    return results


def main(argv=None):
    config = vars(parse_args(argv))
    run_dir = os.path.join(config["out"], config["run"])
    config = check_config(run_dir, config)
    rounds, val, test_puzzles = sample_puzzles(config)
    caps = {s: step_cap(d, config) for batch in rounds + [val, test_puzzles] for d, s in batch}
    for spec in config["teachers"]:
        teacher = make_teacher(spec)
        print(f"=== {config['run']}: {teacher.name} ===", flush=True)
        if not config["test_only"]:
            train(teacher, config, rounds, val, caps, run_dir)
        test(teacher, config, test_puzzles, caps, run_dir)
    print(f"saved to {run_dir}")


# --- reading runs back ---------------------------------------------------------------------------

def load_run(run, out=OUT):
    """{"config", "teachers": {name: {"history", "results": {round: attempts}}}} for a saved run, teachers in
    the order they were given."""
    run_dir = os.path.join(out, run)
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    teachers = {}
    for spec in config["teachers"]:
        teacher = make_teacher(spec)
        teacher_dir = os.path.join(run_dir, folder(teacher))
        entry = {"history": [], "results": {}}
        for key, name in (("history", "history.json"), ("results", "results.json")):
            path = os.path.join(teacher_dir, name)
            if os.path.exists(path):
                with open(path) as f:
                    entry[key] = json.load(f)
        if entry["results"]:
            entry["results"] = {int(r): runs for r, runs in entry["results"]["rounds"].items()}
        teachers[teacher.name] = entry
    return {"config": config, "teachers": teachers}


def mean_se(values):
    """(mean, standard error) of the non-nan values."""
    values = [v for v in values if v == v]
    if not values:
        return float("nan"), float("nan")
    se = statistics.stdev(values) / len(values) ** 0.5 if len(values) > 1 else float("nan")
    return statistics.mean(values), se


def summarize(runs, n_runs, distance_bins=()):
    """Solve rate and moves / optimal as (mean, standard error across
    puzzles), overall and per (lo, hi) distance bin, from test attempts that
    come puzzle by puzzle, `n_runs` each - averaged per puzzle first."""
    groups = [runs[i:i + n_runs] for i in range(0, len(runs), n_runs)]
    solve = [(g[0]["distance"], statistics.mean(float(r["solved"]) for r in g)) for g in groups]
    ratio = [statistics.mean(r["moves"] / r["distance"] for r in g if r["solved"])
             for g in groups if any(r["solved"] for r in g)]
    out = {"solve rate": mean_se([v for _, v in solve]), "moves / optimal": mean_se(ratio)}
    for lo, hi in distance_bins:
        out[(lo, hi)] = mean_se([v for d, v in solve if lo <= d <= hi])
    return out


if __name__ == "__main__":
    main()
