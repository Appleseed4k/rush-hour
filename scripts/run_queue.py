"""A queue of session_experiment.py runs played one after another in a background process that outlives the
notebook (or shell) that started it, and a progress bar with the time left.

From a notebook:
    import run_queue as rq
    rq.launch("teacher_sweep", [("run_a", ["--teachers", "none", "--lr", "1e-4"]), ("run_b", [...])])
    rq.watch("teacher_sweep")      # live bar; stopping the cell stops only the watching
    rq.stop("teacher_sweep")       # stops it, workers included
From a shell (scripts/):
    python run_queue.py watch teacher_sweep     (or: python run_queue.py teacher_sweep - one line; ... stop ...)

Launching a queue that is already running, or whose runs are all saved, does nothing; launching a stopped one
again plays what is missing (session_experiment.py skips the teachers a run has saved). Files:
<out>/queues/<name>.json (the runs, when it was launched, the process) and <name>.log (the runs' output).

Progress is counted in teacher runs (a run's sessions with one teacher, then their tests), from the files
session_experiment.py writes as it goes (session_experiment.progress). The time left assumes the rest goes at the
pace since launch.
"""
import json
import os
import signal
import subprocess
import sys
import time

import session_experiment as se

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_SHARE = 0.75  # share of a teacher run's time spent playing the sessions (the rest testing)


def _path(name, out):
    return os.path.join(out, "queues", f"{name}.json")


def _load(name, out=se.OUT):
    with open(_path(name, out)) as f:
        return json.load(f)


def _save(name, record, out=se.OUT):
    os.makedirs(os.path.join(out, "queues"), exist_ok=True)
    with open(_path(name, out) + ".tmp", "w") as f:
        json.dump(record, f, indent=1)
    os.replace(_path(name, out) + ".tmp", _path(name, out))


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def _done(record, out):
    """Teacher runs done (fractions count), and the one under way as text (None if none is)."""
    done, current = 0.0, None
    for entry in record["runs"]:
        for spec, (played, tested) in se.progress(entry["run"], entry["teachers"], out).items():
            share = SESSION_SHARE * played + (1 - SESSION_SHARE) * tested
            done += share
            if current is None and share < 1 and (played > 0 or tested > 0):
                step = f"sessions {played:.0%}" if played < 1 else f"testing {tested:.0%}"
                current = f"{entry['run']}, {se.teacher_name(spec)} ({step})"
    return done, current


def launch(name, runs, out=se.OUT):
    """Plays `runs` - (run name, session_experiment.py arguments after the run name) pairs - one after another in a
    background process, unless queue `name` is already running."""
    if os.path.exists(_path(name, out)) and status(name, out)["running"]:
        print(f"{name} is already running (pid {_load(name, out)['pid']})")
        return
    record = {"runs": [{"run": run, "argv": list(argv), "teachers": se.parse_args([run, *argv]).teachers}
                       for run, argv in runs], "out": out}
    record["done_at_launch"] = _done(record, out)[0]
    units = sum(len(e["teachers"]) for e in record["runs"])
    if record["done_at_launch"] >= units - 1e-6:
        print(f"{name}: all {units} teacher runs are saved, nothing to play")
        return
    record["launched"] = time.time()
    record["finished"], record["pid"] = None, None
    _save(name, record, out)
    log = os.path.join(out, "queues", f"{name}.log")
    # Started through a shell that exits at once, so the queue doesn't stay this process's child (a finished child
    # would linger as a zombie, still "alive", until this process ends); the queue records its own pid.
    subprocess.run(["/bin/sh", "-c", 'nohup "$0" -u "$1" run "$2" "$3" >> "$4" 2>&1 < /dev/null &',
                    sys.executable, os.path.abspath(__file__), name, out, log], cwd=HERE, check=True)
    for _ in range(100):
        if _load(name, out)["pid"] is not None:
            break
        time.sleep(0.1)
    print(f"launched {name}: {len(runs)} runs, {units} teacher runs (pid {_load(name, out)['pid']}); log in {log}")


def _play(name, out):
    """The background process: each run in turn (a failed run is logged and skipped)."""
    os.setsid()  # its own process group, runs and their workers included, so stop() ends them all
    record = _load(name, out)
    record["pid"] = os.getpid()
    _save(name, record, out)
    failed = []
    for entry in record["runs"]:
        print(f"### {entry['run']}", flush=True)
        code = subprocess.run([sys.executable, "-u", os.path.join(HERE, "session_experiment.py"), entry["run"],
                               *entry["argv"], "--out", out], cwd=HERE).returncode
        if code != 0:
            failed.append(entry["run"])
            print(f"### {entry['run']} failed (exit {code})", flush=True)
    record = _load(name, out)
    record["finished"], record["failed"] = time.time(), failed
    _save(name, record, out)


def stop(name, out=se.OUT):
    """Stops queue `name`: the queue, the run it is playing and that run's worker processes. What the run had
    saved stays; launching the queue again plays the rest (a teacher's sessions start over unless saved)."""
    pid = _load(name, out).get("pid")
    if not _alive(pid):
        print(f"{name} isn't running")
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        try:
            os.killpg(pid, 0)
        except OSError:
            break
        time.sleep(0.2)
    else:
        os.killpg(pid, signal.SIGKILL)
    print(f"stopped {name}")


def _clock(seconds):
    minutes = int(round(seconds / 60))
    return f"{minutes // 60}h{minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def status(name, out=se.OUT):
    """{"total", "done" (teacher runs), "current", "running", "elapsed", "left" (seconds, None until known),
    "failed"}."""
    record = _load(name, out)
    total = sum(len(e["teachers"]) for e in record["runs"])
    done, current = _done(record, out)
    end = record.get("finished") or time.time()
    elapsed = end - record["launched"]
    rate = (done - record["done_at_launch"]) / elapsed if elapsed > 0 else 0
    left = (total - done) / rate if rate > 0 else None
    running = record.get("finished") is None and (_alive(record.get("pid")) or
                                                    record.get("pid") is None and time.time() - record["launched"] < 30)
    return {"total": total, "done": done, "current": current, "running": running,
            "elapsed": elapsed, "left": left, "failed": record.get("failed", [])}


def describe(s):
    if s["running"]:
        left = f"about {_clock(s['left'])} left" if s["left"] is not None else "time left: not known yet"
        return f"{s['done']:.1f}/{s['total']} teacher runs, {_clock(s['elapsed'])} in, {left}" + \
            (f" | now: {s['current']}" if s["current"] else "")
    end = "finished" if s["done"] >= s["total"] - 1e-6 else "stopped"
    return f"{end}: {s['done']:.1f}/{s['total']} teacher runs in {_clock(s['elapsed'])}" + \
        (f"; failed: {', '.join(s['failed'])}" if s["failed"] else "")


def watch(name, every=30, out=se.OUT):
    """A progress bar for queue `name`, refreshed every `every` seconds until it stops."""
    from tqdm.auto import tqdm
    s = status(name, out)
    bar = tqdm(total=s["total"], desc=name, bar_format="{desc}: {percentage:3.0f}%|{bar}| {postfix}")
    try:
        while True:
            bar.n = round(s["done"], 2)
            bar.set_postfix_str(describe(s), refresh=True)
            if not s["running"]:
                break
            time.sleep(every)
            s = status(name, out)
    except KeyboardInterrupt:
        pass
    finally:
        bar.close()


if __name__ == "__main__":
    if sys.argv[1] == "run":
        _play(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else se.OUT)
    elif sys.argv[1] == "watch":
        watch(sys.argv[2])
    elif sys.argv[1] == "stop":
        stop(sys.argv[2])
    else:
        print(describe(status(sys.argv[1])))
