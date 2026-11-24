"""Does the network's depth matter? The trunk's lane blocks (4 in the student) each let a cell gather its row and
column once, so one pass can follow a blocking chain about one link per block (red -> its blocker -> that car's blocker
...). Fewer blocks should make the student worse at moves deep in the chain and further from people mid-puzzle; more
might close part of that gap. Everything else as the student (paths.START): a policy network trained on AND-OR's moves
only and a value network on its distances only, round 1 on the same data and seed, the start's epoch and giving-up
scale fitted to people's first 10 trials as always, the two-mode session at two seeds.

Measures fixed before running: on people's layouts, the chance of an optimal move 7-9 from the goal at each stage
and the total skill misfit; how often the policy picks the deep-chain rule's moves (rule_mining: "depth3 & clears &
chain_fewer") against people; the session's gain in moves / optimal.

    python experiments/network_depth.py pretrain     # apprentice/gamma0.07_free_{novalue,valueonly}_b{2,6}
    python experiments/network_depth.py startfit     # their policies' epochs against people's start: startfit.csv
    python experiments/network_depth.py build        # apprentice/separate_b{2,6}/round1_epochNN.pt (the best epoch)
    python experiments/network_depth.py sessions     # lean_people_all_m10_ao_b{2,6}(_seed1), and their snapshots
    python experiments/network_depth.py summary      # summary.csv, rules.csv (models/summary/network_depth/)
    python experiments/network_depth.py all          # each in turn
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from rushhour import paths

ROOT = paths.ROOT
OUT = os.path.join(paths.MODELS, "summary", "network_depth")
DEPTHS = (2, 6)
PY = sys.executable


def run(args):
    print(">", " ".join(str(a) for a in args), flush=True)
    subprocess.run([PY] + [str(a) for a in args], cwd=ROOT, check=True,
                   env={**os.environ, "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8"})


def policy_run(d):
    return f"gamma0.07_free_novalue_b{d}"


def value_run(d):
    return f"gamma0.07_free_valueonly_b{d}"


def session(d, seed):
    return f"lean_people_all_m10_ao_b{d}" + ("" if seed == 0 else f"_seed{seed}")


def pretrain():
    common = ["--rounds", 1, "--epochs", 11, "--patience", 40]
    for d in DEPTHS:
        for name, extra in ((policy_run(d), ["--value-weight", 0]), (value_run(d), ["--policy-weight", 0])):
            if not os.path.exists(os.path.join(paths.APPRENTICE, name, "history.json")):
                run([os.path.join("experiments", "apprentice.py"), name, *common, *extra, "--blocks", d])


def startfit():
    import separate_value as sv
    df = sv.startfit(runs=[policy_run(d) for d in DEPTHS] + ["policy only"])
    df["blocks"] = df.run.map(lambda r: 4 if r == "policy only" else int(r.rsplit("_b", 1)[1]))
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "startfit.csv"), index=False)
    return df


def best():
    """{blocks: (epoch, giving-up scale)} - each depth's policy epoch that fits people's start best."""
    df = pd.read_csv(os.path.join(OUT, "startfit.csv"))
    out = {}
    for d, g in df.groupby("blocks"):
        r = g.sort_values("misfit").iloc[0]
        out[int(d)] = (int(r.epoch), float(r.scale))
    return out


def build():
    from rushhour.student import PolicyAgent
    out = {}
    for d, (e, scale) in best().items():
        if d == 4:
            continue
        agent = PolicyAgent.load(os.path.join(paths.APPRENTICE, policy_run(d), f"round1_epoch{e:02d}.pt"))
        agent.value_agent = PolicyAgent.load(os.path.join(paths.APPRENTICE, value_run(d), f"round1_epoch{e:02d}.pt"))
        folder = os.path.join(paths.APPRENTICE, f"separate_b{d}")
        os.makedirs(folder, exist_ok=True)
        agent.save(os.path.join(folder, f"round1_epoch{e:02d}.pt"))
        out[d] = (os.path.join(folder, f"round1_epoch{e:02d}.pt"), scale)
    return out


def sessions():
    for d, (init, scale) in build().items():
        for seed in (0, 1):
            name = session(d, seed)
            if not os.path.exists(os.path.join(paths.SESSIONS, name, "no_teacher", "sessions.json")):
                run([os.path.join("experiments", "planner_session.py"), "--lean", "--run", name, "--init", init,
                     "--surrender-scale", scale] + ([] if seed == 0 else ["--seed", seed]))
            if not os.path.exists(os.path.join(paths.MODELS, "summary", f"lean_snapshots_{name}.csv")):
                run([os.path.join("experiments", "fit_memory.py"), "lean-snapshots", "--run", name])


RUNS = {**{(4, s): n for s, n in ((0, "lean_people_all_m10_ao_separate"), (1, "lean_people_all_m10_ao_separate_seed1"))},
        **{(d, s): session(d, s) for d in DEPTHS for s in (0, 1)}}


def summary():
    import fit_memory as fm
    import rule_mining as rm
    from rushhour import human_data as hd
    rows = []
    for (d, s), name in sorted(RUNS.items()):
        f = os.path.join(paths.MODELS, "summary", f"lean_snapshots_{name}.csv")
        if not os.path.exists(f):
            continue
        o, people = fm.summarize_learning(pd.read_csv(f), fm.SNAP_STAGES)
        q = o[[r.snapshot == fm.SNAP_OF[r.stage] for r in o.itertuples()]].set_index("stage")
        recs = json.load(open(os.path.join(paths.SESSIONS, name, "no_teacher", "sessions.json")))
        e = pd.DataFrame([dict(x, subject=k) for k, r in recs.items() for x in r])
        w = e[e.solved].assign(ratio=lambda t: t.moves / t.distance)
        by = w.groupby(pd.cut(w.trial, [0, 9, 27, 45, 72], labels=list(fm.SNAP_STAGES))).ratio.mean()
        for st in fm.SNAP_STAGES:
            rows.append({"blocks": d, "seed": s, "run": name, "stage": st, "session moves / optimal": by[st],
                         **q.loc[st, ["optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                                      "skill misfit", "revisit misfit"]].to_dict()})
    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    people.to_csv(os.path.join(OUT, "people.csv"))
    follows = rm.rule_following({name: os.path.join(paths.SESSIONS, name) for name in RUNS.values()
                                 if os.path.exists(os.path.join(paths.SESSIONS, name, "config.json"))})
    follows.to_csv(os.path.join(OUT, "rules.csv"), index=False)
    return df, follows


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "summary"
    pd.set_option("display.width", 220)
    steps = {"pretrain": pretrain, "startfit": startfit, "build": build, "sessions": sessions, "summary": summary}
    for step in (list(steps) if what == "all" else [what]):
        print(f"== {step}", flush=True)
        out = steps[step]()
        if isinstance(out, pd.DataFrame):
            print(out.round(3).to_string(index=False), flush=True)
        elif isinstance(out, tuple):
            for o in out:
                print(o.round(3).to_string(index=False), flush=True)
