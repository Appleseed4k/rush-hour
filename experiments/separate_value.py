"""The student's network as one trunk with two heads, or as two networks: a policy network and a separate value
network. Writes the tables notebooks/p2.5_value_network.ipynb reads (models/summary/separate_value/).

The shared network (paths.SHARED_START) is one trunk with policy and value heads, trained on the sum of the move and
distance losses. Split, each head gets its own trunk: the policy network is round 1 trained on the move targets alone
(apprentice/gamma0.07_free_novalue), the value network round 1 trained on the distances alone
(apprentice/gamma0.07_free_valueonly), on the same data and seed. They are saved as one checkpoint
(PolicyAgent.value_agent; `build`, which made paths.START); in a session each learns only its own targets
(train.online_update), and flat planning reads the value network (forward.values). The same two policies were
pretrained again from other initial weights on the same data (`_i1` runs: apprentice.py --init-seed 1), to tell the
value loss's effect from run-to-run variation.

    python experiments/separate_value.py build      # models/apprentice/separate/round1_epoch03.pt (paths.START)
    python experiments/separate_value.py curves     # agreement per epoch of every round 1 run: curves.csv
    python experiments/separate_value.py startfit   # each run's epochs fitted to people's first 10 trials (by
                                                    # intuition on pool puzzles, as rushhour.surrender): startfit.csv
    python experiments/separate_value.py quality    # each value head against the true distances: quality.csv
    python experiments/separate_value.py split      # every policy x value network as the two-mode student's start on
                                                    # people's 70 layouts, and each policy by intuition: split.csv
    python experiments/separate_value.py rules      # how often each session's policy follows the rules: rules.csv
    python experiments/planner_session.py --lean --run lean_people_all_m10_ao_separate \\
        --init models/apprentice/separate/round1_epoch03.pt    # the session (and --seed 1: ..._separate_seed1)
"""
import io
import json
import multiprocessing as mp
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from rushhour import board, human_data as hd, paths
from rushhour.student import PolicyAgent

A = paths.APPRENTICE
OUT = os.path.join(paths.MODELS, "summary", "separate_value")
RUNS = {"shared": "gamma0.07_free_nodest", "policy only": "gamma0.07_free_novalue",
        "value only": "gamma0.07_free_valueonly",
        "shared, other weights": "gamma0.07_free_nodest_i1", "policy only, other weights": "gamma0.07_free_novalue_i1"}
SESSIONS = {"shared, seed 0": "lean_people_all_m10_ao_nodest_refit",
            "shared, seed 1": "lean_people_all_m10_ao_nodest_refit_seed1",
            "separate, seed 0": "lean_people_all_m10_ao_separate",
            "separate, seed 1": "lean_people_all_m10_ao_separate_seed1"}


def epoch_file(run, epoch):
    return os.path.join(A, RUNS.get(run, run), f"round1_epoch{epoch:02d}.pt")


def build(epoch=3):
    """The policy-only run's round 1 epoch `epoch` with the value-only run's same epoch as its value network."""
    agent = PolicyAgent.load(epoch_file("policy only", epoch))
    agent.value_agent = PolicyAgent.load(epoch_file("value only", epoch))
    os.makedirs(os.path.join(A, "separate"), exist_ok=True)
    path = os.path.join(A, "separate", f"round1_epoch{epoch:02d}.pt")
    agent.save(path)
    return path


def curves():
    rows = []
    for name, run in RUNS.items():
        f = os.path.join(A, run, "manifest.json")
        if not os.path.exists(f):
            continue
        for m in json.load(open(f)):
            if m["round"] == 1:
                rows.append({"run": name, "epoch": m["epoch"], "agreement": m.get("agreement")})
    return pd.DataFrame(rows)


def startfit(runs=("shared", "policy only", "shared, other weights", "policy only, other weights")):
    """rushhour.surrender's fit of every round 1 epoch of each run (the attempts are its cache, made if missing)."""
    from rushhour import surrender as fs
    (hs, hs_se), (hr, hr_se), first = fs.human_target(10)
    edges, rates, _ = hd.surrender_hazard()
    puzzles = fs.puzzle_set("test")
    scales = np.arange(0.0, 6.0 + 1e-9, 0.05)
    cache_dir = os.path.join(paths.SESSIONS, "surrender_fit")
    rows = []
    for name in runs:
        folder = os.path.join(A, RUNS[name])
        if not os.path.exists(os.path.join(folder, "manifest.json")):
            continue
        for m in json.load(open(os.path.join(folder, "manifest.json"))):
            if m["round"] != 1:
                continue
            cache = os.path.join(cache_dir, f"{RUNS[name]}_{m['file'][:-3]}_test_runs10.json")
            runs_ = json.load(open(cache)) if os.path.exists(cache) else fs.attempts(os.path.join(folder, m["file"]),
                                                                                     puzzles, 10)
            if not os.path.exists(cache):
                json.dump(runs_, open(cache, "w"))
            fits = [(s, *fs.predict(runs_, puzzles, first, edges, rates, s, True)) for s in scales]
            loss = [((ps - hs) / hs_se) ** 2 + ((pr - hr) / hr_se) ** 2 for _, ps, pr in fits]
            k = int(np.argmin(loss))
            rows.append({"run": name, "epoch": m["epoch"], "scale": fits[k][0], "solved": fits[k][1],
                         "moves / optimal": fits[k][2], "misfit": loss[k]})
    return pd.DataFrame(rows)


def boards(n_per=40, seed=0):
    """Boards from people's 70 layouts' state spaces, `n_per` each, with their true distances."""
    rng = random.Random(seed)
    out = []
    for name in sorted(hd.trials().instance.unique()):
        dist = board.multi_bfs(hd.puzzle(name))
        bs = [b for b, d in dist.items() if d >= 1]
        out += [(b, dist[b]) for b in rng.sample(bs, min(n_per, len(bs)))]
    return out


def quality(nets, items):
    """Per value head: rank correlation with the true distance, mean estimate / true distance, mean absolute error."""
    from scipy.stats import spearmanr
    ref = PolicyAgent.load(paths.START)
    x = torch.from_numpy(np.stack([ref.encoder.encode_numpy(b) for b, _ in items])).float()
    true = np.array([d for _, d in items])
    rows = []
    for name, f in nets.items():
        a = PolicyAgent.load(f)
        judge = a.value_agent or a
        judge.net.eval()
        with torch.no_grad():
            v = torch.cat([judge.net.heads(x[k:k + 2048])[1] for k in range(0, len(x), 2048)]).numpy() * 10
        rows.append({"network": name, "spearman with true distance": spearmanr(v, true)[0],
                     "estimate / true": float(np.mean(v / true)), "mean abs error": float(np.mean(np.abs(v - true)))})
    return pd.DataFrame(rows)


def _combo(policy, value):
    agent = PolicyAgent.load(policy)
    agent.value_agent = PolicyAgent.load(value)
    buf = io.BytesIO()
    agent.save(buf)
    return buf.getvalue()


def split(policies, values, runs=16):
    """Each policy x value network as the two-mode student's start (its current weights at trial 1) on people's 70
    layouts, and each policy alone by intuition: fit_memory's summary against people's trials 1-9."""
    import fit_memory as fm
    import forward
    weights, conds = {}, []
    lean = forward.LeanSolver()
    for p, pf in policies.items():
        for v, vf in values.items():
            name = f"{p} | {v}"
            weights[name] = _combo(pf, vf)
            conds.append({"name": name, "snapshot": 0, "kind": "lean", "net": name,
                          "kw": {"lean": True, "w_flat": lean.flat_at(1), "w_ao": lean.w_ao, "opening": True}})
        name = f"{p} | intuition alone"
        weights[name] = open(pf, "rb").read()
        conds.append({"name": name, "snapshot": 0, "kind": "intuition", "net": name,
                      "kw": {"k": 1, "soft_memory": (2, 1)}})
    inst = sorted(hd.trials().instance.unique())
    jobs = [(c, i, f"{r}", c["net"]) for c in conds for i in inst for r in range(runs)]
    random.Random(0).shuffle(jobs)
    with mp.Pool(initializer=forward._init, initargs=(weights,)) as pool:
        df = pd.DataFrame(pool.map(fm.job_snapshot, jobs, chunksize=4))
    out, people = fm.summarize_learning(df, {"1-9": (1, 9)})
    out[["policy", "value"]] = out.condition.str.split(" \\| ", expand=True)
    return out, people


def rules():
    """Per session and stage of people's session: how often the policy of the stage's snapshot (8 students; the
    start for trials 1-9) puts its probability on the rule moves of rule_mining's visited-board list, on people's
    boards; and how often people's own moves are rule moves."""
    import rule_mining as rm
    dl = pd.read_csv(os.path.join(rm.OUT, "decision_list_visited.csv"))
    moves, extra = rm.people_moves({"visited": dl})
    ok = (moves["visited rule"] >= 0).values
    fired = [e[2]["visited"] for e in extra]
    moves["people"] = [np.nan if f is None else float(f[c]) for f, c in zip(fired, moves.chosen)]
    stage = pd.cut(moves.trial, [0, 9, 27, 45, 1000], labels=list(rm.STAGES)).astype(str).values
    rows = [{"who": "people", "stage": st, "follows": moves.people[ok & (stage == st)].mean()} for st in rm.STAGES]
    for name, run in SESSIONS.items():
        nets = rm.session_nets(run)
        nets["0"] = json.load(open(os.path.join(paths.SESSIONS, run, "config.json")))["init"]
        net = rm.network_adherence(extra, "visited", nets)
        for st, n in rm.SNAP_OF.items():
            keys = [k for k in net if (k == "0" if n == 0 else k.split()[0] == str(n))]
            v = np.nanmean(np.stack([net[k] for k in keys]), axis=0)
            rows.append({"who": name, "stage": st, "follows": float(np.nanmean(v[ok & (stage == st)]))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "build"
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 220)
    if what == "build":
        print(build())
    elif what == "curves":
        df = curves()
        df.to_csv(os.path.join(OUT, "curves.csv"), index=False)
        print(df.pivot(index="epoch", columns="run", values="agreement").round(3).to_string())
    elif what == "startfit":
        df = startfit()
        df.to_csv(os.path.join(OUT, "startfit.csv"), index=False)
        print(df.round(3).to_string(index=False))
    elif what == "quality":
        nets = {"shared (its value head)": paths.SHARED_START}
        for e in (1, 2, 3, 5, 8, 11):
            nets[f"value only, epoch {e}"] = epoch_file("value only", e)
        df = quality(nets, boards())
        df.to_csv(os.path.join(OUT, "quality.csv"), index=False)
        print(df.round(3).to_string(index=False))
    elif what == "split":
        policies = {"shared": epoch_file("shared", 3), "policy only": epoch_file("policy only", 3)}
        if os.path.exists(epoch_file("shared, other weights", 3)):
            fit = pd.read_csv(os.path.join(OUT, "startfit.csv"))
            for name in ("shared, other weights", "policy only, other weights"):
                best = fit[fit.run == name].sort_values("misfit").epoch.iloc[0]
                policies[f"{name} (epoch {best})"] = epoch_file(name, int(best))
        values = {"shared head": paths.SHARED_START, "value network": epoch_file("value only", 3)}
        out, people = split(policies, values)
        out.to_csv(os.path.join(OUT, "split.csv"), index=False)
        people.to_csv(os.path.join(OUT, "split_people.csv"))
        print(out[["policy", "value", "optimal 2-3", "optimal 4-6", "optimal 7-9", "optimal 10+", "moves / optimal",
                   "skill misfit"]].round(3).to_string(index=False))
    elif what == "rules":
        df = rules()
        df.to_csv(os.path.join(OUT, "rules.csv"), index=False)
        print(df.pivot(index="stage", columns="who", values="follows").round(3).to_string())
