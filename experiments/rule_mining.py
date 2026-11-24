"""Interpretable move rules for Rush Hour, learned from exact optimal moves (after Butler, Torlak & Popovic 2017,
who synthesized sound condition-action rules for Nonograms).

Nonogram rules can be sound: a deduction is never wrong. Rush Hour is planning - the right move can look wrong locally -
so a rule here is scored instead: when it fires on a board, how often the moves it picks are optimal (precision),
against picking a legal move at random (chance), and on how many boards it fires (coverage).

Concepts come from AND-OR's view of a board (`chain`): red needs its path to the exit; a car in that path is a
blocker (depth 1); a blocker must vacate the path cells it covers, and the cars in the way of its doing so, either
way, are depth 2; and so on (depth 3+ pooled). Each car in the chain has the cells it must vacate (`need`).
Predicates on a move (car, direction, steps):

    red          the car is red                     red_exits    the move solves the puzzle
    depth1/2/3   the car's depth in the chain       off_chain    the car is not red and not in the chain
    clears       afterwards the car covers none of the cells it must vacate
    min_clear    ... and it moved the fewest cells that do that, in this direction
    path_fewer   fewer cars in red's path afterwards     path_more    more cars in red's path afterwards
    chain_fewer  fewer cars in the chain afterwards      chain_more   more cars in the chain afterwards
    frees        a car one level up the chain could not clear before and can now
    open_side    it moves toward the side of its lane with more empty cells
    short_way    a chain car moving the way that clears its cells in fewer steps (or the only way that can)
    max_slide    it slides as far as it can

A rule is a conjunction of up to 3 of these or their negations; it fires on every legal move that satisfies it.
Rules are scored on boards from pool puzzles (`train`: every distance 1+ of each puzzle's state space, sampled), and a
decision list is built greedily: the most precise rule that still covers enough boards, then the next on the boards
left. People's moves and the network's policy on the boards people played are then scored by how often they follow
the list.

    python experiments/rule_mining.py          # writes models/summary/rules/{rules,decision_list,adherence}.csv
    python experiments/rule_mining.py specificity   # the same for other students (the forward student, phase 1's
                                                    # intuition student at x10/x30/x120, ...): specificity.csv
"""
import itertools
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

OUT = os.path.join(paths.MODELS, "summary", "rules")
PREDICATES = ["red", "red_exits", "depth1", "depth2", "depth3", "off_chain", "clears", "min_clear", "path_fewer",
              "path_more", "chain_fewer", "chain_more", "frees", "open_side", "short_way", "max_slide"]
BIT = {p: 1 << i for i, p in enumerate(PREDICATES)}
N = 6


def _occupied(state):
    return {cell: name for name, cells in state for cell in cells}


def _path(state):
    """Red's row and the cells between red and the exit."""
    red = dict(state)["red"]
    r, j = red[0][0], max(c for _, c in red)
    return {(r, c) for c in range(j + 1, N)}


def _shift(cells, direction, k):
    di, dj = board.DELTAS[direction]
    return tuple((i + di * k, j + dj * k) for i, j in cells)


def _clearing_options(cells, horizontal, need):
    """[(direction, k, cells swept into)] - for each direction, the fewest steps after which the car covers none of
    `need`, staying on the board, with the cells it has to pass through or land on."""
    out = []
    for direction in ("lr" if horizontal else "ud"):
        for k in range(1, N):
            moved = _shift(cells, direction, k)
            if not all(0 <= i < N and 0 <= j < N for i, j in moved):
                break
            if not set(moved) & need:
                swept = set()
                for s in range(1, k + 1):
                    swept |= set(_shift(cells, direction, s))
                out.append((direction, k, swept - set(cells)))
                break
    return out


def chain(state, max_depth=3):
    """AND-OR's blocking structure: ({car: depth} for red (0) and every car it depends on, {car: cells to vacate})."""
    occ = _occupied(state)
    sd = dict(state)
    orient = board.car_orientations(state)
    depth, need = {"red": 0}, {}
    frontier = []
    for cell in _path(state):
        car = occ.get(cell)
        if car is not None:
            need.setdefault(car, set()).add(cell)
            if car not in depth:
                depth[car] = 1
                frontier.append(car)
    while frontier:
        nxt = []
        for car in frontier:
            for _, _, swept in _clearing_options(sd[car], orient[car] == "h", need[car]):
                for cell in swept:
                    other = occ.get(cell)
                    if other is None or other in ("red", car):
                        continue
                    need.setdefault(other, set()).add(cell)
                    if other not in depth:
                        depth[other] = min(depth[car] + 1, max_depth)
                        nxt.append(other)
        frontier = nxt
    return depth, need


def _can_clear(state, car, need, orient):
    if car == "red":
        return not (_path(state) & set(_occupied(state)))
    cells = dict(state)[car]
    for (name, _, _), nb in board.neighbors(state, orient):
        if name == car and not set(dict(nb)[car]) & need[car]:
            return True
    return False


def move_masks(state, orient=None):
    """[(move, next board, predicate bitmask)] for every legal move from `state`."""
    orient = orient or board.car_orientations(state)
    depth, need = chain(state)
    occ = _occupied(state)
    path = _path(state)
    n_path = len({occ[c] for c in path if c in occ})
    n_chain = len(depth) - 1
    options = list(board.neighbors(state, orient))
    max_k = {}
    for (car, d, k), _ in options:
        max_k[(car, d)] = max(max_k.get((car, d), 0), k)
    blocked = {p: not _can_clear(state, p, need, orient) for p in depth if p == "red" or p in need}
    out = []
    for (car, d, k), nb in options:
        m = 0
        dep = depth.get(car)
        if car == "red":
            m |= BIT["red"]
            if max(j for _, j in dict(nb)["red"]) == N - 1:
                m |= BIT["red_exits"]
        elif dep is None:
            m |= BIT["off_chain"]
        else:
            m |= BIT[f"depth{dep}"]
            after = set(dict(nb)[car])
            opts = {dd: kk for dd, kk, _ in _clearing_options(dict(state)[car], orient[car] == "h", need[car])}
            if d in opts and opts[d] == min(opts.values()):
                m |= BIT["short_way"]
            if not after & need[car]:
                m |= BIT["clears"]
                if k == opts.get(d, -1):
                    m |= BIT["min_clear"]
            parents = [p for p, dp in depth.items() if dp == dep - 1 and blocked.get(p)]
            if parents:
                d2, n2 = chain(nb)
                if any(p in d2 and (p == "red" or p in n2) and _can_clear(nb, p, {**n2, **{p: need.get(p, set())}},
                                                                          orient) for p in parents):
                    m |= BIT["frees"]
        occ2 = _occupied(nb)
        p2 = len({occ2[c] for c in _path(nb) if c in occ2})
        if p2 < n_path:
            m |= BIT["path_fewer"]
        elif p2 > n_path:
            m |= BIT["path_more"]
        c2 = len(chain(nb)[0]) - 1
        if c2 < n_chain:
            m |= BIT["chain_fewer"]
        elif c2 > n_chain:
            m |= BIT["chain_more"]
        cells = dict(state)[car]
        horizontal = orient[car] == "h"
        lane = [(cells[0][0], c) for c in range(N)] if horizontal else [(r, cells[0][1]) for r in range(N)]
        axis = [c for _, c in cells] if horizontal else [r for r, _ in cells]
        lo, hi = min(axis), max(axis)
        free_lo = sum(1 for cell in lane[:lo] if cell not in occ)
        free_hi = sum(1 for cell in lane[hi + 1:] if cell not in occ)
        toward_hi = d in "rd"
        if (toward_hi and free_hi > free_lo) or (not toward_hi and free_lo > free_hi):
            m |= BIT["open_side"]
        if k == max_k[(car, d)]:
            m |= BIT["max_slide"]
        out.append(((car, d, k), nb, m))
    return out


def _train_puzzle(args):
    """Sampled boards (distance 1+) of one pool puzzle's state space: rows of (board id, mask, optimal, distance)."""
    pid, state, per = args
    dist = board.multi_bfs(state)
    rng = random.Random(pid)
    boards = [b for b, d in dist.items() if d >= 1]
    rows = []
    for bi, b in enumerate(rng.sample(boards, min(per, len(boards)))):
        for _, nb, m in move_masks(b):
            rows.append((pid * 1000 + bi, m, dist[nb] < dist[b], dist[b]))
    return rows


def _visited_init(weights):
    import forward
    forward._init_one(weights)


def _visited_puzzle(args):
    """The boards (distance 1+) the two-mode student (forward.solve_lean, its default weights, the start network)
    visits in `attempts` attempts at one pool puzzle, each once: rows as _train_puzzle's."""
    import forward
    pid, state, attempts = args
    dist = board.multi_bfs(state)
    agent = forward._W["agent"]
    seen = {}
    for a in range(attempts):
        rng = random.Random(pid * 100 + a)
        random.seed(rng.random())
        agent.__dict__.pop("_values", None)
        trace = []
        forward.solve_lean(agent, state, max(50, 10 * dist[state]), opening=True, rng=rng, trace=trace)
        for b in [state] + [nb for _, nb in trace]:
            if dist[b] >= 1:
                seen.setdefault(b, len(seen))
    rows = []
    for b, bi in seen.items():
        for _, nb, m in move_masks(b):
            rows.append((pid * 100000 + bi, m, dist[nb] < dist[b], dist[b]))
    return rows


def _picks(n_puzzles, seed):
    pool = board.load_puzzle_pool()
    human = {hd.puzzle(n) for n in hd.trials().instance.unique()}
    return random.Random(seed).sample([s for d, s in pool if 4 <= d <= 20 and s not in human], n_puzzles)


def training_rows(n_puzzles=300, per=60, seed=0, boards="uniform", attempts=4):
    """Moves on pool puzzles' boards (none of people's layouts): `boards` "uniform", `per` boards sampled from each
    puzzle's whole state space; "visited", the boards the two-mode student reaches in `attempts` attempts."""
    picks = _picks(n_puzzles, seed)
    if boards == "uniform":
        with mp.Pool() as p:
            rows = [r for rs in p.map(_train_puzzle, [(i, s, per) for i, s in enumerate(picks)]) for r in rs]
    else:
        with open(paths.START, "rb") as f:
            weights = f.read()
        with mp.Pool(initializer=_visited_init, initargs=(weights,)) as p:
            rows = [r for rs in p.map(_visited_puzzle, [(i, s, attempts) for i, s in enumerate(picks)]) for r in rs]
    return pd.DataFrame(rows, columns=["board", "mask", "optimal", "distance"])


def all_rules(max_literals=3):
    literals = [(p, True) for p in PREDICATES] + [(p, False) for p in PREDICATES]
    for n in range(1, max_literals + 1):
        for combo in itertools.combinations(literals, n):
            names = [p for p, _ in combo]
            if len(set(names)) < n:
                continue
            pos = sum(BIT[p] for p, v in combo if v)
            neg = sum(BIT[p] for p, v in combo if not v)
            if pos == 0:  # a rule must name something the move does
                continue
            yield combo, pos, neg


def describe(combo):
    return " & ".join(p if v else f"not {p}" for p, v in combo)


class Scorer:
    """Rules scored fast on one table of moves: boards as integer codes, per-board counts by bincount."""

    def __init__(self, df):
        self.code = pd.factorize(df["board"])[0]
        self.nb = self.code.max() + 1
        self.mask = df["mask"].values
        self.opt = df["optimal"].values.astype(float)
        self.dist = df.groupby(self.code).distance.first().values
        self.chance = np.bincount(self.code, self.opt, self.nb) / np.bincount(self.code, minlength=self.nb)

    def fired(self, pos, neg):
        return ((self.mask & pos) == pos) & ((self.mask & neg) == 0)

    def score(self, pos, neg, left=None):
        """(boards fired on, precision, chance) over the boards in `left` (a boolean array; None: all) - precision:
        per board, the share of the moves the rule fires on that are optimal, averaged over the boards it fires on;
        chance: the share of all legal moves that are optimal, on those boards."""
        f = self.fired(pos, neg)
        n = np.bincount(self.code[f], minlength=self.nb)
        on = n > 0 if left is None else (n > 0) & left
        if not on.any():
            return 0, np.nan, np.nan, on
        k = np.bincount(self.code[f], self.opt[f], self.nb)
        return int(on.sum()), float((k[on] / n[on]).mean()), float(self.chance[on].mean()), on


def score(df, pos, neg):
    return Scorer(df).score(pos, neg)[:3]


def decision_list(scorer, rules, min_cover=0.02, max_rules=10, literal_cost=0.02):
    """Greedy: the rule with the best precision less `literal_cost` per literal past the first (so a longer rule must
    be that much more precise: conciseness, as Butler et al.'s cost function), firing on at least `min_cover` of all
    boards among those no earlier rule fires on; then the next, on the boards left. Precision by distance on its own
    boards."""
    left = np.ones(scorer.nb, bool)
    out = []
    for _ in range(max_rules):
        best = None
        for combo, pos, neg in rules:
            n, prec, chance, on = scorer.score(pos, neg, left)
            if n < min_cover * scorer.nb:
                continue
            key = (prec - literal_cost * (len(combo) - 1), -len(combo))
            if best is None or key > best[0]:
                best = (key, combo, pos, neg, n, prec, chance, on)
        if best is None:
            break
        _, combo, pos, neg, n, prec, chance, on = best
        row = {"rule": describe(combo), "pos": pos, "neg": neg, "literals": len(combo), "boards": n,
               "coverage": n / scorer.nb, "precision": prec, "chance": chance}
        f = scorer.fired(pos, neg)
        cnt = np.bincount(scorer.code[f], minlength=scorer.nb)
        k = np.bincount(scorer.code[f], scorer.opt[f], scorer.nb)
        for name, (lo, hi) in {"1-3": (1, 3), "4-9": (4, 9), "10+": (10, 99)}.items():
            sel = on & (scorer.dist >= lo) & (scorer.dist <= hi)
            row[f"boards {name}"] = int(sel.sum())
            row[f"precision {name}"] = float((k[sel] / cnt[sel]).mean()) if sel.any() else np.nan
            row[f"chance {name}"] = float(scorer.chance[sel].mean()) if sel.any() else np.nan
        out.append(row)
        left &= ~on
    out = pd.DataFrame(out)
    out["boards left"] = scorer.nb - out.boards.cumsum()
    return out


def first_rule(masks, dl):
    """Index of the first rule in the decision list that fires on any of these masks, and the fired mask flags."""
    for i, r in enumerate(dl.itertuples()):
        fired = [(m & r.pos) == r.pos and (m & r.neg) == 0 for m in masks]
        if any(fired):
            return i, fired
    return None, None


def people_moves(lists):
    """People's moves 2+ from the goal, with each decision list's first firing rule there: one row per move (who,
    trial, distance, chosen move's index, whether it was optimal) and, per list, the rule and its precision there;
    plus, per move, the board, each legal move's action id and each list's fired flags, for scoring networks."""
    import human_profile as hp
    ref = PolicyAgent.load(paths.START)
    rows, extra, dist = [], [], {}
    for who, trial, instance, attempt, boards, moves, solved, seconds in hp.people_attempts():
        if trial is None:
            continue
        if instance not in dist:
            dist[instance] = board.multi_bfs(boards[0])
        d = dist[instance]
        orient = board.car_orientations(boards[0])
        for b, played in zip(boards, moves):
            if d[b] < 2:
                continue
            mm = move_masks(b, orient)
            mv = [m for m, _, _ in mm]
            opt = np.array([d[nb] < d[b] for _, nb, _ in mm])
            row = {"who": who, "trial": trial, "instance": instance, "distance": d[b], "chosen": mv.index(played),
                   "people optimal": float(opt[mv.index(played)]), "chance optimal": opt.mean()}
            fired_by = {}
            for name, dl in lists.items():
                i, fired = first_rule([m for _, _, m in mm], dl)
                row[f"{name} rule"] = -1 if i is None else i
                fired_by[name] = None if i is None else np.array(fired)
                if i is not None:
                    row[f"{name} precision"] = opt[fired_by[name]].mean()
            rows.append(row)
            extra.append((b, np.array([ref.action_id(b, *m) for m in mv]), fired_by))
    return pd.DataFrame(rows), extra


def network_adherence(extra, list_name, nets, device=None):
    """{net: per move, the policy's probability on the moves list_name's first firing rule picks (nan: none fires)}.
    The boards are encoded once and each network scores them in batches."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ref = PolicyAgent.load(paths.START)
    uniq = {}
    for b, _, _ in extra:
        uniq.setdefault(b, len(uniq))
    enc = torch.from_numpy(np.stack([ref.encoder.encode_numpy(b) for b in uniq])).float()
    idx = np.array([uniq[b] for b, _, _ in extra])
    out = {}
    for name, f in nets.items():
        a = PolicyAgent.load(f).to(device)
        a.net.eval()
        logits = []
        with torch.no_grad():
            for k in range(0, len(enc), 4096):
                logits.append(a.net.heads(enc[k:k + 4096].to(device))[0].cpu())
        logits = torch.cat(logits).numpy()
        vals = np.full(len(extra), np.nan)
        for r, (_, ids, fired_by) in enumerate(extra):
            fired = fired_by[list_name]
            if fired is None:
                continue
            z = logits[idx[r], ids]
            p = np.exp(z - z.max())
            p /= p.sum()
            vals[r] = p[fired].sum()
        out[name] = vals
    return out


def _people_cached():
    """people_moves for the visited-board list, cached in OUT (people_moves.pkl) next to the list it was made with."""
    import pickle
    f = os.path.join(OUT, "people_moves.pkl")
    lf = os.path.join(OUT, "decision_list_visited.csv")
    if os.path.exists(f) and os.path.getmtime(f) > os.path.getmtime(lf):
        with open(f, "rb") as fh:
            return pickle.load(fh)
    moves, extra = people_moves({"visited": pd.read_csv(lf)})
    with open(f, "wb") as fh:
        pickle.dump((moves, extra), fh)
    return moves, extra


def rule_following(runs, trials=(18, 36, 54), students=8):
    """How often each session run's policy follows the visited-board list on people's boards, its snapshot for each
    stage of people's session paired with that stage (the run's own start for trials 1-9; the run's multiplier gives
    the snapshot names), averaged over its students - and people's own moves. `runs`: {name: run folder}. Returns a
    long table: who, stage, by ("all", "distance" or "rule"), key, follows, moves."""
    moves, extra = _people_cached()
    ok = (moves["visited rule"] >= 0).values
    fired = [e[2]["visited"] for e in extra]
    names = dict(enumerate(pd.read_csv(os.path.join(OUT, "decision_list_visited.csv")).rule))
    stage = pd.cut(moves.trial, [0, 9, 27, 45, 1000], labels=list(STAGES)).astype(str).values
    dbin = pd.cut(moves.distance, [1, 3, 9, 99], labels=["2-3", "4-9", "10+"]).astype(str).values
    rule = moves["visited rule"].map(names).values
    people = np.array([np.nan if f is None else float(f[c]) for f, c in zip(fired, moves.chosen)])

    def table(who, values):
        out = []
        for st in STAGES:
            sel = ok & (stage == st)
            out.append({"who": who, "stage": st, "by": "all", "key": "all", "follows": np.nanmean(values[sel]),
                        "moves": int(sel.sum())})
            for by, keys in (("distance", dbin), ("rule", rule)):
                for k in pd.unique(keys[sel]):
                    m = sel & (keys == k)
                    out.append({"who": who, "stage": st, "by": by, "key": k, "follows": np.nanmean(values[m]),
                                "moves": int(m.sum())})
        return out

    rows = table("people", people)
    for who, folder in runs.items():
        config = json.load(open(os.path.join(folder, "config.json")))
        m = int(config.get("multiplier", 10))
        start = os.path.join(folder, "initial.pt")
        nets = {"0": start if os.path.exists(start) else config["init"]}
        for n in trials:
            for k in range(students):
                f = os.path.join(folder, "no_teacher", f"student{k:02d}_puzzle{m * n:03d}.pt")
                if os.path.exists(f):
                    nets[f"{n} s{k:02d}"] = f
        net = network_adherence(extra, "visited", nets)
        paired = np.full(len(moves), np.nan)
        for st, n in SNAP_OF.items():
            keys = [k for k in net if (k == "0" if n == 0 else k.split()[0] == str(n))]
            if keys:
                v = np.nanmean(np.stack([net[k] for k in keys]), axis=0)
                paired[stage == st] = v[stage == st]
        rows += table(who, paired)
    return pd.DataFrame(rows)


def distance_from_people(follows):
    """Per student: the mean absolute difference from people's rule-following, over the 4 stages ("all"), over
    stage x distance cells, and over stage x rule cells with 500+ of people's moves."""
    f = follows.set_index(["by", "key", "stage", "who"]).follows.unstack("who")
    big = follows[(follows.who == "people") & (follows.moves >= 500)].set_index(["by", "key", "stage"]).index
    out = {}
    for who in f.columns:
        if who == "people":
            continue
        d = (f[who] - f["people"]).abs()
        out[who] = {by: d.loc[by][d.loc[by].index.isin(big.droplevel(0)[big.get_level_values(0) == by])].mean()
                    for by in ("all", "distance", "rule")}
    return pd.DataFrame(out).T


STAGES = {"1-9": (1, 9), "10-27": (10, 27), "28-45": (28, 45), "46+": (46, 1000)}
SNAP_OF = {"1-9": 0, "10-27": 18, "28-45": 36, "46+": 54}
RUN = "lean_people_all_m10_ao_nodest_refit"  # the two-mode student as it is now (paths.START, the refitted weights)


def session_nets(run=RUN, students=8):
    """{"0": the start, "<n> sNN": student NN's snapshot after n human trials} for n in 18, 36, 54."""
    folder = os.path.join(paths.SESSIONS, run, "no_teacher")
    nets = {"0": paths.START}
    for n in (18, 36, 54):
        for s_ in range(students):
            f = os.path.join(folder, f"student{s_:02d}_puzzle{10 * n:03d}.pt")
            if os.path.exists(f):
                nets[f"{n} s{s_:02d}"] = f
    return nets


ARCHIVE = os.path.join(os.path.dirname(os.path.dirname(paths.ROOT)), "rush-hour-archive", "models", "sessions")
SPECIFICITY = {  # students that should and shouldn't follow the rules as people do
    "two-mode, separate (s0)": os.path.join(paths.SESSIONS, "lean_people_all_m10_ao_separate"),
    "two-mode, separate (s1)": os.path.join(paths.SESSIONS, "lean_people_all_m10_ao_separate_seed1"),
    "two-mode, shared (s0)": os.path.join(paths.SESSIONS, "lean_people_all_m10_ao_nodest_refit"),
    "two-mode, shared (s1)": os.path.join(paths.SESSIONS, "lean_people_all_m10_ao_nodest_refit_seed1"),
    "two-mode, dest head": os.path.join(paths.SESSIONS, "lean_people_all_m10_ao"),
    "forward student": os.path.join(paths.SESSIONS, "lean_people_all_m10_noao"),
    "intuition, x10": os.path.join(ARCHIVE, "ao_people_all_m10"),
    "intuition, x30": os.path.join(ARCHIVE, "ao_people_all_m30"),
    "intuition, x120": os.path.join(ARCHIVE, "ao_people_all_m120"),
}

if __name__ == "__main__" and sys.argv[1:2] == ["specificity"]:
    pd.set_option("display.width", 220)
    follows = rule_following(SPECIFICITY)
    follows.to_csv(os.path.join(OUT, "specificity.csv"), index=False)
    allr = follows[follows.by == "all"].pivot(index="who", columns="stage", values="follows")[list(STAGES)]
    print(allr.round(3).to_string())
    print(distance_from_people(follows).round(3).to_string())
    sys.exit()

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 70)
    rules = list(all_rules())
    print(f"{len(rules)} candidate rules over {len(PREDICATES)} predicates", flush=True)
    lists = {}
    for kind in ("uniform", "visited"):
        df = training_rows(boards=kind)
        print(f"\n== {kind}: {df.board.nunique()} boards, {len(df)} moves, optimal share {df.optimal.mean():.3f}",
              flush=True)
        sc = Scorer(df)
        if kind == "visited":
            scored = []
            for combo, pos, neg in rules:
                n, prec, chance, _ = sc.score(pos, neg)
                scored.append({"rule": describe(combo), "literals": len(combo), "boards": n,
                               "coverage": n / sc.nb, "precision": prec, "chance": chance})
            pd.DataFrame(scored).to_csv(os.path.join(OUT, "rules.csv"), index=False)
        dl = decision_list(sc, rules)
        dl.to_csv(os.path.join(OUT, f"decision_list_{kind}.csv"), index=False)
        print(dl.drop(columns=["pos", "neg"]).round(3).to_string(), flush=True)
        lists[kind] = dl
    moves, extra = people_moves(lists)
    nets = session_nets()
    net = network_adherence(extra, "visited", nets)
    for n in (0, 18, 36, 54):
        cols = [k for k in net if (k == "0" if n == 0 else k.split()[0] == str(n))]
        moves[f"net {n}"] = np.nanmean(np.stack([net[k] for k in cols]), axis=0)
    ok = (moves["visited rule"] >= 0).values
    fired = [e[2]["visited"] for e in extra]
    moves["chance"] = [np.nan if f is None else f.mean() for f in fired]
    moves["people"] = [np.nan if f is None else float(f[c]) for f, c in zip(fired, moves.chosen)]
    moves["stage"] = pd.cut(moves.trial, [0, 9, 27, 45, 1000], labels=list(STAGES))
    moves["dbin"] = pd.cut(moves.distance, [1, 3, 9, 99], labels=["2-3", "4-9", "10+"])
    moves.to_csv(os.path.join(OUT, "adherence.csv"), index=False)
    m = moves[ok]
    print(f"\n== people's boards: {len(moves)} moves; a rule of the visited list fires on {ok.mean():.3f}")
    for kind in lists:
        has = moves[f"{kind} rule"] >= 0
        print(f"precision on people's boards, {kind} list: {moves.loc[has, f'{kind} precision'].mean():.3f} "
              f"(chance {moves.loc[has, 'chance optimal'].mean():.3f}; fires on {has.mean():.3f})")
    netc = [f"net {n}" for n in (0, 18, 36, 54)]
    print("\nfollowing the rules, by stage of people's session (net n: the 8 students after n trials)")
    print(m.groupby("stage", observed=True)[["chance", "people"] + netc].mean().round(3).to_string())
    paired = {st: m.loc[m.stage == st, f"net {SNAP_OF[st]}"].mean() for st in STAGES}
    print("the network paired with each stage:", {k: round(v, 3) for k, v in paired.items()})
    print("\nby distance")
    print(m.groupby("dbin", observed=True)[["chance", "people"] + netc].mean().round(3).to_string())
    print("\nby rule: people per stage, and the network at its start and after 54 trials")
    byrule = m.pivot_table(index="visited rule", columns="stage", values="people", aggfunc="mean", observed=True)
    byrule["n"] = m.groupby("visited rule").size()
    byrule["chance"] = m.groupby("visited rule")["chance"].mean()
    byrule["net 0"] = m.groupby("visited rule")["net 0"].mean()
    byrule["net 54"] = m.groupby("visited rule")["net 54"].mean()
    byrule.index = [lists["visited"].rule[i] for i in byrule.index]
    print(byrule.round(3).to_string())
    byrule.to_csv(os.path.join(OUT, "adherence_by_rule.csv"))
    t = hd.trials()
    t = t[t.solved & (t.trial <= 72)].assign(ratio=lambda x: x.moves / x.distance)
    half = lambda x: pd.cut(x, [0, 9, 1000], labels=["early", "late"])  # noqa: E731
    eff = t.groupby(["subject", half(t.trial)], observed=True).ratio.mean().unstack()
    adh = m.groupby(["who", half(m.trial)], observed=True).people.mean().unstack()
    per = pd.DataFrame({"rule gain": adh["late"] - adh["early"],
                        "efficiency gain": eff["early"] - eff["late"]}).dropna()
    per.to_csv(os.path.join(OUT, "adherence_people.csv"))
    from scipy.stats import spearmanr
    rho, pval = spearmanr(per["rule gain"], per["efficiency gain"])
    print(f"\npeople ({len(per)}): rule following, trials 1-9 -> 10+, changes by {per['rule gain'].mean():+.3f} on "
          f"average (sd {per['rule gain'].std():.3f}); against their gain in moves / optimal: Spearman {rho:.2f} "
          f"(p {pval:.3f})")
