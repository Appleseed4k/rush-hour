"""How much of optimal move choice is learnable across puzzles?

Trains a policy network on Rush Hour states and measures, on puzzles it never
saw, how often its top legal move is an optimal one (agreement), how close
its distance-to-goal estimate is, and how often its greedy rollouts solve.

Labels come from a breadth-first search over each puzzle's whole state space
(lib.multi_bfs): every optimal move from a state, and the state's distance
to the goal. The loss rewards putting probability on any optimal move.

Conditions vary one thing at a time:
- input "old": the original rl.StateEncoder's per-cell category (empty /
  red / horizontal / vertical), where two cars parked end to end look like
  one long car;
  "new" adds which cells start and end a car, which cars are trucks, and the
  row/column coordinates.
- network "old": the original rl.PolicyNetwork's shape (two 3x3
  convolutions, then one dense layer over all positions); "lane" (what
  rl.PolicyNetwork is now, minus its dest head): residual blocks whose full-row
  and full-column convolutions let every cell see its whole lane, with a
  per-cell (position-shared) move head plus a distance-to-goal head.
- labels "exact" (search) or "traces" (lapsing AND-OR solves at gamma 0.2,
  the best continuation seen per state, as the former rl.PuzzleTrainer
  generated them).

Stage "cluster" trains on the 14 training puzzles of rush_nw_cluster.txt
(rush_hour.ipynb's split); stage "scale" trains on N random rush_nw puzzles.
Both evaluate on the cluster's 2 test puzzles and on held-out rush_nw
puzzles drawn evenly across distance bins."""
import argparse
import json
import random
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import and_or as ao
from lib import car_orientations, multi_bfs, neighbors, parse_puzzle_line, read_puzzles

DIRS = "lrud"
N_ACTIONS = 36 * 4 * 5
BINS = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 30)]


# --- encoding -------------------------------------------------------------------

def action_index(state_dict, car, direction, steps):
    """Index of a move keyed by the car's top/left cell, as in rl.PolicyAgent."""
    r = min(i for i, _ in state_dict[car])
    c = min(j for _, j in state_dict[car])
    return ((r * 6 + c) * 4 + DIRS.index(direction)) * 5 + steps - 1


def encode(state):
    """(7, 6, 6) uint8: empty, red, horizontal, vertical (the original
    rl.StateEncoder's channels), then car start cell, car end cell, truck."""
    x = np.zeros((7, 6, 6), dtype=np.uint8)
    x[0] = 1
    for name, cells in state:
        horizontal = cells[0][0] == cells[1][0]
        category = 1 if name == "red" else (2 if horizontal else 3)
        ordered = sorted(cells)
        for i, j in cells:
            x[0, i, j] = 0
            x[category, i, j] = 1
            x[6, i, j] = len(cells) == 3
        x[4][ordered[0]] = 1
        x[5][ordered[-1]] = 1
    return x


COORDS = torch.stack(torch.meshgrid(torch.linspace(-1, 1, 6), torch.linspace(-1, 1, 6), indexing="ij"))


def to_input(batch_x, input_kind):
    x = torch.as_tensor(batch_x, dtype=torch.float32)
    if input_kind == "old":
        return x[:, :4]
    return torch.cat([x, COORDS.expand(x.shape[0], -1, -1, -1)], dim=1)


def input_channels(input_kind):
    return 4 if input_kind == "old" else 9


# --- labels -----------------------------------------------------------------------

def example(state, distances, orientations):
    d = distances[state]
    sd = dict(state)
    legal, optimal = [], []
    for (car, direction, steps), nxt in neighbors(state, orientations):
        idx = action_index(sd, car, direction, steps)
        legal.append(idx)
        if distances.get(nxt) == d - 1:
            optimal.append(idx)
    return encode(state), d, legal, optimal


def label_puzzle(job):
    """Examples for one puzzle: its start state plus up to `k` other unsolved
    states of its state space (all of them if k is None)."""
    state, k, seed = job
    distances = multi_bfs(state)
    orientations = car_orientations(state)
    others = [s for s, d in distances.items() if d > 0 and s != state]
    if k is not None:
        others = random.Random(seed).sample(others, min(k, len(others)))
    return [example(s, distances, orientations) for s in [state] + others]


def trace_puzzle(job):
    """Labels like the former rl.PuzzleTrainer's for one puzzle: `n` lapsing AND-OR solves
    (gamma 0.2, 200-move cap); per state, the move from the solved trace with
    the fewest moves left. Legal moves come from the state itself."""
    state0, n, gamma, seed = job
    random.seed(seed)
    orientations = car_orientations(state0)
    best = {}
    for _ in range(n):
        state, moves, visited = state0, [], []
        while len(moves) < 200 and ao._red_steps(state) > 0:
            attempt = ao.solve(state, gamma=gamma)
            step = attempt if type(attempt) is list else attempt[1]
            if not step:
                break
            for move in step:
                visited.append((state, move))
                state = ao.AndNode(state, *move).apply(state)
            moves.extend(step)
        if ao._red_steps(state) > 0:
            continue
        for t, (s, move) in enumerate(visited):
            left = len(visited) - t
            if s not in best or left < best[s][1]:
                best[s] = (move, left)
    out = []
    for s, (move, left) in best.items():
        sd = dict(s)
        legal = [action_index(sd, *m) for m, _ in neighbors(s, orientations)]
        out.append((encode(s), left, legal, [action_index(sd, *move)]))
    return out


# --- networks ---------------------------------------------------------------------

class OldNet(nn.Module):
    """The original rl.PolicyNetwork's architecture (no distance head)."""

    def __init__(self, cin, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Conv2d(cin, hidden, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU())
        self.head = nn.Linear(hidden * 36, N_ACTIONS)

    def forward(self, x):
        return self.head(self.trunk(x).flatten(1)), None


class LaneBlock(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.row = nn.Conv2d(h, h, (1, 11), padding=(0, 5))
        self.col = nn.Conv2d(h, h, (11, 1), padding=(5, 0))
        self.local = nn.Conv2d(h, h, 3, padding=1)
        self.norm = nn.BatchNorm2d(h)

    def forward(self, x):
        return F.relu(x + self.norm(self.local(F.relu(self.row(x) + self.col(x)))))


class LaneNet(nn.Module):
    """Residual blocks seeing whole rows and columns; a move head shared
    across positions (20 = 4 directions x 5 steps per cell) and a
    distance-to-goal head."""

    def __init__(self, cin, hidden=96, blocks=4):
        super().__init__()
        self.stem = nn.Conv2d(cin, hidden, 3, padding=1)
        self.blocks = nn.Sequential(*[LaneBlock(hidden) for _ in range(blocks)])
        self.policy = nn.Conv2d(hidden, 20, 1)
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.blocks(F.relu(self.stem(x)))
        logits = self.policy(h).permute(0, 2, 3, 1).reshape(x.shape[0], N_ACTIONS)
        return logits, self.value(h.mean((2, 3))).squeeze(1)


def make_net(kind, cin):
    return OldNet(cin) if kind == "old" else LaneNet(cin)


# --- training and evaluation -----------------------------------------------------------

def masks(examples, device):
    legal = torch.zeros(len(examples), N_ACTIONS, dtype=torch.bool)
    optimal = torch.zeros(len(examples), N_ACTIONS, dtype=torch.bool)
    for r, (_, _, leg, opt) in enumerate(examples):
        legal[r, leg] = True
        optimal[r, opt] = True
    return legal.to(device), optimal.to(device)


def batch_tensors(examples, input_kind, device):
    x = to_input(np.stack([e[0] for e in examples]), input_kind).to(device)
    d = torch.tensor([e[1] for e in examples], dtype=torch.float32, device=device)
    return x, d, *masks(examples, device)


def losses(net, examples, input_kind, device, use_value):
    x, d, legal, optimal = batch_tensors(examples, input_kind, device)
    logits, value = net(x)
    logp = torch.log_softmax(logits.masked_fill(~legal, float("-inf")), dim=1)
    policy = -logp.masked_fill(~optimal, float("-inf")).logsumexp(1).mean()
    if value is None or not use_value:
        return policy, policy
    return policy + 0.5 * F.smooth_l1_loss(value, d / 10), policy


@torch.no_grad()
def evaluate(net, examples, input_kind, device, batch=2048):
    """(agreement, random-pick agreement, distance MAE or None)."""
    net.eval()
    hits, chance, err, n = 0.0, 0.0, 0.0, 0
    has_value = True
    for i in range(0, len(examples), batch):
        chunk = examples[i:i + batch]
        x, d, legal, optimal = batch_tensors(chunk, input_kind, device)
        logits, value = net(x)
        top = logits.masked_fill(~legal, float("-inf")).argmax(1)
        hits += optimal.gather(1, top[:, None]).float().sum().item()
        chance += (optimal.sum(1).float() / legal.sum(1).float()).sum().item()
        if value is None:
            has_value = False
        else:
            err += (value * 10 - d).abs().sum().item()
        n += len(chunk)
    net.train()
    return hits / n, chance / n, (err / n if has_value else None)


def train(net, train_ex, val_ex, input_kind, device, use_value, epochs=40, patience=6, batch=256, lr=1e-3, log=None):
    """Adam on shuffled minibatches; keeps the weights with the best
    validation agreement (validation = held-out *training* puzzles)."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    best, best_state, stale = -1.0, None, 0
    for epoch in range(epochs):
        random.shuffle(train_ex)
        for i in range(0, len(train_ex), batch):
            loss, _ = losses(net, train_ex[i:i + batch], input_kind, device, use_value)
            opt.zero_grad()
            loss.backward()
            opt.step()
        agree = evaluate(net, val_ex, input_kind, device)[0] if val_ex else 0.0
        if log:
            log(f"    epoch {epoch + 1}: val agreement {agree:.1%}")
        if agree > best:
            best, best_state, stale = agree, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net


@torch.no_grad()
def rollout_solve_rate(net, puzzles, input_kind, device, cap=100):
    """Greedy rollouts that skip moves returning to an already-visited board
    (rl.PolicyAgent.policy_avoiding_cycles). Returns (solved fraction, mean
    moves / optimal over solved)."""
    net.eval()
    solved, ratios = 0, []
    for d0, state in puzzles:
        orientations = car_orientations(state)
        visited, moves = {state}, 0
        while moves < cap and max(j for _, j in dict(state)["red"]) != 5:
            options = list(neighbors(state, orientations))
            sd = dict(state)
            logits, _ = net(to_input(encode(state)[None], input_kind).to(device))
            scores = logits[0].cpu()
            ranked = sorted(options, key=lambda o: -scores[action_index(sd, *o[0])].item())
            nxt = next((s for _, s in ranked if s not in visited), ranked[0][1])
            state = nxt
            visited.add(state)
            moves += 1
        if max(j for _, j in dict(state)["red"]) == 5:
            solved += 1
            ratios.append(moves / d0)
    net.train()
    return solved / len(puzzles), (sum(ratios) / len(ratios) if ratios else float("nan"))


# --- data -----------------------------------------------------------------------------

def load_pool(path):
    with open(path) as f:
        return [parse_puzzle_line(line) for line in f]


def sample_by_bin(pool, per_bin, rng, exclude):
    out = []
    for lo, hi in BINS:
        candidates = [p for p in pool if lo <= p[0] <= hi and p[1] not in exclude]
        out += rng.sample(candidates, min(per_bin, len(candidates)))
    return out


def label_all(pool, puzzles, k, seed):
    jobs = [(state, k, hash((seed, i)) & 0xFFFFFFFF) for i, (_, state) in enumerate(puzzles)]
    return pool.map(label_puzzle, jobs, chunksize=4)


def report(name, net, input_kind, device, evals, rollout_sets, log):
    row = {"condition": name}
    for label, ex in evals.items():
        agree, chance, mae = evaluate(net, ex, input_kind, device)
        row[label] = {"agree": agree, "chance": chance, "mae": mae}
    for label, puzzles in rollout_sets.items():
        rate, ratio = rollout_solve_rate(net, puzzles, input_kind, device)
        row[label] = {"solved": rate, "moves_over_opt": ratio}
    parts = []
    for label, v in row.items():
        if label == "condition":
            continue
        if "agree" in v:
            mae = f", dist err {v['mae']:.1f}" if v["mae"] is not None else ""
            parts.append(f"{label}: agree {v['agree']:.0%} (chance {v['chance']:.0%}{mae})")
        else:
            parts.append(f"{label}: solved {v['solved']:.0%} ({v['moves_over_opt']:.1f}x opt)")
    log(f"  {name:<28} " + " | ".join(parts))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["cluster", "scale"], default="cluster")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 10000], help="scale: training puzzles")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="input/net/labels triples like new/lane/exact (default: all four for cluster, "
                         "new/lane/exact for scale)")
    ap.add_argument("--states-per-puzzle", type=int, default=50, help="scale stage and held-out evaluation")
    ap.add_argument("--heldout-per-bin", type=int, default=60)
    ap.add_argument("--rollout-per-bin", type=int, default=20)
    ap.add_argument("--traces", type=int, default=15, help="AND-OR traces per puzzle for 'traces' labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", help="write results as JSON here")
    args = ap.parse_args()

    def log(msg):
        print(msg, flush=True)

    rng = random.Random(args.seed)
    cluster = read_puzzles("../data/rush_nw_cluster.txt")
    c_states = [s for d in sorted(cluster) for s in cluster[d]]
    c_dists = [d for d in sorted(cluster) for _ in cluster[d]]
    c_train, c_test, c_train_d, c_test_d = train_test_split(c_states, c_dists, train_size=0.9, random_state=3)
    nw = load_pool("../data/rush_nw.txt")
    heldout = sample_by_bin(nw, args.heldout_per_bin, rng, set(c_states))
    excluded = set(c_states) | {s for _, s in heldout}

    with Pool() as pool:
        t0 = time.time()
        heldout_ex = [e for ex in label_all(pool, heldout, args.states_per_puzzle, args.seed + 1) for e in ex]
        c_test_ex = [e for ex in label_all(pool, list(zip(c_test_d, c_test)), None, args.seed + 2) for e in ex]
        log(f"labelled {len(heldout)} held-out puzzles ({len(heldout_ex)} states) and the cluster's "
            f"{len(c_test)} test puzzles ({len(c_test_ex)} states) in {time.time() - t0:.0f}s")
        evals = {"cluster test": c_test_ex, "held-out": heldout_ex}
        rollouts = {"held-out rollouts": [p for b in BINS for p in
                                          [h for h in heldout if b[0] <= h[0] <= b[1]][:args.rollout_per_bin]]}

        results = []
        if args.stage == "cluster":
            conditions = args.conditions or ["old/old/traces", "old/old/exact", "new/old/exact", "new/lane/exact"]
            exact_by_puzzle = label_all(pool, list(zip(c_train_d, c_train)), None, args.seed + 3)
            trace_by_puzzle = pool.map(trace_puzzle, [(s, args.traces, 0.2, i) for i, s in enumerate(c_train)])
            log(f"cluster training puzzles: {len(c_train)}; exact-labelled states "
                f"{sum(map(len, exact_by_puzzle))}, trace-labelled states {sum(map(len, trace_by_puzzle))}")
            # Validation (model selection): two of the 14 training puzzles, held out of fitting,
            # always scored against exact labels.
            val_ids = {0, 1}
            val = [e for i, ex in enumerate(exact_by_puzzle) if i in val_ids for e in ex]
            for cond in conditions:
                input_kind, net_kind, labels = cond.split("/")
                source = exact_by_puzzle if labels == "exact" else trace_by_puzzle
                fit = [e for i, ex in enumerate(source) if i not in val_ids for e in ex]
                torch.manual_seed(args.seed)
                net = make_net(net_kind, input_channels(input_kind)).to(args.device)
                t0 = time.time()
                train(net, fit, val, input_kind, args.device, use_value=labels == "exact")
                fit_agree = evaluate(net, fit, input_kind, args.device)[0]
                log(f"  [{cond}: {len(fit)} training states, fit agreement {fit_agree:.0%}, {time.time() - t0:.0f}s]")
                results.append(report(cond, net, input_kind, args.device, evals, rollouts, log))
        else:
            conditions = args.conditions or ["new/lane/exact"]
            biggest = max(args.sizes)
            needed = biggest + max(2, biggest // 10)
            train_pool = sample_by_bin(nw, -(-needed // len(BINS)), rng, excluded)
            rng.shuffle(train_pool)
            t0 = time.time()
            labelled = label_all(pool, train_pool, args.states_per_puzzle, args.seed + 4)
            log(f"labelled {len(train_pool)} training puzzles in {time.time() - t0:.0f}s")
            for n in args.sizes:
                n_val = max(2, n // 10)
                fit = [e for ex in labelled[n_val:n_val + n] for e in ex]
                val = [e for ex in labelled[:n_val] for e in ex]
                for cond in conditions:
                    input_kind, net_kind, _ = cond.split("/")
                    torch.manual_seed(args.seed)
                    net = make_net(net_kind, input_channels(input_kind)).to(args.device)
                    t0 = time.time()
                    train(net, fit, val, input_kind, args.device, use_value=True)
                    fit_agree = evaluate(net, fit[:50_000], input_kind, args.device)[0]
                    log(f"  [{cond} on {n} puzzles: {len(fit)} states, fit agreement {fit_agree:.0%}, "
                        f"{time.time() - t0:.0f}s]")
                    results.append(report(f"{cond} @ {n}", net, input_kind, args.device, evals, rollouts, log))

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
