import argparse
import random
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.cluster._hdbscan.hdbscan")

from rush_hour_and_or import AndNode, OrNode, red_candidates
from rush_hour_lib import parse_puzzle_line

BOARD_SIZE = 6


# Deterministic, tree-capturing mirror of AndNode/OrNode.solve(): uses only
# the pure helper methods (.directions(), .blockers(), .apply()), never
# .solve() itself, so candidate order is whatever those naturally produce
# (nearest-collision-first) instead of GAMMA/heuristic-randomized. This finds
# at most one solution per puzzle - the one a single fixed resolution order
# reaches - which is enough to fingerprint its shape, not to guarantee it
# solves every puzzle.

def move_descriptor(state, car_name, direction, steps):
    """Bucketed, car-identity-free description of one move: its orientation,
    a coarse step-size class, and whether the mover sits against a board
    edge (edge-pinned cars have a restricted move range, which matters more
    to the strategy than their exact row/column)."""
    horizontal = direction in ("l", "r")
    car = dict(state)[car_name]
    cross = car[0][0] if horizontal else car[0][1]
    step_bucket = str(steps) if steps <= 2 else "3+"
    edge_bucket = "edge" if cross in (0, BOARD_SIZE - 1) else "mid"
    return ("h" if horizontal else "v", step_bucket, edge_bucket)


def resolve_and(state, car_name, direction, steps, visited, protected):
    node = AndNode(state, car_name, direction, steps, visited, protected)
    blockers = node.blockers()
    if protected & blockers.keys():
        return None
    working_state = state
    children = []
    next_protected = protected | {car_name}
    for blocker_name, collisions in blockers.items():
        result = resolve_or(working_state, blocker_name, collisions, visited, next_protected)
        if result is None:
            return None
        working_state, child_tree = result
        children.append(child_tree)
    if AndNode(working_state, car_name, direction, steps, visited).blockers():
        return None
    final_state = node.apply(working_state)
    tree = {
        "car": car_name,
        "action": (direction, steps),
        "descriptor": move_descriptor(state, car_name, direction, steps),
        "children": children,
    }
    return final_state, tree


def resolve_or(state, car_name, collisions, visited, protected):
    key = (car_name, frozenset(collisions))
    if key in visited:
        return None
    next_visited = visited | {key}
    node = OrNode(state, car_name, collisions)
    for direction, steps in node.directions():
        result = resolve_and(state, car_name, direction, steps, next_visited, protected)
        if result is not None:
            return result
    return None


def solve_forest(state, max_attempts=50):
    """Deterministic mirror of rush_hour_rl._solve_trace: repeatedly attempts
    the direct slide (or, failing that, a reposition) until red reaches the
    exit. Each attempt is one AND/OR proof tree; returns the list of them (one
    per top-level attempt), or None if it doesn't converge within
    max_attempts."""
    attempts = []
    for _ in range(max_attempts):
        red = dict(state)["red"]
        steps = (BOARD_SIZE - 1) - max(j for _, j in red)
        if steps <= 0:
            return attempts

        result = resolve_and(state, "red", "r", steps, frozenset(), frozenset())
        if result is not None:
            state, tree = result
            attempts.append(tree)
            continue  # direct slide always ends exactly at the exit

        found = False
        for direction, steps2 in red_candidates(state, steps):
            result = resolve_and(state, "red", direction, steps2, frozenset(), frozenset())
            if result is not None:
                state, tree = result
                attempts.append(tree)
                found = True
                break
        if not found:
            return None
    return None


def _depth(tree):
    if not tree["children"]:
        return 1
    return 1 + max(_depth(c) for c in tree["children"])


def _collect_descriptors(tree, out):
    out.append(tree["descriptor"])
    for c in tree["children"]:
        _collect_descriptors(c, out)


DESCRIPTOR_ALPHABET = [(o, s, e) for o in ("h", "v") for s in ("1", "2", "3+") for e in ("edge", "mid")]


def feature_vector(forest):
    """Turns a solve_forest() result into a numeric vector: the normalized
    histogram of move descriptors (so puzzles are compared by the *mix* of
    move types they need, not raw counts) plus a few whole-tree stats
    (attempt count, max depth, total move count) that still matter but
    shouldn't gate matches on their own."""
    descriptors = []
    for tree in forest:
        _collect_descriptors(tree, descriptors)
    counts = Counter(descriptors)
    total = len(descriptors)
    hist = [counts.get(key, 0) / total for key in DESCRIPTOR_ALPHABET]
    max_depth = max(_depth(t) for t in forest)
    return np.array(hist + [len(forest), max_depth, total], dtype=float)


def cluster_features(features, min_cluster_size, min_samples=None):
    """HDBSCAN over z-scored `features` (one row per puzzle). Returns an array
    of cluster labels, -1 for noise. Unlike plain DBSCAN, it extracts stable
    clusters across a range of densities instead of a single fixed radius, so
    it doesn't need re-tuning every time the sample size (and therefore the
    local density of the feature space) changes."""
    X = StandardScaler(copy=False).fit_transform(features)
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(X)


def sample_puzzles(pool_path, count, seed=None):
    """Uniformly samples `count` random (distance, state, line) puzzles from a
    rush_nw.txt-format pool file, without parsing the whole pool. `line` is
    the puzzle's original raw line (no trailing newline), kept so a chosen
    subset can be written back out verbatim."""
    rng = random.Random(seed)
    with open(pool_path) as f:
        lines = f.readlines()
    sampled = rng.sample(lines, min(count, len(lines)))
    return [(*parse_puzzle_line(line), line.rstrip("\n")) for line in sampled]


def cluster_puzzles(entries, min_cluster_size, min_samples=None):
    """Runs solve_forest() + feature_vector() on every entry, then HDBSCAN over
    the results. Returns (clusters, failures): clusters maps cluster id ->
    list of indices into `entries` (noise points are omitted); failures is
    how many entries didn't converge under the deterministic decomposition."""
    converged_idx = []
    feats = []
    failures = 0
    for idx, (_dist, state, _line) in enumerate(entries):
        forest = solve_forest(state)
        if forest is None:
            failures += 1
            continue
        converged_idx.append(idx)
        feats.append(feature_vector(forest))

    if not feats:
        return {}, failures

    labels = cluster_features(np.array(feats), min_cluster_size, min_samples)
    clusters = defaultdict(list)
    for local_idx, label in enumerate(labels):
        if label != -1:
            clusters[int(label)].append(converged_idx[local_idx])
    return dict(clusters), failures


def write_cluster(entries, indices, output_path):
    """Writes the puzzles at `indices` into `entries` to output_path as raw
    "<distance> <bitboard> <count>" lines (rush_nw.txt format), ready to be
    read back with read_puzzles()/sample_puzzles() from rush_hour_lib."""
    lines = [entries[i][2] for i in indices]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Cluster randomly sampled Rush Hour puzzles by AND/OR solution shape, via HDBSCAN over a "
                     "bucketed-move-type feature vector.")
    parser.add_argument("--pool", default="data/rush_nw.txt", help="rush_nw.txt-format puzzle pool")
    parser.add_argument("--count", type=int, default=500, help="number of puzzles to sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=2, help="HDBSCAN minimum puzzles to form a cluster")
    parser.add_argument("--min-samples", type=int, default=None,
                         help="HDBSCAN density-conservativeness knob; higher = stricter, more noise (default: same as --min-cluster-size)")
    parser.add_argument("--min-print-size", type=int, default=2, help="only print clusters with at least this many puzzles")
    parser.add_argument("--output", help="write the chosen cluster's puzzles to this rush_nw.txt-format file")
    parser.add_argument("--cluster-rank", type=int, default=0,
                         help="which cluster to write when --output is given: 0 = most puzzles (default), 1 = next, ...")
    args = parser.parse_args()

    entries = sample_puzzles(args.pool, args.count, args.seed)
    clusters, failures = cluster_puzzles(entries, args.min_cluster_size, args.min_samples)

    print(f"{len(entries)} puzzles sampled from {args.pool}")
    print(f"{failures}/{len(entries)} did not converge under the deterministic decomposition")
    print(f"{len(clusters)} clusters found (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples})\n")

    by_size = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    for cluster_id, members in by_size:
        if len(members) < args.min_print_size:
            break
        dists = sorted({entries[i][0] for i in members})
        print(f"cluster {cluster_id}  ({len(members)} puzzles, distances {dists})")

    if args.output:
        if not by_size:
            print("\nno clusters found; nothing written")
            return
        if not 0 <= args.cluster_rank < len(by_size):
            print(f"\n--cluster-rank {args.cluster_rank} out of range ({len(by_size)} clusters found)")
            return
        cluster_id, members = by_size[args.cluster_rank]
        write_cluster(entries, members, args.output)
        print(f"\nwrote {len(members)} puzzles (cluster {cluster_id}) to {args.output}")


if __name__ == "__main__":
    main()
