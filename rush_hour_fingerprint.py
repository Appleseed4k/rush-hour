import argparse
import random
import warnings
from collections import defaultdict

import numpy as np
from sklearn.cluster import HDBSCAN
from tqdm import tqdm

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
    """Car-identity-free description of one move: its orientation, exact step
    count, and whether the mover sits against a board edge (edge-pinned cars
    have a restricted move range, which matters more to the strategy than
    their exact row/column). Left un-bucketed (unlike earlier versions) so
    move_distance() can score near-variants - e.g. a 2-step vs. 3-step move
    that's otherwise identical - as close rather than an outright mismatch."""
    horizontal = direction in ("l", "r")
    car = dict(state)[car_name]
    cross = car[0][0] if horizontal else car[0][1]
    edge = cross in (0, BOARD_SIZE - 1)
    return (horizontal, steps, edge)


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


def move_distance(a, b):
    """Graded substitution cost between two move descriptors (horizontal,
    steps, edge): 0 for an exact match, small for a close variant (e.g. same
    orientation and edge-status but 2 steps instead of 3 - what you'd see if
    a blocking car were one cell longer), larger for a genuinely different
    kind of move (different orientation, or free-to-move vs. edge-pinned)."""
    horizontal_a, steps_a, edge_a = a
    horizontal_b, steps_b, edge_b = b
    cost = 0.0 if horizontal_a == horizontal_b else 1.0
    cost += min(abs(steps_a - steps_b) / 4.0, 1.0)
    cost += 0.0 if edge_a == edge_b else 0.5
    return cost


def _tree_size(tree):
    """Number of moves (nodes) in this subtree, including itself."""
    return 1 + sum(_tree_size(c) for c in tree["children"])


def subtree_cost(tree, gap_cost):
    """Cost of deleting (or, symmetrically, inserting) this entire subtree
    wholesale: gap_cost per move it contains."""
    return gap_cost * _tree_size(tree)


def _match_children(children_a, children_b, gap_cost):
    """Minimum-cost way to pair up two nodes' children: each pairing costs
    tree_distance() between the two subtrees, and any child left over on
    either side is charged its own subtree_cost() (a whole-subtree
    insert/delete). Order doesn't matter - this considers every possible
    pairing and picks the cheapest, which is what makes independent/
    commutative blockers (children of the same AndNode) free to have been
    resolved in either order. Exact via a bitmask DP over the smaller side,
    fine given how few children an AndNode/OrNode has on a 6x6 board."""
    na, nb = len(children_a), len(children_b)
    if na == 0:
        return sum(subtree_cost(c, gap_cost) for c in children_b)
    if nb == 0:
        return sum(subtree_cost(c, gap_cost) for c in children_a)

    pair_cost = [[tree_distance(ca, cb, gap_cost) for cb in children_b] for ca in children_a]
    delete_cost = [subtree_cost(ca, gap_cost) for ca in children_a]
    insert_cost = [subtree_cost(cb, gap_cost) for cb in children_b]

    dp = {0: 0.0}
    for i in range(na):
        new_dp = {}
        for mask, cost in dp.items():
            candidate = cost + delete_cost[i]
            if candidate < new_dp.get(mask, float("inf")):
                new_dp[mask] = candidate
            for j in range(nb):
                if not (mask & (1 << j)):
                    key = mask | (1 << j)
                    candidate = cost + pair_cost[i][j]
                    if candidate < new_dp.get(key, float("inf")):
                        new_dp[key] = candidate
        dp = new_dp

    return min(
        cost + sum(insert_cost[j] for j in range(nb) if not (mask & (1 << j)))
        for mask, cost in dp.items()
    )


def tree_distance(a, b, gap_cost=1.0):
    """Tree edit distance between two AND/OR proof trees, with unordered
    children: substitution cost of the two nodes' own moves, plus the
    cheapest way to match their children sets (see _match_children).
    A child genuinely must precede its parent (that dependency is real,
    captured by the recursion), but siblings - blockers resolved at the same
    level - can be matched in whatever pairing is cheapest, since swapping
    the order two independent blockers were cleared in isn't a real
    difference between two puzzles."""
    return move_distance(a["descriptor"], b["descriptor"]) + _match_children(a["children"], b["children"], gap_cost)


def forest_distance(forest_a, forest_b, gap_cost=1.0):
    """Distance between two solve_forest() results: a sequence alignment over
    top-level attempts (these genuinely happen one after another - each
    reposition starts from wherever the last one left red, so they're not
    interchangeable like same-level blockers are), where the substitution
    cost between two attempts is their tree_distance(). Normalized by the
    longer forest's total move count."""
    na, nb = len(forest_a), len(forest_b)
    if na == 0 and nb == 0:
        return 0.0
    dp = [[0.0] * (nb + 1) for _ in range(na + 1)]
    for i in range(1, na + 1):
        dp[i][0] = dp[i - 1][0] + subtree_cost(forest_a[i - 1], gap_cost)
    for j in range(1, nb + 1):
        dp[0][j] = dp[0][j - 1] + subtree_cost(forest_b[j - 1], gap_cost)
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            sub = dp[i - 1][j - 1] + tree_distance(forest_a[i - 1], forest_b[j - 1], gap_cost)
            delete = dp[i - 1][j] + subtree_cost(forest_a[i - 1], gap_cost)
            insert = dp[i][j - 1] + subtree_cost(forest_b[j - 1], gap_cost)
            dp[i][j] = min(sub, delete, insert)

    total_a = sum(_tree_size(t) for t in forest_a)
    total_b = sum(_tree_size(t) for t in forest_b)
    return dp[na][nb] / (gap_cost * max(total_a, total_b))


def _merge_clusters_by_epsilon(dist, labels, epsilon):
    """Single-linkage-style merge: unions any two clusters whose closest pair
    of member points is within epsilon of each other, repeatedly (via
    union-find, so chains of near clusters all end up together). This is a
    manual reimplementation of what HDBSCAN's own cluster_selection_epsilon
    is supposed to do - it's needed because that parameter crashes as of
    sklearn 1.9.1 under numpy>=2.0 (TypeError deep in the Cython tree code,
    reproducible even on a random precomputed distance matrix, so it's an
    upstream bug, not something specific to this data)."""
    if epsilon <= 0:
        return labels

    unique = sorted(c for c in set(labels.tolist()) if c != -1)
    if len(unique) < 2:
        return labels

    parent = {c: c for c in unique}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    members = {c: np.where(labels == c)[0] for c in unique}
    for i, a in enumerate(unique):
        for b in unique[i + 1:]:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            if dist[np.ix_(members[a], members[b])].min() <= epsilon:
                parent[ra] = rb

    roots = sorted({find(c) for c in unique})
    canon = {root: i for i, root in enumerate(roots)}
    return np.array([canon[find(c)] if c != -1 else -1 for c in labels])


def cluster_forests(forests, min_cluster_size, min_samples=None, epsilon=0.0):
    """HDBSCAN over the pairwise forest_distance matrix, followed by an
    epsilon merge pass (see _merge_clusters_by_epsilon) to combine clusters
    that sit close together. Returns an array of cluster labels, -1 for
    noise. This is O(n^2) tree comparisons, so unlike the earlier
    histogram-based clustering, it doesn't scale to tens of thousands of
    puzzles - it's meant for hundreds to a couple thousand."""
    n = len(forests)
    dist = np.zeros((n, n))
    pairs = ((i, j) for i in range(n) for j in range(i + 1, n))
    for i, j in tqdm(pairs, total=n * (n - 1) // 2, desc="pairwise distances", unit="pair"):
        d = forest_distance(forests[i], forests[j])
        dist[i, j] = dist[j, i] = d
    labels = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                      metric="precomputed").fit_predict(dist)
    return _merge_clusters_by_epsilon(dist, labels, epsilon)


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


def cluster_puzzles(entries, min_cluster_size, min_samples=None, epsilon=0.0):
    """Runs solve_forest() on every entry, then HDBSCAN over pairwise
    tree-edit-distance forest comparisons (see cluster_forests). Returns
    (clusters, failures): clusters maps cluster id -> list of indices into
    `entries` (noise points omitted); failures is how many entries didn't
    converge under the deterministic decomposition."""
    converged_idx = []
    forests = []
    failures = 0
    for idx, (_dist, state, _line) in enumerate(entries):
        forest = solve_forest(state)
        if forest is None:
            failures += 1
            continue
        converged_idx.append(idx)
        forests.append(forest)

    if not forests:
        return {}, failures

    labels = cluster_forests(forests, min_cluster_size, min_samples, epsilon)
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
        description="Cluster randomly sampled Rush Hour puzzles by AND/OR solution shape, via HDBSCAN over "
                     "pairwise tree-edit-distance comparisons (order-sensitive across attempts, order-free among "
                     "independent blockers).")
    parser.add_argument("--pool", default="data/rush_nw.txt", help="rush_nw.txt-format puzzle pool")
    parser.add_argument("--count", type=int, default=500, help="number of puzzles to sample")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-cluster-size", type=int, default=2, help="HDBSCAN minimum puzzles to form a cluster")
    parser.add_argument("--min-samples", type=int, default=None,
                         help="HDBSCAN density-conservativeness knob; higher = stricter, more noise (default: same as --min-cluster-size)")
    parser.add_argument("--epsilon", type=float, default=0.0,
                         help="HDBSCAN cluster_selection_epsilon: merges clusters/points within this forest_distance into one (default: 0.0, no merging)")
    parser.add_argument("--min-print-size", type=int, default=2, help="only print clusters with at least this many puzzles")
    parser.add_argument("--output", help="write the chosen cluster's puzzles to this rush_nw.txt-format file")
    parser.add_argument("--cluster-rank", type=int, default=0,
                         help="which cluster to write when --output is given: 0 = most puzzles (default), 1 = next, ...")
    args = parser.parse_args()

    entries = sample_puzzles(args.pool, args.count, args.seed)
    clusters, failures = cluster_puzzles(entries, args.min_cluster_size, args.min_samples, args.epsilon)

    print(f"{len(entries)} puzzles sampled from {args.pool}")
    print(f"{failures}/{len(entries)} did not converge under the deterministic decomposition")
    print(f"{len(clusters)} clusters found (min_cluster_size={args.min_cluster_size}, min_samples={args.min_samples}, epsilon={args.epsilon})\n")

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
