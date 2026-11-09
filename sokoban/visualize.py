"""Frame-by-frame pictures of the AND-OR solver playing a level, like
scripts/lib.py's visualize() for Rush Hour.

    python visualize.py                                   # auto-pick a Boxoban hard level
    python visualize.py --source hard/000:12              # a specific level
    python visualize.py --dataset data/microban1.txt --source "microban/microban1:3"
    python visualize.py --gamma 0.05 --want-restart       # show lapses and a restart

Writes into visualization/<level>/:
    move_000.png ...  the board before the first push and after every push.
                      The arrow is the push just played, coloured by what the
                      AND-OR tree played it for: delivering a box (blue), clearing
                      a blocker (orange, with its depth), a lapse (red) or a random
                      fallback (purple). The box being delivered is outlined in blue
                      and its goal ringed. Boxes are lettered so they can be followed.
    trajectory.png    oracle distance to the goal after each push, with restarts
                      and pass boundaries marked.
    solve.gif         the frames as an animation.
"""
import argparse
import os
import random
import shutil
import string

from and_or import play
from label import StateSpace
from level import Level

ROLE_COLORS = {"deliver": "royalblue", "blocker": "darkorange", "lapse": "crimson", "random": "purple"}
WALL, FLOOR, BOX, BOX_ON_GOAL, PLAYER, GOAL = "#3d3d3d", "#efe9dc", "#d9a441", "#6aa84f", "#2b6cb0", "#c0392b"


def draw(ax, level, state, ids, push=None, role=None, target=None, goal=None):
    """Draws `state` onto ax. `ids` maps box cell -> letter. `push` (the push
    just played, as (box cell before, direction)) gets an arrow coloured by
    `role`; `target` (a box cell) is outlined and `goal` ringed in blue."""
    import matplotlib.pyplot as plt

    boxes, player = state
    for i in range(level.height):
        for j in range(level.width):
            c = level.cell(i, j)
            if level.walls >> c & 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=WALL, edgecolor=WALL))
            elif level.floor >> c & 1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=FLOOR, edgecolor="#d8d0bf", linewidth=0.5))
                if level.goals >> c & 1:
                    ax.add_patch(plt.Circle((j + 0.5, i + 0.5), 0.18, facecolor=GOAL, edgecolor="none"))
    if goal is not None:
        gi, gj = level.coords(goal)
        ax.add_patch(plt.Circle((gj + 0.5, gi + 0.5), 0.42, facecolor="none", edgecolor="royalblue",
                                linewidth=2.5, linestyle="--"))
    for c, letter in ids.items():
        i, j = level.coords(c)
        on_goal = level.goals >> c & 1
        ax.add_patch(plt.Rectangle((j + 0.08, i + 0.08), 0.84, 0.84, facecolor=BOX_ON_GOAL if on_goal else BOX,
                                   edgecolor="black", linewidth=1.2))
        ax.text(j + 0.5, i + 0.5, letter, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        if c == target:
            ax.add_patch(plt.Rectangle((j + 0.02, i + 0.02), 0.96, 0.96, facecolor="none", edgecolor="royalblue",
                                       linewidth=3.5))
    pi, pj = level.coords(player)
    ax.add_patch(plt.Circle((pj + 0.5, pi + 0.5), 0.3, facecolor=PLAYER, edgecolor="black", linewidth=1))
    if push is not None:
        b, d = push
        (bi, bj), (ni, nj) = level.coords(b), level.coords(b + level.shifts[d])
        ax.annotate("", xy=(nj + 0.5, ni + 0.5), xytext=(bj + 0.5, bi + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color=ROLE_COLORS.get(role, "black"), lw=3,
                                    mutation_scale=18, shrinkA=0, shrinkB=8))
    ax.set_xlim(0, level.width)
    ax.set_ylim(level.height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


def describe(info, box, level):
    """One line saying what a push was for; `box` is the delivered box's letter."""
    role = info["role"]
    if role == "lapse":
        return "lapse: the pass gives up for a random push"
    if role == "random":
        return "no plan found: random push"
    gi, gj = level.coords(info["goal"])
    if role == "deliver":
        return f"delivering box {box} to goal ({gi},{gj})"
    return f"clearing a blocker (depth {info['depth']}) for box {box}'s delivery"


def _plural(n, word):
    return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith("s") else "s")


def render(level, outcome, out_dir, space=None, gif=True, frame_ms=450):
    """Frames, trajectory and GIF for a play() outcome (see module docstring)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    space = space or StateSpace(level, live_only=False)

    letters = iter(string.ascii_uppercase + string.ascii_lowercase)
    start_ids = {c: next(letters) for c in level.box_cells(level.start)}
    scale = 0.55
    figsize = (max(level.width * scale, 5.5), level.height * scale + 1.1)

    frames, trace, pass_starts, restarts = [], [space.distance(level.start)], [], []

    def save(state, ids, title, subtitle, **kw):
        fig, ax = plt.subplots(figsize=figsize)
        draw(ax, level, state, ids, **kw)
        ax.set_title(f"{title}\n{subtitle}", fontsize=10, loc="left")
        path = os.path.join(out_dir, f"move_{len(frames):03d}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        frames.append(path)

    state, ids, n = level.start, dict(start_ids), 0
    d0 = space.distance(state)
    save(state, ids, f"{level.name}: start", f"optimal {d0} pushes")
    target_id, task = None, None
    for k, entry in enumerate(outcome["log"], 1):
        pass_starts.append(n)
        for push, info in zip(entry["moves"], entry["roles"]):
            ids_before = ids
            if info["role"] in ("deliver", "blocker") and (k, info["box"], info["goal"]) != task:
                # a new delivery: its box hasn't moved yet, so it still stands on info["box"]
                task = (k, info["box"], info["goal"])
                target_id = ids_before.get(info["box"])
            b, d = push
            state = level.apply(state, push)
            ids = {(b + level.shifts[d] if c == b else c): letter for c, letter in ids_before.items()}
            n += 1
            dist = space.distance(state)
            trace.append(dist)
            target = next((c for c, letter in ids.items() if letter == target_id), None) \
                if info["role"] in ("deliver", "blocker") else None
            dist_text = "deadlock" if dist == -1 else f"{dist} to go"
            save(state, ids, f"push {n} - pass {k} ({entry['kind']}) - {dist_text}",
                 describe(info, target_id, level), push=push, role=info["role"], target=target,
                 goal=info.get("goal"))
        if entry["restart"]:
            restarts.append(n)
            state, ids, target_id = level.start, dict(start_ids), None
            trace.append(space.distance(state))
            n += 1
            save(state, ids, f"restart {len(restarts)}", "deadlock detected (or no plan for too long): back to the start")
    if level.solved(state):
        save(state, ids, f"solved in {outcome['pushes']} pushes", f"optimal {d0}, {outcome['restarts']} restart(s)")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    xs = range(len(trace))
    ys = [d if d is not None and d >= 0 else float("nan") for d in trace]
    ax.plot(xs, ys, color="gray", marker="o", markersize=3)
    dead = [x for x, d in zip(xs, trace) if d == -1]
    if dead:
        ax.scatter(dead, [0] * len(dead), color="crimson", marker="x", label="deadlocked", zorder=3)
    for x in pass_starts[1:]:
        ax.axvline(x, color="lightgray", linewidth=0.8, zorder=0)
    for x in restarts:
        ax.axvline(x + 0.5, color="crimson", linestyle="--", label="restart" if x == restarts[0] else None)
    ax.set_xlabel("frame (push, or restart)")
    ax.set_ylabel("pushes to goal (oracle)")
    ax.set_ylim(bottom=0)
    ax.set_title(f"{level.name}: optimal {d0}, solver used {outcome['pushes']} pushes "
                 f"({_plural(len(outcome['log']), 'pass')}, {_plural(outcome['restarts'], 'restart')})", fontsize=10)
    if dead or restarts:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory.png"), dpi=110)
    plt.close(fig)

    if gif:
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in frames]
        w = max(im.width for im in images)
        h = max(im.height for im in images)
        canvas = [Image.new("RGB", (w, h), "white") for _ in images]
        for c, im in zip(canvas, images):
            c.paste(im, (0, 0))
        durations = [frame_ms] * len(canvas)
        durations[0] = durations[-1] = frame_ms * 4
        canvas[0].save(os.path.join(out_dir, "solve.gif"), save_all=True, append_images=canvas[1:],
                       duration=durations, loop=0)
    return frames


def pick(rows, args):
    """The first level (in a seeded shuffle) whose run fits the request."""
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    for optimal, board, source in rows:
        if not args.min_pushes <= optimal <= args.max_pushes:
            continue
        level = Level.from_line(board, source)
        outcome = play(level, gamma=args.gamma, max_pushes=args.max_solver_pushes, seed=args.seed)
        roles = [r for e in outcome["log"] for r in e["roles"]]
        depth = max((r.get("depth", 0) for r in roles), default=0)
        restarts_ok = outcome["restarts"] > 0 if args.want_restart else outcome["restarts"] <= args.max_restarts
        if outcome["solved"] and depth >= args.min_depth and restarts_ok:
            return level, outcome
    raise SystemExit("no level in the dataset fits; loosen --min-depth / --want-restart / the push range")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="data/boxoban_hard.txt")
    parser.add_argument("--source", help="the level's source field, e.g. hard/000:12 (default: auto-pick)")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-pushes", type=int, default=15, help="auto-pick: optimal pushes at least")
    parser.add_argument("--max-pushes", type=int, default=35, help="auto-pick: optimal pushes at most")
    parser.add_argument("--min-depth", type=int, default=2, help="auto-pick: deepest blocker chain at least")
    parser.add_argument("--want-restart", action="store_true", help="auto-pick: a run with a restart")
    parser.add_argument("--max-restarts", type=int, default=0, help="auto-pick: at most this many restarts")
    parser.add_argument("--max-solver-pushes", type=int, default=400)
    parser.add_argument("--out", default=None, help="output folder (default visualization/<level>)")
    args = parser.parse_args(argv)

    from evaluate import read_dataset
    rows = read_dataset(args.dataset)
    if args.source:
        match = [r for r in rows if r[2] == args.source]
        if not match:
            raise SystemExit(f"{args.source} not in {args.dataset}")
        optimal, board, source = match[0]
        level = Level.from_line(board, source)
        outcome = play(level, gamma=args.gamma, max_pushes=args.max_solver_pushes, seed=args.seed)
    else:
        level, outcome = pick(rows, args)
    slug = level.name.replace("/", "_").replace(":", "_").replace(" ", "_").replace("'", "")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization", slug)
    frames = render(level, outcome, out)
    print(f"{level.name}: solved={outcome['solved']} pushes={outcome['pushes']} restarts={outcome['restarts']} "
          f"passes={outcome['passes']}\n{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
