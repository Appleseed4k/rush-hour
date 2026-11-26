"""A video of the AND-OR model solving one of people's layouts, and its trajectory: distance to goal per step.

The model is and_or.solve played to the exit (student.solve.solve_and_or): repeated passes of AND-OR subgoal
decomposition, unordered (no network heuristic), with lapse rate gamma, protection off (the apprentice's expert).
Distance to goal is the board's optimal number of moves left (board.multi_bfs), red's exit counted.

    python experiments/and_or_video.py                                 # prb12715_11, seed 0
    python experiments/and_or_video.py --layout prb10166_16 --seed 3

Writes <layout>_seed<seed>.mp4 and <layout>_seed<seed>_trajectory.png to visualization/and_or_solve/ (git-ignored).
The mp4 needs ffmpeg: on PATH, or given with --ffmpeg.
"""
import argparse
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Rectangle

from rushhour import board, human_data as hd, paths
from rushhour.and_or import AndNode
from rushhour.student.solve import solve_and_or

OUT = os.path.join(paths.ROOT, "visualization", "and_or_solve")
RED, CAR, GRID, INK = "#c0392b", "#b8b8b8", "#e6e6e6", "#333333"


def replay(state, moves):
    """Every board from `state` through each move."""
    states = [state]
    for car, direction, steps in moves:
        state = AndNode(state, car, direction, steps).apply(state)
        states.append(state)
    return states


def draw_board(ax, state):
    ax.clear()
    ax.set_xlim(-0.1, 6.1)
    ax.set_ylim(6.1, -0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    for k in range(7):
        ax.plot([0, 6], [k, k], color=GRID, lw=0.8, zorder=0)
        ax.plot([k, k], [0, 6], color=GRID, lw=0.8, zorder=0)
    # outline, open at the exit (row 2, right side)
    ax.plot([6, 6, 0, 0, 6, 6], [2, 0, 0, 6, 6, 3], color=INK, lw=2)
    for name, cells in state:
        rows = [i for i, _ in cells]
        cols = [j for _, j in cells]
        pad = 0.08
        ax.add_patch(Rectangle((min(cols) + pad, min(rows) + pad), max(cols) - min(cols) + 1 - 2 * pad,
                               max(rows) - min(rows) + 1 - 2 * pad, color=RED if name == "red" else CAR, lw=0))


def video(states, path, ffmpeg, fps):
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg
    fig, ax = plt.subplots(figsize=(4, 4.3), dpi=120)
    fig.subplots_adjust(0.02, 0.02, 0.98, 0.92)
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    with writer.saving(fig, path, dpi=120):
        for step, state in enumerate(states):
            draw_board(ax, state)
            ax.set_title(f"step {step}", fontsize=10, color=INK, loc="left")
            for _ in range(2 if step in (0, len(states) - 1) else 1):  # hold the first and last boards
                writer.grab_frame()
    plt.close(fig)


def trajectory(distances, path):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(range(len(distances)), distances, color=INK, lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("distance to goal")
    ax.set_ylim(bottom=0)
    ax.set_xlim(0, len(distances) - 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", default="prb12715_11")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.07)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    args = parser.parse_args()

    random.seed(args.seed)
    start = hd.puzzle(args.layout)
    moves = solve_and_or(start, max_steps=args.max_steps, gamma=args.gamma)
    if moves is None:
        sys.exit(f"unsolved within {args.max_steps} moves - try another --seed")
    states = replay(start, moves)
    dist = board.multi_bfs(start)
    distances = [dist[s] for s in states]
    print(f"{args.layout} seed {args.seed}: solved in {len(moves)} moves (optimal {distances[0]})")
    print("distance to goal:", distances)

    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"{args.layout}_seed{args.seed}")
    trajectory(distances, stem + "_trajectory.png")
    if not args.ffmpeg:
        sys.exit("no ffmpeg found: give --ffmpeg (the trajectory was written)")
    video(states, stem + ".mp4", args.ffmpeg, args.fps)
    print("wrote", stem + ".mp4", "and", stem + "_trajectory.png")


if __name__ == "__main__":
    main()
