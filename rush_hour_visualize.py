import argparse

from rush_hour_lib import read_puzzles, save_puzzle_images


def main():
    parser = argparse.ArgumentParser(
        description="Render every puzzle in a rush_nw.txt-format file to a PNG in an output folder.")
    parser.add_argument("input", help="rush_nw.txt-format puzzle file, e.g. a cluster written by rush_hour_fingerprint.py")
    parser.add_argument("output_dir", help="folder to write puzzle_NNN.png files into (cleared first)")
    args = parser.parse_args()

    entries = [(dist, state) for dist, states in sorted(read_puzzles(args.input).items()) for state in states]
    states = [state for _, state in entries]
    titles = [f"distance {dist}" for dist, _ in entries]

    save_puzzle_images(states, args.output_dir, titles)
    print(f"wrote {len(states)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
