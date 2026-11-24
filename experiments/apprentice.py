"""Pretrains a network as AND-OR's apprentice by expert iteration (student.expert_iteration, student "and_or"): round 1
learns from plain AND-OR's solved attempts, each later round from AND-OR ordered by the network it has become. The
settings are phase 1's (phase 1, at tag pre-cleanup): the same
1,000 pool puzzles every round (200 per distance bin 1-8 ... 21-30), the same validation puzzles, 4 attempts per
puzzle, up to 40 epochs a round with patience 6 - but the expert's lapse rate (--gamma) and protect_movers
(--protect-movers) can be set. notebooks/archive/p1.3_session.ipynb's gamma section compares plain AND-OR with people at
each setting.

Saved under models/apprentice/<run>/: config.json, and the network before training and after every epoch of every
round (round{r}_epoch{e}.pt), with manifest.json marking the kept ones and history.json (see expert_iteration).
Candidate starting points for a session are then fitted to people's start with `python -m rushhour.surrender --folder`.

    python apprentice.py gamma0.07_free                                 # the session start's run
    python apprentice.py gamma0.01_free --gamma 0.01
    python apprentice.py gamma0.07_protected --protect-movers on        # as phase 1
    python apprentice.py smoke --per-bin 10 --rounds 1 --epochs 2      # a quick check
    python apprentice.py gamma0.07_free_nodest --rounds 1 --epochs 11 --patience 40   # paths.START's run
    python apprentice.py gamma0.07_free_novalue --rounds 1 --epochs 11 --patience 40 --value-weight 0
                                                                       # ... without the value loss too
    python apprentice.py gamma0.07_free_valueonly --rounds 1 --epochs 11 --patience 40 --policy-weight 0
                                                                       # ... a value network alone (same data)
    python apprentice.py gamma0.07_free_nodest_i1 --rounds 1 --epochs 11 --patience 40 --init-seed 1
    python apprentice.py gamma0.07_free_novalue_i1 --rounds 1 --epochs 11 --patience 40 --value-weight 0 --init-seed 1
                                                                       # ... the same data, other initial weights

gamma0.07_free (and its round 1 epoch 2, the start from 2026-11-19 to 2026-11-24) was trained with a dest head as well
(tag pre-dest-removal); gamma0.07_free_nodest is its round 1 again on the same data, without one.
"""
import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BINS = [(1, 8), (9, 12), (13, 16), (17, 20), (21, 30)]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", help="results go to <out>/<run>/")
    ap.add_argument("--gamma", type=float, default=0.07, help="AND-OR's lapse rate while making attempts")
    ap.add_argument("--protect-movers", choices=("on", "off"), default="off",
                    help="and_or.PROTECT_MOVERS for the expert (default off, as and_or's; phase 1 had it on)")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-bin", type=int, default=200, help="training puzzles per distance bin")
    ap.add_argument("--val-per-bin", type=int, default=20)
    ap.add_argument("--attempts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0, help="noise in the network's ordering of options")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-seed", type=int, default=None,
                    help="seed for the network's initial weights and the batch order (default --seed); the puzzles "
                         "and AND-OR's attempts still follow --seed, so runs differing only here train on the same "
                         "data")
    ap.add_argument("--value-weight", type=float, default=0.5,
                    help="the value loss's weight in pretraining (0: no value head training)")
    ap.add_argument("--blocks", type=int, default=4, help="lane blocks in the network's trunk")
    ap.add_argument("--hidden", type=int, default=96, help="channels per cell in the trunk")
    ap.add_argument("--policy-weight", type=float, default=1.0,
                    help="the policy loss's weight (0: a value network only)")
    ap.add_argument("--device", default=None, help="default: cuda, mps or cpu, whichever is there")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "apprentice"))
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Before and_or is first imported, so its default is set here and in every worker (they inherit the environment).
    os.environ["AND_OR_PROTECT_MOVERS"] = "1" if args.protect_movers == "on" else "0"
    sys.path.insert(0, ROOT)
    import torch
    from rushhour import and_or, board, student
    assert and_or.PROTECT_MOVERS == (args.protect_movers == "on")
    device = args.device or ("cuda" if torch.cuda.is_available() else
                             "mps" if torch.backends.mps.is_available() else "cpu")

    run_dir = os.path.join(args.out, args.run)
    if os.path.exists(os.path.join(run_dir, "history.json")):
        raise SystemExit(f"{run_dir} already has a run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {"device": device, "bins": BINS}, f, indent=1)

    pool = board.load_puzzle_pool()
    test = board.sample_by_distance(pool, BINS, 20, seed=args.seed)  # phase 1's test set: kept out, as it was
    val = board.sample_by_distance(pool, BINS, args.val_per_bin, seed=args.seed + 1, exclude=[s for _, s in test])
    train = board.sample_by_distance(pool, BINS, args.per_bin, seed=args.seed + 2,
                                  exclude=[s for _, s in test + val])
    print(f"{len(train)} training and {len(val)} validation puzzles; gamma {args.gamma}, "
          f"protect_movers {args.protect_movers}, device {device}", flush=True)

    init_seed = args.seed if args.init_seed is None else args.init_seed
    random.seed(init_seed)
    torch.manual_seed(init_seed)
    agent = student.PolicyAgent(device=device, temperature=args.temperature, hidden=args.hidden, blocks=args.blocks)
    history = student.expert_iteration(agent, [s for _, s in train], [s for _, s in val], rounds=args.rounds,
                                  n_attempts=args.attempts, gamma=args.gamma, max_steps=args.max_steps,
                                  seed=args.seed, checkpoint_dir=run_dir, epochs=args.epochs,
                                  patience=args.patience, value_weight=args.value_weight,
                                  policy_weight=args.policy_weight)
    for r in history:
        print(f"round {r['round']}: {r['solved']}/{r['attempts']} attempts solved, {r['boards']} boards "
              f"learned from; kept epoch {r['best_epoch']} (move agreement "
              f"{r.get('agreement', float('nan')):.0%})", flush=True)


if __name__ == "__main__":
    main()
