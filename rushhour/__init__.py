"""Rush Hour: a simulated student, the teachers that help it, and their comparison with people.

- board: puzzles, moves, breadth-first search, the puzzle pool; and_or: the AND-OR planner (Olieslagers et al.
  2023, as this repo solves with it); paper_model: the paper's model as published (a move per tree);
- student/: the simulated student (network, solving, labels, learning);
- teachers: when to show the student the optimal move, and the registry of teacher specs;
- sessions: simulated sessions (run, save, load); run_queue: queues of them in the background;
- human_data / play_data: people's play (the paper's 42 participants / the play page in web/);
- analysis, surrender: shared measures for the notebooks, and fitting people's giving up;
- paths: where data and results live.
"""
