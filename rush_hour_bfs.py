from collections import deque

MOVES = {"h": "lr", "v": "ud"}
DELTAS = {"l": (0, -1), "r": (0, 1), "u": (-1, 0), "d": (1, 0)}


def block_orientations(state):
    orientations = {}
    for name, positions in state:
        if positions[0][0] == positions[1][0]:
            orientations[name] = "h"
        else:
            orientations[name] = "v"
    return orientations


def neighbors(state, orientations, n_rows=6, n_cols=6):
    """Yield (move, new_state) pairs reachable from `state` in one move, without mutating anything."""
    occupied = {pos: name for name, positions in state for pos in positions}
    order = [name for name, _ in state]
    state_dict = dict(state)

    for name, positions in state:
        for direction in MOVES[orientations[name]]:
            di, dj = DELTAS[direction]
            new_positions = tuple((i + di, j + dj) for i, j in positions)

            valid = True
            for i, j in new_positions:
                if not (0 <= i < n_rows and 0 <= j < n_cols):
                    valid = False
                    break
                occupant = occupied.get((i, j))
                if occupant is not None and occupant != name:
                    valid = False
                    break
            if not valid:
                continue

            new_state = tuple(
                (n, new_positions if n == name else state_dict[n])
                for n in order
            )
            yield (name, direction), new_state


def find_goal_states(states):
    goal_states = []
    for state in states:
        if state[0][1][1][1] == 5:
            goal_states.append(state)
    return goal_states



def bfs(initial, orientations):
    n_rows, n_cols = 6, 6

    distance = {state: 0 for state in initial}
    queue = deque(initial)

    while queue:
        state = queue.popleft()
        for _, new_state in neighbors(state, orientations, n_rows, n_cols):
            if new_state not in distance:
                distance[new_state] = distance[state] + 1
                queue.append(new_state)

    return distance


def multi_bfs(state):
    orientations = block_orientations(state)
    distances = bfs([state], orientations)
    return bfs(find_goal_states(distances), orientations)


def hint(state, distances):
    """Return a (block, direction) move from `state` that is one step closer to the goal."""
    orientations = block_orientations(state)
    current = distances[state]
    for move, new_state in neighbors(state, orientations):
        if distances[new_state] < current:
            return move
    return None


if __name__ == "__main__":
    puzzle = (('red', ((2, 0), (2, 1))), ('0', ((0, 1), (1, 1))), ('1', ((2, 2), (3, 2))), ('2', ((1, 4), (2, 4))), ('3', ((4, 0), (5, 0))), ('4', ((4, 3), (4, 4), (4, 5))))
    print(list(multi_bfs(puzzle).items())[-10:])
