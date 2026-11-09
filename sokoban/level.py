"""Sokoban levels and push-level moves.

A move is one push: the player walks (for free) to the cell behind a box and
pushes it one cell. That is the counterpart of a Rush Hour slide - walking is
not counted, and distances below are in pushes.

Cells are indexed row-major (`i * width + j`) and sets of cells are int
bitmasks, so the reachable region of the player is a handful of big-int
shifts. A state is `(boxes, player)`: the box mask and the player's cell.
Two states whose player can reach the same cells are the same position for
planning; `key()` folds them together by keeping only the lowest reachable
cell (the usual "normalized" player position).

Board text uses the standard XSB characters: `#` wall, `@` player, `+`
player on goal, `$` box, `*` box on goal, `.` goal, and space, `-` or `_` for
floor. In a dataset line the rows are joined with `/` and floor is `-`.
"""

DIRECTIONS = "udlr"
DELTAS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}
OPPOSITE = {"u": "d", "d": "u", "l": "r", "r": "l"}

_FLOOR_CHARS = " -_"
_BOARD_CHARS = set("#@+$*." + _FLOOR_CHARS)


def _shift(mask, s):
    return mask << s if s > 0 else mask >> -s


def _bits(mask):
    """The set cells of `mask`, lowest first."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class Level:
    """A level's fixed layout (walls, goals, which cells are floor) and its
    start state. `floor` is only the cells the player can reach from the start
    with every box removed, so the area outside the walls is never floor."""

    def __init__(self, rows, name=None):
        rows = [row.rstrip("\n") for row in rows]
        rows = ["".join(" " if c in _FLOOR_CHARS else c for c in row) for row in rows]
        self.name = name
        self.height = len(rows)
        self.width = max(len(row) for row in rows)
        self.n_cells = self.height * self.width
        w = self.width

        walls = goals = boxes = 0
        player = None
        for i, row in enumerate(rows):
            for j, c in enumerate(row.ljust(w)):
                bit = 1 << (i * w + j)
                if c == "#":
                    walls |= bit
                if c in ".+*":
                    goals |= bit
                if c in "$*":
                    boxes |= bit
                if c in "@+":
                    if player is not None:
                        raise ValueError(f"{name}: more than one player")
                    player = i * w + j
        if player is None:
            raise ValueError(f"{name}: no player")

        # Interior floor: flood fill from the player through non-wall cells. A
        # level whose interior touches the border is open, and the bit tricks
        # below (a shift by 1 wraps across rows) assume it never is.
        floor, frontier = 0, [player]
        seen = {player}
        while frontier:
            c = frontier.pop()
            i, j = divmod(c, w)
            if i in (0, self.height - 1) or j in (0, w - 1):
                raise ValueError(f"{name}: level is not enclosed by walls")
            floor |= 1 << c
            for di, dj in DELTAS.values():
                n = (i + di) * w + (j + dj)
                if n not in seen and not walls >> n & 1:
                    seen.add(n)
                    frontier.append(n)

        # A box already on a goal and sealed off from the player (decoration,
        # as in Microban's "The Dungeon") can never move: treat it as wall.
        sealed = boxes & goals & ~floor
        walls |= sealed
        boxes &= ~sealed
        goals &= ~sealed
        if boxes & ~floor or goals & ~floor:
            raise ValueError(f"{name}: a box or goal lies outside the reachable floor")
        if bin(boxes).count("1") != bin(goals).count("1"):
            raise ValueError(f"{name}: {bin(boxes).count('1')} boxes but {bin(goals).count('1')} goals")

        self.walls, self.floor, self.goals = walls, floor, goals
        self.start = (boxes, player)
        self.n_boxes = bin(boxes).count("1")
        self.shifts = {"u": -w, "d": w, "l": -1, "r": 1}
        self.live = self._live_cells()
        self.dead = floor & ~self.live

    # ---- construction from text -------------------------------------------------

    @classmethod
    def from_line(cls, board, name=None):
        """A level from its dataset form: rows joined by '/'."""
        return cls(board.split("/"), name)

    def to_line(self, state=None):
        """The dataset form of `state` (default: the start): rows joined by '/',
        floor as '-', trailing floor dropped from each row."""
        return "/".join(row.rstrip().replace(" ", "-") for row in self.render(state))

    def render(self, state=None):
        """XSB rows for `state` (default: the start)."""
        boxes, player = self.start if state is None else state
        rows = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                c = i * self.width + j
                bit = 1 << c
                if self.walls & bit:
                    row.append("#")
                elif c == player:
                    row.append("+" if self.goals & bit else "@")
                elif boxes & bit:
                    row.append("*" if self.goals & bit else "$")
                elif self.goals & bit:
                    row.append(".")
                else:
                    row.append(" ")
            rows.append("".join(row).rstrip())
        return rows

    def __str__(self):
        return "\n".join(self.render())

    # ---- cells ------------------------------------------------------------------

    def cell(self, i, j):
        return i * self.width + j

    def coords(self, c):
        return divmod(c, self.width)

    def _live_cells(self):
        """Floor cells from which a box alone on the board could still reach a
        goal (everything else is a simple deadlock: a box there is stuck for
        good). Found by pulling a box backwards from every goal: a pull from
        c to c+d needs the player at c+d stepping on to c+2d."""
        live = self.goals
        frontier = list(_bits(self.goals))
        while frontier:
            c = frontier.pop()
            for s in self.shifts.values():
                to, behind = c + s, c + 2 * s
                if (self.floor >> to & 1 and self.floor >> behind & 1
                        and not live >> to & 1):
                    live |= 1 << to
                    frontier.append(to)
        return live

    # ---- moves ------------------------------------------------------------------

    def reach(self, boxes, player):
        """Mask of the cells the player can walk to without pushing anything."""
        free = self.floor & ~boxes
        w = self.width
        region = 1 << player
        while True:
            grown = region | (((region << 1) | (region >> 1) | (region << w) | (region >> w)) & free)
            if grown == region:
                return region
            region = grown

    def pushes_from(self, boxes, region, live_only=False):
        """Every push `(box_cell, direction)` legal for boxes `boxes` and a
        player who can reach `region`. With `live_only`, pushes onto dead
        cells (see _live_cells) are left out."""
        target = (self.live if live_only else self.floor) & ~boxes
        out = []
        for d in DIRECTIONS:
            s = self.shifts[d]
            # box b is pushable in d if the player can stand at b - s and b + s is free
            pushable = boxes & _shift(region, s) & _shift(target, -s)
            out.extend((b, d) for b in _bits(pushable))
        return out

    def pushes(self, state, live_only=False):
        boxes, player = state
        return self.pushes_from(boxes, self.reach(boxes, player), live_only)

    def apply(self, state, push):
        """The state after `push`; the player ends where the box was."""
        boxes, _ = state
        b, d = push
        return (boxes ^ (1 << b)) | (1 << (b + self.shifts[d])), b

    def solved(self, state):
        return state[0] == self.goals

    def key(self, state, region=None):
        """A hashable id shared by every state with the same boxes and the same
        reachable region (pass `region` if already computed)."""
        boxes, player = state
        if region is None:
            region = self.reach(boxes, player)
        return boxes * self.n_cells + ((region & -region).bit_length() - 1)

    def from_key(self, key):
        """A state for `key`, with the player on its lowest reachable cell."""
        boxes, player = divmod(key, self.n_cells)
        return boxes, player

    # ---- deadlocks and box routes -------------------------------------------------

    def _blocked(self, boxes, c, s, seen):
        """Whether the box at c can't move along the axis of shift s: a wall
        on either side, dead cells on both sides, or a neighbouring box that is
        itself frozen (boxes in `seen` count as walls, so cycles end)."""
        a, b = c - s, c + s
        if self.walls >> a & 1 or self.walls >> b & 1:
            return True
        if self.dead >> a & 1 and self.dead >> b & 1:
            return True
        for n in (a, b):
            if boxes >> n & 1 and (n in seen or self._frozen(boxes, n, seen | {c})):
                return True
        return False

    def _frozen(self, boxes, c, seen=frozenset()):
        return self._blocked(boxes, c, 1, seen) and self._blocked(boxes, c, self.width, seen)

    def frozen_off_goal(self, boxes, cells=None):
        """True if some box in `cells` (default: every box) can never move
        again and isn't on a goal - a freeze deadlock. Sound but not complete:
        every state it flags is unsolvable, not every unsolvable state is
        flagged. After a push, checking the moved box and its neighbours
        suffices."""
        cells = _bits(boxes) if cells is None else cells
        return any(boxes >> c & 1 and not self.goals >> c & 1 and self._frozen(boxes, c) for c in cells)

    def deadlocked(self, boxes):
        """A quick, sound deadlock test: a box on a dead cell or a freeze."""
        return bool(boxes & self.dead) or self.frozen_off_goal(boxes)

    def box_distance(self, goal):
        """{cell: pushes} for a lone box to reach `goal` on the empty board
        (walls only; the player is assumed to get anywhere it needs). Cached."""
        cache = self.__dict__.setdefault("_box_distance", {})
        if goal not in cache:
            dist = {goal: 0}
            frontier = [goal]
            for c in frontier:  # BFS: frontier grows as we go
                for s in self.shifts.values():
                    to, behind = c + s, c + 2 * s
                    if to not in dist and self.floor >> to & 1 and self.floor >> behind & 1:
                        dist[to] = dist[c] + 1
                        frontier.append(to)
            cache[goal] = dist
        return cache[goal]

    def box_cells(self, state):
        return list(_bits(state[0]))

    def describe_push(self, push):
        """`push` with the box as (row, col), for printing."""
        b, d = push
        return self.coords(b), d


def read_levels(path):
    """[(name, rows)] for every level in an XSB-style file: boards are runs of
    lines made of board characters that contain a wall; the name is the most
    recent ';' comment or title line before the board (else its index)."""
    levels, board, name = [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for line in lines + [""]:
        if line and "#" in line and set(line) <= _BOARD_CHARS:
            board.append(line)
            continue
        if board:
            levels.append((name if name is not None else str(len(levels)), board))
            board, name = [], None
        stripped = line.strip()
        if stripped.startswith(";"):
            name = stripped.lstrip(";").strip() or name
    return levels
