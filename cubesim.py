"""Fast, dictionary-based re-implementation of the SpOC3 Programmable Cubes move rules.

It follows the official CubeMoveset.py / programmable_cubes_UDP.py logic exactly:
  * a move (cube c, command m) is legal if removing c keeps the ensemble connected
    and one of the pivot patterns for that rotation fits the 5x5 neighbourhood;
  * the first matching pattern (in the official order) decides where the cube lands;
  * illegal commands are skipped but still count towards L(x).
"""
import json
import os
from collections import deque

import numpy as np

# ---- move tables copied from the official CubeMoveset.py -----------------------------------
PAT_OCC = [[7, 8], [6, 11], [15, 16], [12, 17],
           [6, 7], [11, 15], [16, 17], [8, 12],
           [16], [7], [11], [12],
           [12], [11], [16], [7]]
PAT_EMPTY = [[12, 16, 17], [7, 8, 12], [6, 7, 11], [11, 15, 16],
             [11, 15, 16], [12, 16, 17], [7, 8, 12], [6, 7, 11],
             [14, 15, 10, 11, 6, 7], [8, 9, 12, 13, 16, 17], [12, 6, 7, 8, 1, 2], [11, 15, 16, 17, 21, 22],
             [11, 6, 7, 8, 2, 3], [12, 15, 16, 17, 20, 21], [17, 18, 12, 13, 7, 8], [5, 6, 10, 11, 15, 16]]
MAPPING = [[0, 1, 2, 3, 8, 9, 10, 11], [4, 5, 6, 7, 12, 13, 14, 15]]
DIFFS = [[1, 0], [0, 1], [-1, 0], [0, -1],
         [-1, 0], [0, -1], [1, 0], [0, 1],
         [-1, -1], [1, 1], [-1, 1], [1, -1],
         [1, 1], [-1, -1], [1, -1], [-1, 1]]


def _plane(axis, j, i):
    """Map in-plane coordinates (right=j, up=i) to a 3D offset for rotation axis 0=z, 1=x, 2=y."""
    if axis == 0:
        return (j, i, 0)
    if axis == 1:
        return (0, j, i)
    return (j, 0, i)


# GRID[axis][k] = 3D offset of grid cell k (0..23) around the moving cube
GRID = []
for ax in range(3):
    cells = []
    for i in [2, 1, 0, -1, -2]:
        for j in [-2, -1, 0, 1, 2]:
            if i != 0 or j != 0:
                cells.append(_plane(ax, j, i))
    GRID.append(cells)

# For every command m (0..5): list of (occupied offsets, empty offsets, displacement)
MOVES = []
for m in range(6):
    ax, d = m // 2, m % 2
    opts = []
    for p in MAPPING[d]:
        occ = [GRID[ax][k] for k in PAT_OCC[p]]
        emp = [GRID[ax][k] for k in PAT_EMPTY[p]]
        disp = _plane(ax, *DIFFS[p])
        opts.append((occ, emp, disp))
    MOVES.append(opts)

NEIGH6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def land(occupied, p, m, ignore=None):
    """Where a cube at p ends up with command m, or None. `occupied` is a set/dict of positions.
    `ignore` is a position treated as empty (the moving cube's original spot during path search)."""
    for occ, emp, disp in MOVES[m]:
        ok = True
        for o in occ:
            q = (p[0] + o[0], p[1] + o[1], p[2] + o[2])
            if q not in occupied or q == ignore:
                ok = False
                break
        if not ok:
            continue
        for o in emp:
            q = (p[0] + o[0], p[1] + o[1], p[2] + o[2])
            if q in occupied and q != ignore:
                ok = False
                break
        if ok:
            return (p[0] + disp[0], p[1] + disp[1], p[2] + disp[2])
    return None


class Ensemble:
    def __init__(self, positions, types=None):
        self.pos = [tuple(int(v) for v in p) for p in positions]
        self.types = None if types is None else [int(t) for t in types]
        self.at = {p: i for i, p in enumerate(self.pos)}

    def neighbours(self, p):
        at = self.at
        return [at[q] for q in (add(p, n) for n in NEIGH6) if q in at]

    def is_removable(self, c):
        """Same semantics as check_connectivity_of_cubes: removing c keeps the rest connected."""
        p = self.pos[c]
        nb = self.neighbours(p)
        if len(nb) <= 1:
            return True
        targets = set(nb)
        start = nb[0]
        seen = {start, c}
        found = 1
        dq = deque([start])
        while dq:
            u = dq.popleft()
            for v in self.neighbours(self.pos[u]):
                if v not in seen:
                    seen.add(v)
                    if v in targets:
                        found += 1
                        if found == len(targets):
                            return True
                    dq.append(v)
        return False

    def try_move(self, c, m):
        """Apply command (c, m) if legal. Returns True if the ensemble changed."""
        if not self.is_removable(c):
            return False
        p = self.pos[c]
        q = land(self.at, p, m)
        if q is None:
            return False
        del self.at[p]
        self.at[q] = c
        self.pos[c] = q
        return True

    def apply(self, commands):
        """commands: iterable of (cube, move). Returns number of legal moves."""
        n = 0
        for c, m in commands:
            n += self.try_move(c, m)
        return n


class Problem:
    """Loads one SpOC3 cubes problem from the official repository layout."""

    def __init__(self, name, root="."):
        with open(os.path.join(root, "problems", f"{name}.json")) as f:
            self.setup = json.load(f)
        d = os.path.join(root, self.setup["path"])
        self.init_pos = np.load(f"{d}/Initial_Config.npy")
        self.init_types = np.load(f"{d}/Initial_Cube_Types.npy")
        self.target_pos = np.load(f"{d}/Target_Config.npy")
        self.target_types = np.load(f"{d}/Target_Cube_Types.npy")
        self.max_cmds = self.setup["max_cmds"]
        self.offset = self.setup["fitness_offset"]
        self.n = len(self.init_pos)
        self.target = {tuple(int(v) for v in p): int(t) for p, t in zip(self.target_pos, self.target_types)}

    def ensemble(self):
        return Ensemble(self.init_pos, self.init_types)

    def correct(self, ens):
        return sum(1 for i, p in enumerate(ens.pos) if self.target.get(p) == ens.types[i])

    def fitness(self, ens, n_commands):
        F = 0.9 * self.correct(ens) / self.n + 0.1 * (1 - n_commands / self.max_cmds)
        return -(F - self.offset) / (1 - self.offset)

    def chromosome(self, commands):
        x = []
        for c, m in commands:
            x += [int(c), int(m)]
        x.append(-1)
        return np.array(x, dtype=int)
