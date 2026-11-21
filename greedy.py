"""Greedy constructive solver for SpOC3 Programmable Cubes.

Idea: repeatedly take a cube that is not yet on a correct target cell, can be removed without
disconnecting the ensemble, and has the shortest pivot path to an empty target cell of its type.
Move it there along that path. Repeat until no such pair exists.

The path search is a reverse BFS from all empty target cells of one type at once, using exact
predecessors of the official move rules. Each candidate path is then re-checked forwards with the
moving cube removed from the grid (that is what the official evaluator sees) before it is committed.
"""
import argparse
import sys
import time
from collections import deque

import numpy as np

from cubesim import MOVES, Problem, land


def articulation_points(ens):
    """Iterative Tarjan over the face-adjacency graph. Returns a set of cube ids whose removal disconnects."""
    n = len(ens.pos)
    adj = [ens.neighbours(p) for p in ens.pos]
    disc = [-1] * n
    low = [0] * n
    ap = set()
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        children = 0
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, parent, it = stack[-1]
            advanced = False
            for v in it:
                if disc[v] == -1:
                    disc[v] = low[v] = t
                    t += 1
                    if u == root:
                        children += 1
                    stack.append((v, u, iter(adj[v])))
                    advanced = True
                    break
                elif v != parent:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    w = stack[-1][0]
                    low[w] = min(low[w], low[u])
                    if w != root and low[u] >= disc[w]:
                        ap.add(w)
        if children > 1:
            ap.add(root)
    return ap


def predecessors(at, p):
    """All (r, m) with land(at, r, m) == p, where r is empty or occupied (a potential mover)."""
    out = []
    for m in range(6):
        for _, _, disp in MOVES[m]:
            r = (p[0] - disp[0], p[1] - disp[1], p[2] - disp[2])
            if land(at, r, m, ignore=r) == p:
                out.append((r, m))
    return out


def forward_check(at, start, moves):
    """Re-run the path with the mover lifted out of the grid. Returns the end position or None."""
    p = start
    for m in moves:
        q = land(at, p, m, ignore=start)
        if q is None:
            return None
        p = q
    return p


def forward_bfs(at, start, accept, max_nodes=50000):
    """Shortest command list moving the cube at `start` (lifted out of the grid) to a cell where accept(p)."""
    parent = {start: None}
    dq = deque([start])
    while dq and len(parent) < max_nodes:
        p = dq.popleft()
        for m in range(6):
            q = land(at, p, m, ignore=start)
            if q is None or q in parent or (q in at and q != start):
                continue
            parent[q] = (p, m)
            if accept(q):
                moves = []
                while parent[q] is not None:
                    q, mm = parent[q]
                    moves.append(mm)
                return moves[::-1]
            dq.append(q)
    return None


def find_move(prob, ens, movable_of_type, empty_targets_of_type, banned, max_nodes):
    """Reverse BFS from every empty target cell of each type. Returns (cube, moves) for the shortest found."""
    at = ens.at
    best = None
    for t, goals in empty_targets_of_type.items():
        movers = movable_of_type.get(t)
        if not goals or not movers:
            continue
        # parent[p] = (next position towards goal, move used from p)
        parent = {g: None for g in goals}
        dq = deque(goals)
        nodes = 0
        while dq and nodes < max_nodes:
            p = dq.popleft()
            nodes += 1
            for r, m in predecessors(at, p):
                if r in parent:
                    continue
                if r in at:
                    c = at[r]
                    if c in movers and (c, p) not in banned:
                        # rebuild the path r -> ... -> goal
                        moves, q = [m], p
                        while parent[q] is not None:
                            q, mm = parent[q]
                            moves.append(mm)
                        end = forward_check(at, r, moves)
                        if end is not None and end == q:
                            if best is None or len(moves) < len(best[1]):
                                best = (c, moves)
                            dq.clear()
                            break
                        banned.add((c, p))
                    continue
                parent[r] = (p, m)
                dq.append(r)
            if best is not None and dq is not None and len(dq) == 0:
                break
    return best


def repair_move(prob, ens, ap, empty_targets_of_type, max_nodes, seen_states, tries=30):
    """When the greedy is stuck: move an already-correct cube into a walled-in empty target cell of its
    type, if the cell it leaves behind is easier to reach (fewer occupied face neighbours).
    The freed cell can then be filled by a leftover cube in the normal greedy step."""
    done_of_type = {}
    for c, p in enumerate(ens.pos):
        if prob.target.get(p) == ens.types[c] and c not in ap:
            done_of_type.setdefault(ens.types[c], set()).add(c)
    goals = {t: [p for p in g if len(ens.neighbours(p)) >= 3] for t, g in empty_targets_of_type.items()}
    banned = set()
    for _ in range(tries):
        found = find_move(prob, ens, done_of_type, goals, banned, max_nodes)
        if found is None:
            return None
        c, moves = found
        start = ens.pos[c]
        end = start
        p = start
        for m in moves:
            p = land(ens.at, p, m, ignore=start)
        end = p
        # walled-in-ness of the goal vs the cell we leave (counted after the move)
        goal_n = len(ens.neighbours(end))
        left_n = len([q for q in ens.neighbours(start) if q != c]) + (1 if _adjacent(start, end) else 0)
        key = hash(frozenset(ens.at) - {start} | {end})
        if left_n < goal_n and key not in seen_states:
            seen_states.add(key)
            return found
        banned.add((c, _second(start, moves, ens.at)))
    return None


def open_door(prob, ens, ap, empty_targets_of_type, seen_states):
    """Last resort: an empty target cell nobody can reach. Lift a correct neighbour of it out of
    the way (park it on the nearest non-target cell) so the cell opens up; the parked cube then
    goes back through the normal greedy step."""
    goals = sorted((p for g in empty_targets_of_type.values() for p in g),
                   key=lambda p: -len(ens.neighbours(p)))
    for g in goals:
        for d in ens.neighbours(g):
            if d in ap or prob.target.get(ens.pos[d]) != ens.types[d]:
                continue
            start = ens.pos[d]
            mv = forward_bfs(ens.at, start, lambda q: q not in prob.target and q != start)
            if not mv:
                continue
            p = start
            for m in mv:
                p = land(ens.at, p, m, ignore=start)
            key = hash(frozenset(ens.at) - {start} | {p})
            if key in seen_states:
                continue
            seen_states.add(key)
            return d, mv
    return None


def _adjacent(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) == 1


def _second(start, moves, at):
    return land(at, start, moves[0], ignore=start)


def solve(prob, max_nodes=200000, verbose=True, time_limit=None, urgent=3, max_repairs=200):
    ens = prob.ensemble()
    commands = []
    t0 = time.time()
    banned = set()
    repairs, seen_states = 0, set()
    while True:
        ap = articulation_points(ens)
        movable_of_type, empty_targets_of_type = {}, {}
        for c, p in enumerate(ens.pos):
            ty = ens.types[c]
            if prob.target.get(p) != ty and c not in ap:
                movable_of_type.setdefault(ty, set()).add(c)
        for p, ty in prob.target.items():
            if p not in ens.at:
                empty_targets_of_type.setdefault(ty, []).append(p)
        found = None
        # Evacuate wrong-type cubes that sit on target cells first, before they get walled in.
        squatters = {t: {c for c in cs if ens.pos[c] in prob.target} for t, cs in movable_of_type.items()}
        if any(squatters.values()):
            found = find_move(prob, ens, squatters, empty_targets_of_type, banned, max_nodes)
        # A squatter with no reachable goal of its type is parked on the nearest non-target cell instead.
        if found is None and any(squatters.values()):
            for cs in squatters.values():
                for c in sorted(cs):
                    mv = forward_bfs(ens.at, ens.pos[c], lambda q: q not in prob.target)
                    if mv and (found is None or len(mv) < len(found[1])):
                        found = (c, mv)
        # Fill cells that are about to be walled in first (many occupied face neighbours).
        if urgent and found is None:
            for u in range(6, urgent - 1, -1):
                tier = {t: [p for p in g if len(ens.neighbours(p)) >= u] for t, g in empty_targets_of_type.items()}
                found = find_move(prob, ens, movable_of_type, tier, banned, max_nodes)
                if found is not None:
                    break
        if found is None:
            found = find_move(prob, ens, movable_of_type, empty_targets_of_type, banned, max_nodes)
        if found is None and repairs < max_repairs:
            found = repair_move(prob, ens, ap, empty_targets_of_type, max_nodes, seen_states)
            repairs += found is not None
        if found is None and repairs < max_repairs:
            found = open_door(prob, ens, ap, empty_targets_of_type, seen_states)
            repairs += found is not None
        if found is None:
            break
        c, moves = found
        for m in moves:
            ok = ens.try_move(c, m)
            assert ok, "path became illegal"
            commands.append((c, m))
        banned.clear()
        if verbose and len(commands) and (prob.correct(ens) % 10 == 0):
            print(f"  correct {prob.correct(ens)}/{prob.n}  moves {len(commands)}  {time.time()-t0:.0f}s", flush=True)
        if time_limit and time.time() - t0 > time_limit:
            break
    return ens, commands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("--root", default=".", help="folder 'Challenge 3 Programmable Cubes' of esa/SpOC3")
    ap.add_argument("--out", default=None)
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--urgent", type=int, default=3, help="0 disables the walled-in-first rule")
    a = ap.parse_args()
    prob = Problem(a.problem, a.root)
    ens, cmds = solve(prob, time_limit=a.time_limit, urgent=a.urgent)
    print(f"{a.problem}: correct {prob.correct(ens)}/{prob.n}, moves {len(cmds)}, fitness {prob.fitness(ens, len(cmds)):.6f}")
    out = a.out or f"{a.problem}_greedy.npy"
    np.save(out, prob.chromosome(cmds))
    print("saved", out)


if __name__ == "__main__":
    main()
