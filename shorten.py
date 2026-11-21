"""Shorten a finished solution without losing any correct cubes.

Passes (repeated until nothing improves):
  1. drop commands that are illegal (they still count in L(x) but do nothing);
  2. try to delete each run of consecutive moves of one cube, latest first;
  3. re-plan each run as the shortest pivot path between the same two cells.
A change is kept only if the final number of correct cubes does not drop.

python3 shorten.py ISS --root "$CH" --inp solutions/ISS.npy --out solutions/ISS.npy
"""
import argparse
import time

import numpy as np

from cubesim import Ensemble, Problem
from greedy import forward_bfs


def load(x):
    end = int(np.where(x == -1)[0][0]) // 2
    return [(int(x[2 * i]), int(x[2 * i + 1])) for i in range(end)]


def legal_only(prob, start_pos, cmds):
    e = Ensemble(start_pos, prob.init_types)
    out = [cm for cm in cmds if e.try_move(*cm)]
    return out, e


def runs(cmds):
    """Split into maximal runs [(start, end)) of consecutive moves of the same cube."""
    out, i = [], 0
    while i < len(cmds):
        j = i
        while j < len(cmds) and cmds[j][0] == cmds[i][0]:
            j += 1
        out.append((i, j))
        i = j
    return out


def snapshots(prob, cmds, rs):
    e = prob.ensemble()
    snaps = []
    k = 0
    for (a, b) in rs:
        while k < a:
            e.try_move(*cmds[k])
            k += 1
        snaps.append(list(e.pos))
    return snaps


def shorten(prob, cmds, verbose=True, time_limit=None):
    t0 = time.time()
    cmds, e = legal_only(prob, prob.init_pos, cmds)
    goal = prob.correct(e)
    improved = True
    while improved and (time_limit is None or time.time() - t0 < time_limit):
        improved = False
        rs = runs(cmds)
        snaps = snapshots(prob, cmds, rs)
        # latest runs first: deleting them changes the least
        for idx in range(len(rs) - 1, -1, -1):
            a, b = rs[idx]
            if b > len(cmds):
                continue
            c = cmds[a][0]
            # option 1: delete the run; option 2: shortest path to the same end cell
            trial_runs = [[]]
            st = Ensemble(snaps[idx], prob.init_types)
            p_start = st.pos[c]
            probe = Ensemble(snaps[idx], prob.init_types)
            for cm in cmds[a:b]:
                probe.try_move(*cm)
            p_end = probe.pos[c]
            if b - a > 1 and st.is_removable(c):
                mv = forward_bfs(st.at, p_start, lambda q: q == p_end)
                if mv is not None and len(mv) < b - a:
                    trial_runs.append([(c, m) for m in mv])
            for new in trial_runs:
                rest, e2 = legal_only(prob, snaps[idx], new + cmds[b:])
                if prob.correct(e2) >= goal and len(rest) < len(cmds) - a:
                    cmds = cmds[:a] + rest
                    improved = True
                    break
            if improved:
                break
        if verbose:
            print(f"  moves {len(cmds)}  {time.time()-t0:.0f}s", flush=True)
    return cmds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-limit", type=float, default=None)
    a = ap.parse_args()
    prob = Problem(a.problem, a.root)
    cmds = load(np.load(a.inp))
    e = prob.ensemble(); e.apply(cmds)
    print(f"before: correct {prob.correct(e)}/{prob.n} moves {len(cmds)} fitness {prob.fitness(e, len(cmds)):.6f}")
    cmds = shorten(prob, cmds, time_limit=a.time_limit)
    e = prob.ensemble(); e.apply(cmds)
    print(f"after:  correct {prob.correct(e)}/{prob.n} moves {len(cmds)} fitness {prob.fitness(e, len(cmds)):.6f}")
    np.save(a.out, prob.chromosome(cmds))


if __name__ == "__main__":
    main()
