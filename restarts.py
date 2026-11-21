"""Run the greedy solver with many random seeds in parallel and keep the best solution.

python3 restarts.py ISS --root "$CH" --seeds 40 --out solutions/ISS.npy
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

from cubesim import Problem
from greedy import solve

ARGS = None


def run(job):
    seed, urgent = job
    prob = Problem(ARGS.problem, ARGS.root)
    ens, cmds = solve(prob, verbose=False, urgent=urgent, seed=seed, time_limit=ARGS.time_limit,
                      deep_first=ARGS.deep_first)
    return prob.fitness(ens, len(cmds)), prob.correct(ens), seed, urgent, cmds


def init(a):
    global ARGS
    ARGS = a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--urgent", type=int, nargs="+", default=[3])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--time-limit", type=float, default=None)
    ap.add_argument("--deep-first", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prob = Problem(a.problem, a.root)
    best = None
    if os.path.exists(a.out):  # never overwrite a better saved solution
        x = np.load(a.out)
        cmds = [(int(x[2 * i]), int(x[2 * i + 1])) for i in range(int(np.where(x == -1)[0][0]) // 2)]
        e = prob.ensemble()
        e.apply(cmds)
        best = prob.fitness(e, len(cmds))
        print(f"existing {a.out}: fitness {best:.6f}")
    jobs = [(s, u) for s in range(a.seed_offset, a.seed_offset + a.seeds) for u in a.urgent]
    with Pool(a.workers, initializer=init, initargs=(a,)) as pool:
        for f, cor, seed, u, cmds in pool.imap_unordered(run, jobs):
            tag = ""
            if best is None or f < best:
                best = f
                np.save(a.out, prob.chromosome(cmds))
                tag = "  <- new best, saved"
            print(f"seed {seed} urgent {u}: correct {cor}/{prob.n} moves {len(cmds)} fitness {f:.6f}{tag}", flush=True)


if __name__ == "__main__":
    main()
