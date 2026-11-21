"""Iterated local search with the C++ solver: repeatedly cut the best solution at a random point near
its end and re-run the greedy from there with a new seed; keep anything that scores better.
Runs one chain per worker and saves the best into solutions/<problem>.npy (never overwrites a better one).

python3 polish.py JWST --root "$CH" --minutes 10
"""
import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cubesim import Problem
from export_problem import export
from search import HERE, build, load_cmds, run_bin, score_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build()
    prob = Problem(a.problem, a.root)
    out = os.path.join(HERE, "solutions", f"{a.problem}.npy")
    x = np.load(out)
    best, _ = score_file(prob, x)
    tmp = tempfile.mkdtemp()
    ptxt, start = os.path.join(tmp, "problem.txt"), os.path.join(tmp, "start.txt")
    export(prob, ptxt)
    end = int(np.where(x == -1)[0][0]) // 2
    with open(start, "w") as f:
        f.writelines(f"{x[2 * i]} {x[2 * i + 1]}\n" for i in range(end))
    print(f"start: fitness {best:.6f}")

    def chain(k):
        res = os.path.join(tmp, f"chain{k}.txt")
        run_bin(["polish", ptxt, start, res, "100000000", a.seed * 1000 + k, a.minutes * 60])
        return res

    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(chain, range(a.workers)))
    for res in results:
        if not os.path.exists(res):
            continue
        cmds = load_cmds(res)
        e = prob.ensemble()
        legal = e.apply(cmds)
        f = prob.fitness(e, len(cmds))
        print(f"chain: correct {prob.correct(e)}/{prob.n} moves {len(cmds)} fitness {f:.6f}")
        if legal == len(cmds) and f < best:
            best = f
            np.save(out, prob.chromosome(cmds))
            print("  saved")
    print(f"best: {best:.6f}")


if __name__ == "__main__":
    main()
