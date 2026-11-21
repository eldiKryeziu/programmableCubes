"""Run the C++ solver on many seeds in parallel; keep the best solution as solutions/<problem>.npy.

python3 search.py ISS --root "$CH" --seeds 200 --urgent 2 3 4
"""
import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cubesim import Problem
from export_problem import export

HERE = os.path.dirname(os.path.abspath(__file__))
BIN = os.path.join(HERE, "cpp", "cubes")


def build():
    src = os.path.join(HERE, "cpp", "cubes.cpp")
    if not os.path.exists(BIN) or os.path.getmtime(BIN) < os.path.getmtime(src):
        subprocess.check_call(["g++", "-O2", "-march=native", "-std=c++17", "-o", BIN, src])


def load_cmds(path):
    with open(path) as f:
        return [tuple(map(int, line.split())) for line in f if line.strip()]


def score_file(prob, x):
    end = int(np.where(x == -1)[0][0]) // 2
    cmds = [(int(x[2 * i]), int(x[2 * i + 1])) for i in range(end)]
    e = prob.ensemble()
    e.apply(cmds)
    return prob.fitness(e, len(cmds)), prob.correct(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--urgent", type=int, nargs="+", default=[3])
    ap.add_argument("--shakes", type=int, default=300)
    ap.add_argument("--repairs", type=int, default=200)
    ap.add_argument("--seal", type=int, nargs="+", default=[1, 0], help="1 = avoid sealing empty target cells")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    build()
    prob = Problem(a.problem, a.root)
    out = a.out or os.path.join(HERE, "solutions", f"{a.problem}.npy")
    tmp = tempfile.mkdtemp()
    ptxt = os.path.join(tmp, "problem.txt")
    export(prob, ptxt)
    best = score_file(prob, np.load(out))[0] if os.path.exists(out) else None
    print(f"current best in {out}: {best}")

    def run(job):
        seed, u, sl = job
        res = os.path.join(tmp, f"s{seed}_u{u}_{sl}.txt")
        line = subprocess.run([BIN, "solve", ptxt, str(seed), str(u), res, "1e9", str(a.shakes), str(a.repairs), str(sl)],
                              capture_output=True, text=True).stdout.strip() + f" seal {sl}"
        return line, res

    jobs = [(s, u, sl) for s in range(a.seed_offset, a.seed_offset + a.seeds) for u in a.urgent for sl in a.seal]
    with ThreadPoolExecutor(a.workers) as ex:
        for line, res in ex.map(run, jobs):
            cmds = load_cmds(res)
            e = prob.ensemble()
            legal = e.apply(cmds)
            f = prob.fitness(e, len(cmds))
            tag = ""
            if legal == len(cmds) and (best is None or f < best):
                best = f
                np.save(out, prob.chromosome(cmds))
                tag = "  <- new best, saved"
            print(line + tag, flush=True)
            os.remove(res)


if __name__ == "__main__":
    main()
