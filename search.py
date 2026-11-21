"""Run the C++ solver on many seeds in parallel; keep the best solution as solutions/<problem>.npy.

python3 search.py ISS --root "$CH" --seeds 200 --urgent 2 3 4
(Windows: python search.py ISS --root $CH --seeds 200 --urgent 2 3 4)
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cubesim import Problem
from export_problem import export

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOWS = os.name == "nt"
BIN = os.path.join(HERE, "cpp", "cubes.exe" if WINDOWS else "cubes")
SRC = os.path.join(HERE, "cpp", "cubes.cpp")

# Where MSYS2 puts its compilers if they are not on PATH.
MSYS2_DIRS = [r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin", r"C:\msys64\clang64\bin"]


def find_compiler():
    """Return (kind, path) of a C++ compiler: $CXX first, then g++, clang++, MSVC cl, then MSYS2 folders."""
    env = os.environ.get("CXX")
    if env:
        exe = shutil.which(env) or env
        return ("msvc" if os.path.basename(exe).lower().startswith("cl") else "gnu"), exe
    for name in ["g++", "clang++"]:
        exe = shutil.which(name)
        if exe:
            return "gnu", exe
    if WINDOWS:
        exe = shutil.which("cl")
        if exe:
            return "msvc", exe
        for d in MSYS2_DIRS:
            for name in ["g++.exe", "clang++.exe"]:
                exe = os.path.join(d, name)
                if os.path.exists(exe):
                    return "gnu", exe
    return None, None


NO_COMPILER = """No C++ compiler found, so cpp/cubes.cpp cannot be built.

Windows: install MSYS2 (https://www.msys2.org), then in the "MSYS2 UCRT64" shell run
    pacman -S mingw-w64-ucrt-x86_64-gcc
and either add C:\\msys64\\ucrt64\\bin to PATH or just rerun this script (it looks there too).
Alternatively run from a "Developer PowerShell for Visual Studio" (MSVC cl.exe), or set $env:CXX.
Linux/macOS: install g++ or clang++ (e.g. sudo apt install g++, or xcode-select --install).
"""


def build(force=False):
    if not force and os.path.exists(BIN) and os.path.getmtime(BIN) >= os.path.getmtime(SRC):
        return
    kind, cxx = find_compiler()
    if cxx is None:
        sys.exit(NO_COMPILER)
    if kind == "msvc":
        attempts = [[cxx, "/nologo", "/O2", "/EHsc", "/std:c++17", SRC, "/Fe:" + BIN, "/Fo:" + os.path.join(HERE, "cpp") + os.sep]]
    else:
        # -static on Windows so cubes.exe runs without the MinGW DLLs on PATH
        extra = ["-static"] if WINDOWS else []
        base = [cxx, "-O2", "-std=c++17", "-o", BIN, SRC] + extra
        attempts = [base[:2] + ["-march=native"] + base[2:], base]  # retry without -march=native if it fails
    print(f"building {os.path.relpath(BIN, HERE)} with {cxx}", flush=True)
    for cmd in attempts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except OSError as e:
            sys.exit(f"could not run the compiler {cxx}: {e}\n\n{NO_COMPILER}")
        if r.returncode == 0 and os.path.exists(BIN):
            return
    sys.exit(f"compiling cpp/cubes.cpp failed:\n{' '.join(cmd)}\n{r.stdout}{r.stderr}")


def run_bin(args):
    """Run the solver binary; stop with a readable message if it crashes or cannot start."""
    r = subprocess.run([BIN] + [str(x) for x in args], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{os.path.basename(BIN)} failed (exit code {r.returncode}):\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


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
        line = run_bin(["solve", ptxt, seed, u, res, "1e9", a.shakes, a.repairs, sl]) + f" seal {sl}"
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
