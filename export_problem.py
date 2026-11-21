"""Write a problem in the plain-text format read by cpp/cubes.cpp.

python3 export_problem.py ISS --root "$CH" --out cpp/ISS.txt
"""
import argparse

from cubesim import Problem


def export(prob, path):
    with open(path, "w") as f:
        f.write(f"{prob.n} {int(prob.setup['num_cube_types'])} {prob.max_cmds} {prob.offset!r} {len(prob.target_pos)}\n")
        for p, t in zip(prob.init_pos, prob.init_types):
            f.write(f"{p[0]} {p[1]} {p[2]} {t}\n")
        for p, t in zip(prob.target_pos, prob.target_types):
            f.write(f"{p[0]} {p[1]} {p[2]} {t}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    export(Problem(a.problem, a.root), a.out)
