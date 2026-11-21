"""Show a solution in 3D: start shape, your result, and the target, side by side (matplotlib).

Colours are the official ones from problems/<name>.json. In "Your result", cubes on a wrong cell are
drawn red, and target cells that are still empty are drawn as faint grey outlines.

Linux/macOS:  python3 view.py "$CH" JWST solutions/JWST.npy
PowerShell:   python view.py $CH JWST solutions\\JWST.npy
Options:      --save jwst.png      write a picture instead of opening a window
              --types 0 3          only draw these cube types
              --only-wrong         in "Your result", draw only misplaced cubes and empty target cells
Drag with the mouse to rotate; the three views rotate together.
"""
import argparse

import matplotlib
import numpy as np

from cubesim import Problem


def load_cmds(x):
    end = int(np.where(x == -1)[0][0]) // 2
    return [(int(x[2 * i]), int(x[2 * i + 1])) for i in range(end)]


def voxels(ax, cells, colours, lo, shape, alpha=0.6, edge="k"):
    """cells: list of (position, colour index or colour string)."""
    by_colour = {}
    for p, c in cells:
        by_colour.setdefault(c, []).append(p)
    for c, ps in by_colour.items():
        grid = np.zeros(shape, dtype=bool)
        for p in ps:
            grid[p[0] - lo[0], p[1] - lo[1], p[2] - lo[2]] = True
        face = colours[c] if isinstance(c, (int, np.integer)) else c
        ax.voxels(grid, facecolors=face, edgecolors=edge, alpha=alpha, linewidth=0.3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder 'Challenge 3 Programmable Cubes' of esa/SpOC3")
    ap.add_argument("problem", choices=["ISS", "JWST", "Enterprise"])
    ap.add_argument("solution", nargs="?", help=".npy chromosome (default: solutions/<problem>.npy)")
    ap.add_argument("--save", help="save to this image file instead of showing a window")
    ap.add_argument("--types", type=int, nargs="+", help="only these cube types")
    ap.add_argument("--only-wrong", action="store_true")
    a = ap.parse_args()
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prob = Problem(a.problem, a.root)
    colours = prob.setup["colours"]
    x = np.load(a.solution or f"solutions/{a.problem}.npy")
    cmds = load_cmds(x)
    ens = prob.ensemble()
    legal = ens.apply(cmds)
    correct = prob.correct(ens)
    fit = prob.fitness(ens, len(cmds))
    keep = (lambda t: True) if not a.types else (lambda t: t in a.types)

    start = [(tuple(int(v) for v in p), int(t)) for p, t in zip(prob.init_pos, prob.init_types) if keep(int(t))]
    target = [(p, t) for p, t in prob.target.items() if keep(t)]
    result = []
    for c, p in enumerate(ens.pos):
        t = ens.types[c]
        if not keep(t):
            continue
        ok = prob.target.get(p) == t
        if ok and a.only_wrong:
            continue
        result.append((p, t if ok else "red"))
    missing = [(p, "lightgrey") for p, t in prob.target.items()
               if keep(t) and not (p in ens.at and ens.types[ens.at[p]] == t)]

    allp = np.array([p for p, _ in start + target + result] or [(0, 0, 0)])
    lo = allp.min(0)
    shape = tuple(allp.max(0) - lo + 1)

    fig = plt.figure(figsize=(16, 6))
    fig.suptitle(f"{a.problem}: {correct}/{prob.n} cubes correct, {len(cmds)} moves "
                 f"({len(cmds) - legal} illegal), fitness {fit:.6f}   (red = wrong cell, grey = empty target cell)")
    axes = []
    for k, (title, cells) in enumerate([("Start", start), ("Your result", result), ("Target", target)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        ax.set_title(title, y=0.95)
        voxels(ax, cells, colours, lo, shape)
        if title == "Your result" and missing:
            voxels(ax, missing, colours, lo, shape, alpha=0.15, edge="grey")
        ax.set_box_aspect(shape, zoom=1.4)
        ax.set_axis_off()
        ax.set_proj_type("ortho")
        axes.append(ax)

    def sync(event):  # rotate all three views together
        src = next((ax for ax in axes if ax == event.inaxes), None)
        if src is None:
            return
        for ax in axes:
            if ax is not src:
                ax.view_init(elev=src.elev, azim=src.azim)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", sync)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=-0.1)
    if a.save:
        fig.savefig(a.save, dpi=120)
        print("saved", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
