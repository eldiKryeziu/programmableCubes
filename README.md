# SpOC3 Programmable Cubes: starter code

Starter solver for ESA's SpOC3 "Programmable Cubes" challenge. Needs the official repo:
`git clone https://github.com/esa/SpOC3` and `pip install numpy numba matplotlib`.
Below, CH = path to `SpOC3/Challenge 3 Programmable Cubes`. Cloning SpOC3 inside this folder works (it is git-ignored):

```
git clone https://github.com/esa/SpOC3
CH="SpOC3/Challenge 3 Programmable Cubes"
```

| File | What it does |
| --- | --- |
| cubesim.py | Fast re-implementation of the official move rules + fitness (hash map, no numba). |
| test_sim.py | Checks cubesim against the official UDP on random commands: `python3 test_sim.py "$CH"` |
| greedy.py | Greedy constructive solver: `python3 greedy.py ISS --root "$CH" --urgent 3 --seed 8 --out ISS.npy` |
| restarts.py | Many random seeds in parallel, keeps the best (never overwrites a better file): `python3 restarts.py JWST --root "$CH" --seeds 8 --out solutions/JWST.npy` |
| shorten.py | Removes wasted moves without losing correct cubes: `python3 shorten.py ISS --root "$CH" --inp solutions/ISS.npy --out solutions/ISS.npy` |
| verify.py | Scores a saved chromosome with the official UDP: `python3 verify.py "$CH" ISS solutions/ISS.npy` |
| diag.py | Shows which cubes/cells are left unsolved: `python3 diag.py "$CH" ISS solutions/ISS.npy` |
| solutions/ | Best chromosomes so far (.npy, ends with -1). |

How the greedy works, per step: evacuate wrong-type cubes from target cells; fill nearly walled-in
empty targets first (`--urgent`); otherwise take the shortest (movable cube, empty target) pair found
by reverse BFS. When stuck it (1) moves a correct cube into a walled-in cell of its type, (2) "opens a
door" by parking a correct neighbour of an unreachable cell and keeping that cell shut until the
pocket is filled, (3) "shakes" a leftover cube to a random reachable cell. It keeps the best prefix.

Results (official fitness, lower is better, -1 is perfect):

| Problem | Correct | Moves | Fitness |
| --- | --- | --- | --- |
| ISS | 148/148 | 539 | -0.98828 |
| JWST | 634/643 | 3877 | -0.96353 |
| Enterprise | 1451/1472 | 8140 | -0.96979 |

All three checked with the official `programmable_cubes_UDP.fitness` (2026-11-21).
