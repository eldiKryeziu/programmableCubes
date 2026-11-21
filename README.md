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
| cpp/cubes.cpp | C++ simulator + greedy solver + `polish` local search (~100x faster than Python). Built automatically by search.py / polish.py. |
| search.py | Runs the C++ solver on many seeds in parallel, keeps the best: `python3 search.py JWST --root "$CH" --seeds 70 --urgent 2 3` |
| polish.py | Iterated local search: cut the best solution near its end, re-solve from there, keep improvements: `python3 polish.py Enterprise --root "$CH" --minutes 10` |
| export_problem.py | Writes a problem in the text format the C++ program reads. |
| cubesim.py | Python re-implementation of the official move rules + fitness. |
| test_sim.py | Checks cubesim against the official UDP: `python3 test_sim.py "$CH"` |
| greedy.py, restarts.py, shorten.py | Original Python solver, seed runner and move shortener (slower; kept for reference). |
| verify.py | Scores a saved chromosome with the official UDP: `python3 verify.py "$CH" ISS solutions/ISS.npy` |
| diag.py | Shows which cubes/cells are left unsolved. |
| solutions/ | Best chromosomes so far (.npy, ends with -1). Every tool here refuses to overwrite a better file. |

Typical workflow: `search.py` for a good start, then `polish.py` for as long as you like, then `verify.py`.

How the greedy works, per step: evacuate wrong-type cubes from target cells; fill nearly walled-in
empty targets first (`--urgent`); otherwise take the shortest (movable cube, empty target) pair found
by reverse BFS. When stuck it (1) moves a correct cube into a walled-in cell of its type, (2) "opens a
door" by parking a correct neighbour of an unreachable cell and keeping that cell shut until the
pocket is filled, (3) "shakes" a leftover cube to a random reachable cell. It keeps the best prefix. The C++ version also rejects (when it can) a placement that would make a nearby
empty target cell impossible to enter (`seal` option).

Results (official fitness, lower is better, -1 is perfect):

| Problem | Correct | Moves | Fitness |
| --- | --- | --- | --- |
| ISS | 148/148 | 502 | -0.98908 |
| JWST | 642/643 | 4017 | -0.97887 |
| Enterprise | 1467/1472 | 9262 | -0.98226 |

All three checked with the official `programmable_cubes_UDP.fitness` (2026-11-21).
