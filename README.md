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
| greedy.py | Greedy constructive solver: `python3 greedy.py ISS --root "$CH" --urgent 3 --out ISS.npy` |
| verify.py | Scores a saved chromosome with the official UDP: `python3 verify.py "$CH" ISS ISS.npy` |
| diag.py | Shows which cubes/cells are left unsolved: `python3 diag.py "$CH" ISS ISS.npy` |
| solutions/ | Best chromosomes so far (.npy, ends with -1). |

Results (official fitness, lower is better, -1 is perfect):

| Problem | Correct | Moves | Fitness | Runtime |
| --- | --- | --- | --- | --- |
| ISS | 145/148 | 581 | -0.96355 | ~10 s |
| JWST | 611/643 | 3257 | -0.92049 | ~1 min |
| Enterprise | 1422/1472 | 7059 | -0.94582 | ~14 min |
