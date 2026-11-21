import sys, time, numpy as np
CH = sys.argv[1]
sys.path.insert(0, CH); sys.path.insert(0, '.')
import os; os.chdir(CH)
from programmable_cubes_UDP import programmable_cubes_UDP
from cubesim import Problem
rng = np.random.default_rng(0)
for name, L in [('ISS', 3000), ('JWST', 1500)]:
    udp = programmable_cubes_UDP(name); P = Problem(name)
    # random moves biased to cubes that can move: just random
    cmds = [(int(rng.integers(P.n)), int(rng.integers(6))) for _ in range(L)]
    x = P.chromosome(cmds)
    t=time.time(); f_off = udp.fitness(x)[0]; t1=time.time()-t
    e = P.ensemble(); t=time.time(); legal = e.apply(cmds); t2=time.time()-t
    same = np.array_equal(np.array(e.pos), udp.final_cube_positions)
    print(name, 'legal', legal, 'official', f_off, 'mine', P.fitness(e, L), 'same positions', same, f'{t1:.2f}s vs {t2:.2f}s')
