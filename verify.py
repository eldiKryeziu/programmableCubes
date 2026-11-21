import sys, os, numpy as np
CH, name, f = os.path.abspath(sys.argv[1]), sys.argv[2], os.path.abspath(sys.argv[3])
sys.path.insert(0, CH); os.chdir(CH)
from programmable_cubes_UDP import programmable_cubes_UDP
udp = programmable_cubes_UDP(name); x = np.load(f)
print(name, 'official fitness', udp.fitness(x)[0], 'commands', (len(x)-1)//2)
