import sys, numpy as np
from cubesim import Problem
from greedy import articulation_points
CH=sys.argv[1]; P=Problem(sys.argv[2], CH); x=np.load(sys.argv[3])
e=P.ensemble(); e.apply([(x[2*i],x[2*i+1]) for i in range((len(x)-1)//2)])
ap=articulation_points(e)
wrong=[c for c,p in enumerate(e.pos) if P.target.get(p)!=e.types[c]]
print('not done', len(wrong), 'of which articulation', sum(c in ap for c in wrong), 'sitting on wrong-type target', sum(e.pos[c] in P.target for c in wrong))
emp=[p for p in P.target if p not in e.at]
print('empty targets', len(emp), 'types', np.bincount([P.target[p] for p in emp]), 'wrong cubes types', np.bincount([e.types[c] for c in wrong]))
for p in emp: print(p, P.target[p], 'occupied neighbours', len(e.neighbours(p)))
