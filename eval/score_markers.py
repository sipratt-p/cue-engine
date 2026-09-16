import json, sys, numpy as np
truth = json.load(open(sys.argv[1])); M = json.load(open(sys.argv[2]))
T = np.array([r['start'] for r in truth if r['start']], float); P = np.array([m['start'] for m in M if m['start'] > 0], float)
line = f'{sys.argv[3] if len(sys.argv)>3 else "":<34}'
for tol in (10, 20, 30, 45): line += f'  {tol}s {sum(np.min(np.abs(P-t))<=tol for t in T)}/{len(T)}'
print(line + f'   n={len(P)}  median {np.median([np.min(np.abs(P-t)) for t in T]):.0f}s')
