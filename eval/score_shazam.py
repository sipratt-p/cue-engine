import json, sys, re, numpy as np
truth = json.load(open(sys.argv[1])); d = json.load(open(sys.argv[2])); seg = d['segments']
T = np.array([r['start'] for r in truth if r['start']], float); P = np.array([s['start'] for s in seg if s['key']], float)
ident = sum(1 for p in d['probes'] if p.get('key')); err = sum(1 for p in d['probes'] if p.get('error'))
print(f'{len(d["probes"])} probes ({ident} identified, {err} errors) -> {len(P)} songs vs {len(T)} truth')
for tol in (10, 20, 30, 45): print(f'tol {tol:2d}s  recall {sum(np.min(np.abs(P-t))<=tol for t in T)}/{len(T)}  precision {sum(np.min(np.abs(T-p))<=tol for p in P)}/{len(P)}')
print('median |err|:', round(float(np.median([np.min(np.abs(P-t)) for t in T])), 1), 's')
def norm(s): return set(w for w in re.sub(r'[^a-z0-9 ]', ' ', (s or '').lower()).split() if len(w) > 2)
hits = 0
for s in seg:
    if not s['key']: continue
    tr = truth[int(np.argmin(np.abs(T - s['start'])))]
    if norm(s['title']) & norm(tr['title']) or norm(s['artist']) & norm(tr['title']): hits += 1
print('titles/artists agreeing with nearest truth entry:', hits, '/', len(P))
