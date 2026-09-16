import json, sys, numpy as np
truth = [r['start'] for r in json.load(open(sys.argv[1])) if r['start']]
pred = [b['start'] for b in json.load(open(sys.argv[2]))['boundaries']]
truth, pred = np.array(truth, float), np.array(pred, float)
for tol in (10, 20, 30, 45):
    hit_t = sum(np.min(np.abs(pred - t)) <= tol for t in truth); hit_p = sum(np.min(np.abs(truth - p)) <= tol for p in pred)
    print(f'tol {tol:2d}s  recall {hit_t}/{len(truth)}  precision {hit_p}/{len(pred)}')
print('median |err| of matched truths:', round(float(np.median([np.min(np.abs(pred - t)) for t in truth])), 1), 's')
print('truth :', ' '.join(f'{t:.0f}' for t in truth)); print('pred  :', ' '.join(f'{p:.0f}' for p in pred))
