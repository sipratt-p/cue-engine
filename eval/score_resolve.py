"""Score the full pipeline (fingerprint probes + page titles without times + local gap-filling) against DJ timestamps.
usage: score_resolve.py <audio> <truth.json> <probes.json>"""
import json, sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cue_engine')); from resolve import resolve
audio, truth_f, probes_f = sys.argv[1:4]
truth = json.load(open(truth_f)); d = json.load(open(probes_f))
titles = [r['title'] for r in truth]
def score(P, label):
    T = np.array([r['start'] for r in truth if r['start']], float); P = np.array(P, float)
    line = f'{label:<34}'
    for tol in (10, 20, 30, 45): line += f'  {tol}s {sum(np.min(np.abs(P-t))<=tol for t in T)}/{len(T)}'
    print(line + f'   n={len(P)}  median {np.median([np.min(np.abs(P-t)) for t in T]):.0f}s')
with_titles = resolve(audio, d['probes'], d['interval'], d['duration'], titles)
no_titles = resolve(audio, d['probes'], d['interval'], d['duration'], None)
score([m['start'] for m in with_titles if m['start'] > 0], 'probes + page titles (no times)')
score([m['start'] for m in no_titles if m['start'] > 0], 'probes only (no page list)')
fp = [m for m in with_titles if m['source'] == 'fingerprint']
print(f'fingerprint-anchored titles: {len(fp)}/{len(titles)}')
