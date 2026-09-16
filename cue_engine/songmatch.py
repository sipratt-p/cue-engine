"""Song identity helpers shared by the fingerprint providers and the resolver."""
import re

def norm_title(t):
    t = (t or '').lower()
    t = re.sub(r'[\(\[].*?[\)\]]', ' ', t)                     # (feat. X) (remix) [radio edit]
    t = re.sub(r'\b(feat|ft|featuring|remix|mix|edit|version|radio|extended|club|original|vip|dub|instrumental)\b.*', ' ', t)
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', t).split())

def same_song(a, b):
    if not a or not b: return False
    if a['key'] == b['key']: return True
    na, nb = norm_title(a['title']), norm_title(b['title'])
    return bool(na) and na == nb


def merge(rows, interval):
    segs = []
    for r in rows:
        if not r.get('key'):
            if segs and segs[-1]['key'] is None: segs[-1]['end'] = r['t']
            else: segs.append({'key': None, 'title': None, 'artist': None, 'start': r['t'], 'end': r['t'], 'hits': 0, 'starts': []})
            continue
        est = r['t'] - (r.get('offset') or 0)
        if segs and same_song(segs[-1], r):
            segs[-1]['end'] = r['t']; segs[-1]['hits'] += 1; segs[-1]['starts'].append(est)
        elif len(segs) >= 2 and segs[-1]['hits'] <= 1 and same_song(segs[-2], r) and segs[-1]['key'] is not None:
            # A B A flicker: fold B (a single probe) back into A
            b = segs.pop(); a = segs[-1]; a['end'] = r['t']; a['hits'] += 1 + b['hits']; a['starts'] += b['starts'] + [est]
        else:
            segs.append({'key': r['key'], 'title': r['title'], 'artist': r['artist'], 'start': est, 'end': r['t'], 'hits': 1, 'starts': [est]})
    # unidentified gaps shorter than one interval between two songs are noise, not a song
    out = []
    for s in segs:
        if s['key'] is None and out and s['end'] - s['start'] < interval and out[-1]['key'] is not None: continue
        if s['key'] is None and out and out[-1]['key'] is None: out[-1]['end'] = s['end']; continue
        out.append(s)
    for i, s in enumerate(out):
        if s['starts']:
            ss = sorted(s['starts']); s['start'] = max(0.0, ss[len(ss) // 2])
        s['offset_spread'] = round(max(s['starts']) - min(s['starts']), 1) if len(s['starts']) > 1 else 0.0
        if i > 0 and s['start'] < out[i - 1]['start'] + 5: s['start'] = out[i - 1]['start'] + 5
        del s['starts']
    return out

