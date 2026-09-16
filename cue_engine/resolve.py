#!/usr/bin/env python3
"""Turn fingerprint probes + (optional) page tracklist + local detection into song markers.

Anchors come from runs of consecutive probes that agree on a song. Remixes make match offsets
meaningless (the remix is rearranged), so a start is taken from offsets only when they agree;
otherwise from the probe grid. Page titles are matched to anchors by shared words, kept in page
order, and the stretches between matched anchors are split with the local detector using the
number of page titles that fall in between. Without a page list, anchors alone become markers and
long unidentified gaps are split by the detector.

  resolve.py <audio> <probes.json> [--titles tracklist.txt] [--json out.json]
"""
import argparse, json, os, re, subprocess, sys, tempfile, uuid
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); from songmatch import same_song
STOP = {'the', 'and', 'you', 'feat', 'remix', 'mix', 'edit', 'version', 'original', 'jpod', 'dub', 'vip', 'radio', 'extended', 'club', 'instrumental', 'karaoke', 'beat', 'originally', 'performed'}
def words(s): return {w for w in re.sub(r'[^a-z0-9 ]', ' ', (s or '').lower()).split() if len(w) > 2 and w not in STOP}

def runs(probes, interval, audio=None):
    out = []
    for p in probes:
        if not p.get('key'): continue
        if out and same_song(out[-1], p) and p['t'] - out[-1]['last'] <= max(2 * interval, 150):   # remixes drop in and out of recognisability
            r = out[-1]; r['last'] = p['t']; r['hits'] += 1; r['ests'].append(p['t'] - (p['offset'] or 0)); continue
        out.append({'key': p['key'], 'title': p['title'], 'artist': p['artist'], 'first': p['t'], 'last': p['t'], 'hits': 1, 'ests': [p['t'] - (p['offset'] or 0)]})
    for r in out:
        ests = sorted(r['ests']); spread = ests[-1] - ests[0] if len(ests) > 1 else 0
        med = ests[len(ests) // 2]
        # trust offsets only when they agree and place the start before the first sighting
        grid = max(0.0, r['first'] - interval / 2)
        r['start'] = med if (spread <= 20 and r['first'] - 3 * interval <= med <= r['first'] + 2) else grid
        r['offset_ok'] = spread <= 20
    # learned detector: the boundary is the strongest transition between the previous run's last sighting and this
    # run's first sighting (measured better than both offsets and the probe grid on every evaluation mix)
    p = nn_prob(audio) if audio else None
    if p is not None:
        prev_last = 0.0
        for r in out:
            # a song is normally recognised within ~2 minutes of its start; don't search a whole unidentified stretch
            lo = int(max(0, prev_last - 5, r['first'] - 120)); hi = int(min(len(p) - 1, r['first'] + interval / 2))
            if hi - lo >= 6: r['start'] = float(lo + int(np.argmax(p[lo:hi]))); r['offset_ok'] = False
            prev_last = r['last']
    return out

_PROB = {}
def nn_prob(audio):
    """Transition probability per second from the learned detector (~/djmix/detector.pt), or None if unavailable."""
    if audio in _PROB: return _PROB[audio]
    # CUE_DETECTOR may list several models (comma separated); their probability curves are averaged (ensemble)
    models = [os.path.expanduser(m) for m in os.environ.get('CUE_DETECTOR', os.path.join(os.path.dirname(HERE), 'models', 'detector-v1.pt')).split(',') if m.strip()]
    curves = []
    for model in models:
        cache = audio + '.nnprob-' + os.path.basename(model).replace('.pt', '') + '.npy'; p = None
        if os.path.exists(cache): p = np.load(cache)
        elif os.path.exists(model):
            try:
                py = os.path.join(os.path.dirname(HERE), '.venv', 'bin', 'python'); py = py if os.path.exists(py) else sys.executable
                out = os.path.join(tempfile.gettempdir(), 'cue-nn-' + uuid.uuid4().hex[:8] + '.json')
                r = subprocess.run([py, os.path.join(HERE, 'detect_nn.py'), audio, '--model', model, '--json', out], capture_output=True)
                if r.returncode == 0: p = np.array(json.load(open(out))['prob'], float); np.save(cache, p); os.remove(out)
            except Exception: p = None
        if p is not None: curves.append(p)
    p = None
    if curves:
        n = min(len(c) for c in curves); p = np.mean([c[:n] for c in curves], axis=0)
    _PROB[audio] = p; return p

def nn_peaks(p, a, b, k):
    """Exactly k boundaries inside [a, b] maximising summed transition probability with a minimum spacing
    (dynamic programming, same picker as the novelty detector), so songs can't collapse to a few seconds."""
    import segment as S
    lo, hi = int(max(0, a)), int(min(len(p) - 1, b))
    if hi - lo < 6: return []
    seg = np.array(p[lo:hi], float)
    gap = int(np.clip(0.4 * (hi - lo) / (k + 1), 20, 90))
    got = S.pick_n(seg, k, gap, edge=min(20, (hi - lo) // 4))
    return sorted(lo + int(i) for i in got)

def detect(audio, a, b, n):
    """n songs inside [a, b] → n-1 boundaries (mix time). Learned detector when available, else novelty."""
    if n < 2 or b - a < 60: return []
    p = nn_prob(audio)
    if p is not None:
        got = nn_peaks(p, a, b, n - 1)
        if len(got) == n - 1: return got
    stem = os.path.join(tempfile.gettempdir(), 'cue-clip-' + uuid.uuid4().hex[:8]); out = stem + '.json'
    ext = os.path.splitext(audio)[1].lower()
    if ext in ('.m4a', '.mp3', '.aac', '.wav', '.flac'):   # stream copy into the same container
        clip = stem + ext; subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-t', str(b - a), '-i', audio, '-c', 'copy', clip], check=True)
    else:                                                    # anything else: transcode the slice
        clip = stem + '.m4a'; subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-t', str(b - a), '-i', audio, '-c:a', 'aac', '-b:a', '128k', clip], check=True)
    py = os.path.join(os.path.dirname(HERE), '.venv', 'bin', 'python'); py = py if os.path.exists(py) else sys.executable
    r = subprocess.run([py, os.path.join(HERE, 'segment.py'), clip, '--n', str(n), '--kernel', '60', '--json', out], capture_output=True)
    res = [] if r.returncode else [a + x['start'] for x in json.load(open(out))['boundaries']]
    for f in (clip, out, clip + '.feat.npz'):
        try: os.remove(f)
        except OSError: pass
    return res

def resolve(audio, probes, interval, duration, titles=None, min_hits=2, aligned=None, min_margin=0.5, mode='auto'):
    """probes: fingerprint probe rows (may be empty). aligned: align.py anchors (may be None). titles: page tracklist."""
    R = runs(probes, interval, audio) if probes else []
    titles = titles or []
    if aligned and titles and not R:
        # alignment only: "title i is playing at time t"; its start lies between the previous sighting and t
        seen = {}
        for an in aligned:
            if an.get('found') and an.get('margin', 0) >= min_margin: seen[an['index']] = an['at']
        return resolve_sightings(audio, seen, titles, duration, R, interval)
    if titles and R and mode == 'auto':
        # enough fingerprint anchors to trust them as the skeleton? else the page list is the skeleton
        good = [r for r in R if r['hits'] >= min_hits]
        tw = [words(t) for t in titles]
        matched = sum(1 for r in good if max((len((words(r['title']) | words(r['artist'])) & w) for w in tw), default=0) >= 2)
        mode = 'label' if matched >= 0.6 * len(titles) else 'titles'
    if titles and R and mode == 'label':
        # boundaries come from the fingerprints (+ detector inside long gaps); page titles are attached afterwards
        base = resolve(audio, probes, interval, duration, None, min_hits, mode=mode)
        return label_markers(base, titles, R, aligned, min_margin)
    if not titles and not R:
        # nothing to anchor on: segment with the learned detector alone (novelty detector if no model)
        p = nn_prob(audio)
        if p is not None:
            import segment as S
            thr = float(np.percentile(p, 90))
            cand = [i for i in range(1, len(p) - 1) if p[i] >= p[i - 1] and p[i] >= p[i + 1] and p[i] >= max(0.15, thr)]
            picks = []
            for i in sorted(cand, key=lambda i: -p[i]):
                if all(abs(i - j) >= 75 for j in picks) and 30 <= i <= len(p) - 30: picks.append(i)
            picks.sort()
            markers = [{'start': 0.0, 'title': 'Track 1', 'source': 'guess'}] + [{'start': float(t), 'title': f'Track {k + 2}', 'source': 'detector'} for k, t in enumerate(picks)]
            return finish(markers)
        bounds = detect(audio, 0.0, duration, max(2, int(round(duration / 240))))
        return finish([{'start': 0.0, 'title': 'Track 1', 'source': 'guess'}] + [{'start': b, 'title': f'Track {k + 2}', 'source': 'guess'} for k, b in enumerate(bounds)])
    if not titles:
        keep = [r for r in R if r['hits'] >= min_hits]
        markers = []
        for i, r in enumerate(keep):
            markers.append({'start': r['start'], 'title': f"{r['artist']} - {r['title']}" if r['artist'] else r['title'], 'source': 'fingerprint'})
            nxt = keep[i + 1]['start'] if i + 1 < len(keep) else duration
            gap = nxt - (r['last'] + interval)
            if gap >= 150:   # somebody unidentified lives here; guess how many and where
                n = max(1, int(round(gap / 200)))
                bounds = [r['last'] + interval / 2] + detect(audio, r['last'], nxt, n)
                markers += [{'start': b, 'title': 'Unidentified', 'source': 'guess'} for b in bounds]
        if not markers or markers[0]['start'] > 30:
            head = markers[0]['start'] if markers else duration
            markers = [{'start': 0.0, 'title': 'Unidentified' if markers else 'Opening track', 'source': 'guess'}] + markers
        return finish(markers)
    # --- page tracklist available: match anchors to titles, in order ---
    tw = [words(t) for t in titles]
    cand = []
    for i, r in enumerate(R):
        rw = words(r['title']) | words(r['artist'])
        best, score = None, 0
        for j, w in enumerate(tw):
            sc = len(rw & w)
            if sc > score: best, score = j, sc
        if best is not None and (score >= 2 or (score == 1 and r['hits'] >= 2)): cand.append((i, best, score, r['hits']))
    # one candidate per page title: the strongest match (then the longest run)
    by_title = {}
    for c in cand:
        if c[1] not in by_title or (c[2], c[3]) > (by_title[c[1]][2], by_title[c[1]][3]): by_title[c[1]] = c
    cand = sorted(by_title.values())
    # longest increasing subsequence over page index keeps the matches consistent with page order
    dp = [(1, -1)] * len(cand)
    for k in range(len(cand)):
        for m in range(k):
            if cand[m][1] < cand[k][1] and dp[m][0] + 1 > dp[k][0]: dp[k] = (dp[m][0] + 1, m)
    if cand:
        k = max(range(len(cand)), key=lambda x: (dp[x][0], cand[x][3])); seq = []
        while k != -1: seq.append(cand[k]); k = dp[k][1]
        seq.reverse()
    else: seq = []
    matched = {ri: ti for ri, ti, _, _ in seq}
    assigned = {}   # page index -> start
    for ri, ti in matched.items(): assigned[ti] = R[ri]['start']
    # first page title starts at 0 unless an anchor says otherwise
    if 0 not in assigned: assigned[0] = 0.0
    # fill the stretches between assigned titles
    keys = sorted(assigned)
    for a_i, b_i in zip(keys, keys[1:] + [None]):
        lo = assigned[a_i]; hi = assigned[b_i] if b_i is not None else duration
        missing = list(range(a_i + 1, (b_i if b_i is not None else len(titles))))
        if not missing: continue
        # unmatched anchors inside the stretch, in order, can stand in for missing titles one-for-one
        inside = [r for i, r in enumerate(R) if i not in matched and lo < r['first'] < hi and r['hits'] >= 2]
        if len(inside) == len(missing):
            for t_i, r in zip(missing, inside): assigned[t_i] = r['start']
            continue
        # otherwise let the detector split the stretch; anchor-run ends bound the first song's end
        first_end = None
        for i, r in enumerate(R):
            if i in matched and matched[i] == a_i: first_end = r['last'] + interval / 2
        bounds = detect(audio, lo, hi, len(missing) + 1)
        if first_end and bounds and bounds[0] < first_end: bounds[0] = first_end
        for t_i, b in zip(missing, bounds): assigned[t_i] = b
    markers = [{'start': assigned[i], 'title': titles[i], 'source': 'fingerprint' if any(ti == i for ti in matched.values()) else 'guess'} for i in sorted(assigned)]
    return finish(markers)

def resolve_sightings(audio, seen, titles, duration, R, interval):
    """seen: page index -> mix time where that song is audibly playing. Starts are found by the local detector
    between consecutive sightings (one boundary per missing title in between), then everything is ordered."""
    # fingerprint runs can add sightings for titles alignment missed (same word matching as the main path)
    tw = [words(t) for t in titles]
    for r in R:
        if r['hits'] < 2: continue
        rw = words(r['title']) | words(r['artist']); best = max(range(len(tw)), key=lambda j: len(rw & tw[j]), default=None)
        if best is not None and len(rw & tw[best]) >= 2 and best not in seen: seen[best] = r['first']
    # keep sightings consistent with page order (longest increasing run of (index, time))
    items = sorted(seen.items()); keep = []
    for idx, t in items:
        while keep and keep[-1][1] > t: keep.pop()
        keep.append((idx, t))
    seen = dict(keep)
    assigned = {0: 0.0}
    idxs = sorted(seen)
    prev_i, prev_t = 0, 0.0
    for i in idxs:
        if i == 0: prev_t = 0.0; continue
        missing = list(range(prev_i + 1, i + 1))          # titles whose starts lie in (prev_t, seen[i]]
        lo = max(prev_t, assigned.get(prev_i, 0.0)); hi = seen[i]
        bounds = detect(audio, lo, hi, len(missing) + 1)
        if len(bounds) < len(missing):                     # detector failed: spread evenly
            bounds = [lo + (hi - lo) * (k + 1) / (len(missing) + 1) for k in range(len(missing))]
        for t_i, b in zip(missing, bounds): assigned[t_i] = b
        prev_i, prev_t = i, seen[i]
    # titles after the last sighting
    tail = list(range(prev_i + 1, len(titles)))
    if tail:
        bounds = detect(audio, prev_t, duration, len(tail) + 1)
        if len(bounds) < len(tail): bounds = [prev_t + (duration - prev_t) * (k + 1) / (len(tail) + 1) for k in range(len(tail))]
        for t_i, b in zip(tail, bounds): assigned[t_i] = b
    markers = [{'start': assigned[i], 'title': titles[i], 'source': 'aligned' if i in seen else 'guess'} for i in sorted(assigned)]
    return finish(markers)

def label_markers(markers, titles, R, aligned, min_margin):
    """Attach page titles to fingerprint-derived markers. A marker whose fingerprint title shares words with a page
    title takes that title (page order enforced); the rest are filled in order between labelled neighbours. Markers
    beyond the page's count keep their fingerprint name."""
    tw = [words(t) for t in titles]
    labelled = {}
    for i, m in enumerate(markers):
        rw = words(m['title'])
        best = max(range(len(tw)), key=lambda j: len(rw & tw[j]), default=None)
        if best is not None and len(rw & tw[best]) >= 2: labelled[i] = best
    if aligned:
        for an in aligned:
            if an.get('found') and an.get('margin', 0) >= min_margin:
                j = max(range(len(markers)), key=lambda i: markers[i]['start'] if markers[i]['start'] <= an['at'] else -1)
                labelled.setdefault(j, an['index'])
    # enforce page order: keep the longest increasing chain of (marker index, page index)
    items = sorted(labelled.items()); keep = []
    for mi, ti in items:
        while keep and keep[-1][1] >= ti: keep.pop()
        keep.append((mi, ti))
    labelled = dict(keep)
    out = []; next_title = 0
    for i, m in enumerate(markers):
        mm = dict(m)
        if i in labelled:
            next_title = labelled[i]; mm['title'] = titles[next_title]; mm['source'] = 'fingerprint'; next_title += 1
        else:
            # fill in order only if a title is available before the next labelled marker
            upcoming = [labelled[j] for j in range(i + 1, len(markers)) if j in labelled]
            if next_title < len(titles) and (not upcoming or next_title < upcoming[0]):
                mm['title'] = titles[next_title]; mm['source'] = m['source'] if m['source'] != 'fingerprint' else 'guess'; next_title += 1
            elif m['source'] == 'fingerprint': mm['title'] = m['title'] + ' (not in tracklist)'
        out.append(mm)
    return out

def finish(markers):
    markers.sort(key=lambda m: m['start']); out = []
    for m in markers:
        if out and m['start'] - out[-1]['start'] < 5: m['start'] = out[-1]['start'] + 5
        out.append(m)
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('probes', nargs='?'); ap.add_argument('--titles'); ap.add_argument('--aligned'); ap.add_argument('--json'); a = ap.parse_args()
    d = json.load(open(a.probes)) if a.probes else {'probes': [], 'interval': 20, 'duration': float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.audio], capture_output=True, text=True).stdout)}
    aligned = json.load(open(a.aligned))['anchors'] if a.aligned else None
    titles = None
    if a.titles:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'eval')); from tracklist import parse
        text = open(a.titles).read(); rows = parse(text); titles = [r['title'] for r in rows] or [l.strip() for l in text.splitlines() if l.strip()]
    markers = resolve(a.audio, d['probes'], d['interval'], d['duration'], titles)
    if a.json: json.dump(markers, open(a.json, 'w'), indent=1)
    for m in markers: print(f"{int(m['start'])//60:3d}:{int(m['start'])%60:02d}  {m['source']:<11} {m['title']}")
