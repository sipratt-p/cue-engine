#!/usr/bin/env python3
"""Shazam-free anchors: locate known songs inside a mix by aligning free 30 s preview clips.

  align.py <audio> --titles tracklist.txt [--json out.json]

For each title: search iTunes (fallback Deezer) for a preview, compute beat-tolerant chroma, and run
subsequence DTW against the whole mix. The best path gives the mix time where that part of the song
plays plus a confidence (how much better than the second-best, unrelated location). Previews are cut
from the middle of a song, so an anchor says "this song is playing here", not where it began; the
resolver turns ordered anchors into starts.
"""
import argparse, json, os, re, subprocess, sys, tempfile, urllib.parse, urllib.request, hashlib
import numpy as np, librosa
SR = 22050; HOP = 2048
CACHE = os.path.expanduser('~/.cache/cue-previews'); os.makedirs(CACHE, exist_ok=True)

def clean(title):
    t = re.sub(r'\((?:jpod|remix|edit|vip|dub|version|mix)[^)]*\)', ' ', title, flags=re.I)
    t = re.sub(r'\[[^\]]*\]', ' ', t); t = re.sub(r'\b(jpod|remix|remixed|bootleg|edit|vip|interlude|feat\.?|ft\.?)\b', ' ', t, flags=re.I)
    return ' '.join(t.replace('—', '-').replace('–', '-').split())

def search_preview(title):
    q = clean(title)
    key = hashlib.md5(q.encode()).hexdigest()[:12]; cached = os.path.join(CACHE, key + '.json')
    if os.path.exists(cached): return json.load(open(cached))
    res = None
    try:
        r = json.load(urllib.request.urlopen('https://itunes.apple.com/search?media=music&limit=3&term=' + urllib.parse.quote(q), timeout=20))['results']
        if r: res = {'source': 'itunes', 'name': f"{r[0]['artistName']} - {r[0]['trackName']}", 'url': r[0]['previewUrl'], 'duration': r[0].get('trackTimeMillis', 0) / 1000}
    except Exception: pass
    if not res:
        try:
            r = json.load(urllib.request.urlopen('https://api.deezer.com/search?limit=3&q=' + urllib.parse.quote(q), timeout=20)).get('data') or []
            if r and r[0].get('preview'): res = {'source': 'deezer', 'name': f"{r[0]['artist']['name']} - {r[0]['title']}", 'url': r[0]['preview'], 'duration': r[0].get('duration', 0)}
        except Exception: pass
    json.dump(res, open(cached, 'w')); return res

def load(path, sr=SR):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

MODE = os.environ.get('CUE_ALIGN_MODE', 'cqt')
def chroma(y):
    if MODE == 'harmonic': y = librosa.effects.harmonic(y, margin=4)          # drop drums/percussive remix layers
    if MODE == 'cens': C = librosa.feature.chroma_cens(y=y, sr=SR, hop_length=HOP); C = C + 1e-3; return C / np.linalg.norm(C, axis=0, keepdims=True)
    C = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP, bins_per_octave=36)
    C = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, mode='same'), 1, C)   # ~1 s smoothing: harmony, not beats
    C = C + 1e-3                                                                             # silence must not produce NaN cosines
    return C / np.linalg.norm(C, axis=0, keepdims=True)

def align(Cp, Cm):
    """Subsequence DTW of preview chroma Cp (12×n) into mix chroma Cm (12×m). Returns (mix_frame, cost, margin)."""
    D, wp = librosa.sequence.dtw(X=Cp, Y=Cm, subseq=True, metric='cosine', step_sizes_sigma=np.array([[1, 1], [1, 2], [2, 1]]), weights_add=np.array([0, 0, 0]), weights_mul=np.array([1, 1, 1]))
    last = D[-1, :] / Cp.shape[1]
    j = int(np.argmin(last)); best = float(last[j])
    # second best outside ±90 s of the winner: how unique is this location?
    w = int(90 * SR / HOP); mask = np.ones_like(last, dtype=bool); mask[max(0, j - w):j + w] = False
    second = float(np.min(last[mask])) if mask.any() else best
    start_frame = int(wp[-1, 1])   # path runs backwards; last row = beginning of the preview in the mix
    return start_frame, best, second

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--titles', required=True); ap.add_argument('--json'); a = ap.parse_args()
    titles = [l.strip() for l in open(a.titles) if l.strip()]
    y = load(a.audio); dur = len(y) / SR
    cache = a.audio + f'.chroma-{MODE}.npy'
    Cm = np.load(cache) if os.path.exists(cache) else chroma(y)
    if not os.path.exists(cache): np.save(cache, Cm)
    out = []
    for i, t in enumerate(titles):
        p = search_preview(t)
        if not p: out.append({'index': i, 'title': t, 'found': None}); print(f'{i:2d} {t[:45]:<45} no preview', flush=True); continue
        fn = os.path.join(CACHE, hashlib.md5(p['url'].encode()).hexdigest()[:12] + '.m4a')
        if not os.path.exists(fn): urllib.request.urlretrieve(p['url'], fn)
        Cp = chroma(load(fn))
        f, cost, second = align(Cp, Cm)
        margin = (second - cost) / max(second, 1e-6)
        at = f * HOP / SR
        out.append({'index': i, 'title': t, 'found': p['name'], 'source': p['source'], 'at': round(at, 1), 'cost': round(cost, 4), 'margin': round(margin, 3)})
        print(f'{i:2d} {t[:45]:<45} -> {p["name"][:38]:<38} at {int(at)//60:3d}:{int(at)%60:02d}  cost {cost:.3f} margin {margin:.2f}', flush=True)
    if a.json: json.dump({'audio': a.audio, 'duration': dur, 'anchors': out}, open(a.json, 'w'), indent=1)
main()
