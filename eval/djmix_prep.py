"""Build training data for a transition detector from downloaded DJ Mix Dataset mixes.
Per second of audio: log-mel (64) + chroma (12) + onset strength + spectral contrast (7) -> X (T x 84);
labels y[t] = 1 within ±2 s of an annotated track start (dataset timestamps are 1 s..30 s resolution, so
also store the raw boundary list for tolerant evaluation)."""
import djmix as dj, numpy as np, librosa, os, re, sys, subprocess, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cue_engine'))
SR = 22050; HOP = 2048
OUT = os.path.expanduser('~/djmix/features'); os.makedirs(OUT, exist_ok=True)
def load(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)
def per_second(F, fps):
    n = int(F.shape[1] / fps); idx = (np.arange(n + 1) * fps).astype(int)
    return np.stack([F[:, idx[i]:idx[i + 1]].mean(axis=1) for i in range(n)], axis=1)
def feats(y):
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP)); fps = SR / HOP
    mel = librosa.power_to_db(librosa.feature.melspectrogram(S=S ** 2, sr=SR, n_mels=64))
    chroma = librosa.feature.chroma_stft(S=S, sr=SR, n_chroma=12)
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)
    onset = librosa.onset.onset_strength(S=mel, sr=SR, hop_length=HOP)[None, :]
    return np.concatenate([per_second(mel, fps), per_second(chroma, fps), per_second(contrast, fps), per_second(onset, fps)], axis=0).T.astype(np.float32)
def timestamps(m):
    """MixesDB formats: [h:mm:ss], [mm:ss] (precise) and [mm] / [mmm] (minute only, coarse). '?' entries are skipped.
    Returns sorted (seconds, coarse) pairs."""
    out = []
    for t in m.tracklist:
        mm = re.match(r'\[(\d+):(\d\d)(?::(\d\d))?\]', t.title)
        if mm:
            h, mi, s = (int(mm.group(1)), int(mm.group(2)), int(mm.group(3))) if mm.group(3) else (0, int(mm.group(1)), int(mm.group(2)))
            out.append((h * 3600 + mi * 60 + s, False)); continue
        mm = re.match(r'\[(\d{2,3})\]', t.title)
        if mm: out.append((int(mm.group(1)) * 60, True))
    seen = {}
    for t, c in out: seen.setdefault(t, c)
    return sorted(seen.items())
import glob
from multiprocessing import Pool
def prep(args):
    mid, p, ts, title = args
    f = os.path.join(OUT, mid + '.npz')
    if os.path.exists(f): return mid, 'exists'
    try:
        X = feats(load(p)); T = X.shape[0]
        # coarse (minute-only) labels: snap to the strongest local change inside that minute using the same
        # novelty curve the detector uses, so the label sits on the actual transition rather than at :00
        import segment as S
        if any(c for _, c in ts):
            F = {'mfcc': None}
            nov = S.combine({'mfcc': X[:, :64].T, 'chroma': X[:, 64:76].T, 'contrast': X[:, 76:83].T, 'tempo': X[:, 83:84].T})
        fixed = []
        for t, c in ts:
            if c and 0 < t < T:
                lo, hi = t, min(T - 1, t + 60); t = int(lo + np.argmax(nov[lo:hi])) if hi - lo > 5 else t
            fixed.append(t)
        y = np.zeros(T, np.float32)
        for t in fixed:
            if 0 < t < T: y[max(0, t - 2):t + 3] = 1
        np.savez_compressed(f, X=X, y=y, boundaries=np.array(fixed, np.float32), coarse=np.array([c for _, c in ts]), title=title)
        return mid, f'{T}s {len(ts)} boundaries'
    except Exception as e: return mid, f'ERR {e}'
if __name__ == '__main__':
    jobs = []
    for m in dj.mixes:
        cands = [c for c in glob.glob(os.path.join(os.path.expanduser('~/djmix/mixes'), m.id + '.*')) if not c.endswith(('.part', '.ytdl')) and '.part-' not in c]
        if not cands: continue
        ts = timestamps(m)
        if len(ts) < 5: continue
        jobs.append((m.id, cands[0], ts, m.title))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        for mid, msg in pool.imap_unordered(prep, jobs): print(mid, msg, flush=True)
    print('prepared', len(glob.glob(os.path.join(OUT, '*.npz'))))
