#!/usr/bin/env python3
"""Detect song boundaries in a continuous DJ mix from the spectrogram.

Usage: segment.py <audio> [--n N] [--min-gap SEC] [--json out.json]
Prints boundaries (seconds) with novelty scores. With --n, returns exactly N-1 boundaries
(the tracklist length is a strong prior); otherwise peak-picks with an adaptive threshold.
"""
import argparse, json, subprocess, sys, tempfile, os
import numpy as np, librosa, scipy.ndimage as ndi, scipy.signal as sig

SR = 22050
def load(path):
    # ffmpeg decode → mono float32 (libsndfile can't read AAC/m4a)
    cmd = ['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-']
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)

def features(y, hop=2048):
    """Per-frame features (~10.8 fps), then averaged to 1 frame/second."""
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=hop))
    mel = librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=96)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)[1:]   # timbre, drop energy term
    chroma = librosa.feature.chroma_stft(S=S, sr=SR, n_chroma=12)              # harmony / key
    contrast = librosa.feature.spectral_contrast(S=S, sr=SR)
    fps = SR / hop
    def per_second(F):
        n = int(F.shape[1] / fps)
        idx = (np.arange(n + 1) * fps).astype(int)
        return np.stack([F[:, idx[i]:idx[i + 1]].mean(axis=1) for i in range(n)], axis=1)
    feats = {'mfcc': per_second(mfcc), 'chroma': per_second(chroma), 'contrast': per_second(contrast)}
    # local tempo per second from onset strength (tempo jumps mark transitions)
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=SR, hop_length=hop)
    tempo = librosa.feature.tempo(onset_envelope=onset, sr=SR, hop_length=hop, aggregate=None, ac_size=8.0)[None, :]
    feats['tempo'] = per_second(tempo)
    return feats

def novelty(F, kernel_sec=40, smooth=6):
    """Foote checkerboard novelty on a cosine self-similarity matrix of smoothed, z-scored features."""
    F = ndi.uniform_filter1d(F, smooth, axis=1)
    F = (F - F.mean(axis=1, keepdims=True)) / (F.std(axis=1, keepdims=True) + 1e-8)
    Fn = F / (np.linalg.norm(F, axis=0, keepdims=True) + 1e-8)
    S = Fn.T @ Fn
    L = kernel_sec // 2
    g = sig.windows.gaussian(2 * L, std=L / 2.0)
    K = np.outer(g, g) * np.kron(np.array([[1, -1], [-1, 1]]), np.ones((L, L)))
    n = S.shape[0]
    Sp = np.pad(S, L, mode='edge')
    nov = np.array([np.sum(Sp[i:i + 2 * L, i:i + 2 * L] * K) for i in range(n)])
    nov = np.maximum(nov, 0)
    return nov / (nov.max() + 1e-8)

def tempo_novelty(tempo, win=20):
    t = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(tempo[0], win, mode='edge'), 2 * win + 1), axis=1)
    d = np.abs(np.gradient(ndi.uniform_filter1d(t, 8)))
    return d / (d.max() + 1e-8)

def combine(feats):
    nov = 0.45 * novelty(feats['mfcc']) + 0.35 * novelty(feats['chroma']) + 0.2 * novelty(feats['contrast'])
    nov = 0.85 * nov + 0.15 * tempo_novelty(feats['tempo'])
    return ndi.gaussian_filter1d(nov, 3)

def pick_n(nov, k, min_gap, edge=45):
    """Exactly k boundaries maximizing summed novelty with min spacing (DP over local peaks)."""
    peaks, _ = sig.find_peaks(nov, distance=8)
    peaks = peaks[(peaks >= edge) & (peaks <= len(nov) - edge)]
    if len(peaks) < k: return sorted(peaks.tolist())
    P = len(peaks); NEG = -1e9
    best = np.full((k + 1, P), NEG); prev = np.full((k + 1, P), -1, dtype=int)
    best[1] = nov[peaks]
    for j in range(2, k + 1):
        for p in range(P):
            ok = np.where(peaks[:p] <= peaks[p] - min_gap)[0]
            if len(ok) == 0: continue
            q = ok[np.argmax(best[j - 1][ok])]
            if best[j - 1][q] > NEG: best[j][p] = best[j - 1][q] + nov[peaks[p]]; prev[j][p] = q
    p = int(np.argmax(best[k])); out = []
    for j in range(k, 0, -1): out.append(int(peaks[p])); p = prev[j][p]
    return sorted(out)

def pick_auto(nov, min_gap, edge=45):
    thr = np.median(nov) + 1.0 * nov.std()
    peaks, _ = sig.find_peaks(nov, height=thr, distance=min_gap)
    return [int(p) for p in peaks if edge <= p <= len(nov) - edge]

def refine(y, t, radius=6):
    """Snap a boundary to the nearest strong downbeat-ish onset within ±radius seconds."""
    a, b = max(0, int((t - radius) * SR)), int((t + radius) * SR)
    seg = y[a:b]
    if len(seg) < SR: return float(t)
    on = librosa.onset.onset_strength(y=seg, sr=SR, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(on)), sr=SR, hop_length=512) + a / SR
    w = np.exp(-0.5 * ((times - t) / (radius / 2)) ** 2)
    return float(times[np.argmax(on * w)])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--n', type=int); ap.add_argument('--min-gap', type=float, default=0, help='0 = auto from duration/n')
    ap.add_argument('--json'); ap.add_argument('--novelty-out'); ap.add_argument('--kernel', type=int, default=40)
    ap.add_argument('--weights', default='0.45,0.35,0.2,0.15', help='mfcc,chroma,contrast,tempo')
    a = ap.parse_args()
    y = load(a.audio); dur = len(y) / SR
    print(f'{a.audio}: {dur:.1f}s', file=sys.stderr)
    cache = a.audio + '.feat.npz'
    if os.path.exists(cache): F = dict(np.load(cache))
    else: F = features(y); np.savez(cache, **F)
    w = [float(x) for x in a.weights.split(',')]
    nov = 0.85 * (w[0] * novelty(F['mfcc'], a.kernel) + w[1] * novelty(F['chroma'], a.kernel) + w[2] * novelty(F['contrast'], a.kernel)) + w[3] * tempo_novelty(F['tempo'])
    nov = ndi.gaussian_filter1d(nov / nov.max(), 3)
    gap = int(a.min_gap) if a.min_gap > 0 else int(np.clip(0.4 * dur / max(a.n or 12, 1), 25, 90))
    print(f'min gap {gap}s kernel {a.kernel}s', file=sys.stderr)
    secs = pick_n(nov, a.n - 1, gap) if a.n else pick_auto(nov, gap)
    out = [{'start': round(refine(y, s), 2), 'coarse': s, 'score': round(float(nov[s]), 3)} for s in secs]
    if a.novelty_out: np.save(a.novelty_out, nov)
    if a.json: json.dump({'duration': dur, 'boundaries': out}, open(a.json, 'w'), indent=1)
    for b in out: print(f"{b['start']:8.2f}  score={b['score']:.3f}")
if __name__ == "__main__": main()
