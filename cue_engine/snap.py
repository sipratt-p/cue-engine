#!/usr/bin/env python3
"""Snap estimated song starts to the beat grid. DJs cut on phrase boundaries (multiples of 8/16/32 beats),
so a start that lands mid-bar is nearly always a few beats off the true cut.

  snap.py <audio> <markers.json> [--json out.json] [--window 6]

For each marker: beat-track ±window s around it, pick the beat with the strongest onset that also starts an
8-beat group in the local grid, and move the marker there if the move is within the window."""
import argparse, json, subprocess, numpy as np, librosa
SR = 22050
def load(path, a, b):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', str(max(0, a)), '-t', str(b - a), '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)
def snap(audio, t, window=6.0, duration=None):
    a = max(0.0, t - 4 * window); b = t + 4 * window if duration is None else min(duration, t + 4 * window)   # long context for a stable tempo
    y = load(audio, a, b)
    if len(y) < SR * 8: return t, 0.0
    onset = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=SR, hop_length=512, units='frames')
    if len(beats) < 8: return t, 0.0
    times = librosa.frames_to_time(beats, sr=SR, hop_length=512) + a
    strength = onset[beats]
    # phase of the 8-beat grid: the offset whose beats carry the most onset energy (downbeat-ish)
    phases = [strength[k::8].mean() for k in range(8)]; k0 = int(np.argmax(phases))
    cand = [(abs(times[i] - t), times[i], strength[i]) for i in range(k0, len(beats), 8) if abs(times[i] - t) <= window]
    if not cand: return t, 0.0
    # prefer strong onsets, then proximity
    d, tt, s = min(cand, key=lambda c: c[0] - 2.0 * (c[2] / (strength.max() + 1e-9)))
    return float(tt), float(tt - t)
if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('markers'); ap.add_argument('--json'); ap.add_argument('--window', type=float, default=6.0); a = ap.parse_args()
    M = json.load(open(a.markers)); out = []
    for m in M:
        if m['start'] <= 0: out.append(m); continue
        t2, dt = snap(a.audio, m['start'], a.window); mm = dict(m); mm['start'] = t2; mm['snapped'] = round(dt, 2); out.append(mm)
    if a.json: json.dump(out, open(a.json, 'w'), indent=1)
    for m in out: print(f"{int(m['start'])//60:3d}:{int(m['start'])%60:02d}  {m.get('snapped', 0):+6.2f}  {m['title'][:50]}")
