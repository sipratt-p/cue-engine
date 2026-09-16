#!/usr/bin/env python3
"""Identify the songs in a DJ mix by probing Shazam on a fixed grid and merging the results.

  shazam_probe.py <audio> [--interval 20] [--probe 12] [--spacing 3.5] --json out.json

- 12 s probes every `interval` seconds go to Shazam through shazamio (legacy recognize_song;
  the Rust recognize() hangs on macOS). Shazam throttles silently past ~17 req/min, hence spacing.
- Results stream to <json>.probes.jsonl so a run can resume after a crash or throttle.
- Merge: consecutive probes with the same song form one segment. Same-song variants
  (remix/radio edit/feat.) are folded into one song. One-probe flickers inside a longer song are absorbed.
- Start of a segment = median over its probes of (probe time − match offset), i.e. where the song
  actually began in the mix, not where the grid first saw it. Unidentified stretches stay as gaps.
"""
import argparse, asyncio, json, os, re, subprocess, sys, tempfile, time
from shazamio import Shazam
from songmatch import norm_title, same_song, merge

def cut(path, t, dur, out):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(t), '-t', str(dur), '-i', path, '-ac', '1', '-ar', '44100', '-b:a', '128k', out], check=True)

THROTTLED = 0
async def recognize(sh, out):
    """One probe. Shazam signals throttling by hanging, not by 429: after 3 consecutive hangs, back off 5 minutes."""
    global THROTTLED
    for attempt in range(3):
        try:
            r = await asyncio.wait_for(sh.recognize_song(out), timeout=30); THROTTLED = 0; return r
        except Exception as e:
            last = type(e).__name__; THROTTLED += 1
            if THROTTLED >= 3:
                print(f'  throttled: backing off 5 min ({time.strftime("%H:%M")})', file=sys.stderr, flush=True); await asyncio.sleep(300); THROTTLED = 0
            else: await asyncio.sleep(10 * (attempt + 1))
    return {'error': last}

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--interval', type=float, default=20); ap.add_argument('--probe', type=float, default=12)
    ap.add_argument('--spacing', type=float, default=3.5); ap.add_argument('--json', required=True); a = ap.parse_args()
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.audio], capture_output=True, text=True).stdout)
    ts = [i * a.interval for i in range(int(dur // a.interval) + 1)]; ts = [t for t in ts if t + a.probe <= dur]
    jl = a.json + '.probes.jsonl'; done = {}
    if os.path.exists(jl):
        for line in open(jl):
            r = json.loads(line); done[r['t']] = r
    work = tempfile.mkdtemp(prefix='shz-'); sh = Shazam(); t0 = time.time(); errors = 0
    with open(jl, 'a') as log:
        for t in ts:
            if t in done and not done[t].get('error'): continue
            out = os.path.join(work, 'p.mp3'); cut(a.audio, t, a.probe, out)
            await asyncio.sleep(a.spacing)
            r = await recognize(sh, out)
            tr = r.get('track') or {}; m = (r.get('matches') or [{}])[0]
            row = {'t': t, 'title': tr.get('title'), 'artist': tr.get('subtitle'), 'key': tr.get('key'), 'offset': m.get('offset'), 'skew': m.get('frequencyskew'), 'error': r.get('error')}
            if row['error']: errors += 1
            done[t] = row; log.write(json.dumps(row) + '\n'); log.flush()
            print(f"  {int(t)//60:3d}:{int(t)%60:02d} -> {row['title'] or row['error'] or '-'}", file=sys.stderr, flush=True)
    rows = [done[t] for t in ts if t in done]
    segs = merge(rows, a.interval)
    json.dump({'audio': a.audio, 'duration': dur, 'interval': a.interval, 'probes': rows, 'segments': segs}, open(a.json, 'w'), indent=1)
    print(f'{len(rows)} probes, {errors} errors, {time.time()-t0:.0f}s', file=sys.stderr)
    for s in segs: print(f"{int(s['start'])//60:3d}:{int(s['start'])%60:02d}  {s['hits']:2d}x  ±{s['offset_spread']:<5}  {(s['artist'] or '?')} - {(s['title'] or '(unidentified)')}")
asyncio.run(main())
