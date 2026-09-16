"""Ingest many local mixes in parallel (avoids xargs argument limits with long names).
usage: batch_ingest.py [--min-seconds 900] [--workers 4] [--out ~/Mixtapes/Cue] <files or dirs>..."""
import argparse, glob, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument('paths', nargs='+'); ap.add_argument('--min-seconds', type=int, default=900); ap.add_argument('--workers', type=int, default=4)
ap.add_argument('--out', default='out'); ap.add_argument('--provider', default='align'); ap.add_argument('--skip', nargs='*', default=[])
a = ap.parse_args()
files = []
for p in a.paths:
    files += sorted(glob.glob(os.path.join(p, '*'))) if os.path.isdir(p) else [p]
files = [f for f in files if f.lower().endswith(('.m4a', '.mp3', '.wav', '.opus', '.flac', '.aac')) and not any(s in f for s in a.skip)]
def dur(f):
    try: return float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', f], capture_output=True, text=True).stdout)
    except Exception: return 0
files = [f for f in files if dur(f) >= a.min_seconds]
print(len(files), 'mixes', flush=True)
def run(f):
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(HERE, 'ingest.py'), f, '--provider', a.provider, '--out', a.out], capture_output=True, text=True)
    ok = 'songs (' in r.stdout
    return f'{"ok " if ok else "FAIL"} {time.time()-t0:4.0f}s  {os.path.basename(f)[:60]}' + ('' if ok else '\n' + r.stderr[-300:])
with ThreadPoolExecutor(a.workers) as ex:
    for line in ex.map(run, files): print(line, flush=True)
print('batch-done', flush=True)
