"""Fetch DJ Mix Dataset audio in parallel with yt-dlp (the package downloader is single-threaded and slow).
usage: djmix_fetch.py <count> [workers]"""
import djmix as dj, random, sys, os, subprocess, json, time
from concurrent.futures import ThreadPoolExecutor
N = int(sys.argv[1]) if len(sys.argv) > 1 else 80; W = int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUT = os.path.expanduser('~/djmix/mixes'); os.makedirs(OUT, exist_ok=True)
random.seed(7)
mixes = [m for m in dj.mixes if m.num_timestamps and m.num_timestamps >= 8 and m.audio_source in ('soundcloud', 'mixcloud', 'youtube')]
random.shuffle(mixes); mixes = mixes[:N]
log = open(os.path.expanduser('~/djmix/fetch.log'), 'a')
def fetch(m):
    dest = os.path.join(OUT, m.id + '.m4a')
    if os.path.exists(dest): return m.id, 'exists'
    t0 = time.time()
    r = subprocess.run(['yt-dlp', '-q', '-x', '--audio-format', 'm4a', '--audio-quality', '64K', '--no-playlist', '-o', os.path.join(OUT, m.id + '.%(ext)s'), m.audio_url], capture_output=True, text=True, timeout=1800)
    ok = os.path.exists(dest)
    log.write(f'{m.id}\t{"ok" if ok else "fail"}\t{m.audio_source}\t{time.time()-t0:.0f}s\t{(r.stderr or "")[-100:].strip()}\n'); log.flush()
    return m.id, ok
with ThreadPoolExecutor(W) as ex:
    for mid, ok in ex.map(fetch, mixes): print(mid, ok, flush=True)
log.write('fetch-done\n')
