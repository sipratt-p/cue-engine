#!/usr/bin/env python3
"""Turn SoundCloud/YouTube links or local audio files into Cue-ready mixtapes with embedded chapters.

  ingest.py <url-or-file> [...] [--out ~/Mixtapes/Cue] [--tracklist file.txt]

For each input:
  1. download audio (yt-dlp) or copy the local file; grab the page description
  2. parse a tracklist from the description or --tracklist (timestamps optional)
  3. timestamps present  -> use them
     otherwise           -> identify songs with Shazam on a 20 s grid (shazam_probe.py); starts come from
                            the match offsets. If the page lists titles without times, they are matched to
                            the Shazam songs in order so remixes keep the DJ's naming. Unidentified gaps
                            longer than 2 minutes are split with the local detector and labelled as guesses.
     --no-shazam         -> local detection only (old behaviour)
  4. write <title>.m4a with MP4 chapters (+ title/artist tags). Cue reads chapters on import.
AirDrop or copy the output files to the phone and import them from Files.
"""
import argparse, json, os, re, subprocess, sys, tempfile, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'eval'))
from tracklist import parse as parse_tracklist
PY_ = sys.executable
def sh(*cmd, **kw): return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout
def fetch(url, work):
    info = json.loads(sh('yt-dlp', '--skip-download', '-J', url))
    path = os.path.join(work, 'source.m4a')
    subprocess.run(['yt-dlp', '-x', '--audio-format', 'm4a', '--audio-quality', '128K', '-o', os.path.join(work, 'source.%(ext)s'), url], check=True, capture_output=True)
    if not os.path.exists(path):
        cands = [f for f in os.listdir(work) if f.startswith('source.')]; path = os.path.join(work, cands[0])
    text = (info.get('description') or '') + '\n' + '\n'.join(c.get('text', '') for c in (info.get('comments') or []))
    global DATE, GENRE; GENRE = info.get('genre'); DATE = info.get('upload_date') and f"{info['upload_date'][:4]}-{info['upload_date'][4:6]}-{info['upload_date'][6:8]}"
    return path, info.get('title') or 'Mixtape', info.get('uploader') or info.get('artist') or 'Mixtape', text
def detect(path, n):
    out = os.path.join(tempfile.gettempdir(), 'cue-seg.json')
    subprocess.run([PY_, os.path.join(HERE, 'segment.py'), path, '--kernel', '60', '--json', out] + (['--n', str(n)] if n else []), check=True, capture_output=True)
    return [b['start'] for b in json.load(open(out))['boundaries']]
PY312 = os.path.join(os.path.dirname(HERE), '.venv312', 'bin', 'python')
PROVIDER = 'auto'
DATE = None
GENRE = None
def have_audd(): return bool(os.environ.get('AUDD_API_TOKEN') or os.path.exists(os.path.expanduser('~/.audd_key')))
def shazam(path, out_json):
    """Identify songs. Provider 'audd' (whole file, one call) when a token exists, else Shazam grid probing."""
    if PROVIDER == 'audd' or (PROVIDER == 'auto' and have_audd()):
        subprocess.run([PY_, os.path.join(HERE, 'audd_probe.py'), path, '--json', out_json], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([PY312 if os.path.exists(PY312) else PY_, os.path.join(HERE, 'shazam_probe.py'), path, '--interval', '20', '--json', out_json], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out_json))['segments']
def word_set(s):
    return set(w for w in re.sub(r'[^a-z0-9 ]', ' ', (s or '').lower()).split() if len(w) > 2)
def label_from_page(seg, titles, used):
    """Prefer the DJ's own title for this song when one of the page titles shares words with the Shazam result."""
    best, score = None, 0
    for i, t in enumerate(titles):
        if i in used: continue
        w = word_set(t); sc = len(w & word_set(seg['title'])) + len(w & word_set(seg['artist']))
        if sc > score: best, score = i, sc
    if best is not None and score >= 1: used.add(best); return titles[best]
    return f"{seg['artist']} - {seg['title']}" if seg['artist'] else seg['title']
def align(path, titles, out_json):
    """Shazam-free anchors: free 30 s previews of the titles aligned into the mix (align.py)."""
    tf = out_json + '.titles.txt'; open(tf, 'w').write('\n'.join(titles))
    subprocess.run([PY_, os.path.join(HERE, 'align.py'), path, '--titles', tf, '--json', out_json], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out_json))['anchors']
def build_markers_shazam(path, rows, work):
    """Fingerprint and/or alignment anchors + page titles + learned detector, combined by resolve.py."""
    from resolve import resolve
    titles = [r['title'] for r in rows]
    d = {'probes': [], 'interval': 20, 'duration': duration(path)}; aligned = None; how = 'align'
    if PROVIDER != 'align':
        out = os.path.join(work, 'probes.json'); shazam(path, out); d = json.load(open(out))
        how = 'audd' if PROVIDER == 'audd' or (PROVIDER == 'auto' and have_audd()) else 'shazam'
    if titles and PROVIDER in ('align', 'auto'):
        try: aligned = align(path, titles, os.path.join(work, 'align.json')); how = how if how == 'align' else how + '+align'
        except subprocess.CalledProcessError: aligned = None
    markers = resolve(path, d['probes'], d['interval'], d['duration'], titles or None, aligned=aligned)
    if len(markers) < 3 and d['duration'] > 900:
        # the page list gave nothing usable (e.g. one line): segment with the detector alone so skipping still works
        markers = resolve(path, [], d['interval'], d['duration'], None); how = 'detector'
    return [(m['start'], m['title'] + ('' if m['source'] in ('fingerprint', 'aligned') else ' (guess)')) for m in markers], how + '+detector'
def detect_range(path, a, b, n):
    """Run the local detector on a slice of the mix and return boundaries in mix time."""
    clip = os.path.join(tempfile.gettempdir(), 'cue-clip.m4a')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', str(a), '-t', str(b - a), '-i', path, '-c', 'copy', clip], check=True)
    try: return [a + x for x in detect(clip, n)]
    except Exception: return []
def duration(path): return float(sh('ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', path))
def build_markers(path, rows, work, use_shazam=True):
    dur = duration(path)
    timed = [r for r in rows if r['start'] is not None]
    if len(timed) >= 2:
        markers = [(r['start'], r['title']) for r in timed if r['start'] < dur]
        if markers[0][0] > 1: markers.insert(0, (0.0, 'Intro'))
        return markers, 'tracklist'
    if use_shazam:
        try: return build_markers_shazam(path, rows, work)
        except subprocess.CalledProcessError as e: print(f'  shazam failed ({e}); falling back to local detection', file=sys.stderr)
    titles = [r['title'] for r in rows]
    starts = [0.0] + detect(path, len(titles) if titles else None)
    if titles and len(starts) == len(titles): return list(zip(starts, titles)), 'detected'
    return [(s, titles[i] if i < len(titles) else f'Track {i + 1}') for i, s in enumerate(starts)], 'detected'
def write(path, markers, title, artist, out_dir, date=None, genre=None):
    dur = duration(path)
    safe = re.sub(r'[\\/:*?"<>|]+', '-', title).strip() or 'Mixtape'
    dest = os.path.join(out_dir, safe + '.m4a')
    meta = f';FFMETADATA1\ntitle={title}\nartist={artist}\n' + (f'date={date}\n' if date else '') + (f'genre={genre}\n' if genre else '')
    for i, (s, t) in enumerate(markers):
        end = markers[i + 1][0] if i + 1 < len(markers) else dur
        meta += f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={int(s * 1000)}\nEND={int(end * 1000)}\ntitle={t}\n'
    mf = os.path.join(tempfile.gettempdir(), 'cue-meta.txt'); open(mf, 'w').write(meta)
    codec = ['-c', 'copy'] if path.lower().endswith(('.m4a', '.mp4', '.aac')) else ['-c:a', 'aac', '-b:a', '160k']
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-i', mf, '-map', '0:a', '-map_metadata', '1', '-map_chapters', '1', *codec, dest], check=True)
    with open(os.path.splitext(dest)[0] + '.tracklist.txt', 'w') as f:
        for s, t in markers: f.write(f'{int(s)//3600:d}:{(int(s)//60)%60:02d}:{int(s)%60:02d} {t}\n' if s >= 3600 else f'{int(s)//60:d}:{int(s)%60:02d} {t}\n')
    return dest
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('inputs', nargs='+'); ap.add_argument('--out', default='out'); ap.add_argument('--tracklist'); ap.add_argument('--no-shazam', action='store_true', help='local detection only'); ap.add_argument('--provider', choices=['auto', 'audd', 'shazam', 'align'], default='auto')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    global PROVIDER; PROVIDER = a.provider
    for src in a.inputs:
        global DATE, GENRE; DATE = None; GENRE = None
        work = tempfile.mkdtemp(prefix='cue-')
        try:
            if re.match(r'https?://', src): path, title, artist, text = fetch(src, work)
            else:
                path = src; title = os.path.splitext(os.path.basename(src))[0]; artist = 'Local audio'; text = ''
                side = os.path.splitext(src)[0] + '.txt'
                if os.path.exists(side): text = open(side).read()
                meta = os.path.splitext(src)[0] + '.meta.json'
                if os.path.exists(meta):
                    md = json.load(open(meta)); title = md.get('display') or md.get('title') or title; artist = md.get('artist') or artist; DATE = md.get('date'); GENRE = md.get('genre')
            if a.tracklist: text = open(a.tracklist).read()
            rows = parse_tracklist(text)
            markers, how = build_markers(path, rows, work, use_shazam=not a.no_shazam)
            dest = write(path, markers, title, artist, a.out, date=DATE, genre=GENRE)
            print(f'{dest}\n  {len(markers)} songs ({how})')
            for s, t in markers: print(f'  {int(s)//60:3d}:{int(s)%60:02d}  {t}')
        except subprocess.CalledProcessError as e:
            print(f'FAILED {src}: {e.stderr or e}', file=sys.stderr)
        finally: shutil.rmtree(work, ignore_errors=True)
main()
