#!/usr/bin/env python3
"""Identify the songs in a DJ mix with AudD's enterprise endpoint (server-side chunking, timestamps per match).

  audd_probe.py <audio> --json out.json [--skip 2] [--every 1]

Token: ~/.audd_key or $AUDD_API_TOKEN. Billing: 1 request per 12 s scanned; with --skip 2 --every 1 the
server scans 12 s then skips 24 s, i.e. ~100 requests per hour of mix. Output has the same shape as
shazam_probe.py (probes + merged segments) so ingest.py can use either.
"""
import argparse, json, os, re, subprocess, sys, time, urllib.request, uuid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from songmatch import merge, same_song, norm_title

def token():
    p = os.path.expanduser('~/.audd_key')
    t = os.environ.get('AUDD_API_TOKEN') or (open(p).read().strip() if os.path.exists(p) else '')
    if not t: sys.exit('AudD token missing: put it in ~/.audd_key or AUDD_API_TOKEN')
    return t

def mmss(s):
    parts = [int(x) for x in s.split(':')]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))

def post_multipart(url, fields, file_path):
    boundary = uuid.uuid4().hex; body = b''
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    data = open(file_path, 'rb').read()
    body += f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="mix.mp3"\r\nContent-Type: audio/mpeg\r\n\r\n'.encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(url, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=1800) as r: return json.load(r)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--json', required=True)
    ap.add_argument('--skip', type=int, default=2); ap.add_argument('--every', type=int, default=1); a = ap.parse_args()
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', a.audio], capture_output=True, text=True).stdout)
    # upload a compact mono mp3; AudD needs no more than that and it keeps uploads small
    tmp = a.json + '.upload.mp3'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', a.audio, '-ac', '1', '-ar', '44100', '-b:a', '64k', tmp], check=True)
    t0 = time.time()
    resp = post_multipart('https://enterprise.audd.io/', {'api_token': token(), 'accurate_offsets': 'true', 'skip': str(a.skip), 'every': str(a.every)}, tmp)
    os.remove(tmp)
    if resp.get('status') != 'success': sys.exit(f'AudD error: {json.dumps(resp)[:500]}')
    interval = 12 * (a.skip + a.every)
    rows = []
    for chunk in resp.get('result') or []:
        t = mmss(chunk['offset']); songs = chunk.get('songs') or []
        if not songs: rows.append({'t': t, 'title': None, 'artist': None, 'key': None, 'offset': None, 'score': None}); continue
        s = max(songs, key=lambda x: x.get('score', 0))
        # timecode = position inside the original song at the chunk start (+ start_offset ms within the chunk)
        off = mmss(s['timecode']) - (s.get('start_offset') or 0) / 1000.0
        key = s.get('isrc') or (norm_title(s['title']) + '|' + (s.get('artist') or '').lower())
        rows.append({'t': t, 'title': s.get('title'), 'artist': s.get('artist'), 'key': key, 'offset': off, 'score': s.get('score'), 'album': s.get('album'), 'label': s.get('label')})
    rows.sort(key=lambda r: r['t'])
    segs = merge(rows, interval)
    json.dump({'audio': a.audio, 'duration': dur, 'interval': interval, 'provider': 'audd', 'probes': rows, 'segments': segs, 'raw_chunks': len(resp.get('result') or [])}, open(a.json, 'w'), indent=1)
    print(f"{len(rows)} chunks, {sum(1 for r in rows if r['key'])} identified, {time.time()-t0:.0f}s", file=sys.stderr)
    for s in segs: print(f"{int(s['start'])//60:3d}:{int(s['start'])%60:02d}  {s['hits']:2d}x  ±{s['offset_spread']:<5}  {(s['artist'] or '?')} - {(s['title'] or '(unidentified)')}")
main()
