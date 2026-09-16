#!/usr/bin/env python3
"""Parse a tracklist out of free text (SoundCloud description, comments, pasted list).
Returns [{'start': seconds|None, 'title': str}] in order. Timestamps optional."""
import re, sys, json
TS = re.compile(r'(?<![\d:])(\d{1,2}:)?(\d{1,2}):(\d{2})(?![\d:])')
LEAD = re.compile(r'^\s*[\[\(]?\s*(?:(?:\d{1,2}:)?\d{1,2}:\d{2})?\s*[\]\)]?\s*[-–—:.)]?\s*(?:\d{1,3}\s*[.)\-:]\s*)?')
def to_sec(m):
    h = int(m.group(1)[:-1]) if m.group(1) else 0
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3))
def parse(text):
    rows = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) < 3: continue
        m = TS.search(s)
        title = TS.sub('', s) if m else s
        title = LEAD.sub('', title).strip(' -–—:|\t')
        # a tracklist line normally looks like "Artist - Title" or has a timestamp or a leading number
        numbered = re.match(r'^\s*\d{1,3}\s*[.)\-:]', s)
        if not (m or numbered or re.search(r'\s[-–—]\s', title)): continue
        if len(title) < 3: continue
        rows.append({'start': to_sec(m) if m else None, 'title': title})
    # keep only the timestamped monotonic run if timestamps exist
    timed = [r for r in rows if r['start'] is not None]
    if len(timed) >= 3:
        out, last = [], -1
        for r in timed:
            if r['start'] > last: out.append(r); last = r['start']
        return out
    return rows
if __name__ == '__main__':
    print(json.dumps(parse(sys.stdin.read()), indent=1))
