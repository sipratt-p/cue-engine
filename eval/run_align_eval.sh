#!/bin/zsh
# Align previews for every truth mix, then score: align-only, shazam-only (if probed), and both.
cd ~/mixtape-cue/analysis; export PYTHONWARNINGS=ignore
for id in fatos mix-568b70f7 mix-ff3e4cdd mix-c811cc2c mix-342dd4bf mix-b2f003eb mix-358ebdaa mix-8f9a7f47; do
  [ -f eval/$id.m4a ] || continue
  python3 -c "import json; print('\n'.join(r['title'] for r in json.load(open('eval/$id-truth.json'))))" > /tmp/$id-titles.txt
  [ -f eval/$id-align.json ] || .venv/bin/python align.py eval/$id.m4a --titles /tmp/$id-titles.txt --json eval/$id-align.json > eval/$id-align.txt 2>&1
  echo "== $id ($(wc -l < /tmp/$id-titles.txt) titles)"
  .venv/bin/python resolve.py eval/$id.m4a --titles /tmp/$id-titles.txt --aligned eval/$id-align.json --json /tmp/$id-m-align.json >/dev/null 2>&1 && .venv/bin/python eval/score_markers.py eval/$id-truth.json /tmp/$id-m-align.json "align + titles (no shazam)"
  if [ -f eval/$id-shazam.json ]; then
    .venv/bin/python resolve.py eval/$id.m4a eval/$id-shazam.json --titles /tmp/$id-titles.txt --json /tmp/$id-m-shz.json >/dev/null 2>&1 && .venv/bin/python eval/score_markers.py eval/$id-truth.json /tmp/$id-m-shz.json "shazam + titles"
    .venv/bin/python resolve.py eval/$id.m4a eval/$id-shazam.json --titles /tmp/$id-titles.txt --aligned eval/$id-align.json --json /tmp/$id-m-both.json >/dev/null 2>&1 && .venv/bin/python eval/score_markers.py eval/$id-truth.json /tmp/$id-m-both.json "align + shazam + titles"
  fi
done
echo align-eval-done
