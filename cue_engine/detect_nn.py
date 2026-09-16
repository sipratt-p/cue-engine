#!/usr/bin/env python3
"""Run the trained transition detector on a mix. Prints boundary candidates with probabilities.
usage: detect_nn.py <audio> [--model ~/djmix/detector.pt] [--json out] [--thr 0.3] [--n N]"""
import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'eval'))
from djmix_prep import feats, load
from train_detector import Net, peaks
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('audio'); ap.add_argument('--model', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'detector-v1.pt')); ap.add_argument('--json'); ap.add_argument('--thr', type=float, default=0.3); ap.add_argument('--n', type=int); a = ap.parse_args()
    X = feats(load(a.audio)); net = Net(); net.load_state_dict(torch.load(a.model, map_location='cpu')); net.eval()
    with torch.no_grad(): p = torch.sigmoid(net(torch.tensor(X[None])))[0].numpy()
    P = peaks(p, thr=a.thr)
    if a.n and len(P) > a.n - 1: P = np.array(sorted(sorted(P, key=lambda i: -p[int(i)])[:a.n - 1]))
    out = [{'start': float(t), 'prob': float(p[int(t)])} for t in P]
    if a.json: json.dump({'duration': len(X), 'boundaries': out, 'prob': p.round(3).tolist()}, open(a.json, 'w'))
    for b in out: print(f"{int(b['start'])//60:3d}:{int(b['start'])%60:02d}  p={b['prob']:.2f}")
if __name__ == '__main__': main()
