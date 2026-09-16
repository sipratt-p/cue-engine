"""Train a transition detector on per-second features from the DJ Mix Dataset.
X: (T, 84) per mix; y: 1 within ±2 s of an annotated start. Model: 1D CNN over a ±32 s context.
usage: train_detector.py [--epochs 20] [--holdout 0.2]"""
import argparse, glob, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, random, json
FEAT = os.path.expanduser('~/djmix/features'); dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
class Net(nn.Module):
    def __init__(s, f=84):
        super().__init__()
        s.norm = nn.BatchNorm1d(f)
        s.c1 = nn.Conv1d(f, 96, 5, padding=2); s.c2 = nn.Conv1d(96, 96, 5, padding=4, dilation=2); s.c3 = nn.Conv1d(96, 96, 5, padding=8, dilation=4)
        s.c4 = nn.Conv1d(96, 96, 5, padding=16, dilation=8); s.c5 = nn.Conv1d(96, 64, 5, padding=32, dilation=16); s.out = nn.Conv1d(64, 1, 1)
        s.drop = nn.Dropout(0.3)
    def forward(s, x):            # x: (B, T, F) -> logits (B, T)
        h = s.norm(x.transpose(1, 2))
        if s.training: h = h + 0.1 * torch.randn_like(h)      # feature noise: small data, strong augmentation
        for c in (s.c1, s.c2, s.c3, s.c4, s.c5): h = s.drop(F.gelu(c(h))) + (h if c.in_channels == c.out_channels else 0)
        return s.out(h).squeeze(1)
def load_all():
    data = []
    for f in sorted(glob.glob(os.path.join(FEAT, '*.npz'))):
        d = np.load(f); data.append((os.path.basename(f)[:-4], d['X'], d['y'], d['boundaries']))
    return data
def batches(data, win=600, bs=8):
    while True:
        xs, ys = [], []
        for _ in range(bs):
            _, X, y, _ = random.choice(data)
            if len(X) <= win: a = 0
            else: a = random.randint(0, len(X) - win)
            xs.append(X[a:a + win]); ys.append(y[a:a + win])
        L = min(len(x) for x in xs)
        yield torch.tensor(np.stack([x[:L] for x in xs])).to(dev), torch.tensor(np.stack([y[:L] for y in ys])).to(dev)
def peaks(p, min_gap=60, thr=0.3):
    out = []
    for i in range(1, len(p) - 1):
        if p[i] >= thr and p[i] >= p[i - 1] and p[i] >= p[i + 1] and (not out or i - out[-1] >= min_gap): out.append(i)
    return np.array(out, float)
def evaluate(net, data):
    net.eval(); res = {}
    with torch.no_grad():
        for name, X, y, B in data:
            p = torch.sigmoid(net(torch.tensor(X[None]).to(dev)))[0].cpu().numpy()
            P = peaks(p); B = B[(B > 0) & (B < len(X))]
            hits = {tol: int(sum(np.min(np.abs(P - b)) <= tol for b in B)) if len(P) else 0 for tol in (10, 20, 30)}
            prec = {tol: int(sum(np.min(np.abs(B - q)) <= tol for q in P)) if len(B) and len(P) else 0 for tol in (10, 20, 30)}
            res[name] = {'n_truth': int(len(B)), 'n_pred': int(len(P)), 'recall': hits, 'precision': prec}
    net.train(); return res
def summarize(res, label):
    nt = sum(r['n_truth'] for r in res.values()); npd = sum(r['n_pred'] for r in res.values())
    line = f'{label:<10}'
    for tol in (10, 20, 30): line += f'  {tol}s R {sum(r["recall"][tol] for r in res.values())}/{nt} P {sum(r["precision"][tol] for r in res.values())}/{npd}'
    print(line, flush=True)
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--epochs', type=int, default=20); ap.add_argument('--steps', type=int, default=150); ap.add_argument('--holdout', type=float, default=0.2); ap.add_argument('--save', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'detector-v1.pt')); a = ap.parse_args()
    data = load_all(); random.seed(1); random.shuffle(data); k = max(1, int(len(data) * a.holdout)); test, train = data[:k], data[k:]
    print(f'{len(train)} train mixes, {len(test)} test mixes, device {dev}', flush=True)
    net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    pos = torch.tensor([8.0]).to(dev); gen = batches(train)
    best = (-1, None)
    for ep in range(a.epochs):
        tot = 0
        for _ in range(a.steps):
            x, y = next(gen); loss = F.binary_cross_entropy_with_logits(net(x), y, pos_weight=pos); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        res = evaluate(net, test); f = fscore(res, 30)
        print(f'epoch {ep + 1} loss {tot / a.steps:.4f}  test F1@30s {f:.3f}', flush=True); summarize(res, 'test')
        if f > best[0]: best = (f, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}); torch.save(best[1], a.save)
    print(f'best test F1@30s {best[0]:.3f}; saved {a.save}')
def fscore(res, tol):
    nt = sum(r['n_truth'] for r in res.values()); npd = sum(r['n_pred'] for r in res.values())
    R_ = sum(r['recall'][tol] for r in res.values()) / max(nt, 1); P_ = sum(r['precision'][tol] for r in res.values()) / max(npd, 1)
    return 2 * R_ * P_ / max(R_ + P_, 1e-9)
if __name__ == '__main__': main()
