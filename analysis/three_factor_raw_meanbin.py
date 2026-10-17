"""Raw-unit binned plot data: dW~ (paper units) vs coactivity (F/std-1 units), no
z-scoring, per condition (mean_bin_plot convention: 5 equal-count bins). x =
mean coactivity per pseudo-trial, and x = coactivity change (post - pre), each
centred within session (session offsets removed). Prints JSON.
Reads three_factor_control_vs_learning<TAG>.npy (MAG=1)."""
import json
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
COL = 5
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and r.get('mag') is not None]


def mean_bin(xx, yy, col=COL):
    b = np.argsort(xx)
    x, y = xx[b], yy[b]
    row = len(x) // col
    x = x[:row * col].reshape(col, row)
    y = y[:row * col].reshape(col, row)
    return x.mean(axis=1).tolist(), y.mean(axis=1).tolist(), (y.std(axis=1) / np.sqrt(row)).tolist()


out = {}
for cond in ['control', 'learning_spont']:
    ss = [r for r in res if r['condition'] == cond]
    for xname in ['cc_mean', 'dcc']:
        xs, ys = [], []
        for r in ss:
            m = r['mag']
            if xname == 'dcc':
                if 'cc_pre' not in m:
                    continue
                x = (m['cc_post'] - m['cc_pre']).astype(float)
            else:
                x = m['cc_mean'].astype(float)
            y = m['dw'].astype(float)
            xs.append(x - x.mean())
            ys.append(y - y.mean())
        x, y = np.concatenate(xs), np.concatenate(ys)
        bx, by, be = mean_bin(x, y)
        rr, pp = sps.pearsonr(x, y)
        out[f'{cond}/{xname}'] = dict(bx=bx, by=by, be=be, r=float(rr), p=float(pp), n=int(len(x)),
                                      sd_x=float(x.std()), sd_y=float(y.std()),
                                      slope=float(np.sum(x * y) / np.sum(x * x)))
        print(f'{cond:15s} {xname:8s} n={len(x):7d} sd_x={x.std():.4f} sd_y={y.std():.4f} '
              f'slope={out[f"{cond}/{xname}"]["slope"]:+.4f} r={rr:+.4f} p={pp:.2g}')
print('JSON' + json.dumps(out))
