"""Binned 3-factor curve (held-out predicted vs actual dW), all mice pooled,
per condition, from three_factor_control_vs_learning.npy. Predictions are
rank-normal scored within each session (so sessions with small prediction
scale aren't compressed), then binned by pooled quantile; mean ± SEM over pairs.
Prints JSON for plotting."""
import json
import os
import sys
import numpy as np
from scipy import stats as sps

N_BINS = 6
MIN_GROUPS = 10
conds = sys.argv[1:] or ['control', 'learning_pseudo']
res = np.load('/root/capsule/results/three_factor_control_vs_learning'
              + os.environ.get('TAG', '') + '.npy', allow_pickle=True)
out = {}
for cond in conds:
    ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
    x = np.concatenate([sps.norm.ppf((sps.rankdata(r['Y_pred_all']) - 0.5) / len(r['Y_pred_all']))
                        for r in ss])
    y = np.concatenate([r['Y_test_all'] for r in ss])
    xr = np.concatenate([r['Y_pred_all'] for r in ss])
    r_p, p_p = sps.pearsonr(xr, y)
    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, N_BINS - 1)
    out[cond] = dict(bx=[float(x[idx == b].mean()) for b in range(N_BINS)],
                     by=[float(y[idx == b].mean()) for b in range(N_BINS)],
                     be=[float(y[idx == b].std() / np.sqrt(np.sum(idx == b))) for b in range(N_BINS)],
                     r=float(r_p), p=float(p_p), n_pairs=int(len(y)), n_sess=len(ss),
                     n_sig=int(sum(r['p_test'] < 0.05 and r['r_test'] > 0 for r in ss)))
print('JSON' + json.dumps(out))
