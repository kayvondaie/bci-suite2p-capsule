"""Pooled Pearson vs Spearman of held-out predicted vs actual dW per condition,
from three_factor_control_vs_learning<TAG>.npy (TAG env var, e.g. _test100).
Spearman on raw pooled predictions and on within-session ranks."""
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
print(f'file tag: {tag or "(none)"}')
for cond in ['control', 'learning', 'learning_pseudo']:
    ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
    if not ss:
        continue
    x = np.concatenate([r['Y_pred_all'] for r in ss])
    y = np.concatenate([r['Y_test_all'] for r in ss])
    xr = np.concatenate([sps.rankdata(r['Y_pred_all']) / len(r['Y_pred_all']) for r in ss])
    yr = np.concatenate([sps.rankdata(r['Y_test_all']) / len(r['Y_test_all']) for r in ss])
    pr, pp = sps.pearsonr(x, y)
    sr, sp = sps.spearmanr(x, y)
    wr, wp = sps.pearsonr(xr, yr)
    print(f'{cond:16s} {len(ss):2d} sess {len(x):7d} pairs | Pearson r={pr:+.4f} p={pp:.2g} | '
          f'Spearman rho={sr:+.4f} p={sp:.2g} | within-session-rank rho={wr:+.4f} p={wp:.2g}')
