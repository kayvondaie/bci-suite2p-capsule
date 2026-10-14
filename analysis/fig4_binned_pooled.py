"""Binned Fig. 4h curve (dA vs dW, target dW regressed out), all mice pooled,
per condition. Prints JSON for plotting. Putative SST outliers were already
excluded when the results were built.
  python fig4_binned_pooled.py            -> fig4_dcorr_dw_control_vs_learning.npy (paper dA_rho)
  python fig4_binned_pooled.py <form> ... -> fig4_dcorr_variants.npy, cont_halves, given forms
                                             (paper, dC, CdW, dCu)"""
import json
import sys
import numpy as np
from scipy import stats as sps

N_BINS = 8
MIN_GROUPS = 10
R = '/root/capsule/results/'


def binned(y, x, c):
    A = np.column_stack([np.ones_like(x), x, c])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    cov = (resid @ resid / (len(y) - 3)) * np.linalg.inv(A.T @ A)
    p = 2 * sps.t.sf(abs(beta[1] / np.sqrt(cov[1, 1])), len(y) - 3)
    B = np.column_stack([np.ones_like(c), c])
    yr = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    xr = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    edges = np.percentile(xr, np.linspace(0, 100, N_BINS + 1))
    idx = np.clip(np.digitize(xr, edges[1:-1]), 0, N_BINS - 1)
    return dict(bx=[float(xr[idx == b].mean()) for b in range(N_BINS)],
                by=[float(yr[idx == b].mean()) for b in range(N_BINS)],
                be=[float(yr[idx == b].std() / np.sqrt(np.sum(idx == b))) for b in range(N_BINS)],
                beta=float(beta[1]), p=float(p), n_pairs=int(len(y)))


out = {}
if len(sys.argv) == 1:
    res = np.load(R + 'fig4_dcorr_dw_control_vs_learning.npy', allow_pickle=True)
    for cond in ['control', 'learning']:
        ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
        cat = lambda k: np.concatenate([r['dW'][k] for r in ss])
        out[cond] = dict(binned(cat('y'), cat('x'), cat('c')), n_sess=len(ss))
else:
    res = np.load(R + 'fig4_dcorr_variants.npy', allow_pickle=True)
    for form in sys.argv[1:]:
        k = ('cont_halves', form)
        for cond in ['control', 'learning']:
            ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS
                  and k in r.get('raw', {})]
            cat = lambda f: np.concatenate([r['raw'][k][f] for r in ss])
            out[f'{form}/{cond}'] = dict(binned(cat('y'), cat('x'), cat('c')), n_sess=len(ss))
print('JSON' + json.dumps(out))
