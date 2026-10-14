"""Binned Fig. 4h curve (dA_rho vs dW, target dW regressed out), all mice pooled,
per condition, from fig4_dcorr_dw_control_vs_learning.npy. Prints JSON for plotting.
Putative SST outliers were already excluded when the results were built."""
import json
import numpy as np
from scipy import stats as sps

N_BINS = 8
MIN_GROUPS = 10
res = np.load('/root/capsule/results/fig4_dcorr_dw_control_vs_learning.npy', allow_pickle=True)
out = {}
for cond in ['control', 'learning']:
    ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
    y = np.concatenate([r['dW']['y'] for r in ss])
    x = np.concatenate([r['dW']['x'] for r in ss])
    c = np.concatenate([r['dW']['c'] for r in ss])
    A = np.column_stack([np.ones_like(x), x, c])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    cov = (resid @ resid / (len(y) - 3)) * np.linalg.inv(A.T @ A)
    t = beta[1] / np.sqrt(cov[1, 1])
    p = 2 * sps.t.sf(abs(t), len(y) - 3)
    B = np.column_stack([np.ones_like(c), c])
    yr = y - B @ np.linalg.lstsq(B, y, rcond=None)[0]
    xr = x - B @ np.linalg.lstsq(B, x, rcond=None)[0]
    edges = np.percentile(xr, np.linspace(0, 100, N_BINS + 1))
    idx = np.clip(np.digitize(xr, edges[1:-1]), 0, N_BINS - 1)
    out[cond] = dict(
        bx=[float(xr[idx == b].mean()) for b in range(N_BINS)],
        by=[float(yr[idx == b].mean()) for b in range(N_BINS)],
        be=[float(yr[idx == b].std() / np.sqrt(np.sum(idx == b))) for b in range(N_BINS)],
        beta=float(beta[1]), p=float(p), n_pairs=int(len(y)), n_sess=len(ss),
        n_mice=len({r['mouse'] for r in ss}))
print('JSON' + json.dumps(out))
