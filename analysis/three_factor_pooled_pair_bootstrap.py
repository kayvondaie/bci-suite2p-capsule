"""Pair-level bootstrap for the pooled 3-factor comparison.

For each condition, pool held-out (Y_pred, Y_test) pairs from all sessions,
resample pairs with replacement N_BOOT times, and recompute pooled r.
Reports 95% CI per condition, CI of learning - control, and a two-sided
bootstrap p for the difference. Predictions use the same rank-based normal
scores (within session) as three_factor_control_vs_learning_figure.py.
Reads three_factor_control_vs_learning.npy.
"""
import numpy as np
from scipy.stats import norm, rankdata

RESULTS = '/root/capsule/results/three_factor_control_vs_learning.npy'
MIN_GROUPS = 10
N_BOOT = 2000

res = [r for r in np.load(RESULTS, allow_pickle=True).tolist()
       if r['n_groups'] >= MIN_GROUPS]


def rank_z(x):
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def pooled(cond):
    rows = [r for r in res if r['condition'] == cond]
    x = np.concatenate([rank_z(r['Y_pred_all']) for r in rows])
    y = np.concatenate([r['Y_test_all'] for r in rows])
    return x, y, len(rows)


def corr(x, y):
    x = x - x.mean()
    y = y - y.mean()
    return (x @ y) / np.sqrt((x @ x) * (y @ y))


def boot_r(x, y, rng):
    out = np.empty(N_BOOT)
    n = len(x)
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        out[b] = corr(x[idx], y[idx])
    return out


rng = np.random.default_rng(0)
xc, yc, nc = pooled('control')
bc = boot_r(xc, yc, rng)
print(f"control : {nc} sessions, {len(xc)} pairs, r = {corr(xc, yc):+.4f} "
      f"(95% CI {np.percentile(bc, 2.5):+.4f} to {np.percentile(bc, 97.5):+.4f})")

for lv in ['learning', 'learning_pseudo']:
    xl, yl, nl = pooled(lv)
    bl = boot_r(xl, yl, rng)
    d = bl - bc
    p = min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))
    print(f"\n{lv}: {nl} sessions, {len(xl)} pairs, r = {corr(xl, yl):+.4f} "
          f"(95% CI {np.percentile(bl, 2.5):+.4f} to {np.percentile(bl, 97.5):+.4f})")
    print(f"  {lv} - control = {corr(xl, yl) - corr(xc, yc):+.4f} "
          f"(95% CI {np.percentile(d, 2.5):+.4f} to {np.percentile(d, 97.5):+.4f}), "
          f"pair-bootstrap p = {max(p, 1 / N_BOOT):.4f}"
          f"{' (< 1/N_BOOT)' if p == 0 else ''}")
