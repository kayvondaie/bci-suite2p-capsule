"""3-factor MLR, water control vs learning: pooled held-out predictions per
condition (all mice), pooled r with its Pearson p, and the condition
difference from one pooled regression with a condition x prediction
interaction (Wald p) plus a Fisher z-test on the two pooled r's. No resampling.
Reads three_factor_control_vs_learning.npy (outliers already excluded)."""
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
res = np.load('/root/capsule/results/three_factor_control_vs_learning.npy', allow_pickle=True)
qc = [r for r in res if r['n_groups'] >= MIN_GROUPS]


def pooled(cond):
    ss = [r for r in qc if r['condition'] == cond]
    x = np.concatenate([r['Y_pred_all'] for r in ss])
    y = np.concatenate([r['Y_test_all'] for r in ss])
    return ss, x, y


def ols(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    cov = (resid @ resid / dof) * np.linalg.inv(A.T @ A)
    t = beta / np.sqrt(np.diag(cov))
    return beta, 2 * sps.t.sf(np.abs(t), dof)


ss_c, xc, yc = pooled('control')
r_c, p_c = sps.pearsonr(xc, yc)
print(f"control        : {len(ss_c):2d} sess, {len(xc):6d} pairs, pooled r={r_c:+.4f} p={p_c:.3g}, "
      f"sessions p<.05 & r>0: {sum(r['p_test'] < 0.05 and r['r_test'] > 0 for r in ss_c)}/{len(ss_c)}")
for lv in ['learning', 'learning_pseudo']:
    ss_l, xl, yl = pooled(lv)
    r_l, p_l = sps.pearsonr(xl, yl)
    x = np.concatenate([xc, xl])
    y = np.concatenate([yc, yl])
    L = np.concatenate([np.zeros(len(xc)), np.ones(len(xl))])
    beta, p = ols(y, np.column_stack([x, L, L * x]))
    z = (np.arctanh(r_l) - np.arctanh(r_c)) / np.sqrt(1 / (len(xl) - 3) + 1 / (len(xc) - 3))
    print(f"{lv:15s}: {len(ss_l):2d} sess, {len(xl):6d} pairs, pooled r={r_l:+.4f} p={p_l:.3g}, "
          f"sessions p<.05 & r>0: {sum(r['p_test'] < 0.05 and r['r_test'] > 0 for r in ss_l)}/{len(ss_l)}")
    print(f"   {lv} - control: interaction slope={beta[3]:+.4f}, Wald p={p[3]:.3g}; "
          f"Fisher z={z:+.2f}, p={2 * sps.norm.sf(abs(z)):.3g}")
