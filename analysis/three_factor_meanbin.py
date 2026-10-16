"""Pooled predicted-vs-actual dW per condition in the three_factor_variance_explained_figure
convention (pf.mean_bin_plot): pooled held-out predictions z-scored, 5 equal-count
bins of sorted predictions, mean +/- SEM of actual dW (z) per bin, pooled Pearson r.
Learning - water r difference by pair bootstrap. Reads
three_factor_control_vs_learning<TAG>.npy; prints a JSON block."""
import json
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
COL = 5
N_BOOT = 1000
N_BOOT_SESS = 5000
tag = os.environ.get('TAG', '')
conds = os.environ.get('CONDS', 'control,learning_spont').split(',')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)


def mean_bin(xx, yy, col=COL):
    """Same as plotting_functions.mean_bin_plot (no plotting)."""
    b = np.argsort(xx)
    x, y = xx[b], yy[b]
    row = len(x) // col
    x = x[:row * col].reshape(col, row)
    y = y[:row * col].reshape(col, row)
    return x.mean(axis=1), y.mean(axis=1), y.std(axis=1) / np.sqrt(row)


out, pooled = {}, {}
for cond in conds:
    ss = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
    yp = np.concatenate([r['Y_pred_all'] for r in ss])
    yt = np.concatenate([r['Y_test_all'] for r in ss])
    xz = (yp - yp.mean()) / yp.std()
    X, Y, E = mean_bin(xz, yt)
    r, p = sps.pearsonr(yp, yt)
    pooled[cond] = (yp, yt)
    out[cond] = dict(bx=X.tolist(), by=Y.tolist(), be=E.tolist(), r=float(r), p=float(p),
                     n_pairs=int(len(yt)), n_sess=len(ss),
                     n_sig=int(sum(r_['p_test'] < 0.05 and r_['r_test'] > 0 for r_ in ss)))
    print(f'{cond:15s}: {len(ss)} sess, {len(yt)} pairs, pooled r={r:+.4f} p={p:.2g}, '
          f'sessions sig {out[cond]["n_sig"]}/{len(ss)}')

if len(conds) == 2:
    rng = np.random.default_rng(0)

    def boot_r(x, y):
        i = rng.integers(0, len(x), len(x))
        return np.corrcoef(x[i], y[i])[0, 1]

    (xc, yc), (xl, yl) = pooled[conds[0]], pooled[conds[1]]
    obs = np.corrcoef(xl, yl)[0, 1] - np.corrcoef(xc, yc)[0, 1]
    diffs = np.array([boot_r(xl, yl) - boot_r(xc, yc) for _ in range(N_BOOT)])
    ci = np.percentile(diffs, [2.5, 97.5])
    p = max(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1 / N_BOOT)
    out['boot'] = dict(diff=float(obs), ci=ci.tolist(), p=float(p))
    print(f'{conds[1]} - {conds[0]}: r diff={obs:+.4f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}], '
          f'pair bootstrap p={p:.3g}')

    # Session bootstrap: resample whole sessions with replacement within each
    # condition, recompute the pooled r from the resampled sessions' pairs
    def sums(r_):
        x, y = r_['Y_pred_all'], r_['Y_test_all']
        return np.array([len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])

    def pooled_r(S):
        n, sx, sy, sxy, sxx, syy = S.sum(axis=0)
        return (sxy - sx * sy / n) / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))

    S = {c: np.array([sums(r_) for r_ in res if r_['condition'] == c and r_['n_groups'] >= MIN_GROUPS])
         for c in conds}
    bs = {c: np.array([pooled_r(S[c][rng.integers(0, len(S[c]), len(S[c]))])
                       for _ in range(N_BOOT_SESS)]) for c in conds}
    d_s = bs[conds[1]] - bs[conds[0]]
    ci_s = np.percentile(d_s, [2.5, 97.5])
    p_s = max(2 * min(np.mean(d_s <= 0), np.mean(d_s >= 0)), 1 / N_BOOT_SESS)
    out['boot_session'] = dict(diff=float(obs), ci=ci_s.tolist(), p=float(p_s),
                               ci_by_cond={c: np.percentile(bs[c], [2.5, 97.5]).tolist() for c in conds},
                               p_gt0_by_cond={c: float(max(2 * np.mean(bs[c] <= 0), 1 / N_BOOT_SESS))
                                              for c in conds})
    for c in conds:
        print(f'  session bootstrap {c:15s}: r 95% CI [{np.percentile(bs[c], 2.5):+.4f}, '
              f'{np.percentile(bs[c], 97.5):+.4f}], fraction r<=0 = {np.mean(bs[c] <= 0):.3f}')
    print(f'{conds[1]} - {conds[0]}: r diff={obs:+.4f}, session bootstrap 95% CI '
          f'[{ci_s[0]:+.4f}, {ci_s[1]:+.4f}], p={p_s:.3g}')
print('JSON' + json.dumps(out))
