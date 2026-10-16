"""3-factor held-out predictions split by non-target distance from the stim site.
Within-session percentiles are computed over all of a session's held-out pairs,
then pairs are split into distance bands. Per band and condition: binned
predicted-vs-actual curve, pooled within-session rank rho, and mean |dW| (raw
held-out dW, z-scored by the training folds). Reads
three_factor_control_vs_learning<TAG>.npy; prints tables and a JSON block.
Learning vs water: pair bootstrap (pairs resampled within condition)."""
import json
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
N_BINS = 6
BANDS = [(30, 100), (100, 200), (200, 400), (400, 1000)]
N_BOOT = 1000
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and 'dist_test_all' in r]
CONDS = sorted({r['condition'] for r in res}, key=lambda c: c != 'control')


def ranks(v):
    return (sps.rankdata(v) - 0.5) / len(v)


out = {}
PAIRS = {}
print(f'file tag: {tag}')
for cond in CONDS:
    ss = [r for r in res if r['condition'] == cond]
    x = np.concatenate([ranks(r['Y_pred_all']) for r in ss])
    y = np.concatenate([ranks(r['Y_test_all']) for r in ss])
    a = np.concatenate([np.abs(r['Y_test_all']) for r in ss])
    d = np.concatenate([r['dist_test_all'] for r in ss])
    for lo, hi in BANDS:
        m = (d >= lo) & (d < hi)
        if m.sum() < 100:
            continue
        rho, p = sps.pearsonr(x[m], y[m])
        idx = np.clip((x[m] * N_BINS).astype(int), 0, N_BINS - 1)
        xm, ym = x[m], y[m]
        key = f'{cond}/{lo}-{hi}'
        out[key] = dict(rho=rho, p=p, n=int(m.sum()), n_sess=len(ss),
                        mean_abs_dw=float(a[m].mean()), se_abs_dw=float(a[m].std() / np.sqrt(m.sum())),
                        bx=[float(xm[idx == b].mean()) for b in range(N_BINS)],
                        by=[float(ym[idx == b].mean()) for b in range(N_BINS)],
                        be=[float(ym[idx == b].std() / np.sqrt(np.sum(idx == b))) for b in range(N_BINS)])
        print(f'  {cond:15s} {lo:4d}-{hi:<4d} um: rho={rho:+.4f} p={p:.2g} n={m.sum():7d} | '
              f'mean |dW| = {a[m].mean():.3f} ± {a[m].std() / np.sqrt(m.sum()):.3f}')
    PAIRS[cond] = (x, y, d)


def boot_rho(x, y, rng):
    i = rng.integers(0, len(x), len(x))
    return np.corrcoef(x[i], y[i])[0, 1]


# Pair bootstrap of learning - water: resample pairs with replacement within each
# condition (within-session percentiles fixed), recompute rho, take the difference
print(f'\n=== pair bootstrap, learning - water (N_BOOT={N_BOOT}) ===')
rng = np.random.default_rng(0)
for lo, hi in [(30, 1000)] + BANDS:
    if 'control' not in PAIRS or 'learning_spont' not in PAIRS:
        break
    sel = {c: (PAIRS[c][2] >= lo) & (PAIRS[c][2] < hi) for c in ['control', 'learning_spont']}
    xc, yc = PAIRS['control'][0][sel['control']], PAIRS['control'][1][sel['control']]
    xl, yl = PAIRS['learning_spont'][0][sel['learning_spont']], PAIRS['learning_spont'][1][sel['learning_spont']]
    obs = np.corrcoef(xl, yl)[0, 1] - np.corrcoef(xc, yc)[0, 1]
    diffs = np.array([boot_rho(xl, yl, rng) - boot_rho(xc, yc, rng) for _ in range(N_BOOT)])
    ci = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    out[f'boot/{lo}-{hi}'] = dict(diff=float(obs), ci=[float(ci[0]), float(ci[1])], p=float(max(p, 1 / N_BOOT)))
    print(f'  {lo:4d}-{hi:<4d} um: rho(learning) - rho(water) = {obs:+.4f}, 95% CI '
          f'[{ci[0]:+.4f}, {ci[1]:+.4f}], p = {max(p, 1 / N_BOOT):.3g}')
print('JSON' + json.dumps(out))
