"""Does spont_post coactivity predict dW better than spont_pre coactivity?
Per condition, pooled over sessions (three_factor_variance_explained_figure convention):
  model : held-out predictions of the 3-factor fit using only spont_pre windows vs
          only spont_post windows (DIAG 'pre' / 'post' variants)
  coact : single regressor, mean coactivity over spont_pre vs spont_post pseudo-
          trials (MAG cc_pre / cc_post), z-scored within session, vs dW z-scored
          within session (no fitting)
Pooled Pearson r, post - pre by pair bootstrap (same pairs resampled for pre and
post), and post - pre difference between conditions. Prints JSON incl. mean-bin
curves. Reads three_factor_control_vs_learning<TAG>.npy (DIAG=1, MAG=1)."""
import json
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
N_BOOT = 1000
COL = 5
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and r.get('diag') is not None
       and r.get('mag') is not None and 'cc_pre' in r['mag']
       and 'pre' in r['diag'] and 'post' in r['diag']]
rng = np.random.default_rng(0)


def z(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def mean_bin(xx, yy):
    xx = (xx - xx.mean()) / xx.std()
    b = np.argsort(xx)
    x, y = xx[b], yy[b]
    row = len(x) // COL
    x, y = x[:row * COL].reshape(COL, row), y[:row * COL].reshape(COL, row)
    return x.mean(1).tolist(), y.mean(1).tolist(), (y.std(1) / np.sqrt(row)).tolist()


out, data = {}, {}
for cond in ['control', 'learning_spont']:
    ss = [r for r in res if r['condition'] == cond]
    d = {}
    # model variants: pre and post held-out predictions are on the same pairs/order
    for ep in ['pre', 'post']:
        d[f'model_{ep}'] = (np.concatenate([r['diag'][ep]['Y_pred'] for r in ss]),
                            np.concatenate([r['diag'][ep]['Y_test'] for r in ss]))
    ycoact = np.concatenate([z(r['mag']['dw'].astype(float)) for r in ss])
    for ep in ['pre', 'post']:
        d[f'coact_{ep}'] = (np.concatenate([z(r['mag'][f'cc_{ep}'].astype(float)) for r in ss]), ycoact)
    data[cond] = d
    print(f'\n### {cond}: {len(ss)} sessions')
    for k, (x, y) in d.items():
        rr, pp = sps.pearsonr(x, y)
        bx, by, be = mean_bin(x, y)
        out[f'{cond}/{k}'] = dict(r=float(rr), p=float(pp), n=int(len(x)), n_sess=len(ss), bx=bx, by=by, be=be)
        print(f'  {k:11s} r={rr:+.4f} p={pp:.2g} (n={len(x)})')
    for kind in ['model', 'coact']:
        xa, ya = d[f'{kind}_pre']
        xb, yb = d[f'{kind}_post']
        n = len(ya)
        obs = np.corrcoef(xb, yb)[0, 1] - np.corrcoef(xa, ya)[0, 1]
        diffs = []
        for _ in range(N_BOOT):
            i = rng.integers(0, n, n)
            diffs.append(np.corrcoef(xb[i], yb[i])[0, 1] - np.corrcoef(xa[i], ya[i])[0, 1])
        diffs = np.array(diffs)
        ci = np.percentile(diffs, [2.5, 97.5])
        p = max(2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)), 1 / N_BOOT)
        out[f'{cond}/{kind}_post_minus_pre'] = dict(diff=float(obs), ci=ci.tolist(), p=float(p), boot=diffs.tolist())
        print(f'  {kind}: post - pre = {obs:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], pair bootstrap p={p:.3g}')

print('\n### learning - water, (post - pre)')
for kind in ['model', 'coact']:
    a = np.array(out[f'control/{kind}_post_minus_pre']['boot'])
    b = np.array(out[f'learning_spont/{kind}_post_minus_pre']['boot'])
    obs = out[f'learning_spont/{kind}_post_minus_pre']['diff'] - out[f'control/{kind}_post_minus_pre']['diff']
    dd = b - a
    ci = np.percentile(dd, [2.5, 97.5])
    p = max(2 * min(np.mean(dd <= 0), np.mean(dd >= 0)), 1 / N_BOOT)
    out[f'diff_of_diff/{kind}'] = dict(diff=float(obs), ci=ci.tolist(), p=float(p))
    print(f'  {kind}: {obs:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}], p={p:.3g}')
for k in list(out):
    out[k].pop('boot', None)
print('JSON' + json.dumps(out))
