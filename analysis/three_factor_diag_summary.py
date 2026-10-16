"""Summaries of the DIAG=1 3-factor fits (spont vs spont): pooled within-session
rank correlation for the main model and each variant, per-session and per-mouse
breakdown, Hebbian (sum) slope signs, mean window weights by epoch. Reads
three_factor_control_vs_learning<TAG>.npy. Prints tables and a JSON block."""
import json
import os
import numpy as np
from scipy import stats as sps

MIN_GROUPS = 10
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and r.get('diag') is not None]
CONDS = ['control', 'learning_spont']
INDICATOR = {'BCI103': '6s'}


def ranks(v):
    return (sps.rankdata(v) - 0.5) / len(v)


def ws_rho(pairs):
    """Pooled within-session rank correlation; pairs = list of (pred, test)."""
    if not pairs:
        return np.nan, np.nan, 0
    x = np.concatenate([ranks(p) for p, t in pairs])
    y = np.concatenate([ranks(t) for p, t in pairs])
    r, p = sps.pearsonr(x, y)
    return r, p, len(x)


def get(r, variant):
    if variant == 'main':
        return r['Y_pred_all'], r['Y_test_all']
    d = r['diag'].get(variant)
    return (d['Y_pred'], d['Y_test']) if d is not None else None


out = {}
print(f'file tag: {tag}')
print('\n=== pooled within-session rank rho, by variant ===')
for variant in ['main', 'group', 'sum', 'pre', 'post']:
    line = []
    for cond in CONDS:
        pr = [get(r, variant) for r in res if r['condition'] == cond]
        pr = [x for x in pr if x is not None]
        rho, p, n = ws_rho(pr)
        out[f'{variant}/{cond}'] = dict(rho=rho, p=p, n=n, n_sess=len(pr))
        line.append(f'{cond}: rho={rho:+.4f} p={p:.2g} ({len(pr)} sess, {n} pairs)')
    a, b = out[f'{variant}/control'], out[f'{variant}/learning_spont']
    if a['n'] and b['n']:
        z = (np.arctanh(b['rho']) - np.arctanh(a['rho'])) / np.sqrt(1 / (a['n'] - 3) + 1 / (b['n'] - 3))
        line.append(f'diff z={z:+.2f} p={2 * sps.norm.sf(abs(z)):.2g}')
    print(f'  {variant:6s} | ' + ' | '.join(line))

print('\n=== per session (main model, within-session rank rho) ===')
sess_rows = []
for cond in CONDS:
    for r in res:
        if r['condition'] != cond:
            continue
        rho, p, n = ws_rho([get(r, 'main')])
        rs, ps_, _ = ws_rho([get(r, 'sum')])
        sess_rows.append(dict(cond=cond, mouse=r['mouse'], session=r['session'], rho=rho, p=p, n=n,
                              rho_sum=rs, slope=r['diag']['sum_slope']))
        print(f"  {cond:15s} {r['mouse']:7s} {r['session']} rho={rho:+.4f} p={p:.2g} n={n:6d} | "
              f"sum-model rho={rs:+.4f} | Hebbian slope r={r['diag']['sum_slope']:+.4f}")
for cond in CONDS:
    v = [x['rho'] for x in sess_rows if x['cond'] == cond]
    sl = [x['slope'] for x in sess_rows if x['cond'] == cond]
    print(f"  {cond}: sessions rho>0 {sum(np.array(v) > 0)}/{len(v)}, median {np.median(v):+.4f}; "
          f"Hebbian slope >0 {sum(np.array(sl) > 0)}/{len(sl)}, median {np.median(sl):+.4f}")

print('\n=== per mouse (main model, pooled within-session rank rho) ===')
for cond in CONDS:
    for m in sorted({r['mouse'] for r in res if r['condition'] == cond}):
        pr = [get(r, 'main') for r in res if r['condition'] == cond and r['mouse'] == m]
        rho, p, n = ws_rho(pr)
        out[f'mouse/{cond}/{m}'] = dict(rho=rho, p=p, n=n, n_sess=len(pr))
        print(f"  {cond:15s} {m:7s} ({INDICATOR.get(m, '8s')}) rho={rho:+.4f} p={p:.2g} "
              f"({len(pr)} sess, {n} pairs)")

print('\n=== mean full-model window weight by epoch ===')
for cond in CONDS:
    for ep in ['pre', 'post', 'mixed']:
        b = np.concatenate([r['diag']['beta'][r['diag']['win_epoch'] == ep]
                            for r in res if r['condition'] == cond])
        if b.size:
            print(f"  {cond:15s} {ep:5s}: mean beta={b.mean():+.4f} ± {b.std(ddof=1) / np.sqrt(b.size):.4f} "
                  f"(>0 in {np.mean(b > 0):.0%} of {b.size} windows)")
out['sessions'] = sess_rows
print('JSON' + json.dumps(out, default=float))
