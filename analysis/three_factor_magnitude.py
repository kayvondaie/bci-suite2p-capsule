"""Magnitude comparison, water vs learning (spont vs spont), in units comparable
across sessions (no z-scoring):
  dW        paper-1 Delta PS response (noise-normalised, unitless)
  cc_mean   mean coactivity per 10 s pseudo-trial, dF/F = F/std(F) - 1 units
  dcc       cc_post - cc_pre (change in coactivity from spont_pre to spont_post)
  gain      pooled slope of dW on cc_mean (and on dcc), both centred within
            session (= change in dW per unit coactivity)
Reported for all mice, GCaMP8s mice only, and BCI103 (GCaMP6s). Learning -
water by pair bootstrap. Reads three_factor_control_vs_learning<TAG>.npy (MAG=1)."""
import json
import os
import numpy as np

MIN_GROUPS = 10
N_BOOT = 1000
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and r.get('mag') is not None]
CONDS = ['control', 'learning_spont']
GROUPS = {'all': lambda m: True, '8s': lambda m: m != 'BCI103', '6s (BCI103)': lambda m: m == 'BCI103'}
rng = np.random.default_rng(0)


def gather(cond, gsel):
    ss = [r for r in res if r['condition'] == cond and gsel(r['mouse'])]
    if not ss:
        return None
    g = {}
    for k in ['dw', 'cc_mean']:
        g[k] = np.concatenate([r['mag'][k].astype(float) for r in ss])
        g[k + '_c'] = np.concatenate([r['mag'][k] - r['mag'][k].mean() for r in ss]).astype(float)
    ok = [r for r in ss if 'cc_pre' in r['mag']]
    g['dcc'] = np.concatenate([(r['mag']['cc_post'] - r['mag']['cc_pre']).astype(float) for r in ok]) if ok else None
    if ok:
        g['dw_ok_c'] = np.concatenate([r['mag']['dw'] - r['mag']['dw'].mean() for r in ok]).astype(float)
        d = [(r['mag']['cc_post'] - r['mag']['cc_pre']).astype(float) for r in ok]
        g['dcc_c'] = np.concatenate([x - x.mean() for x in d])
    g['sess'] = [(r['mouse'], r['session'], float(np.std(r['mag']['dw'])), float(np.std(r['mag']['cc_mean'])),
                  float(np.polyfit(r['mag']['cc_mean'] - r['mag']['cc_mean'].mean(),
                                   r['mag']['dw'] - r['mag']['dw'].mean(), 1)[0])) for r in ss]
    g['n_sess'] = len(ss)
    return g


def slope(x, y):
    return float(np.sum(x * y) / np.sum(x * x))


STATS = {
    'SD of dW': lambda g: (float(np.std(g['dw'])), lambda i: np.std(g['dw'][i]), len(g['dw'])),
    'mean |dW|': lambda g: (float(np.mean(np.abs(g['dw']))), lambda i: np.mean(np.abs(g['dw'][i])), len(g['dw'])),
    'SD of coactivity': lambda g: (float(np.std(g['cc_mean'])), lambda i: np.std(g['cc_mean'][i]), len(g['cc_mean'])),
    'mean coactivity': lambda g: (float(np.mean(g['cc_mean'])), lambda i: np.mean(g['cc_mean'][i]), len(g['cc_mean'])),
    'SD of d(coactivity) post-pre': lambda g: ((float(np.std(g['dcc'])), lambda i: np.std(g['dcc'][i]), len(g['dcc']))
                                              if g['dcc'] is not None else None),
    'gain: dW per unit coactivity': lambda g: (slope(g['cc_mean_c'], g['dw_c']),
                                               lambda i: slope(g['cc_mean_c'][i], g['dw_c'][i]), len(g['dw_c'])),
    'gain: dW per unit d(coactivity)': lambda g: ((slope(g['dcc_c'], g['dw_ok_c']),
                                                   lambda i: slope(g['dcc_c'][i], g['dw_ok_c'][i]), len(g['dcc_c']))
                                                  if g['dcc'] is not None else None),
}

out = {}
print(f'file tag: {tag}')
for gname, gsel in GROUPS.items():
    G = {c: gather(c, gsel) for c in CONDS}
    if any(v is None for v in G.values()):
        continue
    print(f"\n### {gname}: water {G['control']['n_sess']} sess, learning {G['learning_spont']['n_sess']} sess")
    for sname, fn in STATS.items():
        a, b = fn(G['control']), fn(G['learning_spont'])
        if a is None or b is None:
            continue
        (va, fa, na), (vb, fb, nb) = a, b
        bc = np.array([fa(rng.integers(0, na, na)) for _ in range(N_BOOT)])
        bl = np.array([fb(rng.integers(0, nb, nb)) for _ in range(N_BOOT)])
        d = bl - bc
        ci = np.percentile(d, [2.5, 97.5])
        p = max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / N_BOOT)
        out[f'{gname}/{sname}'] = dict(water=va, learning=vb, diff=vb - va, ci=ci.tolist(), p=p,
                                       water_ci=np.percentile(bc, [2.5, 97.5]).tolist(),
                                       learning_ci=np.percentile(bl, [2.5, 97.5]).tolist())
        print(f'  {sname:32s} water={va:+.5f}  learning={vb:+.5f}  diff={vb - va:+.5f} '
              f'[{ci[0]:+.5f}, {ci[1]:+.5f}]  p={p:.3g}')
    for c in CONDS:
        sl = np.array([x[4] for x in G[c]['sess']])
        print(f'  {c:15s} per-session gain >0: {np.sum(sl > 0)}/{len(sl)}, median {np.median(sl):+.4f}; '
              f'per-session SD(dW) median {np.median([x[2] for x in G[c]["sess"]]):.3f}')
    out[f'{gname}/sessions'] = {c: G[c]['sess'] for c in CONDS}
print('JSON' + json.dumps(out))
