"""Fitted 3-factor window weights (beta, mean over CV folds) per session, aligned
to window centre relative to the spont_pre / spont_post boundary (minutes), per
condition: mean +/- SEM across sessions, plus per-session values. Reads
three_factor_control_vs_learning<TAG>.npy (DIAG=1). Prints JSON."""
import json
import os
import numpy as np

MIN_GROUPS = 10
PSEUDO_SEC = 10
tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
res = [r for r in res if r['n_groups'] >= MIN_GROUPS and r.get('diag') is not None
       and r['diag'].get('n_pre') is not None]
out = {}
for cond in ['control', 'learning_spont']:
    rows = {}
    for r in res:
        if r['condition'] != cond:
            continue
        d = r['diag']
        t = np.round((d['win_starts'] + d['win_size'] / 2 - d['n_pre']) * PSEUDO_SEC / 60, 2)
        for ti, b in zip(t, d['beta']):
            rows.setdefault(float(ti), []).append(float(b))
    ts = sorted(rows)
    out[cond] = dict(t=ts, mean=[float(np.mean(rows[t])) for t in ts],
                     sem=[float(np.std(rows[t], ddof=1) / np.sqrt(len(rows[t]))) if len(rows[t]) > 1 else 0.0
                          for t in ts],
                     n=[len(rows[t]) for t in ts], values={str(t): rows[t] for t in ts})
    print(f'{cond}:')
    for t in ts:
        v = np.array(rows[t])
        print(f'  t={t:+.2f} min  beta mean={v.mean():+.4f} ± {v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0:.4f}'
              f'  (>0 in {np.sum(v > 0)}/{len(v)})')
print('JSON' + json.dumps(out))
