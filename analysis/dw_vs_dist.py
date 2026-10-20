"""dW and |dW| vs distance, learning vs water, per indicator group.
dW = AMP1 - AMP0 (raw dF/F) for ordinary non-target pairs (30-1000 um from the
group's targets; not SST outlier, not garbage; groups with >= 1 responsive
target). No spont needed, so every session with photostim maps is used.
Per session and distance bin: mean dW, mean |dW|, mean W0, mean |W0|, n.
Summary per group x condition x bin: session mean +- SEM (sessions as unit)
and pooled pair mean +- SEM. Writes /root/capsule/results/dw_vs_dist.json."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'session_offsets.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('rows = []')])

EDGES = [30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]
NB = len(EDGES) - 1
recs = []
jobs = [(m, s, 'learning') for m, s in LEARNING_SESSIONS if (m, s) not in EXCLUDE] + \
       [(m, s, 'water') for m, s in CONTROL_SESSIONS]
for mouse, sess, cond in jobs:
    try:
        folder, AMP, D = load_amp(mouse, sess, cond)
    except Exception as e:
        print(f'{cond} {mouse} {sess}: skipped ({e})'); continue
    n, G = D.shape
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    bad = garbage(folder, AMP, D, n)
    dW, W0, dist = [], [], []
    for g in range(G):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out & ~bad)[0]
        if cl.size == 0:
            continue
        i = np.where((D[:, g] > dist_nontarg_min) & (D[:, g] < dist_nontarg_max) & ~is_out & ~bad &
                     np.isfinite(AMP[0][:, g]) & np.isfinite(AMP[1][:, g]))[0]
        dW += list(AMP[1][i, g] - AMP[0][i, g]); W0 += list(AMP[0][i, g]); dist += list(D[i, g])
    if not dW:
        continue
    dW, W0, dist = map(np.asarray, (dW, W0, dist))
    b = np.digitize(dist, EDGES) - 1
    rec = dict(mouse=mouse, session=sess, cond=cond, group=GROUP.get(mouse, 'ribo8'),
               dW=[float(np.mean(dW[b == k])) if np.any(b == k) else None for k in range(NB)],
               adW=[float(np.mean(np.abs(dW[b == k]))) if np.any(b == k) else None for k in range(NB)],
               W0=[float(np.mean(W0[b == k])) if np.any(b == k) else None for k in range(NB)],
               aW0=[float(np.mean(np.abs(W0[b == k]))) if np.any(b == k) else None for k in range(NB)],
               sd=[float(np.std(dW[b == k])) if np.any(b == k) else None for k in range(NB)],
               n=[int(np.sum(b == k)) for k in range(NB)])
    recs.append(rec)
    print(f"{cond} {mouse} {sess} [{rec['group']}]: dW by bin " + ' '.join(f'{x:+.3f}' if x is not None else 'nan' for x in rec['dW']), flush=True)

out = {'edges': EDGES, 'sessions': recs, 'summary': {}}
for grp in ['ribo8', 'cyto8', 'cyto6']:
    for cond in ['learning', 'water']:
        R = [r for r in recs if r['group'] == grp and r['cond'] == cond]
        if not R:
            continue
        S = {}
        for q in ['dW', 'adW', 'W0', 'aW0']:
            M = np.array([[np.nan if x is None else x for x in r[q]] for r in R])
            N = np.array([r['n'] for r in R], float)
            S[q] = dict(sess_mean=np.nanmean(M, 0).tolist(), sess_sem=(np.nanstd(M, 0) / np.sqrt(np.sum(np.isfinite(M), 0))).tolist(),
                        pair_mean=(np.nansum(M * N, 0) / np.sum(N * np.isfinite(M), 0)).tolist())
        S['n_sess'] = len(R)
        S['n_pairs'] = np.sum([r['n'] for r in R], 0).tolist()
        out['summary'][f'{grp}/{cond}'] = S
        print(f"{grp} {cond} ({len(R)} sess): dW " + ' '.join(f'{x:+.4f}' for x in S['dW']['sess_mean']) +
              ' | |dW| ' + ' '.join(f'{x:.3f}' for x in S['adW']['sess_mean']))
with open('/root/capsule/results/dw_vs_dist.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
