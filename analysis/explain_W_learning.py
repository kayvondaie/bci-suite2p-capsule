"""What explains connectivity W (before: W0, after: W1) in the BCI learning
sessions? Candidate explainers for each pair (non-target i, photostim group g):
  dist_*        distance of i from the nearest target, one-hot bins
  cell_comp     i's mean W to all OTHER groups (leave-one-group-out), same map
  group_comp    mean W of all OTHER non-targets to group g (leave-one-cell-out)
  tgt_drive     summed AMP of g's responsive targets in the same map
  n_tgt         number of responsive targets
  bright_nt     log mean raw F of i (spont_pre)
  bright_tgt    log mean raw F of g's responsive targets
  corr_<ep>     mean Pearson correlation of i with g's responsive targets in
                epoch ep (spont_pre, spont_post, bci); '_1s' = 1 s bins
W = AMP (compute_amp_from_photostim). Targets: < 10 um, AMP > 0.1 in both
maps, non-outlier. Non-targets 30-1000 um, putative SST outliers excluded.
Reports, per map:
  A  single-explainer r, pooled over sessions (within-session z-scored), and
     sessions individually significant
  B  cross-validated (GroupKFold by photostim group) full-model R2 and drop-one
     unique R2 per explainer family
  C  correlation explainers vs the pair-specific residual of W after removing
     distance, cell_comp, group_comp, tgt_drive, n_tgt (within-session OLS)
Loaders / config from three_factor_control_vs_learning.py. Prints JSON."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

import json
from scipy import stats as sps
from sklearn.model_selection import GroupKFold

DIST_EDGES = [30, 50, 75, 100, 150, 200, 300, 500, 1000]
N_FOLDS = 5


def corr_mat(x):
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).mean(axis=1))
    sd[sd == 0] = np.inf
    x = x / sd[:, None]
    return (x @ x.T) / x.shape[1]


def bin_frames(x, n):
    m = x.shape[1] // n
    return x[:, :m * n].reshape(x.shape[0], m, n).mean(axis=2)


def z(v):
    v = np.asarray(v, dtype=float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def load_learning(mouse, session):
    folder = learning_folder(mouse, session)
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
    AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
    fs = 1.0 / float(data['dt_si'])
    acts = {'bci': np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float))}
    bright = None
    try:
        raw = load_learning_spont(folder)
        bright = np.log(np.clip(np.nanmean(raw[0], axis=1), 1e-3, None))
        names = ['spont_pre', 'spont_post'][:len(raw)]
        for n, F in zip(names, raw):
            acts[n] = np.nan_to_num(spont_dff(F))
    except FileNotFoundError:
        pass
    return AMP, stimDist, acts, bright, fs


def session_table(mouse, session):
    AMP, D, acts, bright, fs = load_learning(mouse, session)
    n_cells, n_groups = D.shape
    for k, a in acts.items():
        if a.shape[0] != n_cells:
            raise RuntimeError(f'{k} has {a.shape[0]} cells, photostim {n_cells}')
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    nb = max(1, int(round(fs)))
    C = {}
    for k, a in acts.items():
        C[f'corr_{k}'] = corr_mat(a)
        C[f'corr_{k}_1s'] = corr_mat(bin_frames(a, nb))
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ~is_out[:, None]
    rows = []
    for g in range(n_groups):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        if cl.size == 0:
            continue
        nt = np.where(NT[:, g] & np.isfinite(AMP[0][:, g]) & np.isfinite(AMP[1][:, g]))[0]
        for i in nt:
            r = dict(g=g, i=i, dist=D[i, g], n_tgt=cl.size,
                     bright_nt=bright[i] if bright is not None else np.nan,
                     bright_tgt=np.mean(bright[cl]) if bright is not None else np.nan)
            for m in range(2):
                W = AMP[m]
                r[f'W{m}'] = W[i, g]
                others_g = NT[i, :].copy(); others_g[g] = False
                r[f'cell_comp{m}'] = np.nanmean(W[i, others_g]) if others_g.any() else np.nan
                others_i = NT[:, g].copy(); others_i[i] = False
                r[f'group_comp{m}'] = np.nanmean(W[others_i, g]) if others_i.any() else np.nan
                r[f'tgt_drive{m}'] = np.nansum(W[cl, g])
            for k, Ck in C.items():
                r[k] = Ck[i, cl].mean()
            rows.append(r)
    if not rows:
        raise RuntimeError('no pairs')
    keys = rows[0].keys()
    T = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    T['n_groups'] = len(np.unique(T['g']))
    return T


def dist_onehot(d):
    idx = np.clip(np.digitize(d, DIST_EDGES[1:-1]), 0, len(DIST_EDGES) - 2)
    return np.eye(len(DIST_EDGES) - 1)[idx]


def families(T, m):
    F = {'distance': dist_onehot(T['dist'])[:, 1:],      # drop one bin (intercept)
         'cell_comp': z(T[f'cell_comp{m}'])[:, None],
         'group_comp': z(T[f'group_comp{m}'])[:, None],
         'tgt_drive': np.column_stack([z(T[f'tgt_drive{m}']), z(T['n_tgt'])])}
    if np.isfinite(T['bright_nt']).all():
        F['brightness'] = np.column_stack([z(T['bright_nt']), z(T['bright_tgt'])])
    for k in sorted(k for k in T if k.startswith('corr_')):
        F[k] = z(T[k])[:, None]
    return F


def cv_r2(X, y, groups):
    yp = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=N_FOLDS).split(X, groups=groups):
        A = np.column_stack([np.ones(len(tr)), X[tr]])
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        yp[te] = np.column_stack([np.ones(len(te)), X[te]]) @ b
    return 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)


results = []
for mouse, session in LEARNING_SESSIONS:
    if (mouse, session) in EXCLUDE:
        continue
    try:
        T = session_table(mouse, session)
        if T['n_groups'] < MIN_GROUPS:
            print(f'{mouse} {session}: only {T["n_groups"]} groups, skipped')
            continue
        ok = np.isfinite(T['cell_comp0']) & np.isfinite(T['group_comp0']) & \
             np.isfinite(T['cell_comp1']) & np.isfinite(T['group_comp1'])
        T = {k: (v[ok] if isinstance(v, np.ndarray) else v) for k, v in T.items()}
        res = dict(mouse=mouse, session=session, n_pairs=int(len(T['W0'])), single={}, cv={}, resid={})
        for m in range(2):
            y = z(T[f'W{m}'])
            F = families(T, m)
            for k, X in F.items():
                if k == 'distance':
                    continue
                for j in range(X.shape[1]):
                    nm = k if X.shape[1] == 1 else f'{k}[{j}]'
                    rr, pp = sps.pearsonr(X[:, j], y)
                    res['single'][f'W{m}/{nm}'] = (float(rr), float(pp))
            fam_names = list(F)
            Xall = np.column_stack([F[k] for k in fam_names])
            full = cv_r2(Xall, y, T['g'])
            res['cv'][f'W{m}/full'] = float(full)
            for k in fam_names:
                Xd = np.column_stack([F[q] for q in fam_names if q != k])
                res['cv'][f'W{m}/drop_{k}'] = float(full - cv_r2(Xd, y, T['g']))
            # residual of W after non-specific structure, vs correlation explainers
            Xn = np.column_stack([np.ones(len(y))] + [F[q] for q in ['distance', 'cell_comp', 'group_comp', 'tgt_drive']])
            resid = y - Xn @ np.linalg.lstsq(Xn, y, rcond=None)[0]
            res[f'resid_W{m}'] = resid
            for k in fam_names:
                if k.startswith('corr_'):
                    rr, pp = sps.pearsonr(F[k][:, 0], resid)
                    res['resid'][f'W{m}/{k}'] = (float(rr), float(pp))
        res['table'] = {k: T[k] for k in T if k.startswith('corr_')}
        res['y'] = {f'W{m}': z(T[f'W{m}']) for m in range(2)}
        results.append(res)
        print(f"{mouse} {session}: {res['n_pairs']} pairs | CV R2 W0={res['cv']['W0/full']:.3f} "
              f"W1={res['cv']['W1/full']:.3f} | resid r corr_spont_pre: "
              f"{res['resid'].get('W0/corr_spont_pre', (np.nan,))[0]:+.3f} / "
              f"{res['resid'].get('W1/corr_spont_pre', (np.nan,))[0]:+.3f}  corr_bci: "
              f"{res['resid']['W0/corr_bci'][0]:+.3f} / {res['resid']['W1/corr_bci'][0]:+.3f}")
    except FileNotFoundError as e:
        print(f'{mouse} {session}: skipped ({e})')
    except Exception as e:
        print(f'{mouse} {session}: FAILED {e}')
        traceback.print_exc()

out = {}
print(f'\n{len(results)} sessions')
for m in range(2):
    print(f'\n######## W{m} ({"before" if m == 0 else "after"}) ########')
    keys = sorted({k.split('/', 1)[1] for r in results for k in r['single'] if k.startswith(f'W{m}/')})
    print('A) single-explainer r: median session r | sessions r>0 & p<.05 | sessions r<0 & p<.05')
    for k in keys:
        v = [r['single'][f'W{m}/{k}'] for r in results if f'W{m}/{k}' in r['single']]
        rr = np.array([a for a, _ in v]); pp = np.array([b for _, b in v])
        out[f'A/W{m}/{k}'] = dict(median=float(np.median(rr)), n_pos=int(np.sum((rr > 0) & (pp < .05))),
                                  n_neg=int(np.sum((rr < 0) & (pp < .05))), n=len(rr))
        print(f'   {k:22s} {np.median(rr):+.4f} | {np.sum((rr > 0) & (pp < .05)):2d}/{len(rr)} | '
              f'{np.sum((rr < 0) & (pp < .05)):2d}/{len(rr)}')
    print('B) CV R2 (GroupKFold by group): median full model, and median unique R2 when dropped')
    full = np.array([r['cv'][f'W{m}/full'] for r in results])
    print(f'   full model: median {np.median(full):.4f} (range {full.min():.4f} to {full.max():.4f})')
    out[f'B/W{m}/full'] = dict(median=float(np.median(full)), values=full.tolist())
    for k in sorted({q.split('drop_')[1] for r in results for q in r['cv'] if q.startswith(f'W{m}/drop_')}):
        v = np.array([r['cv'][f'W{m}/drop_{k}'] for r in results if f'W{m}/drop_{k}' in r['cv']])
        out[f'B/W{m}/{k}'] = dict(median=float(np.median(v)), n_pos=int(np.sum(v > 0)), n=len(v))
        print(f'   drop {k:22s} unique R2 median {np.median(v):+.5f}, >0 in {np.sum(v > 0)}/{len(v)}')
    print('C) correlation vs pair-specific residual: pooled r (within-session z) | sessions sig +/-')
    for k in sorted({q.split('/', 1)[1] for r in results for q in r['resid'] if q.startswith(f'W{m}/')}):
        S = [r for r in results if f'W{m}/{k}' in r['resid']]
        x = np.concatenate([z(r['table'][k]) for r in S])
        y = np.concatenate([z(r[f'resid_W{m}']) for r in S])
        rr, pp = sps.pearsonr(x, y)
        v = [r['resid'][f'W{m}/{k}'] for r in S]
        npos = sum(a > 0 and b < .05 for a, b in v); nneg = sum(a < 0 and b < .05 for a, b in v)
        out[f'C/W{m}/{k}'] = dict(r=float(rr), p=float(pp), n_pos=int(npos), n_neg=int(nneg), n=len(S))
        print(f'   {k:22s} pooled r={rr:+.4f} p={pp:.2g} | +{npos}/{len(S)}  -{nneg}/{len(S)}')
print('JSON' + json.dumps(out))
