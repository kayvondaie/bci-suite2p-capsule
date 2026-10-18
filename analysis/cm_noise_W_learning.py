"""Common-mode correction and activity-based explainers of photostim W
(learning sessions).

W here is built from per-repeat responses r(cell, repeat) (paper-1 windows,
repeat_responses_paper): W_m[i, g] = mean of r over group g's repeats in map
m (0 = photostim, 1 = photostim2).

PART 1, common mode. Per repeat t (group g), c(t) = mean response of cells
> CM_FAR um from every site of g (putative SST outliers excluded; leave-one-
out for the cell itself). Variants:
  raw : W
  cm1 : mean_t (r_i(t) - c(t))                      full subtraction
  cmb : mean_t (r_i(t) - b_i c(t)), b_i = cell i's slope on c across repeats
        after removing group means (its coupling to population noise)
For each variant, reliability of the two-way decomposition of the non-target
matrix (row = cell-wide, col = group-wide, pair = residual), within map (odd
vs even repeats) and across maps at matched repeat count (odd half of map 0 vs
even half of map 1 and vice versa). Within >> across means the component is
systematic within a map but changes between maps (state); both low means
noise; both high means a fixed property.

PART 2, explainers of the pair-specific part of W. Pair (non-target i, group
g), activity features = mean over g's responsive targets:
  infl_x       repeat-level influence: corr over g's repeats of r_i(t) with
               the targets' mean response, partial on c(t); computed in one
               map and tested against W of the OTHER map (cross-validated)
  corr_bci     frame-level correlation, BCI epoch (df_closedloop)
  corr_bci_1s  1 s bins
  nc_bci       noise correlation: trial F minus the trial-averaged PSTH,
               frame level
  nc_trial     trial-count noise correlation (mean F over the trial, across
               trials, mean removed)
  dcorr_bci / dcorr_spont   correlation of the positive first difference of
               dF/F (fast rise events), BCI / spont_pre
  lag_bci_s / lag_bci_l / lag_spont_s   directed asymmetry
               corr(tgt(t), i(t+L)) - corr(i(t), tgt(t+L)), L ~ 35 ms (s)
               or ~ 100 ms (l)
Targets: < 10 um, AMP > 0.1 in both maps, non-outlier. Non-targets 30-1000 um,
outliers excluded. Targets of the residual: Wbar = (W0 + W1) / 2 and each map,
residualised (within session OLS) on distance bins, cell_comp, group_comp.
Pooled r (within-session z) + sessions individually significant.
MICE env var restricts mice. Prints JSON."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

import json
from scipy import stats as sps

MICE = os.environ.get('MICE', '').split(',') if os.environ.get('MICE') else None
CM_FAR = float(os.environ.get('CM_FAR', 300))
DIST_EDGES = [30, 50, 75, 100, 150, 200, 300, 500, 1000]
VARIANTS = ['raw', 'cm1', 'cmb']


def z(v):
    v = np.asarray(v, dtype=float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1.0)


def stdz(x):
    x = x - x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    sd[sd == 0] = np.inf
    return x / sd


def corr_mat(x):
    Z = stdz(x)
    return (Z @ Z.T) / Z.shape[1]


def lag_asym(x, L):
    """A[a, b] = corr(a(t), b(t+L)) - corr(b(t), a(t+L))."""
    Z = stdz(x)
    C = (Z[:, :-L] @ Z[:, L:].T) / (Z.shape[1] - L)
    return C - C.T


def bin_frames(x, n):
    m = x.shape[1] // n
    return x[:, :m * n].reshape(x.shape[0], m, n).mean(axis=2)


def two_way(W, M):
    """Additive row + col fit on entries M; returns row, col, residual."""
    X = np.where(M, W, np.nan)
    mu = np.nanmean(X)
    row = np.zeros(W.shape[0]); col = np.zeros(W.shape[1])
    for _ in range(20):
        row = np.nan_to_num(np.nanmean(X - mu - col[None, :], axis=1))
        col = np.nan_to_num(np.nanmean(X - mu - row[:, None], axis=0))
    res = X - mu - row[:, None] - col[None, :]
    return row, col, res


def rel(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def common_mode(r, s, D, ok_cell):
    """c(t) and leave-one-out version per cell (cells x repeats)."""
    n_cells, n_rep = r.shape
    c_loo = np.full(r.shape, np.nan)
    c = np.full(n_rep, np.nan)
    for t in range(n_rep):
        g = s[t]
        if g < 0 or g >= D.shape[1]:
            continue
        far = (D[:, g] > CM_FAR) & ok_cell & np.isfinite(r[:, t])
        if far.sum() < 10:
            continue
        tot = r[far, t].sum(); n = far.sum()
        c[t] = tot / n
        c_loo[:, t] = c[t]
        c_loo[far, t] = (tot - r[far, t]) / (n - 1)
    return c, c_loo


def group_means(r, s, n_groups, idx=None):
    W = np.full((r.shape[0], n_groups), np.nan)
    for g in range(n_groups):
        k = np.where(s == g)[0]
        if idx is not None:
            k = k[idx(k)]
        if k.size:
            with np.errstate(all='ignore'):
                W[:, g] = np.nanmean(r[:, k], axis=1)
    return W


def variants(r, s, D, ok_cell):
    n_groups = D.shape[1]
    c, c_loo = common_mode(r, s, D, ok_cell)
    # b_i: slope of r_i on c after removing group means
    rr = r.copy()
    cc = np.where(np.isfinite(c_loo), c_loo, np.nan)
    for g in range(n_groups):
        k = s == g
        if k.any():
            with np.errstate(all='ignore'):
                rr[:, k] -= np.nanmean(rr[:, k], axis=1, keepdims=True)
                cc[:, k] -= np.nanmean(cc[:, k], axis=1, keepdims=True)
    okk = np.isfinite(rr) & np.isfinite(cc)
    num = np.nansum(np.where(okk, rr * cc, 0), axis=1)
    den = np.nansum(np.where(okk, cc * cc, 0), axis=1)
    b = num / np.where(den > 0, den, np.nan)
    out = {'raw': r, 'cm1': r - c_loo, 'cmb': r - b[:, None] * c_loo}
    return out, c, c_loo, b


def load(mouse, session):
    folder = learning_folder(mouse, session)
    data = ddct.load_hdf5(folder, ['F', 'df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    dt = float(data['dt_si'])
    reps = [repeat_responses_paper(ps_file(folder, k), dt, D) for k in ['photostim', 'photostim2']]
    acts = {'bci': np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float))}
    try:
        acts['spont'] = np.nan_to_num(spont_dff(load_learning_spont(folder)[0]))
    except FileNotFoundError:
        pass
    F = np.asarray(data['F'], dtype=float)                 # frames x cells x trials
    return AMP, D, dt, reps, acts, F


def activity_features(acts, F, dt):
    fs = 1.0 / dt
    Ls = max(1, int(round(0.035 * fs))); Ll = max(1, int(round(0.1 * fs)))
    C = {}
    a = acts['bci']
    C['corr_bci'] = corr_mat(a)
    C['corr_bci_1s'] = corr_mat(bin_frames(a, max(1, int(round(fs)))))
    da = np.clip(np.diff(a, axis=1), 0, None)
    C['dcorr_bci'] = corr_mat(da)
    C['lag_bci_s'] = lag_asym(a, Ls)
    C['lag_bci_l'] = lag_asym(a, Ll)
    if 'spont' in acts:
        sp = acts['spont']
        C['dcorr_spont'] = corr_mat(np.clip(np.diff(sp, axis=1), 0, None))
        C['lag_spont_s'] = lag_asym(sp, Ls)
    # noise correlations from the trial array
    with np.errstate(all='ignore'):
        R = F - np.nanmean(F, axis=2, keepdims=True)
    R = np.nan_to_num(np.transpose(R, (1, 0, 2)).reshape(F.shape[1], -1))
    C['nc_bci'] = corr_mat(R)
    tsta = np.arange(0, 12, dt); tsta = tsta - tsta[int(2 / dt)]
    win = np.where((tsta > 0) & (tsta < 10))[0]
    win = win[win < F.shape[0]]
    with np.errstate(all='ignore'):
        tm = np.nanmean(F[win], axis=0)                  # cells x trials
    tm = np.nan_to_num(tm - np.nanmean(tm, axis=1, keepdims=True))
    C['nc_trial'] = corr_mat(tm)
    return C


def influence(r, s, c, g, cl, nt):
    """corr over g's repeats of r_i with targets' mean response, partial on c."""
    k = np.where(s == g)[0]
    out = np.full(len(nt), np.nan)
    if k.size < 8:
        return out
    d = np.nanmean(r[np.ix_(cl, k)], axis=0)
    cg = c[k]
    Y = r[np.ix_(nt, k)]
    ok = np.isfinite(d) & np.isfinite(cg)
    if ok.sum() < 8:
        return out
    d, cg, Y = d[ok], cg[ok], Y[:, ok]
    A = np.column_stack([np.ones_like(cg), cg])
    dr = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
    Yf = np.nan_to_num(Y - np.nanmean(Y, axis=1, keepdims=True))
    Yr = Yf - (Yf @ A) @ np.linalg.pinv(A.T @ A) @ A.T
    num = Yr @ dr
    den = np.sqrt((Yr ** 2).sum(axis=1) * (dr ** 2).sum())
    out[:] = num / np.where(den > 0, den, np.nan)
    return out


def dist_onehot(d):
    idx = np.clip(np.digitize(d, DIST_EDGES[1:-1]), 0, len(DIST_EDGES) - 2)
    return np.eye(len(DIST_EDGES) - 1)[idx][:, 1:]


def session(mouse, session_id):
    AMP, D, dt, reps, acts, F = load(mouse, session_id)
    n_cells, n_groups = D.shape
    for k, a in acts.items():
        if a.shape[0] != n_cells:
            raise RuntimeError(f'{k} has {a.shape[0]} cells, photostim {n_cells}')
    if F.shape[1] != n_cells:
        raise RuntimeError(f'F has {F.shape[1]} cells, photostim {n_cells}')
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    ok_cell = ~is_out
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ok_cell[:, None]
    TG = [np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                   (AMP[1][:, g] > amp1_thr) & ok_cell)[0] for g in range(n_groups)]
    keep_g = np.array([t.size > 0 for t in TG])
    M = NT & keep_g[None, :]

    # ---------------- PART 1
    V, CMs = [], []
    for r, s in reps:
        v, c, c_loo, b = variants(r, s, D, ok_cell)
        V.append(v); CMs.append((c, s))
    rel_out = {}
    odd = lambda k: np.arange(k.size) % 2 == 0
    even = lambda k: np.arange(k.size) % 2 == 1
    for vn in VARIANTS:
        full = [group_means(V[m][vn], reps[m][1], n_groups) for m in range(2)]
        half = [[group_means(V[m][vn], reps[m][1], n_groups, f) for f in (odd, even)] for m in range(2)]
        dec = lambda W: two_way(W, M & np.isfinite(W))
        comp = {}
        for m in range(2):
            comp[f'w{m}'] = [dec(W) for W in half[m]]
        comp['full'] = [dec(W) for W in full]
        res = {}
        for ci, cname in enumerate(['row', 'col', 'pair']):
            pick = (lambda x: x[ci]) if cname != 'pair' else (lambda x: x[2][M])
            if cname == 'col':
                pick = lambda x: x[1][keep_g]
            if cname == 'row':
                pick = lambda x: x[0][M.any(axis=1)]
            within = np.nanmean([rel(pick(comp[f'w{m}'][0]), pick(comp[f'w{m}'][1])) for m in range(2)])
            across = np.nanmean([rel(pick(comp['w0'][0]), pick(comp['w1'][1])),
                                 rel(pick(comp['w0'][1]), pick(comp['w1'][0]))])
            res[cname] = dict(within=float(within), across=float(across),
                              full_across=rel(pick(comp['full'][0]), pick(comp['full'][1])))
        res['total_across'] = rel(full[0][M], full[1][M])
        res['col_sd'] = float(np.nanstd(comp['full'][0][1][keep_g]))
        res['pair_sd'] = float(np.nanstd(comp['full'][0][2][M]))
        rel_out[vn] = res
    Wfull = {vn: [group_means(V[m][vn], reps[m][1], n_groups) for m in range(2)] for vn in VARIANTS}

    # ---------------- PART 2
    C = activity_features(acts, F, dt)
    rows = {k: [] for k in ['g', 'i', 'dist'] + [f'W{m}_{vn}' for m in range(2) for vn in VARIANTS] +
            [f'cell_comp{m}' for m in range(2)] + [f'group_comp{m}' for m in range(2)] +
            ['infl0', 'infl1'] + list(C)}
    for g in np.where(keep_g)[0]:
        cl = TG[g]
        nt = np.where(M[:, g] & np.isfinite(Wfull['raw'][0][:, g]) & np.isfinite(Wfull['raw'][1][:, g]))[0]
        if nt.size == 0:
            continue
        rows['g'] += [g] * nt.size; rows['i'] += list(nt); rows['dist'] += list(D[nt, g])
        for m in range(2):
            for vn in VARIANTS:
                rows[f'W{m}_{vn}'] += list(Wfull[vn][m][nt, g])
            W = Wfull['raw'][m]
            Wm = np.where(M, W, np.nan)
            cs = np.nansum(Wm[nt], axis=1) - np.nan_to_num(Wm[nt, g])
            cn = np.sum(np.isfinite(Wm[nt]), axis=1) - 1
            rows[f'cell_comp{m}'] += list(cs / np.where(cn > 0, cn, np.nan))
            gs = np.nansum(Wm[:, g]); gn = np.sum(np.isfinite(Wm[:, g]))
            rows[f'group_comp{m}'] += list((gs - Wm[nt, g]) / (gn - 1))
            r, s = reps[m]
            rows[f'infl{m}'] += list(influence(r, s, CMs[m][0], g, cl, nt))
        for k, Ck in C.items():
            rows[k] += list(Ck[np.ix_(cl, nt)].mean(axis=0))
    T = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    return rel_out, T


def residual(y, T, m):
    cc = T[f'cell_comp{m}'] if m in (0, 1) else (T['cell_comp0'] + T['cell_comp1']) / 2
    gc = T[f'group_comp{m}'] if m in (0, 1) else (T['group_comp0'] + T['group_comp1']) / 2
    X = np.column_stack([np.ones(len(y)), dist_onehot(T['dist']), z(cc), z(gc)])
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    out = np.full(len(y), np.nan)
    out[ok] = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    return out


results = []
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        rel_out, T = session(mouse, sess)
        if len(np.unique(T['g'])) < MIN_GROUPS:
            print(f'{mouse} {sess}: too few groups, skipped')
            continue
        Wbar = (T['W0_raw'] + T['W1_raw']) / 2
        targets = {'Wbar': residual(Wbar, T, 'bar'), 'W0': residual(T['W0_raw'], T, 0),
                   'W1': residual(T['W1_raw'], T, 1)}
        feats = [k for k in T if k.startswith(('corr_', 'nc_', 'dcorr_', 'lag_'))]
        tests = {}
        for tn, y in targets.items():
            for f in feats:
                tests[f'{tn}/{f}'] = (T[f], y)
        # influence cross-validated across maps
        tests['W1/infl_x'] = (T['infl0'], targets['W1'])
        tests['W0/infl_x'] = (T['infl1'], targets['W0'])
        tests['Wbar/infl_x'] = (np.nanmean([T['infl0'], T['infl1']], axis=0), targets['Wbar'])
        tests['Wbar_raw/infl_x'] = (np.nanmean([T['infl0'], T['infl1']], axis=0), Wbar)
        # infl0 vs infl1 reliability
        tests['infl0/infl1'] = (T['infl0'], T['infl1'])
        single = {}
        for k, (x, y) in tests.items():
            ok = np.isfinite(x) & np.isfinite(y)
            rr, pp = sps.pearsonr(x[ok], y[ok])
            single[k] = (float(rr), float(pp), z(x[ok]), z(y[ok]))
        results.append(dict(mouse=mouse, session=sess, rel=rel_out, single=single, n=len(Wbar)))
        rw = rel_out['raw']; c1 = rel_out['cm1']
        print(f"{mouse} {sess}: n={len(Wbar)} | col within/across raw {rw['col']['within']:+.2f}/{rw['col']['across']:+.2f}"
              f" cm1 {c1['col']['within']:+.2f}/{c1['col']['across']:+.2f} | pair raw {rw['pair']['within']:+.3f}/"
              f"{rw['pair']['across']:+.3f} cm1 {c1['pair']['within']:+.3f}/{c1['pair']['across']:+.3f} | "
              f"infl_x {single['Wbar/infl_x'][0]:+.3f} nc_bci {single['Wbar/nc_bci'][0]:+.3f}", flush=True)
    except FileNotFoundError as e:
        print(f'{mouse} {sess}: skipped ({e})')
    except Exception as e:
        print(f'{mouse} {sess}: FAILED {e}')
        traceback.print_exc()

out = {'rel': {}, 'single': {}}
print(f'\n{len(results)} sessions')
print('\nPART 1: reliability (median over sessions) within-map split-half | across-map matched | full across')
for vn in VARIANTS:
    for cname in ['row', 'col', 'pair']:
        w = np.array([r['rel'][vn][cname]['within'] for r in results])
        a = np.array([r['rel'][vn][cname]['across'] for r in results])
        f = np.array([r['rel'][vn][cname]['full_across'] for r in results])
        out['rel'][f'{vn}/{cname}'] = dict(within=float(np.nanmedian(w)), across=float(np.nanmedian(a)),
                                           full=float(np.nanmedian(f)), within_all=w.tolist(), across_all=a.tolist())
        print(f'   {vn:4s} {cname:5s} within {np.nanmedian(w):+.3f} | across {np.nanmedian(a):+.3f} | full {np.nanmedian(f):+.3f}'
              f'   (across<within in {np.sum(a < w)}/{len(w)})')
    t = np.array([r['rel'][vn]['total_across'] for r in results])
    cs = np.array([r['rel'][vn]['col_sd'] for r in results]); ps = np.array([r['rel'][vn]['pair_sd'] for r in results])
    out['rel'][f'{vn}/total'] = float(np.nanmedian(t))
    print(f'   {vn:4s} total r(W0,W1) median {np.nanmedian(t):+.3f}; col SD {np.nanmedian(cs):.4f}, pair SD {np.nanmedian(ps):.4f}')

print('\nPART 2: features vs pair-specific residual of W: pooled r (within-session z) p | sessions sig + / -')
keys = list(results[0]['single']) if results else []
for k in sorted({k for r in results for k in r['single']}):
    S = [r for r in results if k in r['single']]
    x = np.concatenate([r['single'][k][2] for r in S]); y = np.concatenate([r['single'][k][3] for r in S])
    rr, pp = sps.pearsonr(x, y)
    v = [r['single'][k][:2] for r in S]
    npos = sum(a > 0 and b < .05 for a, b in v); nneg = sum(a < 0 and b < .05 for a, b in v)
    out['single'][k] = dict(r=float(rr), p=float(pp), n_pos=int(npos), n_neg=int(nneg), n=len(S),
                            per_session=[float(a) for a, _ in v])
    print(f'   {k:22s} r={rr:+.4f} p={pp:.2g} | +{npos}/{len(S)}  -{nneg}/{len(S)}')
print('JSON' + json.dumps(out))
