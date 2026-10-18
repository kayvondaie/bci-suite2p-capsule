"""What drives the W-vs-correlation slope among robust connections? (the
negative slope in BCI103). Robust connections selected cross-validated as in
robust_prepost_meanbin.py (W0 for pairs selected p < P_SEL positive in map 1,
W1 for pairs selected in map 0). Correlation (1 s bins, epoch CORR_EP) of the
non-target i with g's responsive targets is split into
  corr_tgt   mean correlation of g's targets with all other cells (target hubness)
  corr_nt    mean correlation of i with all other cells (non-target hubness)
  corr_spec  pair correlation minus the additive fit mu + a_i + b_g, with a, b
             from the full (all cells x all targets) matrix (pair-specific part)
plus
  nt_bright  log mean raw F of i (spont_pre)
  nt_noise   frame-to-frame noise of i in dF/F units: median |diff F| / mean F
  tgt_amp    mean AMP of g's responsive targets in the same map
  tgt_bright log mean raw F of g's responsive targets
All z-scored within session. Reports pooled r with W, session counts, the
feature-feature correlations, and a joint OLS (standardised betas, pooled).
Writes /root/capsule/results/robust_neg_decomp.json (meanbin data too)."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robust_prepost_meanbin.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index("pool = {'s6'")])

P_SEL = float(os.environ.get('P_SEL', 0.001))
CORR_EP = os.environ.get('CORR_EP', 'bci')
# DECONV=1: deconvolve the activity used for correlations (OASIS from suite2p,
# AR(1) inverse filter as fallback); decay tau TAU6 (BCI103, GCaMP6s) or TAU8
DECONV = os.environ.get('DECONV', '0') == '1'
TAU6 = float(os.environ.get('TAU6', 1.25))
TAU8 = float(os.environ.get('TAU8', 0.125))
_DC_CACHE = {}
try:
    from numba import njit
    DC_METHOD = 'oasis'
except Exception:
    njit = None
    DC_METHOD = 'ar1'
print(f'deconvolution: {DECONV}, method {DC_METHOD}, tau6 {TAU6}, tau8 {TAU8}')

if njit is not None:
    @njit
    def _oasis_trace(F, v, w, t, l, s, tau, fs):
        # AR(1) OASIS, as suite2p.extraction.dcnv.oasis_trace
        NT = F.shape[0]
        g = -1. / (tau * fs)
        it = 0
        ip = 0
        while it < NT:
            v[ip], w[ip], t[ip], l[ip] = F[it], 1, it, 1
            while ip > 0:
                if v[ip - 1] * np.exp(g * l[ip - 1]) > v[ip]:
                    f1 = np.exp(g * l[ip - 1])
                    f2 = np.exp(2 * g * l[ip - 1])
                    wnew = w[ip - 1] + w[ip] * f2
                    v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                    w[ip - 1] = wnew
                    l[ip - 1] = l[ip - 1] + l[ip]
                    ip -= 1
                else:
                    break
            it += 1
            ip += 1
        s[t[1:ip]] = v[1:ip] - v[:ip - 1] * np.exp(g * l[:ip - 1])


def _maximin(x, fs):
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d
    win = max(3, int(60.0 * fs))
    lo = gaussian_filter1d(x, 10.0, axis=1)
    lo = minimum_filter1d(lo, win, axis=1)
    lo = maximum_filter1d(lo, win, axis=1)
    return x - lo


def deconv(act, fs, mouse):
    key = (mouse, act.shape, float(np.sum(act[:, :200])))
    if key in _DC_CACHE:
        return _DC_CACHE[key]
    tau = TAU6 if mouse == 'BCI103' else TAU8
    x = np.asarray(act, dtype=np.float64)
    if njit is not None:
        xb = _maximin(x, fs)
        sp = np.zeros_like(xb)
        T = xb.shape[1]
        for n in range(xb.shape[0]):
            v = np.zeros(T); w = np.zeros(T); t = np.zeros(T, np.int64); l = np.zeros(T)
            s_ = np.zeros(T)
            _oasis_trace(xb[n], v, w, t, l, s_, tau, fs)
            sp[n] = s_
    else:
        g = np.exp(-1.0 / (tau * fs))
        sp = np.clip(x[:, 1:] - g * x[:, :-1], 0, None)
        sp = np.concatenate([np.zeros((x.shape[0], 1)), sp], axis=1)
    _DC_CACHE.clear()
    _DC_CACHE[key] = sp
    return sp


FEATS = ['corr_pair', 'corr_spec', 'corr_tgt', 'corr_nt', 'nt_bright', 'nt_noise', 'tgt_amp', 'tgt_bright']
# NEU=1: neuropil contamination indices of the non-target (spont_pre suite2p
# F / Fneu, iscell cells): neu_corr = corr(F, Fneu); neu_ratio = mean Fneu /
# mean F; neu_resid = corr(F - 0.7 Fneu, Fneu); plus tgt_neu_corr (mean
# neu_corr of g's targets)
NEU = os.environ.get('NEU', '0') == '1'
NEU_FEATS = ['neu_corr', 'neu_ratio', 'neu_resid', 'tgt_neu_corr']
if NEU:
    FEATS = FEATS + NEU_FEATS


def neu_indices(folder):
    for pair in SPONT_FOLDER_PAIRS:
        d = os.path.join(folder, pair[0], 'plane0')
        if os.path.exists(os.path.join(d, 'Fneu.npy')):
            F = np.load(os.path.join(d, 'F.npy'), allow_pickle=True)
            N = np.load(os.path.join(d, 'Fneu.npy'), allow_pickle=True)
            ic = np.asarray(np.load(os.path.join(d, 'iscell.npy'), allow_pickle=True))[:, 0] == 1
            F = np.asarray(F[ic], float); N = np.asarray(N[ic], float)
            def rowcorr(a, b):
                a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
                return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))
            return dict(neu_corr=rowcorr(F, N), neu_ratio=N.mean(1) / np.where(F.mean(1) > 0, F.mean(1), np.nan),
                        neu_resid=rowcorr(F - 0.7 * N, N))
    raise FileNotFoundError(f'no spont Fneu in {folder}')


def table(mouse, sess):
    folder = learning_folder(mouse, sess)
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    dt = float(data['dt_si']); fs = 1.0 / dt; nb = max(1, int(round(fs)))
    n_cells, n_groups = D.shape
    reps = [repeat_responses_paper(ps_file(folder, k), dt, D) for k in ['photostim', 'photostim2']]
    raw = load_learning_spont(folder)[0]
    if raw.shape[0] != n_cells:
        raise RuntimeError('spont cell count mismatch')
    mF = np.nanmean(raw, axis=1)
    bright = np.log(np.clip(mF, 1e-3, None))
    noise = np.nanmedian(np.abs(np.diff(raw, axis=1)), axis=1) / np.where(mF > 0, mF, np.nan)
    if CORR_EP == 'bci':
        act = np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float))
    else:
        act = np.nan_to_num(spont_dff(raw))
    if DECONV:
        act = deconv(act, fs, mouse)
    C = corr_mat(bin_frames(act, nb))
    np.fill_diagonal(C, np.nan)
    cell_hub = np.nanmean(C, axis=1)
    NI = neu_indices(folder) if NEU else None
    if NI is not None and len(NI['neu_corr']) != n_cells:
        raise RuntimeError('Fneu cell count mismatch')
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ~is_out[:, None]
    groups = []
    for g in range(n_groups):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        nt = np.where(NT[:, g])[0]
        if cl.size and nt.size:
            groups.append((g, cl, nt))
    # pair-level correlation matrix (cells x groups) for the additive fit
    P = np.full((n_cells, n_groups), np.nan)
    for g, cl, nt in groups:
        P[nt, g] = np.nanmean(C[np.ix_(cl, nt)], axis=0)
    M = np.isfinite(P)
    mu = np.nanmean(P); a = np.zeros(n_cells); b = np.zeros(n_groups)
    for _ in range(20):
        a = np.nan_to_num(np.nanmean(P - mu - b[None, :], axis=1))
        b = np.nan_to_num(np.nanmean(P - mu - a[:, None], axis=0))
    spec = P - mu - a[:, None] - b[None, :]
    rows = {k: [] for k in ['g', 'mu0', 'mu1', 't0', 't1', 'p0', 'p1', 'tgt_amp0', 'tgt_amp1'] + FEATS}
    for g, cl, nt in groups:
        for m in range(2):
            mu_, t, p = tstats(reps[m][0], reps[m][1], g, nt)
            rows[f'mu{m}'] += list(mu_); rows[f't{m}'] += list(t); rows[f'p{m}'] += list(p)
            rows[f'tgt_amp{m}'] += [float(np.nanmean(AMP[m][cl, g]))] * nt.size
        rows['g'] += [g] * nt.size
        rows['corr_pair'] += list(P[nt, g]); rows['corr_spec'] += list(spec[nt, g])
        rows['corr_tgt'] += [float(np.nanmean(cell_hub[cl]))] * nt.size
        rows['corr_nt'] += list(cell_hub[nt])
        rows['nt_bright'] += list(bright[nt]); rows['nt_noise'] += list(noise[nt])
        rows['tgt_bright'] += [float(np.mean(bright[cl]))] * nt.size
        rows['tgt_amp'] += [np.nan] * nt.size
        if NI is not None:
            for k in ['neu_corr', 'neu_ratio', 'neu_resid']:
                rows[k] += list(NI[k][nt])
            rows['tgt_neu_corr'] += [float(np.nanmean(NI['neu_corr'][cl]))] * nt.size
    T = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    ok = np.isfinite(T['p0']) & np.isfinite(T['p1'])
    T = {k: v[ok] for k, v in T.items()}
    for m in range(2):
        T[f'W{m}'] = z(T[f'mu{m}'])
        T[f'tgt_amp{m}'] = z(T[f'tgt_amp{m}'])
    for k in FEATS:
        if k != 'tgt_amp':
            T[k] = z(T[k])
    return T


pool = {'s6': [], 's8': []}
for si, (mouse, sess) in enumerate(LEARNING_SESSIONS):
    if (mouse, sess) in EXCLUDE:
        continue
    try:
        T = table(mouse, sess)
        if len(np.unique(T['g'])) < MIN_GROUPS:
            continue
    except Exception as e:
        print(f'{mouse} {sess}: {e}'); continue
    T['sid'] = np.full(len(T['g']), si, float)
    pool['s6' if mouse == 'BCI103' else 's8'].append(T)
    print(f'{mouse} {sess}: {len(T["g"])} pairs', flush=True)

out = {}
for ind, S in pool.items():
    cat = {k: np.concatenate([s[k] for s in S]) for k in S[0]}
    print(f'\n######## {ind} ({len(S)} sessions), correlation epoch {CORR_EP}')
    for m in range(2):
        s_ = (cat[f'p{1 - m}'] < P_SEL) & (cat[f't{1 - m}'] > 0)
        y = cat[f'W{m}'][s_]
        F = {k: (cat[f'tgt_amp{m}'] if k == 'tgt_amp' else cat[k])[s_] for k in FEATS}
        sid = cat['sid'][s_]
        print(f' W{m} (selected in map {1 - m}), n={s_.sum()}')
        for k in FEATS:
            res = meanbin(F[k], y, sid)
            if res:
                out[f'{ind}/W{m}/{k}'] = res
                print(f'   {k:11s} r={res["r"]:+.3f} p={res["p"]:.2g}  sig+ {res["n_sig"]}/{res["n_sess"]}')
        if NEU:
            base = [k for k in FEATS if k not in ['corr_pair'] + NEU_FEATS]
            Xb = np.column_stack([F[k] for k in base])
            okb = np.isfinite(Xb).all(axis=1) & np.isfinite(y) & np.isfinite(np.column_stack([F[k] for k in NEU_FEATS])).all(axis=1)
            Xbs = (Xb[okb] - Xb[okb].mean(0)) / Xb[okb].std(0); ybs = (y[okb] - y[okb].mean()) / y[okb].std()
            bb = np.linalg.lstsq(np.column_stack([np.ones(okb.sum()), Xbs]), ybs, rcond=None)[0]
            print('   joint OLS WITHOUT neuropil features (same pairs): ' +
                  ', '.join(f'{k} {bb[j + 1]:+.3f}' for j, k in enumerate(base)))
            out[f'{ind}/W{m}/ols_noneu'] = {k: float(bb[j + 1]) for j, k in enumerate(base)}
        X = np.column_stack([F[k] for k in FEATS if k != 'corr_pair'])
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        Xs = (X[ok] - X[ok].mean(0)) / X[ok].std(0); ys = (y[ok] - y[ok].mean()) / y[ok].std()
        A = np.column_stack([np.ones(ok.sum()), Xs])
        beta, *_ = np.linalg.lstsq(A, ys, rcond=None)
        resid = ys - A @ beta
        cov = np.linalg.inv(A.T @ A) * (resid @ resid) / (ok.sum() - A.shape[1])
        se = np.sqrt(np.diag(cov))
        names = [k for k in FEATS if k != 'corr_pair']
        print('   joint OLS (standardised beta, p):')
        out[f'{ind}/W{m}/ols'] = {}
        for j, k in enumerate(names):
            pv = 2 * sps.t.sf(abs(beta[j + 1] / se[j + 1]), ok.sum() - A.shape[1])
            out[f'{ind}/W{m}/ols'][k] = (float(beta[j + 1]), float(pv))
            print(f'     {k:11s} {beta[j + 1]:+.3f}  p={pv:.2g}')
        if m == 0 and NEU:
            print('   neuropil indices vs hub-ness / brightness (pairs):')
            for k in NEU_FEATS:
                for k2 in ['corr_nt', 'corr_tgt', 'nt_bright', 'nt_noise', 'corr_spec']:
                    okk = np.isfinite(F[k]) & np.isfinite(F[k2])
                    rr = np.corrcoef(F[k][okk], F[k2][okk])[0, 1]
                    out[f'{ind}/neucorr/{k}/{k2}'] = float(rr)
                print('     ' + k + ': ' + ', '.join(f"{k2} {out[f'{ind}/neucorr/{k}/{k2}']:+.2f}"
                                                for k2 in ['corr_nt', 'corr_tgt', 'nt_bright', 'nt_noise', 'corr_spec']))
        if m == 0:
            print('   feature correlations with corr_pair:')
            for k in FEATS[1:]:
                okk = np.isfinite(F[k]) & np.isfinite(F['corr_pair'])
                rr = np.corrcoef(F[k][okk], F['corr_pair'][okk])[0, 1]
                out[f'{ind}/fcorr/{k}'] = float(rr)
                print(f'     corr_pair~{k:11s} r={rr:+.3f}')
with open(f'/root/capsule/results/robust_neg_decomp_{CORR_EP}' + ('_neu' if NEU else '') + ('_dc' if DECONV else '') + '.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
