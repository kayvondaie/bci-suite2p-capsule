"""W before (W0) and after (W1) vs correlation, robust connections only, maps
kept separate and selection cross-validated: W0 plotted for pairs selected
(p < P_SEL, positive) in map 1, W1 for pairs selected in map 0. Also the
same-map selection for comparison. Correlation = mean correlation of the
non-target with g's responsive targets, 1 s bins, in spont_pre, BCI, and
spont_post (z within session). W = per-repeat mean (paper windows), z within
session. pf.mean_bin_plot style (5 equal-count bins, mean +- SEM, pooled r).
Writes /root/capsule/results/robust_prepost.json; per indicator (s8 / s6)."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tail_replication_W.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('FEATS = [')])

P_SEL = float(os.environ.get('P_SEL', 0.01))
EPOCHS = ['spont_pre', 'bci', 'spont_post']


def sess_table(mouse, sess):
    folder = learning_folder(mouse, sess)
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    dt = float(data['dt_si']); fs = 1.0 / dt; nb = max(1, int(round(fs)))
    n_cells, n_groups = D.shape
    reps = [repeat_responses_paper(ps_file(folder, k), dt, D) for k in ['photostim', 'photostim2']]
    acts = {'bci': np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float))}
    try:
        sp = load_learning_spont(folder)
        for n, F in zip(['spont_pre', 'spont_post'], sp):
            acts[n] = np.nan_to_num(spont_dff(F))
    except FileNotFoundError:
        pass
    C = {k: corr_mat(bin_frames(a, nb)) for k, a in acts.items() if a.shape[0] == n_cells}
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ~is_out[:, None]
    rows = {k: [] for k in ['g', 'dist', 'mu0', 'mu1', 't0', 't1', 'p0', 'p1'] + EPOCHS}
    for g in range(n_groups):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        nt = np.where(NT[:, g])[0]
        if cl.size == 0 or nt.size == 0:
            continue
        for m in range(2):
            mu, t, p = tstats(reps[m][0], reps[m][1], g, nt)
            rows[f'mu{m}'] += list(mu); rows[f't{m}'] += list(t); rows[f'p{m}'] += list(p)
        rows['g'] += [g] * nt.size; rows['dist'] += list(D[nt, g])
        for k in EPOCHS:
            rows[k] += list(C[k][np.ix_(cl, nt)].mean(axis=0)) if k in C else [np.nan] * nt.size
    T = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    ok = np.isfinite(T['p0']) & np.isfinite(T['p1'])
    T = {k: v[ok] for k, v in T.items()}
    for m in range(2):
        T[f'W{m}'] = z(T[f'mu{m}'])
    for k in EPOCHS:
        T[k] = z(T[k]) if np.isfinite(T[k]).any() else T[k]
    return T


def meanbin(x, y, sess_id):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, sid = x[ok], y[ok], sess_id[ok]
    if x.size < 20:
        return None
    e = np.quantile(x, np.linspace(0, 1, 6))
    b = np.clip(np.digitize(x, e[1:-1]), 0, 4)
    r, p = sps.pearsonr(x, y)
    n_sig = 0; n_s = 0
    for s in np.unique(sid):
        m = sid == s
        if m.sum() >= 8:
            n_s += 1
            rr, pp = sps.pearsonr(x[m], y[m])
            n_sig += int(rr > 0 and pp < 0.05)
    return dict(bx=[float(x[b == k].mean()) for k in range(5)], by=[float(y[b == k].mean()) for k in range(5)],
                be=[float(y[b == k].std() / np.sqrt(np.sum(b == k))) for k in range(5)],
                r=float(r), p=float(p), n=int(x.size), n_sig=n_sig, n_sess=n_s)


pool = {'s6': [], 's8': []}
for si, (mouse, sess) in enumerate(LEARNING_SESSIONS):
    if (mouse, sess) in EXCLUDE:
        continue
    try:
        T = sess_table(mouse, sess)
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
    sel = {m: (cat[f'p{m}'] < P_SEL) & (cat[f't{m}'] > 0) for m in range(2)}
    for m in range(2):
        for mode, s_ in [('cross', sel[1 - m]), ('same', sel[m])]:
            for k in EPOCHS:
                res = meanbin(cat[k][s_], cat[f'W{m}'][s_], cat['sid'][s_])
                if res:
                    out[f'{ind}/W{m}/{mode}/{k}'] = res
                    print(f"{ind} W{m} sel={mode:5s} {k:10s} r={res['r']:+.3f} p={res['p']:.2g} n={res['n']} "
                          f"sig+ {res['n_sig']}/{res['n_sess']}")
with open('/root/capsule/results/robust_prepost.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
