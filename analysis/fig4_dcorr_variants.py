#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Paper-1 Fig. 4h (Delta-correlation -> Delta-W) with different definitions of
# Delta-correlation from the BCI training activity. Photostim metric (W~, dW~,
# masks, outlier exclusion) is identical to fig4_dcorr_dw_control_vs_learning.py,
# whose CELLS 1-3 (config + loading) are executed here.
#
# Delta-correlation variants (C0 = early, C1 = late):
#   cont_halves   continuous activity (BCI df_closedloop / control spont), halves
#   cont_slope    continuous activity, OLS slope of C over 8 equal blocks
#   trial_halves  trial-aligned F, first vs second half of trials
#   trial_first40 trial-aligned F, trials 1-20 vs 21-40 (paper-1 trial range)
#   trial_slope   trial-aligned F, OLS slope of C over 10-trial windows (step 5)
#   signal_halves correlation of the trial-averaged response (per half)
#   noise_halves  correlation of residuals around the trial average (per half)
#   prepost       spont_post vs spont_pre (water controls and learning_spont)
# cont_* also exist for water controls (spont); trial_* are learning only.
# learning_spont: learning sessions with their own spont_pre / spont_post
# (suite2p spont folders, iscell cells, 10th-percentile F0) -> 'prepost' with
# activity matched to the water sessions; only what happened in between differs.
# Each pair variant gives two regressors:
#   paper : sum_j (C1_ij W~1_j - C0_ij W~0_j)       (paper-1 definition)
#   dC    : sum_j (C1_ij - C0_ij) (W~0_j + W~1_j)/2 (correlation change only)
#   CdW   : sum_j (C0_ij + C1_ij)/2 (W~1_j - W~0_j)   (correlation-weighted
#           target change; paper = dC + CdW exactly)
#   dCu   : mean_j (C1_ij - C0_ij) over responsive targets, unweighted, with
#           the 3-factor target definition (< 10 um, AMP > 0.1 in both maps);
#           no target-amplitude weights, so no target-change term at all
# joint fits: dW ~ dC + CdW + target dW, to see which part of the paper
# regressor carries the effect once the summed target change is regressed out.
# Slopes give dC only (S_ij in place of C1 - C0, per unit session time).
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fig4_dcorr_dw_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('# CELL 4: Per-session regressors and fits')])

N_BLOCKS = 8
TRIAL_WIN = 10
TRIAL_STEP = 5
VARIANTS = ['cont_halves', 'cont_slope', 'trial_halves', 'trial_first40',
            'trial_slope', 'signal_halves', 'noise_halves', 'prepost']
# SPONT_MAX_MIN env var: cap every spont epoch at this many minutes, identically
# in water and learning sessions (pre: last N min, post: first N min, i.e. the
# parts closest to the intervening BCI / water period)
SPONT_MAX_MIN = float(os.environ['SPONT_MAX_MIN']) if os.environ.get('SPONT_MAX_MIN') else None


def cap_epoch(Fs, fs, which):
    if SPONT_MAX_MIN is None:
        return Fs
    n = int(round(SPONT_MAX_MIN * 60 * fs))
    return Fs[:, -n:] if which == 0 else Fs[:, :n]
TAG = '' if SPONT_MAX_MIN is None else f'_spont{SPONT_MAX_MIN:g}min'
RUN_CONDITIONS = os.environ.get('CONDITIONS', 'learning,control,learning_spont').split(',')
SPONT_FOLDER_PAIRS = [('suite2p_spont_pre', 'suite2p_spont_post'),
                      ('suite2p_spont', 'suite2p_spont2')]


def spont_epochs(mouse, session, condition):
    """dF/F (10th-percentile F0) of the spont epochs before / after, and fs."""
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy',
                     allow_pickle=True).item()
        fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            raw = [np.asarray(f[k][:], dtype=float) for k in ['spont_pre', 'spont_post']]
    else:
        folder = learning_folder(mouse, session)
        fs = 1.0 / float(ddct.load_hdf5(folder, ['dt_si'], [])['dt_si'])
        raw = None
        for pair in SPONT_FOLDER_PAIRS:
            dirs = [os.path.join(folder, n, 'plane0') for n in pair]
            if all(os.path.exists(os.path.join(d, 'F.npy')) for d in dirs):
                raw = []
                for d in dirs:
                    F = np.load(os.path.join(d, 'F.npy'), allow_pickle=True)
                    isc = np.load(os.path.join(d, 'iscell.npy'), allow_pickle=True)
                    raw.append(np.asarray(F[np.asarray(isc)[:, 0] == 1], dtype=float))
                break
        if raw is None:
            raise FileNotFoundError(f'no spont pre/post pair in {folder}')
    out = []
    for ei, Fs in enumerate(raw):
        Fs = cap_epoch(Fs, fs, ei)
        F0 = np.nanpercentile(Fs, 10, axis=1, keepdims=True)
        F0[F0 <= 0] = np.nan
        out.append(np.nan_to_num((Fs - F0) / F0, nan=0.0))
    return out, fs

#%% ============================================================================
# CELL 2: Correlation helpers
# ============================================================================
def corr_cols(x):
    """Pearson correlation of rows of x (cells x samples). Columns that are
    mostly NaN (past the end of a trial) are dropped; remaining NaNs are set to
    the cell's mean so they add nothing to the covariance."""
    x = x[:, np.mean(np.isfinite(x), axis=0) > 0.5]
    mu = np.nanmean(x, axis=1, keepdims=True)
    x = np.where(np.isfinite(x), x, mu) - mu
    x = np.nan_to_num(x)
    sd = np.sqrt((x ** 2).mean(axis=1))
    sd[sd == 0] = np.inf
    x = x / sd[:, None]
    C = (x @ x.T) / x.shape[1]
    np.fill_diagonal(C, 0.0)
    return C


def trial_cols(F, idx):
    """frames x cells x trials -> cells x (frames * trials)."""
    X = F[:, :, idx]
    return X.transpose(1, 0, 2).reshape(X.shape[1], -1)


def slope_matrix(Cs):
    t = np.linspace(0, 1, len(Cs))
    t = t - t.mean()
    return sum(ti * C for ti, C in zip(t, Cs)) / np.sum(t ** 2)


def correlation_variants(segs, F):
    out = {}
    h = len(segs) // 2
    out['cont_halves'] = ('pair', corr_matrix(segs[:h]), corr_matrix(segs[h:2 * h]))
    blocks = np.array_split(np.arange(len(segs)), N_BLOCKS)
    out['cont_slope'] = ('slope', slope_matrix([corr_matrix([segs[k] for k in b])
                                                for b in blocks]))
    if F is None:
        return out
    n_tr = F.shape[2]
    ht = n_tr // 2
    out['trial_halves'] = ('pair', corr_cols(trial_cols(F, np.arange(ht))),
                           corr_cols(trial_cols(F, np.arange(ht, 2 * ht))))
    n40 = min(n_tr, 40) // 2
    out['trial_first40'] = ('pair', corr_cols(trial_cols(F, np.arange(n40))),
                            corr_cols(trial_cols(F, np.arange(n40, 2 * n40))))
    starts = np.arange(0, n_tr - TRIAL_WIN + 1, TRIAL_STEP)
    out['trial_slope'] = ('slope', slope_matrix(
        [corr_cols(trial_cols(F, np.arange(s, s + TRIAL_WIN))) for s in starts]))
    sig, noi = [], []
    for idx in [np.arange(ht), np.arange(ht, 2 * ht)]:
        X = F[:, :, idx]
        psth = np.nanmean(X, axis=2)                      # frames x cells
        sig.append(corr_cols(psth.T))
        noi.append(corr_cols(trial_cols(X - psth[:, :, None], np.arange(len(idx)))))
    out['signal_halves'] = ('pair', sig[0], sig[1])
    out['noise_halves'] = ('pair', noi[0], noi[1])
    return out


#%% ============================================================================
# CELL 3: Per-session regressors and fits
# ============================================================================
def ols(y, X):
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    cov = (resid @ resid / dof) * np.linalg.pinv(A.T @ A)
    t = beta / np.sqrt(np.diag(cov))
    return beta[1:], t[1:], 2 * sps.t.sf(np.abs(t[1:]), dof)


def zs(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def session_fit(mouse, session, condition):
    S = load_session(mouse, session, 'learning' if condition == 'learning_spont' else condition)
    F = None
    if condition == 'learning':
        folder = learning_folder(mouse, session)
        F = np.asarray(ddct.load_hdf5(folder, ['F'], [])['F'], dtype=float)
    CV = correlation_variants(S['segs'], F) if condition != 'learning_spont' else {}
    del F, S['segs']
    if condition in ('control', 'learning_spont'):
        eps, fs = spont_epochs(mouse, session, condition)
        if eps[0].shape[0] != S['MU'][0].shape[0]:
            raise RuntimeError(f"spont has {eps[0].shape[0]} cells, photostim {S['MU'][0].shape[0]}")
        CV['prepost'] = ('pair', corr_matrix(segments(eps[0], fs)),
                         corr_matrix(segments(eps[1], fs)))
        spont_min = [e.shape[1] / fs / 60 for e in eps]
    else:
        spont_min = None

    AMP, D, MU, SD = S['AMP'], S['D'], S['MU'], S['SD']
    Wt = [MU[0] / SD[0], MU[1] / SD[1]]
    dWt = (MU[1] - MU[0]) / np.sqrt(SD[0] ** 2 + SD[1] ** 2)
    n_groups = Wt[0].shape[1]
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[np.minimum(D[0], D[1]) < NONTARG_UM[0]] = np.nan
    mean_nt = np.nanmean(amp_avg, axis=1)
    is_out = mean_nt > np.nanmean(mean_nt) + N_SD_OUTLIER * np.nanstd(mean_nt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False

    keys = [(v, form) for v, c in CV.items()
            for form in (['paper', 'dC', 'CdW', 'dCu'] if c[0] == 'pair' else ['dC', 'dCu'])]
    rows = {k: [] for k in keys}
    y_list, c_list, n_used = [], [], 0
    for g in range(n_groups):
        # outliers carry no opsin: never targets, but keep the group
        tg = np.where((D[0][:, g] < TARGET_UM) & (D[1][:, g] < TARGET_UM) & ~is_out)[0]
        if tg.size == 0:
            continue
        nt = np.where((D[0][:, g] > NONTARG_UM[0]) & (D[0][:, g] < NONTARG_UM[1]) &
                      (D[1][:, g] > NONTARG_UM[0]) & (D[1][:, g] < NONTARG_UM[1]) &
                      ~is_out & np.isfinite(dWt[:, g]))[0]
        if nt.size == 0:
            continue
        w0 = np.nan_to_num(Wt[0][tg, g])
        w1 = np.nan_to_num(Wt[1][tg, g])
        wbar = (w0 + w1) / 2
        tg3 = np.where((D[0][:, g] < 10) & (D[1][:, g] < 10) & ~is_out &
                       (AMP[0][:, g] > amp_thr) & (AMP[1][:, g] > amp_thr))[0]
        y_list.append(dWt[nt, g])
        c_list.append(np.full(nt.size, np.nansum(dWt[tg, g])))
        for v, c in CV.items():
            if c[0] == 'pair':
                C0, C1 = c[1][np.ix_(nt, tg)], c[2][np.ix_(nt, tg)]
                rows[v, 'paper'].append(C1 @ w1 - C0 @ w0)
                rows[v, 'dC'].append((C1 - C0) @ wbar)
                rows[v, 'CdW'].append(((C0 + C1) / 2) @ (w1 - w0))
                dCu = (c[2][np.ix_(nt, tg3)] - c[1][np.ix_(nt, tg3)]).mean(axis=1)
            else:
                rows[v, 'dC'].append(c[1][np.ix_(nt, tg)] @ wbar)
                dCu = c[1][np.ix_(nt, tg3)].mean(axis=1)
            rows[v, 'dCu'].append(dCu if tg3.size else np.full(nt.size, np.nan))
        n_used += 1
    if n_used == 0:
        raise RuntimeError('no valid groups')
    y_raw, c_raw = np.concatenate(y_list), np.concatenate(c_list)
    y, c = zs(y_raw), zs(c_raw)
    res = dict(mouse=mouse, session=session, condition=condition, n_groups=n_used, spont_min=spont_min,
               n_pairs=len(y), fits={}, joint={}, raw={})
    for k in keys:
        x_raw = np.concatenate(rows[k])
        ok = np.isfinite(x_raw)
        if ok.sum() < 20:
            continue
        yk, ck, x = zs(y_raw[ok]), zs(c_raw[ok]), zs(x_raw[ok])
        b, t, p = ols(yk, np.column_stack([x, ck]))
        res['fits'][k] = dict(beta=b[0], p=p[0], n=int(ok.sum()),
                              suff=_suff(yk, np.column_stack([x, ck])))
        if k[0] == 'cont_halves':
            res['raw'][k] = dict(y=yk, x=x, c=ck)
    for v, cv in CV.items():
        if cv[0] != 'pair':
            continue
        A = np.column_stack([zs(np.concatenate(rows[v, 'dC'])),
                             zs(np.concatenate(rows[v, 'CdW'])), c])
        b, t, p = ols(y, A)
        res['joint'][v] = dict(beta=b, p=p, suff=_suff(y, A))
    return res


def _suff(y, A):
    return A.T @ A, A.T @ y, y @ y, len(y)


all_results = []
for condition, sessions in [(c, s) for c, s in [('learning', LEARNING_SESSIONS),
                                                ('control', CONTROL_SESSIONS),
                                                ('learning_spont', LEARNING_SESSIONS)]
                            if c in RUN_CONDITIONS]:
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            r = session_fit(mouse, session, condition)
            all_results.append(r)
            print(f"  {r['n_groups']} groups, {r['n_pairs']} pairs, spont min: {r['spont_min']}")
            for (v, form), f in r['fits'].items():
                print(f"    {v:14s} {form:5s} beta={f['beta']:+.4f} p={f['p']:.3g}")
        except FileNotFoundError as e:
            print(f"  Skipping -- file not found ({e}).")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

np.save(os.path.join(RESULTS_DIR, f'fig4_dcorr_variants{TAG}.npy'), all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 4: Pooled stats per variant
# ============================================================================
def pooled_fit(sufs):
    XtX = sum(s[0] for s in sufs)
    Xty = sum(s[1] for s in sufs)
    yty = sum(s[2] for s in sufs)
    n = sum(s[3] for s in sufs)
    beta = np.linalg.solve(XtX, Xty)
    s2 = (yty - beta @ Xty) / (n - len(beta))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    return beta[0], se[0], 2 * sps.norm.sf(abs(beta[0] / se[0])), n


def pooled_diff_p(sufs_c, sufs_l, col=0):
    """Learning - control difference in one pooled model with condition-
    specific coefficients and a common residual variance (equivalent to a
    condition x regressor interaction); Wald p, no resampling."""
    rss, n, k, var, b = 0.0, 0, 0, 0.0, []
    for sufs in (sufs_c, sufs_l):
        XtX = sum(s[0] for s in sufs)
        Xty = sum(s[1] for s in sufs)
        beta = np.linalg.solve(XtX, Xty)
        rss += sum(s[2] for s in sufs) - beta @ Xty
        n += sum(s[3] for s in sufs)
        k += len(beta)
        var += np.linalg.inv(XtX)[col, col]
        b.append(beta[col])
    s2 = rss / (n - k)
    z = (b[1] - b[0]) / np.sqrt(s2 * var)
    return b[1] - b[0], 2 * sps.norm.sf(abs(z))


def pooled_fit_all(sufs):
    XtX = sum(s[0] for s in sufs)
    Xty = sum(s[1] for s in sufs)
    yty = sum(s[2] for s in sufs)
    n = sum(s[3] for s in sufs)
    beta = np.linalg.solve(XtX, Xty)
    s2 = (yty - beta @ Xty) / (n - len(beta))
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    return beta, se, 2 * sps.norm.sf(np.abs(beta / se))


qc = [r for r in all_results if r['n_groups'] >= MIN_GROUPS]
summary = {}
print("\n================ POOLED dW ~ dA + target dW ================")
for mg, mice in MOUSE_GROUPS.items():
    print(f"\n### {mg}")
    for cond in ['learning', 'learning_spont', 'control']:
        sess = [r for r in qc if r['condition'] == cond and r['mouse'] in mice]
        keys = sorted({k for r in sess for k in r['fits']},
                      key=lambda k: (VARIANTS.index(k[0]), k[1] != 'paper'))
        for k in keys:
            ss = [r for r in sess if k in r['fits']]
            b, se, p, n = pooled_fit([r['fits'][k]['suff'] for r in ss])
            nsig = sum(r['fits'][k]['p'] < 0.05 and r['fits'][k]['beta'] > 0 for r in ss)
            npos = sum(r['fits'][k]['beta'] > 0 for r in ss)
            summary[mg, cond, k] = dict(beta=b, se=se, p=p, n_sess=len(ss), nsig=nsig, npos=npos)
            print(f"  {cond:8s} {k[0]:14s} {k[1]:5s} beta={b:+.4f} ± {se:.4f} p={p:.2g} | "
                  f"sess beta>0 {npos}/{len(ss)}, beta>0&p<.05 {nsig}/{len(ss)}")

print("\n================ JOINT dW ~ dC + CdW + target dW (pooled) ================")
for mg, mice in MOUSE_GROUPS.items():
    print(f"\n### {mg}")
    for cond in ['learning', 'learning_spont', 'control']:
        sess = [r for r in qc if r['condition'] == cond and r['mouse'] in mice]
        for v in VARIANTS:
            ss = [r for r in sess if v in r.get('joint', {})]
            if not ss:
                continue
            b, se, p = pooled_fit_all([r['joint'][v]['suff'] for r in ss])
            print(f"  {cond:8s} {v:14s} dC={b[0]:+.4f} (p={p[0]:.2g})  CdW={b[1]:+.4f} "
                  f"(p={p[1]:.2g})  target dW={b[2]:+.4f} (p={p[2]:.2g})  n={len(ss)}")
    for v in VARIANTS:
        sl = [r['joint'][v]['suff'] for r in qc if r['condition'] == 'learning'
              and r['mouse'] in mice and v in r.get('joint', {})]
        sc = [r['joint'][v]['suff'] for r in qc if r['condition'] == 'control'
              and r['mouse'] in mice and v in r.get('joint', {})]
        if sl and sc:
            d0, p0 = pooled_diff_p(sc, sl, 0)
            d1, p1 = pooled_diff_p(sc, sl, 1)
            print(f"  learning - control {v:14s} dC diff={d0:+.4f} (p={p0:.2g})  "
                  f"CdW diff={d1:+.4f} (p={p1:.2g})")

print("\n================ learning - control, single-regressor fits (pooled) ================")
for mg, mice in MOUSE_GROUPS.items():
    for lv, vs in [('learning', ['cont_halves', 'cont_slope']), ('learning_spont', ['prepost'])]:
        for k in [(v, f) for v in vs for f in ['paper', 'dC', 'CdW', 'dCu']]:
            sl = [r['fits'][k]['suff'] for r in qc if r['condition'] == lv
                  and r['mouse'] in mice and k in r['fits']]
            sc = [r['fits'][k]['suff'] for r in qc if r['condition'] == 'control'
                  and r['mouse'] in mice and k in r['fits']]
            if sl and sc:
                d, p = pooled_diff_p(sc, sl)
                print(f"  {mg:13s} {k[0]:12s} {k[1]:5s} {lv} - control = {d:+.4f}, p={p:.2g} "
                      f"({len(sl)} vs {len(sc)} sess)")

#%% ============================================================================
# CELL 5: Figure — pooled beta per variant
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
for ax, mg in zip(axes, MOUSE_GROUPS):
    labels, xi = [], 0
    for v in VARIANTS:
        for form in ['paper', 'dC', 'CdW', 'dCu']:
            for cond, color, dx in [('learning', '#e74c3c', -0.2), ('learning_spont', '#eda100', 0.0),
                                    ('control', '#3498db', 0.2)]:
                s = summary.get((mg, cond, (v, form)))
                if s is None:
                    continue
                ax.errorbar(xi + dx, s['beta'], yerr=1.96 * s['se'], fmt='o', color=color,
                            capsize=3, mfc=color if s['p'] < 0.05 else 'white')
            if any((mg, c, (v, form)) in summary for c in ['learning', 'learning_spont', 'control']):
                labels.append(f'{v}\n{form}')
                xi += 1
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=8)
    ax.set_title(mg)
axes[0].set_ylabel(r'pooled $\beta$ ($\Delta A \rightarrow \Delta W$) ± 95% CI')
axes[0].plot([], [], 'o', color='#e74c3c', label='learning')
axes[0].plot([], [], 'o', color='#3498db', label='water control')
axes[0].legend(fontsize=9)
fig.suptitle('Fig. 4h with different Delta-correlation definitions (filled: p<0.05)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'fig4_dcorr_variants{TAG}.png'), dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
