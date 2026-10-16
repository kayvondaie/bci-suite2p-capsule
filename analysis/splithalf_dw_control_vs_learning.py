#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Split-half reliability of connectivity change, water control vs BCI learning.
# No activity is used: only the two photostim maps, identically in both
# conditions.
#
# Each map's repeats of each group are split into odd / even halves, giving two
# independent estimates of every pair's response (mean dF/F, Fstim windows as
# in fig4_dcorr_dw_control_vs_learning.py):
#     dW_h = W1_h - W0_h,   h = odd, even
# Measurement noise is independent across halves, so cov(dW_odd, dW_even)
# estimates the variance of the TRUE change; unconnected / unchanged pairs add
# nothing in expectation (no pair selection needed).
#
# The cells x groups dW matrix (non-target entries only) is decomposed per half
# with a two-way additive model  dW_ij = mu + a_i + b_j + e_ij :
#     a_i  postsynaptic-cell component (cell changes its response to all groups:
#          excitability / drift)
#     b_j  group component (all non-targets change their response to group j:
#          change in target drive)
#     e_ij pair-specific residual (synapse-specific change)
# Reliable variance of each component = covariance of that component between
# halves. Each is normalised by the reliable variance of the same component of
# the mean map W = (W0 + W1)/2, i.e. "reliable change relative to reliable
# connectivity".
#
# Robustness: each cell's per-repeat responses are divided by that cell's SD
# across all repeats of the map (noise-normalised response, cf. paper-1 W~)
# and clipped at +/- CLIP_SD, identically in both conditions. Sessions whose
# pair-specific W split-half r < MIN_PAIR_R_W (no measurable pair-level
# connectivity) are left out of the pooled stats.
#
# Stats (pooled, no resampling): per-unit products of the normalised half
# estimates, pooled over all sessions of a condition; mean ± SE, z-test vs 0
# and between conditions. Putative SST outliers excluded (as postsynaptic cells,
# and groups dropped if a responsive target is one).
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'fig4_dcorr_dw_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('# CELL 4: Per-session regressors and fits')])

NONTARG_FIT = (30, 1000)      # pairs used for the two-way decomposition
NONTARG_CLOSE = 100           # pair-specific component also reported < 100 um
MIN_HALF_REPEATS = 5
CLIP_SD = 5
MIN_PAIR_R_W = 0.05
N_ITER = 30

#%% ============================================================================
# CELL 2: Loading (photostim only)
# ============================================================================
def load_maps(mouse, session, condition):
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy',
                     allow_pickle=True).item()
        dt = 1.0 / float(sh['metadata']['hRoiManager']['scanFrameRate'])
    else:
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
        dt = float(data['dt_si'])
    AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
    k = np.nanmedian(stimDist / np.where(data['photostim2']['stimDist'] == 0, np.nan,
                                         data['photostim2']['stimDist']))
    D = [data['photostim']['stimDist'] * k, data['photostim2']['stimDist'] * k]
    halves = []                                   # [map][half] -> cells x groups
    for pi, pkey in enumerate(['photostim', 'photostim2']):
        r, s, _ = repeat_responses(ps_file(folder, pkey), mouse, dt, D[pi])
        sd = np.nanstd(r, axis=1, keepdims=True)
        sd[~(sd > 0)] = np.nan
        r = np.clip(r / sd, -CLIP_SD, CLIP_SD)
        n_groups = D[pi].shape[1]
        H = np.full((2, r.shape[0], n_groups), np.nan)
        for g in range(n_groups):
            idx = np.where(s == g)[0]
            for h in range(2):
                R = r[:, idx[h::2]]
                n = np.sum(np.isfinite(R), axis=1)
                ok = n >= MIN_HALF_REPEATS
                if ok.any():
                    H[h, ok, g] = np.nanmean(R[ok], axis=1)
        halves.append(H)
    return AMP, D, halves


#%% ============================================================================
# CELL 3: Two-way decomposition and reliable variances
# ============================================================================
def two_way(X, M):
    """Additive fit X_ij = mu + a_i + b_j + e_ij over entries where M is True."""
    Xm = np.where(M, X, 0.0)
    nr, nc = M.sum(axis=1), M.sum(axis=0)
    mu = Xm.sum() / M.sum()
    a, b = np.zeros(X.shape[0]), np.zeros(X.shape[1])
    for _ in range(N_ITER):
        a = np.where(nr > 0, (np.where(M, X - mu - b[None, :], 0).sum(axis=1)) / np.maximum(nr, 1), 0)
        b = np.where(nc > 0, (np.where(M, X - mu - a[:, None], 0).sum(axis=0)) / np.maximum(nc, 1), 0)
    e = np.where(M, X - mu - a[:, None] - b[None, :], np.nan)
    return a, b, e


def rel_cov(u, v):
    u, v = u - u.mean(), v - v.mean()
    return np.mean(u * v)


def session_splithalf(mouse, session, condition):
    AMP, D, H = load_maps(mouse, session, condition)
    n_cells, n_groups = D[0].shape
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[np.minimum(D[0], D[1]) < NONTARG_FIT[0]] = np.nan
    mean_nt = np.nanmean(amp_avg, axis=1)
    is_out = mean_nt > np.nanmean(mean_nt) + N_SD_OUTLIER * np.nanstd(mean_nt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False

    dmin, dmax = np.minimum(D[0], D[1]), np.maximum(D[0], D[1])
    M = (dmin > NONTARG_FIT[0]) & (dmax < NONTARG_FIT[1]) & ~is_out[:, None]
    for g in range(n_groups):
        resp_t = np.where((D[0][:, g] < 10) & (D[1][:, g] < 10) &
                          (AMP[0][:, g] > amp_thr) & (AMP[1][:, g] > amp_thr))[0]
        if resp_t.size == 0 or np.any(is_out[resp_t]):
            M[:, g] = False
    for X in [H[0][0], H[0][1], H[1][0], H[1][1]]:
        M &= np.isfinite(X)
    # keep cells / groups with enough entries for stable row / column effects
    for _ in range(3):
        M &= (M.sum(axis=1) >= 5)[:, None]
        M &= (M.sum(axis=0) >= 5)[None, :]
    if M.sum() < 200:
        raise RuntimeError(f'only {M.sum()} pairs')
    rows, cols = M.any(axis=1), M.any(axis=0)

    comp = {}
    for name, Xh in [('dW', [H[1][h] - H[0][h] for h in range(2)]),
                     ('W', [(H[1][h] + H[0][h]) / 2 for h in range(2)])]:
        fits = [two_way(Xh[h], M) for h in range(2)]
        comp[name] = dict(
            row=(fits[0][0][rows], fits[1][0][rows]),
            col=(fits[0][1][cols], fits[1][1][cols]),
            pair=(fits[0][2][M], fits[1][2][M]),
            total=(Xh[0][M], Xh[1][M]))
    close = (dmin[M] < NONTARG_CLOSE)

    out = dict(mouse=mouse, session=session, condition=condition,
               n_pairs=int(M.sum()), n_cells=int(rows.sum()), n_groups=int(cols.sum()),
               n_close=int(close.sum()), rel={}, prod={})
    for k in ['total', 'row', 'col', 'pair']:
        rW = rel_cov(*comp['W'][k])
        rD = rel_cov(*comp['dW'][k])
        out['rel'][k] = dict(dW=rD, W=rW, ratio=rD / rW if rW > 0 else np.nan)
        # normalised per-unit products for pooling: mean of products = ratio
        if rW > 0:
            u, v = comp['dW'][k]
            out['prod'][k] = (u - u.mean()) * (v - v.mean()) / rW
    # close pairs: residuals from the all-pair decomposition, normalised by the
    # close pairs' own reliable pair-specific W variance
    uc, vc = (x[close] for x in comp['dW']['pair'])
    wc = [x[close] for x in comp['W']['pair']]
    rWc = rel_cov(*wc) if close.sum() > 20 else np.nan
    rDc = rel_cov(uc, vc) if close.sum() > 20 else np.nan
    out['rel']['pair_close'] = dict(dW=rDc, W=rWc, ratio=rDc / rWc if rWc > 0 else np.nan)
    if rWc > 0:
        out['prod']['pair_close'] = (uc - uc.mean()) * (vc - vc.mean()) / rWc
        out['r_pair_W_close'] = float(np.corrcoef(*wc)[0, 1])
    if out['rel']['pair']['W'] > 0:
        u, v = comp['dW']['pair']
        # split-half correlation of the pair-specific change itself
        out['r_pair'] = float(np.corrcoef(u, v)[0, 1])
        out['r_pair_W'] = float(np.corrcoef(*comp['W']['pair'])[0, 1])
    return out


all_results = []
for condition, sessions in [('control', CONTROL_SESSIONS), ('learning', LEARNING_SESSIONS)]:
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            r = session_splithalf(mouse, session, condition)
            all_results.append(r)
            print(f"  {r['n_pairs']} pairs ({r['n_close']} < {NONTARG_CLOSE} um), "
                  f"{r['n_cells']} cells, {r['n_groups']} groups")
            for k in ['total', 'row', 'col', 'pair', 'pair_close']:
                q = r['rel'][k]
                print(f"    {k:5s}: reliable var dW={q['dW']:+.3e}  W={q['W']:+.3e}  "
                      f"dW/W={q['ratio']:+.3f}")
            if 'r_pair' in r:
                print(f"    split-half r, pair-specific: dW={r['r_pair']:+.3f}  W={r['r_pair_W']:+.3f}")
        except FileNotFoundError as e:
            print(f"  Skipping -- file not found ({e}).")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

np.save(os.path.join(RESULTS_DIR, 'splithalf_dw_control_vs_learning.npy'), all_results,
        allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 4: Pooled stats
# ============================================================================
COMPONENTS = ['total', 'row', 'col', 'pair', 'pair_close']
for cond in ['control', 'learning']:
    rr = [r for r in all_results if r['condition'] == cond]
    ok = [r for r in rr if r.get('r_pair_W', 0) >= MIN_PAIR_R_W]
    print(f"QC {cond}: {len(ok)}/{len(rr)} sessions with pair-specific W split-half r >= "
          f"{MIN_PAIR_R_W}: " + ', '.join(f"{r['mouse']} {r['session']} ({r['r_pair_W']:.2f})" for r in ok))
summary = {}
print("\n======== reliable change / reliable connectivity (pooled products) ========")
for mg, mice in MOUSE_GROUPS.items():
    print(f"\n### {mg}")
    for k in COMPONENTS:
        st = {}
        for cond in ['control', 'learning']:
            ss = [r for r in all_results if r['condition'] == cond and r['mouse'] in mice
                  and k in r['prod'] and r.get('r_pair_W', 0) >= MIN_PAIR_R_W]
            if not ss:
                continue
            x = np.concatenate([r['prod'][k] for r in ss])
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
            ratios = [r['rel'][k]['ratio'] for r in ss]
            st[cond] = dict(m=m, se=se, n=len(x), n_sess=len(ss),
                            n_pos=int(np.sum(np.array(ratios) > 0)),
                            med_ratio=float(np.nanmedian(ratios)))
            print(f"  {k:10s} {cond:8s}: {m:+.4f} ± {se:.4f} (p vs 0 = "
                  f"{2 * sps.norm.sf(abs(m / se)):.2g}), {len(x)} units, {len(ss)} sess, "
                  f"session ratio >0: {st[cond]['n_pos']}/{len(ss)}, median {st[cond]['med_ratio']:+.3f}")
        if len(st) == 2:
            z = (st['learning']['m'] - st['control']['m']) / np.sqrt(
                st['learning']['se'] ** 2 + st['control']['se'] ** 2)
            st['p_diff'] = 2 * sps.norm.sf(abs(z))
            print(f"  {k:10s} learning - control = {st['learning']['m'] - st['control']['m']:+.4f}, "
                  f"z={z:+.2f}, p={st['p_diff']:.2g}")
        summary[mg, k] = st

#%% ============================================================================
# CELL 5: Figure
# ============================================================================
fig, axes = plt.subplots(1, len(COMPONENTS), figsize=(16, 4), sharey=False)
labels = {'total': 'total', 'row': 'postsynaptic\ncell', 'col': 'stim group',
          'pair': 'pair-specific', 'pair_close': 'pair-specific\n< 100 um'}
for ax, k in zip(axes, COMPONENTS):
    st = summary.get(('combined', k), {})
    for xi, (cond, color) in enumerate([('control', '#3498db'), ('learning', '#e74c3c')]):
        if cond not in st:
            continue
        ax.bar(xi, st[cond]['m'], yerr=st[cond]['se'], color=color, alpha=0.7, capsize=4)
        ratios = [r['rel'][k]['ratio'] for r in all_results
                  if r['condition'] == cond and k in r['prod']
                  and r.get('r_pair_W', 0) >= MIN_PAIR_R_W]
        rng = np.random.default_rng(0)
        ax.plot(xi + rng.uniform(-0.15, 0.15, len(ratios)), ratios, 'k.', alpha=0.5)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['control', 'learning'])
    ax.set_title(f"{labels[k]}\np={st.get('p_diff', np.nan):.2g}")
axes[0].set_ylabel('reliable dW variance / reliable W variance')
fig.suptitle('Split-half reliable connectivity change, all mice pooled (bars: pooled mean ± SE; '
             'dots: sessions)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_splithalf_dw_control_vs_learning.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
