#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# BCI paper 1, Fig. 4 analysis (correlation -> W, Delta-correlation -> Delta-W)
# applied to water control vs BCI learning sessions, with matched activity
# preprocessing: both conditions are cut into consecutive 10 s pseudo-trials
#   control : spont_pre + spont_post dF/F (photostim maps before/after water)
#   learning: BCI-epoch df_closedloop (photostim maps before/after BCI)
# Within a session, Delta-correlation = correlation over the late half of
# pseudo-trials minus the early half (no day-to-day split exists here).
# learning_matched uses only the first/last K pseudo-trials of the BCI epoch,
# K = median control half length, so correlation estimates have equal noise.
#
# Paper-1 metrics (Methods, "Causal connectivity" / "fitting"):
#   repeat response   r_{i,g,a} = <F>_after - <F>_before   (Fstim, per repeat)
#   PS response       W~_{i,g}  = mean_a / std_a           (>= MIN_REPEATS)
#   Delta PS response dW~_{i,g} = (mu1 - mu0) / sqrt(sd0^2 + sd1^2)
#   targets d < 20 um, non-targets 30 < d < 100 um, in both maps
#   A_rho_{i,g}   = sum_j C_ij W~_{j,g}               control W~_g  = sum_j W~_{j,g}
#   dA_rho_{i,g}  = sum_j (C1_ij W~1_{j,g} - C0_ij W~0_{j,g})   control dW~_g
#   W ~ A_rho + W~_g ;  dW ~ dA_rho + dW~_g   (OLS, Wald test on the first term)
# Putative SST outliers (pan-neuronal mice) excluded as in the other control
# scripts: as non-targets, and groups dropped if a responsive target is one.
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import traceback
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps
import data_dict_create_module_test as ddct
from BCI_data_helpers import compute_amp_from_photostim

CO_DATA = '/root/capsule/data'
LEARNING_ROOTS = [
    f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal',
    f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025',
]
CONTROL_ROOT = os.path.dirname(
    (glob.glob(f'{CO_DATA}/*/731015_2025-01-27') + [f'{CO_DATA}/x/x'])[0])
RESULTS_DIR = '/root/capsule/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

SUBJECT_ID = {'BCI88': '731015', 'BCI93': '740369', 'BCI103': '772414',
              'BCI105': '754303', 'BCI107': '767715'}

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12})
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
PAN_NEURONAL_MICE = ['BCI88', 'BCI93', 'BCI107']
E_ONLY_MICE = ['BCI103', 'BCI105']
MOUSE_GROUPS = {
    'combined': PAN_NEURONAL_MICE + E_ONLY_MICE,
    'E-only': E_ONLY_MICE,
    'pan-neuronal': PAN_NEURONAL_MICE,
}

CONTROL_SESSIONS = [
    ('BCI88',  '012725'), ('BCI93',  '012825'), ('BCI93',  '022125'),
    ('BCI107', '021925'), ('BCI103', '013025'), ('BCI103', '021925'),
    ('BCI105', '013025'),
]
LEARNING_SESSIONS = [
    ('BCI88', '011025'), ('BCI88', '011325'), ('BCI88', '012425'),
    ('BCI88', '012825'), ('BCI88', '013125'),
    ('BCI93', '010925'), ('BCI93', '011025'), ('BCI93', '011325'),
    ('BCI93', '012425'), ('BCI93', '020325'), ('BCI93', '020425'),
    ('BCI107', '013025'), ('BCI107', '013125'), ('BCI107', '020325'),
    ('BCI107', '020625'),
    ('BCI103', '012125'), ('BCI103', '012325'), ('BCI103', '012425'),
    ('BCI103', '012725'), ('BCI103', '012825'), ('BCI103', '012925'),
    ('BCI103', '020325'), ('BCI103', '020425'), ('BCI103', '020525'),
    ('BCI103', '020625'),
    ('BCI105', '012225'), ('BCI105', '012325'), ('BCI105', '012725'),
    ('BCI105', '012925'), ('BCI105', '013125'), ('BCI105', '020325'),
    ('BCI105', '020425'),
]
EXCLUDE = {('BCI88', '013125'), ('BCI107', '013125')}

PSEUDO_SEC = 10        # pseudo-trial length
TARGET_UM = 20         # paper 1 target radius
NONTARG_UM = (30, 100) # paper 1 non-target annulus
MIN_REPEATS = 10
N_SD_OUTLIER = 2
amp_thr = 0.1          # "responsive target" for outlier-group exclusion
MIN_GROUPS = 10
N_PERM = 5000
N_BINS = 6

CONDITION_LABELS = {
    'control': 'Control (water; spont pseudo-trials)',
    'learning': 'Learning (BCI pseudo-trials, halves)',
    'learning_matched': 'Learning (BCI, first/last K pseudo-trials)',
}

#%% ============================================================================
# CELL 3: Loading
# ============================================================================
def control_tag(mouse, session):
    return f"{SUBJECT_ID[mouse]}_20{session[4:6]}-{session[0:2]}-{session[2:4]}"


def learning_folder(mouse, session):
    for r in LEARNING_ROOTS:
        folder = os.path.join(r, mouse, session, 'pophys') + '/'
        if os.path.isdir(folder):
            return folder
    raise FileNotFoundError(f'{mouse} {session}')


def ps_file(folder, pkey):
    files = sorted(glob.glob(folder + 'data_photostim*.h5'))
    if pkey == 'photostim':
        files = [f for f in files if not os.path.basename(f).startswith('data_photostim2')]
    else:
        files = [f for f in files if os.path.basename(f).startswith('data_photostim2')]
    if not files:
        raise FileNotFoundError(f'{pkey} h5 in {folder}')
    return files[0]


def segments(dff, fs):
    """Split cells x frames into consecutive PSEUDO_SEC segments."""
    L = int(round(PSEUDO_SEC * fs))
    n = dff.shape[1] // L
    return [dff[:, k * L:(k + 1) * L] for k in range(n)]


def corr_matrix(segs):
    x = np.concatenate(segs, axis=1)
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).mean(axis=1))
    sd[sd == 0] = np.inf
    x = x / sd[:, None]
    C = (x @ x.T) / x.shape[1]
    np.fill_diagonal(C, 0.0)
    return C


def repeat_responses(path, mouse, dt, stimDist_um):
    """Per-repeat response (cells x repeats) and group index per repeat."""
    with h5py.File(path, 'r') as f:
        favg_raw = f['favg_raw'][:]
        seq = np.asarray(f['seq'][:]).astype(int).ravel() - 1
        artifact = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        artifact = artifact - np.nanmean(artifact[0:4])
        artifact = np.where(artifact > 0.5)[0]
        artifact = artifact[artifact < 40]
        if artifact.size == 0:
            raise RuntimeError(f'no stim artifact in {os.path.basename(path)}')
        after = int(np.floor((0.5 if mouse == 'BCI103' else 0.4) / dt))
        before = int(np.floor(0.2 / dt))
        pre = (max(0, int(artifact[0] - before)), int(artifact[0] - 2))
        post = (int(artifact[-1] + 2), int(artifact[-1] + after))
        win = f['Fstim'][pre[0]:post[1], :, :]          # frames x cells x repeats
    resp_all = (np.nanmean(win[post[0] - pre[0]:], axis=0) -
                np.nanmean(win[:pre[1] - pre[0]], axis=0))  # cells x repeats

    # seq / Fstim alignment: offset maximizing the nearest-cell target response
    n_groups = stimDist_um.shape[1]
    nearest = np.argmin(stimDist_um, axis=0)
    best, best_off, best_sr = -np.inf, 0, (seq, resp_all)
    for off in range(-10, 11):
        if off > 0:
            s, r = seq[:-off], resp_all[:, off:]
        elif off < 0:
            s, r = seq[-off:], resp_all[:, :off]
        else:
            s, r = seq, resp_all
        n = min(len(s), r.shape[1])
        s, r = s[:n], r[:, :n]
        v = np.nanmean([np.nanmean(r[nearest[g], s == g]) for g in range(n_groups)
                        if np.any(s == g)])
        if v > best:
            best, best_off, best_sr = v, off, (s, r)
    s, r = best_sr
    r = r.copy()
    # Neurons targeted on the previous stim still carry elevated activity
    for t in range(1, len(s)):
        if 0 <= s[t - 1] < n_groups:
            r[stimDist_um[:, s[t - 1]] < TARGET_UM, t] = np.nan
    return r, s, best_off


def group_stats(r, s, n_groups):
    n_cells = r.shape[0]
    mu = np.full((n_cells, n_groups), np.nan)
    sd = np.full((n_cells, n_groups), np.nan)
    for g in range(n_groups):
        R = r[:, s == g]
        n = np.sum(np.isfinite(R), axis=1)
        ok = n >= MIN_REPEATS
        if not np.any(ok):
            continue
        mu[ok, g] = np.nanmean(R[ok], axis=1)
        sd[ok, g] = np.nanstd(R[ok], axis=1, ddof=1)
    sd[sd == 0] = np.nan
    return mu, sd


def load_session(mouse, session, condition):
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy',
                     allow_pickle=True).item()
        fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
        segs = []
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            for k in ['spont_pre', 'spont_post']:
                Fs = np.asarray(f[k][:], dtype=float)
                F0 = np.nanpercentile(Fs, 10, axis=1, keepdims=True)
                F0[F0 <= 0] = np.nan
                segs += segments(np.nan_to_num((Fs - F0) / F0, nan=0.0), fs)
    else:
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        fs = 1.0 / float(data['dt_si'])
        segs = segments(np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float),
                                      nan=0.0), fs)
    dt = 1.0 / fs

    # Per-map stimDist in um (compute_amp returns the second map's; rescale)
    um_per_pix = stimDist / np.where(data['photostim2']['stimDist'] == 0, np.nan,
                                     data['photostim2']['stimDist'])
    k = np.nanmedian(um_per_pix)
    D = [data['photostim']['stimDist'] * k, data['photostim2']['stimDist'] * k]

    MU, SD, offs = [], [], []
    for pi, pkey in enumerate(['photostim', 'photostim2']):
        r, s, off = repeat_responses(ps_file(folder, pkey), mouse, dt, D[pi])
        mu, sd = group_stats(r, s, D[pi].shape[1])
        MU.append(mu)
        SD.append(sd)
        offs.append(off)
    return dict(AMP=AMP, D=D, MU=MU, SD=SD, segs=segs, offsets=offs)


#%% ============================================================================
# CELL 4: Per-session regressors and fits
# ============================================================================
def ols(y, X):
    """OLS with intercept; returns coefs, Wald t and p (for X columns)."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = resid @ resid / dof
    cov = s2 * np.linalg.pinv(A.T @ A)
    t = beta / np.sqrt(np.diag(cov))
    p = 2 * sps.t.sf(np.abs(t), dof)
    return beta[1:], t[1:], p[1:]


def zs(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def session_regressors(S, mouse, early, late):
    AMP, D, MU, SD = S['AMP'], S['D'], S['MU'], S['SD']
    Wt = [MU[0] / SD[0], MU[1] / SD[1]]
    dWt = (MU[1] - MU[0]) / np.sqrt(SD[0] ** 2 + SD[1] ** 2)
    n_cells, n_groups = Wt[0].shape
    if len(S['segs']) and S['segs'][0].shape[0] != n_cells:
        raise RuntimeError(f"activity has {S['segs'][0].shape[0]} cells, photostim {n_cells}")

    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[np.minimum(D[0], D[1]) < NONTARG_UM[0]] = np.nan
    mean_nt = np.nanmean(amp_avg, axis=1)
    is_out = mean_nt > np.nanmean(mean_nt) + N_SD_OUTLIER * np.nanstd(mean_nt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False

    C0, C1, Call = corr_matrix(early), corr_matrix(late), corr_matrix(early + late)
    rows = {k: [] for k in ['dW', 'dA', 'dWg', 'W1', 'A1', 'W1g', 'W0', 'A0', 'W0g']}
    n_used = 0
    for g in range(n_groups):
        tg = np.where((D[0][:, g] < TARGET_UM) & (D[1][:, g] < TARGET_UM))[0]
        if tg.size == 0:
            continue
        resp_t = tg[(AMP[0][tg, g] > amp_thr) & (AMP[1][tg, g] > amp_thr)]
        if np.any(is_out[resp_t]):
            continue
        nt = np.where((D[0][:, g] > NONTARG_UM[0]) & (D[0][:, g] < NONTARG_UM[1]) &
                      (D[1][:, g] > NONTARG_UM[0]) & (D[1][:, g] < NONTARG_UM[1]) &
                      ~is_out & np.isfinite(dWt[:, g]) &
                      np.isfinite(Wt[0][:, g]) & np.isfinite(Wt[1][:, g]))[0]
        if nt.size == 0:
            continue
        w0 = np.nan_to_num(Wt[0][tg, g])
        w1 = np.nan_to_num(Wt[1][tg, g])
        rows['dW'].append(dWt[nt, g])
        rows['dA'].append(C1[np.ix_(nt, tg)] @ w1 - C0[np.ix_(nt, tg)] @ w0)
        rows['dWg'].append(np.full(nt.size, np.nansum(dWt[tg, g])))
        rows['W1'].append(Wt[1][nt, g])
        rows['A1'].append(Call[np.ix_(nt, tg)] @ w1)
        rows['W1g'].append(np.full(nt.size, w1.sum()))
        rows['W0'].append(Wt[0][nt, g])
        rows['A0'].append(Call[np.ix_(nt, tg)] @ w0)
        rows['W0g'].append(np.full(nt.size, w0.sum()))
        n_used += 1
    if n_used == 0:
        raise RuntimeError('no valid groups')
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out['n_groups'] = n_used
    out['n_outliers'] = int(is_out.sum())
    return out


def fit_block(R, y, x, ctrl):
    yz, xz, cz = zs(R[y]), zs(R[x]), zs(R[ctrl])
    b, t, p = ols(yz, np.column_stack([xz, cz]))
    return dict(beta=b[0], t=t[0], p=p[0], beta_ctrl=b[1], p_ctrl=p[1],
                y=yz, x=xz, c=cz)


def run_session(mouse, session, condition, S, k_match=None):
    segs = S['segs']
    if condition == 'learning_matched':
        if k_match is None or 2 * k_match > len(segs):
            raise RuntimeError(f'cannot match K={k_match} with {len(segs)} pseudo-trials')
        early, late = segs[:k_match], segs[-k_match:]
    else:
        h = len(segs) // 2
        early, late = segs[:h], segs[h:2 * h]
    if len(early) < 3:
        raise RuntimeError(f'only {len(segs)} pseudo-trials')
    R = session_regressors(S, mouse, early, late)
    res = dict(mouse=mouse, session=session, condition=condition,
               n_groups=R['n_groups'], n_pairs=len(R['dW']), n_half=len(early),
               n_outliers=R['n_outliers'], offsets=S['offsets'])
    res['dW'] = fit_block(R, 'dW', 'dA', 'dWg')     # Fig 4h
    res['W1'] = fit_block(R, 'W1', 'A1', 'W1g')     # Fig 4e, map after activity
    res['W0'] = fit_block(R, 'W0', 'A0', 'W0g')     # Fig 4e, map before activity
    return res


def report(r):
    print(f"  [{r['condition']}] {r['n_groups']} groups, {r['n_pairs']} pairs, "
          f"{r['n_half']} pseudo-trials/half, outliers {r['n_outliers']}, "
          f"seq offsets {r['offsets']}")
    for blk in ['dW', 'W1', 'W0']:
        b = r[blk]
        print(f"    {blk:3s}: beta={b['beta']:+.4f} p={b['p']:.3g} | "
              f"target-ctrl beta={b['beta_ctrl']:+.3f} p={b['p_ctrl']:.2g}")


all_results = []
control_halves = []
for condition, sessions in [('control', CONTROL_SESSIONS),
                            ('learning', LEARNING_SESSIONS)]:
    k_match = int(np.median(control_halves)) if control_halves else None
    if condition == 'learning':
        print(f"\nK (matched pseudo-trials per half) = {k_match}")
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            S = load_session(mouse, session, condition)
        except FileNotFoundError as e:
            print(f"  Skipping -- file not found ({e}).")
            continue
        except Exception as e:
            print(f"  LOAD FAILED: {e}")
            traceback.print_exc()
            continue
        conds = ['control'] if condition == 'control' else ['learning', 'learning_matched']
        for cond in conds:
            try:
                r = run_session(mouse, session, cond, S, k_match)
                all_results.append(r)
                if cond == 'control':
                    control_halves.append(r['n_half'])
                report(r)
            except Exception as e:
                print(f"  [{cond}] FAILED: {e}")
                traceback.print_exc()
        del S

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'fig4_dcorr_dw_control_vs_learning.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 6: Pooled stats
# ============================================================================
# Pool per-session z-scored pairs within a condition, fit y ~ x + target ctrl.
# Condition difference: beta_learning - beta_control, null from shuffling
# condition labels across sessions.
qc = [r for r in all_results if r['n_groups'] >= MIN_GROUPS]
for r in all_results:
    if r['n_groups'] < MIN_GROUPS:
        print(f"QC exclude: {r['condition']} {r['mouse']} {r['session']} ({r['n_groups']} groups)")


def suff(r, blk):
    b = r[blk]
    A = np.column_stack([b['x'], b['c']])
    return A.T @ A, A.T @ b['y'], b['y'] @ b['y'], len(b['y'])


def pooled_fit(sufs):
    XtX = sum(s[0] for s in sufs)
    Xty = sum(s[1] for s in sufs)
    yty = sum(s[2] for s in sufs)
    n = sum(s[3] for s in sufs)
    beta = np.linalg.solve(XtX, Xty)
    s2 = (yty - beta @ Xty) / (n - 2)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(XtX)))
    t = beta / se
    return beta[0], t[0], 2 * sps.norm.sf(abs(t[0])), n


pooled = {}
for blk in ['dW', 'W1', 'W0']:
    for mg, mice in MOUSE_GROUPS.items():
        ctrl = [r for r in qc if r['condition'] == 'control' and r['mouse'] in mice]
        sc = [suff(r, blk) for r in ctrl]
        bc, tc, pc, nc = pooled_fit(sc)
        for lv in ['learning', 'learning_matched']:
            lrn = [r for r in qc if r['condition'] == lv and r['mouse'] in mice]
            if not ctrl or not lrn:
                continue
            sl = [suff(r, blk) for r in lrn]
            bl, tl, pl, nl = pooled_fit(sl)
            allsuf = sc + sl
            rng = np.random.default_rng(0)
            null = np.empty(N_PERM)
            for i in range(N_PERM):
                perm = rng.permutation(len(allsuf))
                null[i] = (pooled_fit([allsuf[k] for k in perm[len(sc):]])[0] -
                           pooled_fit([allsuf[k] for k in perm[:len(sc)]])[0])
            p_diff = (np.sum(np.abs(null) >= abs(bl - bc)) + 1) / (N_PERM + 1)
            pooled[blk, mg, lv] = dict(bc=bc, pc=pc, nc=nc, bl=bl, pl=pl, nl=nl,
                                       p_diff=p_diff, ctrl=ctrl, lrn=lrn)
            sig_c = sum(r[blk]['p'] < 0.05 and r[blk]['beta'] > 0 for r in ctrl)
            sig_l = sum(r[blk]['p'] < 0.05 and r[blk]['beta'] > 0 for r in lrn)
            print(f"\n=== {blk} | {mg} | control vs {lv} ===")
            print(f"  control : {len(ctrl):2d} sess, {nc:6d} pairs, pooled beta={bc:+.4f} "
                  f"p={pc:.3g}; sessions beta>0 & p<.05: {sig_c}/{len(ctrl)}; "
                  f"beta>0: {sum(r[blk]['beta'] > 0 for r in ctrl)}/{len(ctrl)}")
            print(f"  {lv:9s}: {len(lrn):2d} sess, {nl:6d} pairs, pooled beta={bl:+.4f} "
                  f"p={pl:.3g}; sessions beta>0 & p<.05: {sig_l}/{len(lrn)}; "
                  f"beta>0: {sum(r[blk]['beta'] > 0 for r in lrn)}/{len(lrn)}")
            print(f"  difference {bl - bc:+.4f}, session-label permutation p={p_diff:.4f}")

#%% ============================================================================
# CELL 7: Figure — binned dA_rho vs dW (target control regressed out)
# ============================================================================
def binned(sessions, blk):
    x = np.concatenate([r[blk]['x'] for r in sessions])
    y = np.concatenate([r[blk]['y'] for r in sessions])
    c = np.concatenate([r[blk]['c'] for r in sessions])
    A = np.column_stack([np.ones_like(c), c])
    y = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    x = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    edges = np.percentile(x, np.linspace(0, 100, N_BINS + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, N_BINS - 1)
    bx = np.array([x[idx == b].mean() for b in range(N_BINS)])
    by = np.array([y[idx == b].mean() for b in range(N_BINS)])
    be = np.array([y[idx == b].std() / np.sqrt(np.sum(idx == b)) for b in range(N_BINS)])
    return bx, by, be


COLORS = {'control': '#3498db', 'learning': '#e74c3c', 'learning_matched': '#c0392b'}
YLAB = {'dW': r'$\Delta W$ (z, target $\Delta W$ regressed)',
        'W1': r'$W$ after (z, target $W$ regressed)',
        'W0': r'$W$ before (z, target $W$ regressed)'}
XLAB = {'dW': r'$\Delta$ correlation with targets, $\Delta A^\rho$ (z)',
        'W1': r'Correlation with targets, $A^\rho$ (z)',
        'W0': r'Correlation with targets, $A^\rho$ (z)'}
for mg in MOUSE_GROUPS:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for row, lv in enumerate(['learning', 'learning_matched']):
        for col, blk in enumerate(['dW', 'W1', 'W0']):
            ax = axes[row, col]
            P = pooled.get((blk, mg, lv))
            if P is None:
                ax.axis('off')
                continue
            for cond, group, b, p in [('control', P['ctrl'], P['bc'], P['pc']),
                                      (lv, P['lrn'], P['bl'], P['pl'])]:
                bx, by, be = binned(group, blk)
                ax.errorbar(bx, by, yerr=be, fmt='o-', color=COLORS[cond], capsize=3,
                            lw=2, label=f"{CONDITION_LABELS[cond]}\n"
                                        f"beta={b:+.3f}, p={p:.2g}, n={len(group)} sess")
            ax.axhline(0, color='k', ls='--', alpha=0.3)
            ax.set_xlabel(XLAB[blk])
            ax.set_ylabel(YLAB[blk])
            ax.set_title(f"{blk}: difference p={P['p_diff']:.3f} (session perm)")
            ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(f'Paper-1 Fig. 4 metric, water control vs BCI learning ({mg}, '
                 f'mean ± SEM over pairs)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f'fig4_dcorr_dw_{mg}.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

# Per-session coefficients for the dW fit
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
for ax, mg in zip(axes, MOUSE_GROUPS):
    rng = np.random.default_rng(0)
    for xi, cond in enumerate(['control', 'learning', 'learning_matched']):
        for r in [r for r in qc if r['condition'] == cond and r['mouse'] in MOUSE_GROUPS[mg]]:
            ax.plot(xi + rng.uniform(-0.12, 0.12), r['dW']['beta'], 'o',
                    color=COLORS[cond], alpha=0.7,
                    markeredgecolor='k' if r['dW']['p'] < 0.05 else 'none')
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Control', 'Learning\n(halves)', 'Learning\n(matched K)'])
    ax.set_title(mg)
axes[0].set_ylabel(r'$\beta$ ($\Delta A^\rho \rightarrow \Delta W$), per session')
fig.suptitle('Per-session Fig. 4h coefficient (black edge: p<0.05)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_dcorr_dw_sessions.png'), dpi=150,
            bbox_inches='tight')
plt.close(fig)
print("Figures saved.")
