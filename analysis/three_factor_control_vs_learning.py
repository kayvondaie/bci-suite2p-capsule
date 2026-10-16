#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# three_factor_variance_explained.py (dW ~ sum_w beta_w * CC_w, pinv, 5-fold CV
# across pairs) applied to E->E pairs in BCI learning sessions and in water
# control sessions.
#   learning: pre-epoch (-10..0 s before trial start) activity from trial F
#   control : no trials; spont_pre + spont_post (dF/F) cut into consecutive
#             10 s pseudo-trials, each standing in for one trial's pre epoch
#   learning_spont: learning sessions' own spont_pre + spont_post (suite2p
#             spont folders, iscell cells, 10th-percentile F0), cut into 10 s
#             pseudo-trials exactly like the controls -> activity matched to
#             the water sessions; only what happened in between differs
#   learning_pseudo: learning sessions run through the control procedure
#             (continuous BCI-epoch df_closedloop cut into 10 s pseudo-trials),
#             so control vs learning can be compared with identical activity
#             preprocessing
# Everything downstream (baseline, sliding windows, design matrix, CV fit) is
# identical. Pair selection / E->E restriction / putative-SST exclusion match
# connectivity_change_control_vs_learning.py. Test r compared learning vs
# control with a within-mouse permutation test.
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import traceback
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, wilcoxon
from sklearn.model_selection import KFold
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

plt.rcParams.update({'font.size': 13, 'axes.titlesize': 13})
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
PAN_NEURONAL_MICE = ['BCI88', 'BCI93', 'BCI107']
E_ONLY_MICE = ['BCI103', 'BCI105']
MOUSE_GROUPS = {
    'pan-neuronal': PAN_NEURONAL_MICE,
    'E-only': E_ONLY_MICE,
    'combined': PAN_NEURONAL_MICE + E_ONLY_MICE,
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

# three_factor_variance_explained.py settings
N_BASELINE = 20
WIN_SIZE = 10
WIN_STEP = 5
n_cv_folds = 5
PRE_EPOCH_SEC = 10          # trial pre epoch length = control pseudo-trial length

# Pair selection
dist_target_lt = 10
dist_nontarg_min = 30
dist_nontarg_max = float(os.environ.get('NONTARG_MAX', 1000))
# output files get a suffix when the non-target range is not the default
TAG = '' if dist_nontarg_max == 1000 else f'_nt{int(dist_nontarg_max)}'
# TEST_MAX: fit on all pairs, but score held-out predictions only on pairs
# whose non-target is closer than TEST_MAX um to the stim site
TEST_MAX = float(os.environ['TEST_MAX']) if os.environ.get('TEST_MAX') else None


# SPONT_MAX_MIN env var: cap every spont epoch at this many minutes, identically
# in water and learning sessions (pre: last N min, post: first N min, i.e. the
# parts closest to the intervening BCI / water period)
SPONT_MAX_MIN = float(os.environ['SPONT_MAX_MIN']) if os.environ.get('SPONT_MAX_MIN') else None


def cap_epoch(Fs, fs, which):
    if SPONT_MAX_MIN is None:
        return Fs
    n = int(round(SPONT_MAX_MIN * 60 * fs))
    return Fs[:, -n:] if which == 0 else Fs[:, :n]
if SPONT_MAX_MIN is not None:
    TAG += f'_spont{SPONT_MAX_MIN:g}min'
# DW_DEF='paper': dW = paper-1 Delta PS response, (mu1 - mu0) / sqrt(sd0^2 + sd1^2)
# from per-repeat Fstim responses (300 ms after / 200 ms before the stim,
# >= 10 repeats, previous-target repeats skipped). Default 'amp': AMP1 - AMP0.
DW_DEF = os.environ.get('DW_DEF', 'amp')
if DW_DEF == 'paper':
    TAG += '_psdef'
PS_MIN_REPEATS = 10
PS_TARGET_UM = 20
# W0_COV=1 (DW_DEF='paper' only): regression-to-the-mean control. Before the
# CV fit, each session's dW~ is residualised on the pair's baseline W~0
# (= mu0/sigma0), W~0^2, and the group's summed target W~0 (first map).
W0_COV = os.environ.get('W0_COV', '0') == '1'
if W0_COV:
    TAG += '_w0cov'
if TEST_MAX is not None:
    TAG += f'_test{int(TEST_MAX)}'
amp0_thr = 0.1
amp1_thr = 0.1
N_SD_OUTLIER = 2
MIN_GROUPS = 10

N_PERM = 10000

# Learning sessions with blown-up dW (median |dW| and W0 regression far
# outside every other session); excluded.
EXCLUDE = {('BCI88', '013125'), ('BCI107', '013125')}

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


def bin_pseudo_trials(dff, fs):
    """Mean dF/F in consecutive PRE_EPOCH_SEC bins (cells x n_bins)."""
    bin_len = int(round(PRE_EPOCH_SEC * fs))
    n_bins = dff.shape[1] // bin_len
    dff = dff[:, :n_bins * bin_len].reshape(dff.shape[0], n_bins, bin_len)
    return np.nanmean(dff, axis=2)


SPONT_FOLDER_PAIRS = [('suite2p_spont_pre', 'suite2p_spont_post'),
                      ('suite2p_spont', 'suite2p_spont2')]


def load_learning_spont(folder):
    """Raw F of the learning session's spont epochs (pre, post order), cells
    selected with each folder's iscell as in the data dict."""
    for pair in SPONT_FOLDER_PAIRS:
        dirs = [os.path.join(folder, n, 'plane0') for n in pair]
        found = [p for p in dirs if os.path.exists(os.path.join(p, 'F.npy'))]
        if found:
            out = []
            for p in found:
                F = np.load(os.path.join(p, 'F.npy'), allow_pickle=True)
                iscell = np.load(os.path.join(p, 'iscell.npy'), allow_pickle=True)
                out.append(np.asarray(F[np.asarray(iscell)[:, 0] == 1], dtype=float))
            return out
    raise FileNotFoundError(f'no suite2p spont folder in {folder}')


def ps_file(folder, pkey):
    files = sorted(glob.glob(folder + 'data_photostim*.h5'))
    if pkey == 'photostim':
        files = [f for f in files if not os.path.basename(f).startswith('data_photostim2')]
    else:
        files = [f for f in files if os.path.basename(f).startswith('data_photostim2')]
    if not files:
        raise FileNotFoundError(f'{pkey} h5 in {folder}')
    return files[0]


def repeat_responses_paper(path, dt, stimDist_um):
    """Per-repeat responses (cells x repeats), paper-1 windows."""
    with h5py.File(path, 'r') as f:
        favg_raw = f['favg_raw'][:]
        seq = np.asarray(f['seq'][:]).astype(int).ravel() - 1
        art = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
        art = art - np.nanmean(art[0:4])
        art = np.where(art > 0.5)[0]
        art = art[art < 40]
        if art.size == 0:
            raise RuntimeError(f'no stim artifact in {os.path.basename(path)}')
        n_after = max(1, int(round(0.3 / dt)))
        n_before = max(1, int(round(0.2 / dt)))
        pre = (max(0, int(art[0] - 2 - n_before)), int(art[0] - 2))
        post = (int(art[-1] + 2), int(art[-1] + 2 + n_after))
        win = f['Fstim'][pre[0]:post[1], :, :]
    resp = (np.nanmean(win[post[0] - pre[0]:], axis=0) - np.nanmean(win[:pre[1] - pre[0]], axis=0))
    n_groups = stimDist_um.shape[1]
    nearest = np.argmin(stimDist_um, axis=0)
    best, best_sr = -np.inf, (seq, resp)
    for off in range(-10, 11):
        if off > 0:
            s_, r_ = seq[:-off], resp[:, off:]
        elif off < 0:
            s_, r_ = seq[-off:], resp[:, :off]
        else:
            s_, r_ = seq, resp
        n = min(len(s_), r_.shape[1])
        s_, r_ = s_[:n], r_[:, :n]
        v = np.nanmean([np.nanmean(r_[nearest[g], s_ == g]) for g in range(n_groups) if np.any(s_ == g)])
        if v > best:
            best, best_sr = v, (s_, r_)
    s_, r_ = best_sr
    r_ = r_.copy()
    for t in range(1, len(s_)):
        if 0 <= s_[t - 1] < n_groups:
            r_[stimDist_um[:, s_[t - 1]] < PS_TARGET_UM, t] = np.nan
    return r_, s_


def paper_dW(folder, dt, stimDist_um):
    """Paper-1 Delta PS response, cells x groups (NaN where < 10 repeats)."""
    MU, SD = [], []
    for pkey in ['photostim', 'photostim2']:
        r, s_ = repeat_responses_paper(ps_file(folder, pkey), dt, stimDist_um)
        mu = np.full(stimDist_um.shape, np.nan)
        sd = np.full(stimDist_um.shape, np.nan)
        for g in range(stimDist_um.shape[1]):
            R = r[:, s_ == g]
            ok = np.sum(np.isfinite(R), axis=1) >= PS_MIN_REPEATS
            if ok.any():
                mu[ok, g] = np.nanmean(R[ok], axis=1)
                sd[ok, g] = np.nanstd(R[ok], axis=1, ddof=1)
        MU.append(mu)
        SD.append(sd)
    den = np.sqrt(SD[0] ** 2 + SD[1] ** 2)
    den[~(den > 0)] = np.nan
    return (MU[1] - MU[0]) / den, MU[0] / SD[0]


def load_session(mouse, session, condition):
    """AMP, stimDist and per-(pseudo)trial pre-epoch activity (cells x trials)."""
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
        dt = 1.0 / fs
        bins = []
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            for ei, k in enumerate(['spont_pre', 'spont_post']):
                Fs = cap_epoch(np.asarray(f[k][:], dtype=float), fs, ei)
                F0 = np.nanpercentile(Fs, 10, axis=1, keepdims=True)
                F0[F0 <= 0] = np.nan
                bins.append(bin_pseudo_trials((Fs - F0) / F0, fs))
        epoch_act = np.nan_to_num(np.concatenate(bins, axis=1), nan=0.0)
    elif condition == 'learning_spont':
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        fs = 1.0 / float(data['dt_si'])
        dt = 1.0 / fs
        bins = []
        for ei, Fs in enumerate(load_learning_spont(folder)):
            Fs = cap_epoch(Fs, fs, ei)
            F0 = np.nanpercentile(Fs, 10, axis=1, keepdims=True)
            F0[F0 <= 0] = np.nan
            bins.append(bin_pseudo_trials((Fs - F0) / F0, fs))
        epoch_act = np.nan_to_num(np.concatenate(bins, axis=1), nan=0.0)
    elif condition == 'learning_pseudo':
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        dt = float(data['dt_si'])
        epoch_act = np.nan_to_num(
            bin_pseudo_trials(np.asarray(data['df_closedloop'], dtype=float),
                              1.0 / float(data['dt_si'])), nan=0.0)
    else:
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['F', 'dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        dt_si = data['dt_si']
        dt = float(dt_si)
        F = data['F']                                      # frames x cells x trials
        tsta = np.arange(0, 12, dt_si)
        tsta = tsta - tsta[int(2 / dt_si)]
        F_nan = F.copy()
        F_nan[np.isnan(F_nan)] = 0
        ts_pre = np.where((tsta > -PRE_EPOCH_SEC) & (tsta < 0))[0]
        epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1] + 1, :, :], axis=0)
    dWp, W0p = paper_dW(folder, dt, stimDist) if DW_DEF == 'paper' else (None, None)
    return AMP, stimDist, epoch_act, dWp, W0p


#%% ============================================================================
# CELL 4: Per-session MLR fit (as in three_factor_variance_explained.py)
# ============================================================================
def session_fit(mouse, session, condition):
    AMP, stimDist, epoch_act, dWp, W0p = load_session(mouse, session, condition)
    if AMP[0].shape != AMP[1].shape:
        raise RuntimeError(f'before/after maps differ {AMP[0].shape} vs {AMP[1].shape}')
    n_neurons, trl = epoch_act.shape
    if n_neurons != AMP[0].shape[0]:
        raise RuntimeError(f'activity has {n_neurons} cells, photostim {AMP[0].shape[0]}')

    # Putative SST cells (pan-neuronal mice only), average of both maps
    amp_masked = (AMP[0] + AMP[1]) / 2
    amp_masked[stimDist < 30] = np.nan
    mean_amp_nontarg = np.nanmean(amp_masked, axis=1)
    is_outlier = mean_amp_nontarg > (np.nanmean(mean_amp_nontarg) +
                                     N_SD_OUTLIER * np.nanstd(mean_amp_nontarg))
    if mouse not in PAN_NEURONAL_MICE:
        is_outlier = np.zeros_like(is_outlier)
    is_E = ~is_outlier & np.isfinite(mean_amp_nontarg)

    # ---- Pair selection (E->E) ----
    dw_list, pair_cl_list, pair_nt_list, dist_list = [], [], [], []
    w0_list, tw0_list = [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < dist_target_lt) &
                      (AMP[0][:, gi] > amp0_thr) & (AMP[1][:, gi] > amp1_thr))[0]
        # outliers carry no opsin: never real targets, but keep the group
        cl = cl[~is_outlier[cl]]
        if cl.size == 0:
            continue
        nontarg = np.where((stimDist[:, gi] > dist_nontarg_min) &
                           (stimDist[:, gi] < dist_nontarg_max) & is_E &
                           (np.isfinite(dWp[:, gi]) if dWp is not None else True))[0]
        if nontarg.size == 0:
            continue
        dw_list.append(dWp[nontarg, gi] if dWp is not None
                       else AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
        pair_cl_list.append(cl)
        pair_nt_list.append(nontarg)
        dist_list.append(stimDist[nontarg, gi])
        if W0p is not None:
            w0_list.append(W0p[nontarg, gi])
            tw0_list.append(np.full(nontarg.size, np.nansum(W0p[cl, gi])))
    n_groups = len(dw_list)
    if n_groups == 0:
        raise RuntimeError('no valid pairs')

    Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
    r_y_w0 = np.nan
    if W0_COV and W0p is not None:
        w0 = np.nan_to_num(np.concatenate(w0_list))
        tw0 = np.concatenate(tw0_list)
        r_y_w0 = float(np.corrcoef(Y, w0)[0, 1])
        A = np.column_stack([np.ones_like(w0), w0, w0 ** 2, tw0])
        Y = Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]
    all_nt = np.concatenate(pair_nt_list)
    pair_dist = np.concatenate(dist_list)
    n_pairs = len(Y)
    cl_weights = np.zeros((n_pairs, n_neurons))
    offset = 0
    for cl, nt in zip(pair_cl_list, pair_nt_list):
        cl_weights[offset:offset + len(nt), cl] = 1.0 / len(cl)
        offset += len(nt)

    # ---- CC per (pseudo)trial, dev2 ----
    bl_mean = np.nanmean(epoch_act[:, :min(N_BASELINE, trl)], axis=1)
    r_pre = cl_weights @ epoch_act
    r_post_dev = epoch_act[all_nt, :] - bl_mean[all_nt, np.newaxis]
    cc_trial = r_pre * r_post_dev

    win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
    n_wins = len(win_starts)
    if n_wins < 3:
        raise RuntimeError(f'only {n_wins} windows')
    X = np.zeros((n_pairs, n_wins))
    for wi, ws in enumerate(win_starts):
        X[:, wi] = np.sum(cc_trial[:, ws:ws + WIN_SIZE], axis=1)
    sig_x = X.std(axis=0)
    sig_x[sig_x == 0] = 1.0
    X = (X - X.mean(axis=0)) / sig_x

    # ---- Cross-validated pinv fit ----
    cv = KFold(n_splits=n_cv_folds, shuffle=True, random_state=42)
    r_te_f, r_tr_f, p_te_f = [], [], []
    Y_test_all, Y_pred_all = [], []
    for train_idx, test_idx in cv.split(X):
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        mu_y, sig_y = Y_tr.mean(), Y_tr.std()
        if sig_y == 0 or not np.isfinite(sig_y):
            sig_y = 1.0
        beta = np.linalg.pinv(X[train_idx]) @ ((Y_tr - mu_y) / sig_y)
        if TEST_MAX is not None:
            test_idx = test_idx[pair_dist[test_idx] < TEST_MAX]
            Y_te = Y[test_idx]
        pred_tr, pred_te = X[train_idx] @ beta, X[test_idx] @ beta
        r_tr_f.append(pearsonr(pred_tr, (Y_tr - mu_y) / sig_y)[0]
                      if np.std(pred_tr) > 0 else 0.0)
        if len(pred_te) > 2 and np.std(pred_te) > 0:
            r, p = pearsonr(pred_te, (Y_te - mu_y) / sig_y)
        else:
            r, p = 0.0, 1.0
        r_te_f.append(r)
        p_te_f.append(p)
        Y_test_all.append((Y_te - mu_y) / sig_y)
        Y_pred_all.append(pred_te)
    return {
        'mouse': mouse, 'session': session, 'condition': condition,
        'n_groups': n_groups, 'n_pairs': n_pairs, 'n_trials': trl, 'n_windows': n_wins, 'r_y_w0': r_y_w0,
        'r_test': np.mean(r_te_f), 'r_train': np.mean(r_tr_f),
        'p_test': np.exp(np.mean(np.log(np.clip(p_te_f, 1e-300, 1.0)))),
        'Y_test_all': np.concatenate(Y_test_all),
        'Y_pred_all': np.concatenate(Y_pred_all),
    }


all_results = []
# CONDITIONS env var (comma-separated) restricts which conditions are run
RUN_CONDITIONS = os.environ.get('CONDITIONS', 'control,learning,learning_pseudo').split(',')
for condition, sessions in [(c, s) for c, s in [('control', CONTROL_SESSIONS),
                                                ('learning', LEARNING_SESSIONS),
                                                ('learning_pseudo', LEARNING_SESSIONS),
                                                ('learning_spont', LEARNING_SESSIONS)]
                            if c in RUN_CONDITIONS]:
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            r = session_fit(mouse, session, condition)
            all_results.append(r)
            sig = '*' if r['p_test'] < 0.05 else ''
            print(f"  {r['n_groups']} groups, {r['n_pairs']} pairs, {r['n_trials']} trials, "
                  f"{r['n_windows']} wins | r(dW,W0)={r['r_y_w0']:+.3f} | train r={r['r_train']:.3f}, "
                  f"test r={r['r_test']:.3f}, p={r['p_test']:.4f} {sig}")
        except FileNotFoundError:
            print("  Skipping -- file not found.")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, f'three_factor_control_vs_learning{TAG}.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 6: Stats — test r, learning vs control
# ============================================================================
qc_all = [r for r in all_results if r['n_groups'] >= MIN_GROUPS]
for r in all_results:
    if r['n_groups'] < MIN_GROUPS:
        print(f"QC exclude: {r['condition']} {r['mouse']} {r['session']} "
              f"({r['n_groups']} groups)")


def within_mouse_permutation(ctrl, lrn, key, mice, n_perm=N_PERM, seed=0):
    """Mean over mice of (learning mean - control mean); p from shuffling
    learning/control labels among each mouse's own sessions."""
    per_mouse = []
    for m in mice:
        v = np.array([r[key] for r in ctrl if r['mouse'] == m] +
                     [r[key] for r in lrn if r['mouse'] == m])
        per_mouse.append((v, sum(r['mouse'] == m for r in ctrl)))
    stat = lambda pm: np.mean([v[n:].mean() - v[:n].mean() for v, n in pm])
    obs = stat(per_mouse)
    rng = np.random.default_rng(seed)
    null = np.array([stat([(rng.permutation(v), n) for v, n in per_mouse])
                     for _ in range(n_perm)])
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)


def one_sample_p(v):
    try:
        return wilcoxon(v).pvalue
    except Exception:
        return np.nan


LEARNING_VARIANTS = {k: v for k, v in {'learning': 'Learning (BCI trials)',
                                        'learning_pseudo': 'Learning (BCI pseudo-trials)',
                                        'learning_spont': 'Learning (spont pseudo-trials)'}.items()
                     if k in RUN_CONDITIONS}
stats = {}
for lv in LEARNING_VARIANTS:
    for mg, mice in MOUSE_GROUPS.items():
        ctrl = [r for r in qc_all if r['condition'] == 'control' and r['mouse'] in mice]
        lrn = [r for r in qc_all if r['condition'] == lv and r['mouse'] in mice]
        mice_both = sorted({r['mouse'] for r in ctrl} & {r['mouse'] for r in lrn})
        diff, p = within_mouse_permutation(ctrl, lrn, 'r_test', mice_both)
        stats[lv, mg] = {'ctrl': ctrl, 'lrn': lrn, 'mice_both': mice_both, 'p': p}
        rc = np.array([r['r_test'] for r in ctrl])
        rl = np.array([r['r_test'] for r in lrn])
        print(f"\n=== control vs {lv} | {mg} {mice_both}: "
              f"{len(ctrl)} / {len(lrn)} sessions ===")
        print(f"  test r  control  median {np.median(rc):+.4f}, >0 in {np.sum(rc > 0)}/{len(rc)}, "
              f"sig {np.sum([r['p_test'] < 0.05 for r in ctrl])}/{len(rc)}, "
              f"Wilcoxon vs 0 p={one_sample_p(rc):.3f}")
        print(f"  test r  {lv} median {np.median(rl):+.4f}, >0 in {np.sum(rl > 0)}/{len(rl)}, "
              f"sig {np.sum([r['p_test'] < 0.05 for r in lrn])}/{len(rl)}, "
              f"Wilcoxon vs 0 p={one_sample_p(rl):.3f}")
        print(f"  {lv} - control (within mouse) {diff:+.4f}, perm p={p:.3f}")
        print('  per mouse (ctrl / learn): ' + ', '.join(
            f"{m} {np.mean([r['r_test'] for r in ctrl if r['mouse'] == m]):+.4f}/"
            f"{np.mean([r['r_test'] for r in lrn if r['mouse'] == m]):+.4f}"
            for m in mice_both))

#%% ============================================================================
# CELL 7: Figure — test r per session, control vs learning
# ============================================================================
mouse_list = sorted({r['mouse'] for r in qc_all})
mouse_colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(mouse_list)}
fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True)
for ax, (lv, mg) in zip(axes.flat, [(lv, mg) for lv in LEARNING_VARIANTS
                                    for mg in MOUSE_GROUPS]):
    S = stats[lv, mg]
    rng = np.random.default_rng(0)
    for xi, group in [(0, S['ctrl']), (1, S['lrn'])]:
        for r in group:
            ax.plot(xi + rng.uniform(-0.12, 0.12), r['r_test'], 'o',
                    color=mouse_colors[r['mouse']], alpha=0.6,
                    markeredgecolor='k' if r['p_test'] < 0.05 else 'none')
    for m in S['mice_both']:
        vc = np.mean([r['r_test'] for r in S['ctrl'] if r['mouse'] == m])
        vl = np.mean([r['r_test'] for r in S['lrn'] if r['mouse'] == m])
        ax.plot([0, 1], [vc, vl], '-', color=mouse_colors[m], linewidth=2)
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control\n(spont pseudo-trials)',
                        LEARNING_VARIANTS[lv].replace(' (', '\n(')])
    ax.set_xlim(-0.5, 1.5)
    ax.set_title(f'{mg}\nwithin-mouse p={S["p"]:.3f}')
for ax in axes[:, 0]:
    ax.set_ylabel('Test r (5-fold CV), dW ~ CC_windows')
handles = [plt.Line2D([], [], marker='o', ls='', color=mouse_colors[m], label=m)
           for m in mouse_list]
fig.legend(handles=handles, loc='lower center', ncol=len(mouse_list),
           bbox_to_anchor=(0.5, -0.05))
fig.suptitle('3-factor MLR (E->E): BCI learning vs water control '
             '(black edge: session p<0.05)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'fig_three_factor_control_vs_learning{TAG}.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")

#%% ============================================================================
# CELL 8: Pooled comparison — all sessions per condition, no per-mouse pairing
# ============================================================================
# Pool every session's held-out (Y_pred, Y_test) pairs within a condition.
#   pooled r      : Pearson r over all pooled pairs
#   p (difference): shuffle condition labels across sessions (ignoring mouse),
#                   recompute pooled r_learning - r_control
#   binned curve  : predicted-dW percentile bins, 95% CI from bootstrapping
#                   sessions within each condition
N_POOL_PERM = 5000
N_POOL_BOOT = 1000
N_POOL_BINS = 5


def session_sums(r):
    x, y = r['Y_pred_all'], r['Y_test_all']
    return np.array([len(x), x.sum(), y.sum(), (x * y).sum(), (x * x).sum(), (y * y).sum()])


def pooled_r(sums):
    n, sx, sy, sxy, sxx, syy = sums.sum(axis=0)
    cov = sxy - sx * sy / n
    return cov / np.sqrt((sxx - sx ** 2 / n) * (syy - sy ** 2 / n))


def binned_curve(sessions):
    x = np.concatenate([r['Y_pred_all'] for r in sessions])
    y = np.concatenate([r['Y_test_all'] for r in sessions])
    edges = np.percentile(x, np.linspace(0, 100, N_POOL_BINS + 1))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, N_POOL_BINS - 1)
    return (np.array([x[idx == b].mean() for b in range(N_POOL_BINS)]),
            np.array([y[idx == b].mean() for b in range(N_POOL_BINS)]))


pooled = {}
for lv in LEARNING_VARIANTS:
    ctrl = [r for r in qc_all if r['condition'] == 'control']
    lrn = [r for r in qc_all if r['condition'] == lv]
    S_c = np.array([session_sums(r) for r in ctrl])
    S_l = np.array([session_sums(r) for r in lrn])
    r_c, r_l = pooled_r(S_c), pooled_r(S_l)
    obs = r_l - r_c

    S_all = np.vstack([S_c, S_l])
    rng = np.random.default_rng(0)
    null = np.empty(N_POOL_PERM)
    for i in range(N_POOL_PERM):
        perm = rng.permutation(len(S_all))
        null[i] = pooled_r(S_all[perm[len(S_c):]]) - pooled_r(S_all[perm[:len(S_c)]])
    p_diff = (np.sum(np.abs(null) >= abs(obs)) + 1) / (N_POOL_PERM + 1)

    # Bootstrap sessions within each condition for pooled-r and curve CIs
    boot = {}
    for name, group, S in [('control', ctrl, S_c), (lv, lrn, S_l)]:
        rs, xs, ys = [], [], []
        for _ in range(N_POOL_BOOT):
            pick = rng.integers(0, len(group), len(group))
            rs.append(pooled_r(S[pick]))
            bx, by = binned_curve([group[k] for k in pick])
            xs.append(bx)
            ys.append(by)
        boot[name] = dict(r_ci=np.percentile(rs, [2.5, 97.5]),
                          x=np.median(xs, axis=0), y=np.median(ys, axis=0),
                          y_lo=np.percentile(ys, 2.5, axis=0),
                          y_hi=np.percentile(ys, 97.5, axis=0))
    pooled[lv] = dict(r_c=r_c, r_l=r_l, p=p_diff, boot=boot,
                      n_c=(len(ctrl), int(S_c[:, 0].sum())),
                      n_l=(len(lrn), int(S_l[:, 0].sum())))
    print(f"\n=== POOLED control vs {lv} ===")
    print(f"  control : {len(ctrl)} sessions, {int(S_c[:, 0].sum())} pairs, pooled r = "
          f"{r_c:+.4f} (95% CI {boot['control']['r_ci'][0]:+.4f} to {boot['control']['r_ci'][1]:+.4f})")
    print(f"  {lv:8s}: {len(lrn)} sessions, {int(S_l[:, 0].sum())} pairs, pooled r = "
          f"{r_l:+.4f} (95% CI {boot[lv]['r_ci'][0]:+.4f} to {boot[lv]['r_ci'][1]:+.4f})")
    print(f"  difference {obs:+.4f}, session-label permutation p = {p_diff:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, lv in zip(axes, LEARNING_VARIANTS):
    P = pooled[lv]
    for name, color, label in [('control', '#3498db', f'Control (water), r={P["r_c"]:.3f}'),
                               (lv, '#e74c3c', f'{LEARNING_VARIANTS[lv]}, r={P["r_l"]:.3f}')]:
        b = P['boot'][name]
        ax.errorbar(b['x'], b['y'], yerr=[b['y'] - b['y_lo'], b['y_hi'] - b['y']],
                    fmt='o-', color=color, capsize=4, linewidth=2, markersize=7,
                    label=label)
    ax.axhline(0, color='k', ls='--', alpha=0.3)
    ax.set_xlabel(r'Predicted $\Delta W$ (z, held out)')
    ax.set_title(f'Pooled: control vs {LEARNING_VARIANTS[lv]}\n'
                 f'difference p={P["p"]:.4f} (session-label permutation)')
    ax.legend(fontsize=10)
axes[0].set_ylabel(r'Actual $\Delta W$ (z)')
fig.suptitle('3-factor MLR, E->E pairs, all mice pooled (95% CI: session bootstrap)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, f'fig_three_factor_pooled_control_vs_learning{TAG}.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Pooled figure saved.")
