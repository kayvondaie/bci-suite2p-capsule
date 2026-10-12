#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# three_factor_variance_explained.py (dW ~ sum_w beta_w * CC_w, pinv, 5-fold CV
# across pairs) applied to E->E pairs in BCI learning sessions and in water
# control sessions.
#   learning: pre-epoch (-10..0 s before trial start) activity from trial F
#   control : no trials; spont_pre + spont_post (dF/F) cut into consecutive
#             10 s pseudo-trials, each standing in for one trial's pre epoch
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
dist_nontarg_max = 1000
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
        bins = []
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            for k in ['spont_pre', 'spont_post']:
                Fs = np.asarray(f[k][:], dtype=float)
                F0 = np.nanpercentile(Fs, 10, axis=1, keepdims=True)
                F0[F0 <= 0] = np.nan
                bins.append(bin_pseudo_trials((Fs - F0) / F0, fs))
        epoch_act = np.nan_to_num(np.concatenate(bins, axis=1), nan=0.0)
    elif condition == 'learning_pseudo':
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        epoch_act = np.nan_to_num(
            bin_pseudo_trials(np.asarray(data['df_closedloop'], dtype=float),
                              1.0 / float(data['dt_si'])), nan=0.0)
    else:
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['F', 'dt_si'], ['stimDist', 'favg_raw'])
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        dt_si = data['dt_si']
        F = data['F']                                      # frames x cells x trials
        tsta = np.arange(0, 12, dt_si)
        tsta = tsta - tsta[int(2 / dt_si)]
        F_nan = F.copy()
        F_nan[np.isnan(F_nan)] = 0
        ts_pre = np.where((tsta > -PRE_EPOCH_SEC) & (tsta < 0))[0]
        epoch_act = np.nanmean(F_nan[ts_pre[0]:ts_pre[-1] + 1, :, :], axis=0)
    return AMP, stimDist, epoch_act


#%% ============================================================================
# CELL 4: Per-session MLR fit (as in three_factor_variance_explained.py)
# ============================================================================
def session_fit(mouse, session, condition):
    AMP, stimDist, epoch_act = load_session(mouse, session, condition)
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
    dw_list, pair_cl_list, pair_nt_list = [], [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < dist_target_lt) &
                      (AMP[0][:, gi] > amp0_thr) & (AMP[1][:, gi] > amp1_thr))[0]
        if cl.size == 0 or np.any(is_outlier[cl]):
            continue
        nontarg = np.where((stimDist[:, gi] > dist_nontarg_min) &
                           (stimDist[:, gi] < dist_nontarg_max) & is_E)[0]
        if nontarg.size == 0:
            continue
        dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
        pair_cl_list.append(cl)
        pair_nt_list.append(nontarg)
    n_groups = len(dw_list)
    if n_groups == 0:
        raise RuntimeError('no valid pairs')

    Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
    all_nt = np.concatenate(pair_nt_list)
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
    for train_idx, test_idx in cv.split(X):
        Y_tr, Y_te = Y[train_idx], Y[test_idx]
        mu_y, sig_y = Y_tr.mean(), Y_tr.std()
        if sig_y == 0 or not np.isfinite(sig_y):
            sig_y = 1.0
        beta = np.linalg.pinv(X[train_idx]) @ ((Y_tr - mu_y) / sig_y)
        pred_tr, pred_te = X[train_idx] @ beta, X[test_idx] @ beta
        r_tr_f.append(pearsonr(pred_tr, (Y_tr - mu_y) / sig_y)[0]
                      if np.std(pred_tr) > 0 else 0.0)
        if np.std(pred_te) > 0:
            r, p = pearsonr(pred_te, (Y_te - mu_y) / sig_y)
        else:
            r, p = 0.0, 1.0
        r_te_f.append(r)
        p_te_f.append(p)
    return {
        'mouse': mouse, 'session': session, 'condition': condition,
        'n_groups': n_groups, 'n_pairs': n_pairs, 'n_trials': trl, 'n_windows': n_wins,
        'r_test': np.mean(r_te_f), 'r_train': np.mean(r_tr_f),
        'p_test': np.exp(np.mean(np.log(np.clip(p_te_f, 1e-300, 1.0)))),
    }


all_results = []
for condition, sessions in [('control', CONTROL_SESSIONS),
                            ('learning', LEARNING_SESSIONS),
                            ('learning_pseudo', LEARNING_SESSIONS)]:
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            r = session_fit(mouse, session, condition)
            all_results.append(r)
            sig = '*' if r['p_test'] < 0.05 else ''
            print(f"  {r['n_groups']} groups, {r['n_pairs']} pairs, {r['n_trials']} trials, "
                  f"{r['n_windows']} wins | train r={r['r_train']:.3f}, "
                  f"test r={r['r_test']:.3f}, p={r['p_test']:.4f} {sig}")
        except FileNotFoundError:
            print("  Skipping -- file not found.")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'three_factor_control_vs_learning.npy'),
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


LEARNING_VARIANTS = {'learning': 'Learning (BCI trials)',
                     'learning_pseudo': 'Learning (BCI pseudo-trials)'}
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
plt.savefig(os.path.join(RESULTS_DIR, 'fig_three_factor_control_vs_learning.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
