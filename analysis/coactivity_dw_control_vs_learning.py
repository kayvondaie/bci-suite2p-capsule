#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Does E->E plasticity depend on coactivity differently in BCI learning vs
# water-control sessions?
#   dW        : after - before photostim connectivity (compute_amp_from_photostim)
#   coactivity: correlation of pre (stim group's target cells, mean trace) and
#               post (non-target cell) activity between the two photostim maps
#                 learning: BCI epoch (df_closedloop)
#                 control : spont_pre + spont_post
# Per session: dW ~ b_coact * coactivity + b_w0 * W0 (W0 guards against
# regression to the mean). b_coact compared learning vs control with a
# within-mouse permutation test.
# Pair selection, E->E restriction and putative-SST exclusion match
# connectivity_change_control_vs_learning.py.
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import traceback
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr
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

# Pair selection (same as connectivity_change_control_vs_learning.py)
dist_target_lt = 10
dist_nontarg_min = 30
dist_nontarg_max = 1000
amp0_thr = 0.1
amp1_thr = 0.1
N_SD_OUTLIER = 2
MIN_GROUPS = 10

DETREND_SEC = 30      # moving-average window subtracted before correlation
N_PERM = 10000
N_COACT_BINS = 8

#%% ============================================================================
# CELL 3: Loading
# ============================================================================
def control_tag(mouse, session):
    return f"{SUBJECT_ID[mouse]}_20{session[4:6]}-{session[0:2]}-{session[2:4]}"


def load_amp(mouse, session, condition):
    """AMP (before, after) and stimDist, as in the connectivity-change script."""
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
    else:
        folder = next(os.path.join(r, mouse, session, 'pophys') + '/'
                      for r in LEARNING_ROOTS
                      if os.path.isdir(os.path.join(r, mouse, session, 'pophys')))
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
    AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
    return AMP, stimDist, folder


def preprocess(F, fs):
    """Detrend (minus moving average) and z-score each cell; F is cells x time."""
    F = np.asarray(F, dtype=float)
    row_mean = np.nanmean(F, axis=1, keepdims=True)
    F = np.where(np.isfinite(F), F, row_mean)
    F = F - uniform_filter1d(F, size=max(3, int(DETREND_SEC * fs)), axis=1,
                             mode='nearest')
    sd = F.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (F - F.mean(axis=1, keepdims=True)) / sd


def load_activity(mouse, session, condition, folder):
    """z-scored activity between the two photostim maps (cells x time)."""
    if condition == 'control':
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy',
                     allow_pickle=True).item()
        fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            segs = [preprocess(f[k][:], fs) for k in ['spont_pre', 'spont_post']]
        return np.concatenate(segs, axis=1)
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], [])
    return preprocess(data['df_closedloop'], 1.0 / float(data['dt_si']))


#%% ============================================================================
# CELL 4: Per-session coactivity vs dW
# ============================================================================
def session_coactivity_dw(mouse, session, condition):
    AMP, stimDist, folder = load_amp(mouse, session, condition)
    if AMP[0].shape != AMP[1].shape:
        raise RuntimeError(f'before/after maps differ {AMP[0].shape} vs {AMP[1].shape}')
    Z = load_activity(mouse, session, condition, folder)
    if Z.shape[0] != AMP[0].shape[0]:
        raise RuntimeError(f'activity has {Z.shape[0]} cells, photostim {AMP[0].shape[0]}')
    T = Z.shape[1]

    # Putative SST cells (pan-neuronal mice only), average of both maps
    amp_masked = (AMP[0] + AMP[1]) / 2
    amp_masked[stimDist < 30] = np.nan
    mean_amp_nontarg = np.nanmean(amp_masked, axis=1)
    is_outlier = mean_amp_nontarg > (np.nanmean(mean_amp_nontarg) +
                                     N_SD_OUTLIER * np.nanstd(mean_amp_nontarg))
    if mouse not in PAN_NEURONAL_MICE:
        is_outlier = np.zeros_like(is_outlier)
    is_E = ~is_outlier & np.isfinite(mean_amp_nontarg)

    dw, w0, coact = [], [], []
    n_groups = 0
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < dist_target_lt) &
                      (AMP[0][:, gi] > amp0_thr) & (AMP[1][:, gi] > amp1_thr))[0]
        if cl.size == 0 or np.any(is_outlier[cl]):
            continue
        nt = np.where((stimDist[:, gi] > dist_nontarg_min) &
                      (stimDist[:, gi] < dist_nontarg_max) & is_E)[0]
        if nt.size == 0:
            continue
        pre = Z[cl].mean(axis=0)
        pre = (pre - pre.mean()) / (pre.std() or 1)
        coact.append(Z[nt] @ pre / T)
        dw.append(AMP[1][nt, gi] - AMP[0][nt, gi])
        w0.append(AMP[0][nt, gi])
        n_groups += 1
    if n_groups == 0:
        raise RuntimeError('no valid pairs')

    dw, w0, coact = map(np.concatenate, (dw, w0, coact))
    ok = np.isfinite(dw) & np.isfinite(w0) & np.isfinite(coact)
    dw, w0, coact = dw[ok], w0[ok], coact[ok]

    # dW ~ b0 + b_coact * z(coact) + b_w0 * z(W0)
    zc = (coact - coact.mean()) / coact.std()
    zw = (w0 - w0.mean()) / w0.std()
    X = np.column_stack([np.ones_like(zc), zc, zw])
    b = np.linalg.lstsq(X, dw, rcond=None)[0]
    return {
        'mouse': mouse, 'session': session, 'condition': condition,
        'n_groups': n_groups, 'n_pairs': len(dw), 'n_frames': T,
        'b_coact': b[1], 'b_w0': b[2],
        'rho_coact_dw': spearmanr(coact, dw)[0],
        'coact_mean': coact.mean(), 'coact_sd': coact.std(),
        'coact': coact, 'dw': dw,
    }


all_results = []
for condition, sessions in [('control', CONTROL_SESSIONS),
                            ('learning', LEARNING_SESSIONS)]:
    for mouse, session in sessions:
        print(f"\n--- {condition}: {mouse} {session} ---")
        try:
            r = session_coactivity_dw(mouse, session, condition)
            all_results.append(r)
            print(f"  {r['n_groups']} groups, {r['n_pairs']} pairs, {r['n_frames']} frames, "
                  f"coact {r['coact_mean']:+.3f}+-{r['coact_sd']:.3f}, "
                  f"b_coact={r['b_coact']:+.4f}, b_w0={r['b_w0']:+.4f}, "
                  f"rho={r['rho_coact_dw']:+.3f}")
        except StopIteration:
            print("  Skipping -- file not found.")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

#%% ============================================================================
# CELL 5: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'coactivity_dw_control_vs_learning.npy'),
        [{k: v for k, v in r.items() if k not in ('coact', 'dw')} for r in all_results],
        allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 6: Stats — learning vs control, within mouse
# ============================================================================
METRICS = [('b_coact', 'dW ~ coactivity slope (W0-controlled)'),
           ('rho_coact_dw', 'Spearman rho(coactivity, dW)'),
           ('coact_mean', 'Mean E->E coactivity')]

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


stats = {}
for mg, mice in MOUSE_GROUPS.items():
    ctrl = [r for r in qc_all if r['condition'] == 'control' and r['mouse'] in mice]
    lrn = [r for r in qc_all if r['condition'] == 'learning' and r['mouse'] in mice]
    mice_both = sorted({r['mouse'] for r in ctrl} & {r['mouse'] for r in lrn})
    stats[mg] = {'ctrl': ctrl, 'lrn': lrn, 'mice_both': mice_both}
    print(f"\n=== {mg} {mice_both}: {len(ctrl)} control / {len(lrn)} learning sessions ===")
    for key, label in METRICS:
        diff, p = within_mouse_permutation(ctrl, lrn, key, mice_both)
        stats[mg][key] = p
        per = ', '.join(
            f"{m} {np.mean([r[key] for r in ctrl if r['mouse'] == m]):+.4f}/"
            f"{np.mean([r[key] for r in lrn if r['mouse'] == m]):+.4f}"
            for m in mice_both)
        print(f"{label:40s} ctrl {np.median([r[key] for r in ctrl]):+.4f}  "
              f"learn {np.median([r[key] for r in lrn]):+.4f}  "
              f"learn-ctrl {diff:+.4f}  perm p={p:.3f}")
        print(f"      per mouse (ctrl/learn): {per}")

#%% ============================================================================
# CELL 7: Figure — per-session slopes and pooled binned dW vs coactivity
# ============================================================================
mouse_list = sorted({r['mouse'] for r in qc_all})
mouse_colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(mouse_list)}

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, mg in enumerate(MOUSE_GROUPS):
    S = stats[mg]
    # Top: per-session W0-controlled slope
    ax = axes[0, col]
    rng = np.random.default_rng(0)
    for xi, group in [(0, S['ctrl']), (1, S['lrn'])]:
        for r in group:
            ax.plot(xi + rng.uniform(-0.12, 0.12), r['b_coact'], 'o',
                    color=mouse_colors[r['mouse']], alpha=0.6)
    for m in S['mice_both']:
        vc = np.mean([r['b_coact'] for r in S['ctrl'] if r['mouse'] == m])
        vl = np.mean([r['b_coact'] for r in S['lrn'] if r['mouse'] == m])
        ax.plot([0, 1], [vc, vl], '-', color=mouse_colors[m], linewidth=2)
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control\n(spont)', 'Learning\n(BCI)'])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('dW ~ coactivity slope')
    ax.set_title(f'{mg}\nwithin-mouse p={S["b_coact"]:.3f}')

    # Bottom: dW vs within-session coactivity percentile, pooled per condition
    ax = axes[1, col]
    for group, color, name in [(S['ctrl'], '#3498db', 'Control'),
                               (S['lrn'], '#e74c3c', 'Learning')]:
        curves = []
        for r in group:
            edges = np.percentile(r['coact'], np.linspace(0, 100, N_COACT_BINS + 1))
            idx = np.clip(np.digitize(r['coact'], edges[1:-1]), 0, N_COACT_BINS - 1)
            curves.append([np.mean(r['dw'][idx == b]) for b in range(N_COACT_BINS)])
        if not curves:
            continue
        curves = np.array(curves)
        x = (np.arange(N_COACT_BINS) + 0.5) / N_COACT_BINS * 100
        ax.errorbar(x, curves.mean(0), yerr=curves.std(0) / np.sqrt(len(curves)),
                    fmt='o-', color=color, capsize=3, linewidth=2,
                    label=f'{name} (n={len(curves)})')
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_xlabel('Coactivity (within-session percentile)')
    ax.set_ylabel('Mean dW')
    ax.legend(fontsize=10)

handles = [plt.Line2D([], [], marker='o', ls='', color=mouse_colors[m], label=m)
           for m in mouse_list]
fig.legend(handles=handles, loc='lower center', ncol=len(mouse_list),
           bbox_to_anchor=(0.5, -0.03))
fig.suptitle('E->E dW vs coactivity: BCI learning (BCI epoch) vs water control '
             '(spont)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_coactivity_dw_control_vs_learning.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
