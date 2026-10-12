#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# Compare photostim connectivity change (after - before) between
#   - control sessions: photostim -> spont -> water in bowl (no BCI) -> spont -> photostim
#   - learning sessions: photostim -> BCI -> photostim
# Two mouse groups, E->E pairs only:
#   pan-neuronal (ai230 x slc17a7 riboGCaMP8s: BCI88, BCI93, BCI107): putative
#     SST (outlier) cells are excluded both as stimulated targets and as
#     postsynaptic cells.
#   E-only (BCI103, BCI105): all cells are E, no exclusion.
# Connectivity, pair selection and dW are otherwise computed exactly as in
# sliding_window_temporal_offset.py (compute_amp_from_photostim, targets
# <10 um with AMP>0.1 in both epochs, non-targets 30-1000 um).
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import h5py
import data_dict_create_module_test as ddct
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, wilcoxon, mannwhitneyu
import traceback
import importlib
import BCI_data_helpers
importlib.reload(BCI_data_helpers)
from BCI_data_helpers import *

# Environment toggle: 'local' or 'codeocean'
RUN_ENV = 'codeocean'

if RUN_ENV == 'codeocean':
    # Learning sessions: pan-neuronal mice (BCI88/93/107) and the rest live
    # in two assets with the scratch <mouse>/<session>/pophys layout.
    # Control sessions: bci-suite2p-capsule results (<subject>_<date>/).
    CO_DATA = '/root/capsule/data'
    LEARNING_ROOTS = [
        f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal',
        f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025',
    ]
    CONTROL_ROOT = os.path.dirname(
        (glob.glob(f'{CO_DATA}/*/731015_2025-01-27') +
         glob.glob(f'{CO_DATA}/*/*/731015_2025-01-27') + [f'{CO_DATA}/x/x'])[0])
    RESULTS_DIR = '/root/capsule/results'
else:
    LEARNING_ROOTS = [r'//allen/aind/scratch/BCI/2p-raw']
    CONTROL_ROOT = None   # controls staged into scratch pophys folders
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'meta_analysis_results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Labtracks subject IDs (control results are named <subject>_<YYYY-MM-DD>)
SUBJECT_ID = {'BCI88': '731015', 'BCI93': '740369', 'BCI103': '772414',
              'BCI105': '754303', 'BCI107': '767715'}

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
})

print("Setup complete!")

#%% ============================================================================
# CELL 2: Session lists
# ============================================================================
# (mouse, session, FOV) from 'Surgeries and experiments - BCI.xlsx'.
PAN_NEURONAL_MICE = ['BCI88', 'BCI93', 'BCI107']
E_ONLY_MICE = ['BCI103', 'BCI105']

CONTROL_SESSIONS = [
    ('BCI88',  '012725', 'FOV_04'),
    ('BCI93',  '012825', 'FOV_05'),
    ('BCI93',  '022125', 'FOV_07'),
    ('BCI107', '021925', 'FOV_03'),
    ('BCI103', '013025', 'FOV_01'),
    ('BCI103', '021925', 'FOV_03'),
    ('BCI105', '013025', 'FOV_01'),
]

# 'BCI + photostim before-after' sessions from the same mice
# (BCI_ai230_slc17a7_riboGcamp8s and BCI_three_factor_2025 assets).
LEARNING_SESSIONS = [
    ('BCI88',  '011025', 'FOV_04'), ('BCI88',  '011325', 'FOV_04'),
    ('BCI88',  '012425', 'FOV_04'), ('BCI88',  '012825', 'FOV_04'),
    ('BCI88',  '013125', 'FOV_04'),
    ('BCI93',  '010925', 'FOV_05'), ('BCI93',  '011025', 'FOV_05'),
    ('BCI93',  '011325', 'FOV_05'), ('BCI93',  '012425', 'FOV_05'),
    ('BCI93',  '020325', 'FOV_06'), ('BCI93',  '020425', 'FOV_06'),
    ('BCI107', '013025', 'FOV_01'), ('BCI107', '013125', 'FOV_01'),
    ('BCI107', '020325', 'FOV_02'), ('BCI107', '020625', 'FOV_01'),
    ('BCI103', '012125', 'FOV_01'), ('BCI103', '012225', 'FOV_01'),
    ('BCI103', '012325', 'FOV_01'), ('BCI103', '012425', 'FOV_01'),
    ('BCI103', '012725', 'FOV_01'), ('BCI103', '012825', 'FOV_01'),
    ('BCI103', '012925', 'FOV_01'),
    ('BCI103', '020325', 'FOV_02'), ('BCI103', '020425', 'FOV_02'),
    ('BCI103', '020525', 'FOV_02'), ('BCI103', '020625', 'FOV_02'),
    ('BCI105', '012125', 'FOV_01'), ('BCI105', '012225', 'FOV_01'),
    ('BCI105', '012325', 'FOV_01'), ('BCI105', '012425', 'FOV_01'),
    ('BCI105', '012725', 'FOV_01'), ('BCI105', '012925', 'FOV_01'),
    ('BCI105', '013125', 'FOV_01'),
    ('BCI105', '020325', 'FOV_02'), ('BCI105', '020425', 'FOV_02'),
]

# Pair selection (same as sliding_window_temporal_offset.py)
dist_target_lt = 10
dist_nontarg_min = 30
dist_nontarg_max = 1000
amp0_thr = 0.1
amp1_thr = 0.1

DIST_BINS = np.array([30, 60, 100, 200, 400, 1000])

# QC: sessions with fewer valid stim groups are kept in all_results but
# excluded from stats/figures.
MIN_GROUPS = 10

# Putative SST (outlier) cells in pan-neuronal mice: mean non-target amp >
# mean + N SD. Excluded as targets and as postsynaptic cells.
N_SD_OUTLIER = 2
CELL_GROUPS = ['E']

# Mouse groups reported separately and together
MOUSE_GROUPS = {
    'pan-neuronal': PAN_NEURONAL_MICE,
    'E-only': E_ONLY_MICE,
    'combined': PAN_NEURONAL_MICE + E_ONLY_MICE,
}

# Within-mouse label permutations for the learning-vs-control test
N_PERM = 10000

#%% ============================================================================
# CELL 3: Per-session connectivity change
# ============================================================================
def load_session(mouse, session, condition):
    """Return (data, folder) ready for compute_amp_from_photostim."""
    photostim_keys = ['stimDist', 'favg_raw']
    if condition == 'control' and CONTROL_ROOT is not None:
        # bci-suite2p-capsule output: <subject>_<YYYY-MM-DD>/photostim[2]_compact.h5
        # plus suite2p_photostim_single/plane0/siHeader.npy. No BCI epoch, so
        # compute_amp_from_photostim takes dt_si from that siHeader.
        tag = f"{SUBJECT_ID[mouse]}_20{session[4:6]}-{session[0:2]}-{session[2:4]}"
        folder = os.path.join(CONTROL_ROOT, tag) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in photostim_keys}
        return data, folder
    # Learning sessions get dt_si from data_main; locally staged controls
    # fall back to the suite2p_photostim_single siHeader.
    bci_keys = ['dt_si'] if condition == 'learning' else []
    for root in LEARNING_ROOTS:
        folder = os.path.join(root, mouse, session, 'pophys') + '/'
        if os.path.isdir(folder):
            return ddct.load_hdf5(folder, bci_keys, photostim_keys), folder
    raise FileNotFoundError(f'{mouse} {session} not found in {LEARNING_ROOTS}')


def session_connectivity_change(mouse, session, condition):
    data, folder = load_session(mouse, session, condition)

    AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
    if len(AMP) < 2:
        raise RuntimeError('only one photostim epoch')
    if AMP[0].shape != AMP[1].shape:
        raise RuntimeError(f'before/after maps differ in shape '
                           f'{AMP[0].shape} vs {AMP[1].shape}')

    # Putative SST cells (criterion of sliding_window_temporal_offset_v3_by_group.py):
    # mean non-target (>30 um) amp > mean + N_SD_OUTLIER * SD. Uses the average
    # of the before and after maps so selection doesn't bias dW toward either.
    amp_masked = (AMP[0] + AMP[1]) / 2
    amp_masked[stimDist < 30] = np.nan
    mean_amp_nontarg = np.nanmean(amp_masked, axis=1)
    outlier_threshold = (np.nanmean(mean_amp_nontarg) +
                         N_SD_OUTLIER * np.nanstd(mean_amp_nontarg))
    is_outlier = mean_amp_nontarg > outlier_threshold
    if mouse not in PAN_NEURONAL_MICE:
        is_outlier = np.zeros_like(is_outlier)   # E-only: every cell is E
    cell_masks = {'E': ~is_outlier & np.isfinite(mean_amp_nontarg)}

    dw_list, amp0_list, amp1_list, dist_list, nt_list = [], [], [], [], []
    targ0, targ1 = [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where(
            (stimDist[:, gi] < dist_target_lt) &
            (AMP[0][:, gi] > amp0_thr) &
            (AMP[1][:, gi] > amp1_thr)
        )[0]
        if cl.size == 0 or np.any(is_outlier[cl]):
            continue
        nontarg = np.where(
            (stimDist[:, gi] > dist_nontarg_min) &
            (stimDist[:, gi] < dist_nontarg_max)
        )[0]
        if nontarg.size == 0:
            continue
        amp0_list.append(AMP[0][nontarg, gi])
        amp1_list.append(AMP[1][nontarg, gi])
        dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
        dist_list.append(stimDist[nontarg, gi])
        nt_list.append(nontarg)
        targ0.append(np.nanmean(AMP[0][cl, gi]))
        targ1.append(np.nanmean(AMP[1][cl, gi]))

    if len(dw_list) == 0:
        raise RuntimeError('no valid pairs')

    dw_all = np.concatenate(dw_list)
    a0_all = np.concatenate(amp0_list)
    a1_all = np.concatenate(amp1_list)
    dist_all = np.concatenate(dist_list)
    nt_all = np.concatenate(nt_list)
    targ0, targ1 = np.array(targ0), np.array(targ1)

    # One result per postsynaptic cell group; targets are not restricted.
    results = []
    for cell_group, cmask in cell_masks.items():
        ok = (np.isfinite(dw_all) & np.isfinite(a0_all) & np.isfinite(a1_all) &
              cmask[nt_all])
        if np.sum(ok) < 20:
            continue
        dw, a0, a1, dist = dw_all[ok], a0_all[ok], a1_all[ok], dist_all[ok]

        dw_by_dist = np.full(len(DIST_BINS) - 1, np.nan)
        absdw_by_dist = np.full(len(DIST_BINS) - 1, np.nan)
        for bi in range(len(DIST_BINS) - 1):
            m = (dist >= DIST_BINS[bi]) & (dist < DIST_BINS[bi + 1])
            if np.sum(m) > 10:
                dw_by_dist[bi] = np.median(dw[m])
                absdw_by_dist[bi] = np.median(np.abs(dw[m]))

        results.append({
            'mouse': mouse,
            'session': session,
            'condition': condition,
            'cell_group': cell_group,
            'n_cells': int(np.sum(cmask)),
            'n_outliers': int(np.sum(is_outlier)),
            'n_groups': len(targ0),
            'n_pairs': len(dw),
            'mean_dw': np.mean(dw),
            'median_dw': np.median(dw),
            'mean_abs_dw': np.mean(np.abs(dw)),
            'median_abs_dw': np.median(np.abs(dw)),
            'std_dw': np.std(dw),
            'iqr_dw': np.subtract(*np.percentile(dw, [75, 25])),
            'w_corr': pearsonr(a0, a1)[0],
            'mean_w0': np.mean(a0),
            'target_amp0': np.mean(targ0),
            'target_amp1': np.mean(targ1),
            'target_change': np.mean(targ1 - targ0),
            'dw_by_dist': dw_by_dist,
            'absdw_by_dist': absdw_by_dist,
        })
    return results


all_results = []
for condition, sessions in [('control', CONTROL_SESSIONS),
                            ('learning', LEARNING_SESSIONS)]:
    for mouse, session, fov in sessions:
        print(f"\n--- {condition}: {mouse} {session} {fov} ---")
        try:
            for r in session_connectivity_change(mouse, session, condition):
                r['fov'] = fov
                all_results.append(r)
                print(f"  {r['cell_group']:8s} {r['n_cells']:4d} cells, "
                      f"{r['n_groups']} groups, {r['n_pairs']} pairs, "
                      f"median|dW|={r['median_abs_dw']:.4f}, "
                      f"r(W0,W1)={r['w_corr']:.3f}")
        except FileNotFoundError:
            print("  Skipping -- file not found.")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

for condition in ['control', 'learning']:
    n = sum(r['condition'] == condition for r in all_results)
    print(f"{condition}: {n} sessions")

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'connectivity_change_control_vs_learning.npy'),
        all_results, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Load
# ============================================================================
all_results = list(np.load(
    os.path.join(RESULTS_DIR, 'connectivity_change_control_vs_learning.npy'),
    allow_pickle=True))
print(f"Loaded {len(all_results)} sessions")

#%% ============================================================================
# CELL 6: Stats — session level and within-mouse
# ============================================================================
METRICS = [
    ('median_abs_dw', 'Median |dW|'),
    ('iqr_dw', 'IQR of dW'),
    ('w_corr', 'r(W before, W after)'),
    ('median_dw', 'Median dW'),
    ('target_change', 'Target response change'),
]

qc_all = [r for r in all_results if r['n_groups'] >= MIN_GROUPS]
for r in all_results:
    if r['n_groups'] < MIN_GROUPS and r['cell_group'] == 'all':
        print(f"QC exclude: {r['condition']} {r['mouse']} {r['session']} "
              f"({r['n_groups']} groups)")


def mouse_mean(results, mouse, key):
    v = [r[key] for r in results if r['mouse'] == mouse and np.isfinite(r[key])]
    return np.mean(v) if v else np.nan


def within_mouse_permutation(ctrl, lrn, key, mice, n_perm=N_PERM, seed=0):
    """Mean over mice of (learning mean - control mean); p from shuffling
    learning/control labels among each mouse's own sessions."""
    per_mouse = []
    for m in mice:
        v = np.array([r[key] for r in ctrl if r['mouse'] == m] +
                     [r[key] for r in lrn if r['mouse'] == m])
        n_c = sum(r['mouse'] == m for r in ctrl)
        per_mouse.append((v, n_c))
    stat = lambda pm: np.mean([v[n_c:].mean() - v[:n_c].mean() for v, n_c in pm])
    obs = stat(per_mouse)
    rng = np.random.default_rng(seed)
    null = np.array([stat([(rng.permutation(v), n_c) for v, n_c in per_mouse])
                     for _ in range(n_perm)])
    p = (np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)
    return obs, p


stats = {}
for cg in MOUSE_GROUPS:
    qc = [r for r in qc_all if r['mouse'] in MOUSE_GROUPS[cg]]
    ctrl = [r for r in qc if r['condition'] == 'control']
    lrn = [r for r in qc if r['condition'] == 'learning']
    mice_both = sorted({r['mouse'] for r in ctrl} & {r['mouse'] for r in lrn})
    n_cells_c = np.median([r['n_cells'] for r in ctrl]) if ctrl else np.nan
    n_cells_l = np.median([r['n_cells'] for r in lrn]) if lrn else np.nan
    print(f"\n=== Mice: {cg} {MOUSE_GROUPS[cg]}, E->E pairs (median {n_cells_c:.0f} control / "
          f"{n_cells_l:.0f} learning cells per session) ===")
    stats[cg] = {'ctrl': ctrl, 'lrn': lrn, 'mice_both': mice_both}
    for key, label in METRICS:
        c = np.array([r[key] for r in ctrl])
        l = np.array([r[key] for r in lrn])
        p_sess = mannwhitneyu(c, l).pvalue if len(c) > 1 and len(l) > 1 else np.nan
        mc = np.array([mouse_mean(ctrl, m, key) for m in mice_both])
        ml = np.array([mouse_mean(lrn, m, key) for m in mice_both])
        diff, p_mouse = within_mouse_permutation(ctrl, lrn, key, mice_both)
        stats[cg][key] = dict(mc=mc, ml=ml, p_sess=p_sess, p_mouse=p_mouse)
        print(f"{label:28s} control {np.median(c) if len(c) else np.nan:+.4f} (n={len(c)})  "
              f"learning {np.median(l) if len(l) else np.nan:+.4f} (n={len(l)})  "
              f"session MWU p={p_sess:.3f}  within-mouse learn-ctrl {diff:+.4f} "
              f"perm p={p_mouse:.3f} (n={len(mice_both)} mice)")
        print('      per mouse (ctrl / learn): ' + ', '.join(
            f'{m} {vc:+.4f}/{vl:+.4f}' for m, vc, vl in zip(mice_both, mc, ml)))

#%% ============================================================================
# CELL 7: Figures — control vs learning, one per postsynaptic cell group
# ============================================================================
mouse_list = sorted({r['mouse'] for r in qc_all})
mouse_colors = {m: plt.cm.tab10(i % 10) for i, m in enumerate(mouse_list)}

for cg in MOUSE_GROUPS:
    S = stats[cg]
    ctrl, lrn, mice_both = S['ctrl'], S['lrn'], S['mice_both']
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (key, label) in zip(axes.flat, METRICS):
        s = S[key]
        rng = np.random.default_rng(0)
        for xi, group in [(0, ctrl), (1, lrn)]:
            for r in group:
                ax.plot(xi + rng.uniform(-0.12, 0.12), r[key], 'o',
                        color=mouse_colors[r['mouse']], alpha=0.5, markersize=6)
        for m, vc, vl in zip(mice_both, s['mc'], s['ml']):
            ax.plot([0, 1], [vc, vl], '-', color=mouse_colors[m], linewidth=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control\n(water)', 'Learning\n(BCI)'])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel(label)
        ax.set_title(f'{label}\nsession p={s["p_sess"]:.3f}, within-mouse p={s["p_mouse"]:.3f}',
                     fontsize=12)
        if key in ('median_dw', 'target_change'):
            ax.axhline(0, color='k', alpha=0.3)

    # Distance profile of median |dW|
    ax = axes.flat[-1]
    centers = np.sqrt(DIST_BINS[:-1] * DIST_BINS[1:])
    for group, color, name in [(ctrl, '#3498db', 'Control'), (lrn, '#e74c3c', 'Learning')]:
        if len(group) == 0:
            continue
        prof = np.array([r['absdw_by_dist'] for r in group])
        mu = np.nanmedian(prof, axis=0)
        se = np.nanstd(prof, axis=0) / np.sqrt(np.sum(np.isfinite(prof), axis=0))
        ax.errorbar(centers, mu, yerr=se, fmt='o-', color=color, capsize=4,
                    linewidth=2, label=f'{name} (n={len(group)})')
    ax.set_xscale('log')
    ax.set_xlabel('Distance from stim site (um)')
    ax.set_ylabel('Median |dW|')
    ax.legend()

    handles = [plt.Line2D([], [], marker='o', ls='', color=mouse_colors[m], label=m)
               for m in mouse_list]
    fig.legend(handles=handles, loc='lower center', ncol=len(mouse_list),
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f'E->E photostim connectivity change, {cg} mice: '
                 'water control vs BCI learning', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR,
                             f'fig_connectivity_change_control_vs_learning_E_{cg}.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    plt.close(fig)
print("Figures saved.")
