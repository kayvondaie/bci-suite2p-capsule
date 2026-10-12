#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
# QC for the water-control sessions (bci-suite2p-capsule output):
#   fig 1: photostim response vs distance from stim site, before and after maps
#   fig 2: target response amplitude, before vs after, one point per stim group
# AMP is computed with compute_amp_from_photostim, as in
# connectivity_change_control_vs_learning.py.
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
from BCI_data_helpers import compute_amp_from_photostim

CO_DATA = '/root/capsule/data'
CONTROL_ROOT = os.path.dirname(
    (glob.glob(f'{CO_DATA}/*/731015_2025-01-27') + [f'{CO_DATA}/x/x'])[0])
RESULTS_DIR = '/root/capsule/results'

CONTROL_SESSIONS = [
    ('BCI88',  '012725', 'FOV_04'),
    ('BCI93',  '012825', 'FOV_05'),
    ('BCI93',  '022125', 'FOV_07'),
    ('BCI107', '021925', 'FOV_03'),
    ('BCI103', '013025', 'FOV_01'),
    ('BCI103', '021925', 'FOV_03'),
    ('BCI105', '013025', 'FOV_01'),
]
SUBJECT_ID = {'BCI88': '731015', 'BCI93': '740369', 'BCI103': '772414',
              'BCI105': '754303', 'BCI107': '767715'}

dist_target_lt = 10
DIST_EDGES = np.array([0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000])

plt.rcParams.update({'font.size': 11, 'axes.titlesize': 12})

#%% ============================================================================
# CELL 2: Load AMP for each control session
# ============================================================================
sessions = []
for mouse, session, fov in CONTROL_SESSIONS:
    tag = f"{SUBJECT_ID[mouse]}_20{session[4:6]}-{session[0:2]}-{session[2:4]}"
    folder = os.path.join(CONTROL_ROOT, tag) + '/'
    data = {}
    for pkey in ['photostim', 'photostim2']:
        with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
            data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
    AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)

    # Mean response of the targeted cells (< 10 um) for each stim group
    t0, t1 = [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where(stimDist[:, gi] < dist_target_lt)[0]
        if cl.size:
            t0.append(np.nanmean(AMP[0][cl, gi]))
            t1.append(np.nanmean(AMP[1][cl, gi]))
    t0, t1 = np.array(t0), np.array(t1)
    ok = np.isfinite(t0) & np.isfinite(t1)

    sessions.append(dict(mouse=mouse, session=session, fov=fov, AMP=AMP,
                         stimDist=stimDist, t0=t0[ok], t1=t1[ok]))
    print(f"{mouse} {session} {fov}: {AMP[0].shape[0]} cells x {AMP[0].shape[1]} groups, "
          f"{ok.sum()} groups with targets, median target amp before "
          f"{np.median(t0[ok]):.3f} / after {np.median(t1[ok]):.3f}, "
          f"frac > 0.1 before {np.mean(t0[ok] > 0.1):.2f} / after {np.mean(t1[ok] > 0.1):.2f}")

#%% ============================================================================
# CELL 3: Figure 1 — response vs distance from stim site
# ============================================================================
centers = np.sqrt(np.maximum(DIST_EDGES[:-1], 1) * DIST_EDGES[1:])
rng = np.random.default_rng(0)
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for ax, s in zip(axes.flat, sessions):
    d = s['stimDist'].ravel()
    for ei, (color, name) in enumerate([('#3498db', 'before'), ('#e74c3c', 'after')]):
        a = s['AMP'][ei].ravel()
        ok = np.isfinite(a) & np.isfinite(d)
        sub = rng.choice(np.where(ok)[0], size=min(4000, ok.sum()), replace=False)
        ax.plot(np.maximum(d[sub], 1), a[sub], '.', color=color, alpha=0.08,
                markersize=2)
        med = [np.nanmedian(a[ok & (d >= lo) & (d < hi)])
               if np.sum(ok & (d >= lo) & (d < hi)) > 5 else np.nan
               for lo, hi in zip(DIST_EDGES[:-1], DIST_EDGES[1:])]
        ax.plot(centers, med, 'o-', color=color, linewidth=2, markersize=4,
                label=name)
    ax.axhline(0, color='k', alpha=0.3)
    ax.axvline(dist_target_lt, color='k', ls='--', alpha=0.3)
    ax.set_xscale('log')
    ax.set_ylim(-0.5, 2.5)
    ax.set_title(f"{s['mouse']} {s['session']} {s['fov']}")
    ax.set_xlabel('Distance from stim site (um)')
axes[0, 0].set_ylabel('Photostim response (AMP)')
axes[1, 0].set_ylabel('Photostim response (AMP)')
axes[0, 0].legend()
axes.flat[-1].axis('off')
fig.suptitle('Control sessions: response vs distance (points: random subset of '
             'cell x group pairs; line: binned median)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'qc_control_response_vs_distance.png'),
            dpi=130, bbox_inches='tight')
plt.close(fig)

#%% ============================================================================
# CELL 4: Figure 2 — target response before vs after
# ============================================================================
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, s in zip(axes.flat, sessions):
    t0, t1 = s['t0'], s['t1']
    lim = max(1.0, np.nanpercentile(np.r_[t0, t1], 98) * 1.1)
    ax.plot(t0, t1, 'o', color='#2c3e50', alpha=0.6, markersize=5)
    ax.plot([-0.5, lim], [-0.5, lim], 'k--', alpha=0.4)
    ax.axhline(0.1, color='#e74c3c', alpha=0.3)
    ax.axvline(0.1, color='#e74c3c', alpha=0.3)
    r = pearsonr(t0, t1)[0] if len(t0) > 2 else np.nan
    ax.set_xlim(-0.5, lim)
    ax.set_ylim(-0.5, lim)
    ax.set_aspect('equal')
    ax.set_title(f"{s['mouse']} {s['session']} {s['fov']}\n"
                 f"{len(t0)} groups, r={r:.2f}")
    ax.set_xlabel('Target response before')
    ax.set_ylabel('Target response after')
axes.flat[-1].axis('off')
fig.suptitle('Control sessions: target (<10 um) response, before vs after '
             '(red lines: 0.1 pair-selection threshold)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'qc_control_target_before_after.png'),
            dpi=130, bbox_inches='tight')
plt.close(fig)
print('Figures saved.')
