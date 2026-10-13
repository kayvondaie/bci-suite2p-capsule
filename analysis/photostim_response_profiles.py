#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Single-cell photostim response profiles, E-only vs pan-neuronal mice.

Per cell, from the average of the before/after photostim maps (non-target
responses, stimDist > 30 um):
  1. spatial profile
       r_30_100, r_100_200, r_200_400, r_400p : mean response in distance bins
       slope_logd : slope of response vs log10(distance), 30-1000 um
       far_minus_near : mean(200-1000 um) - mean(30-100 um)
  2. suppression / excitation
       noise = robust SD (1.4826 * MAD) of the cell's responses to groups
               > 400 um away
       frac_sup / frac_exc : fraction of groups at 30-300 um with response
               < -2 noise / > +2 noise
Features are z-scored within session. Comparison as in
pairwise_correlation_profiles.py: E-only cells as reference, photostim
outliers tracked, Mahalanobis excess above the E-only 99th percentile.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import matplotlib.pyplot as plt
import numpy as np
import data_dict_create_module_test as ddct
from BCI_data_helpers import compute_amp_from_photostim

CO_DATA = '/root/capsule/data'
ROOTS = {
    'E-only': (f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025',
               ['BCI102', 'BCI103', 'BCI104', 'BCI105', 'BCI106', 'BCI109']),
    'pan-neuronal': (f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal',
                     ['BCI88', 'BCI93', 'BCI107']),
}
RESULTS_DIR = '/root/capsule/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12})

N_SD_OUTLIER = 2
DIST_BINS = [(30, 100), (100, 200), (200, 400), (400, 1000)]
FEATURES = ['r_30_100', 'r_100_200', 'r_200_400', 'r_400p', 'slope_logd',
            'far_minus_near', 'frac_sup', 'frac_exc']
MIN_GROUPS_PER_BIN = 3
print("Setup complete!")

#%% ============================================================================
# CELL 2: Per-cell features
# ============================================================================
def cell_features(A, D):
    """A, D: (n_cells, n_groups) response and distance (um)."""
    n = A.shape[0]
    f = np.full((n, len(FEATURES)), np.nan)
    for i in range(n):
        a, d = A[i], D[i]
        ok = np.isfinite(a) & np.isfinite(d) & (d > 30) & (d < 1000)
        if ok.sum() < 10:
            continue
        a, d = a[ok], d[ok]
        for bi, (lo, hi) in enumerate(DIST_BINS):
            m = (d >= lo) & (d < hi)
            if m.sum() >= MIN_GROUPS_PER_BIN:
                f[i, bi] = a[m].mean()
        x = np.log10(d)
        if np.std(x) > 0:
            f[i, 4] = np.polyfit(x, a, 1)[0]
        near, far = a[d < 100], a[d >= 200]
        if len(near) >= MIN_GROUPS_PER_BIN and len(far) >= MIN_GROUPS_PER_BIN:
            f[i, 5] = far.mean() - near.mean()
        noise_ref = a[d >= 400]
        if len(noise_ref) >= 5:
            noise = 1.4826 * np.median(np.abs(noise_ref - np.median(noise_ref)))
            test = a[d < 300]
            if noise > 0 and len(test) >= 5:
                z = (test - np.median(noise_ref)) / noise
                f[i, 6] = np.mean(z < -2)
                f[i, 7] = np.mean(z > 2)
    return f


cells = []
for cohort, (root, mice) in ROOTS.items():
    for mouse in mice:
        mdir = os.path.join(root, mouse)
        if not os.path.isdir(mdir):
            continue
        for session in sorted(os.listdir(mdir)):
            folder = os.path.join(mdir, session, 'pophys') + '/'
            if not os.path.isdir(folder):
                continue
            try:
                data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
                AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
                if len(AMP) < 2 or AMP[0].shape != AMP[1].shape:
                    raise RuntimeError('maps differ / single map')
            except Exception as e:
                print(f"{cohort} {mouse} {session}: skipped ({e})")
                continue
            A = (AMP[0] + AMP[1]) / 2
            a = A.copy()
            a[stimDist < 30] = np.nan
            m = np.nanmean(a, axis=1)
            out = m > np.nanmean(m) + N_SD_OUTLIER * np.nanstd(m)
            raw = cell_features(A, stimDist)
            fz = (raw - np.nanmean(raw, axis=0)) / np.nanstd(raw, axis=0)
            for i in range(len(fz)):
                cells.append({'cohort': cohort, 'mouse': mouse, 'session': session,
                              'z': fz[i], 'raw': raw[i], 'photostim_outlier': bool(out[i])})
            print(f"{cohort:12s} {mouse:7s} {session}: {len(fz)} cells, {A.shape[1]} groups, "
                  f"{out.sum()} outliers, median frac_sup {np.nanmedian(raw[:, 6]):.3f}, "
                  f"frac_exc {np.nanmedian(raw[:, 7]):.3f}")

np.save(os.path.join(RESULTS_DIR, 'photostim_response_profiles.npy'), cells,
        allow_pickle=True)
print(f"\nSaved {len(cells)} cells.")

#%% ============================================================================
# CELL 3: Comparison
# ============================================================================
Zall = np.array([c['z'] for c in cells])
Rall = np.array([c['raw'] for c in cells])
cohort = np.array([c['cohort'] for c in cells])
pso = np.array([c['photostim_outlier'] for c in cells])
groups = [('E-only', (cohort == 'E-only') & ~pso),
          ('E-only outliers', (cohort == 'E-only') & pso),
          ('pan non-outliers', (cohort == 'pan-neuronal') & ~pso),
          ('pan outliers', (cohort == 'pan-neuronal') & pso)]

print("\nPer-feature medians, within-session z  |  raw")
print(f"  {'feature':15s} " + ' '.join(f'{g:>17s}' for g, _ in groups))
for j, fn in enumerate(FEATURES):
    zs = ' '.join(f"{np.nanmedian(Zall[m, j]):+8.2f}|{np.nanmedian(Rall[m, j]):+8.3f}"
                  for _, m in groups)
    print(f"  {fn:15s} {zs}")

# Shape-only novelty (features not used to define outliers): slope, far-near, frac_sup, frac_exc
SHAPE = [4, 5, 6, 7]
ok = np.all(np.isfinite(Zall[:, SHAPE]), axis=1)
E = Zall[ok & (cohort == 'E-only')][:, SHAPE]
mu, icov = E.mean(axis=0), np.linalg.inv(np.cov(E.T))
d = np.full(len(Zall), np.nan)
d[ok] = np.sqrt(np.einsum('ij,jk,ik->i', Zall[ok][:, SHAPE] - mu, icov,
                          Zall[ok][:, SHAPE] - mu))
thr = np.nanpercentile(d[cohort == 'E-only'], 99)
print(f"\nShape-feature Mahalanobis threshold (E-only 99th pct) = {thr:.2f}")
for name, mask in groups:
    v = d[mask & ok]
    print(f"  {name:18s} n={len(v):6d}  fraction above threshold = {np.mean(v > thr):.3f}")

for name, j in [('frac_sup', 6), ('frac_exc', 7)]:
    print(f"\n{name}: fraction of cells with raw value >= 0.2")
    for g, m in groups:
        v = Rall[m, j]
        v = v[np.isfinite(v)]
        print(f"  {g:18s} {np.mean(v >= 0.2):.3f}  (median {np.median(v):.3f}, n={len(v)})")

#%% ============================================================================
# CELL 4: Figures
# ============================================================================
colors = {'E-only': '0.3', 'E-only outliers': '#e67e22',
          'pan non-outliers': '#3498db', 'pan outliers': '#e74c3c'}

# Population distance profiles (raw, mean +- SEM across cells)
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
centers = [np.sqrt(lo * hi) for lo, hi in DIST_BINS]
for g, m in groups:
    prof = Rall[m][:, :4]
    mu_p = np.nanmean(prof, axis=0)
    se_p = np.nanstd(prof, axis=0) / np.sqrt(np.sum(np.isfinite(prof), axis=0))
    ax.errorbar(centers, mu_p, yerr=se_p, fmt='o-', color=colors[g], capsize=3,
                linewidth=2, label=f'{g} (n={m.sum()})')
ax.axhline(0, color='k', alpha=0.3)
ax.set_xscale('log')
ax.set_xlabel('Distance from stim site (um)')
ax.set_ylabel('Mean response (AMP)')
ax.set_title('Spatial response profile')
ax.legend(fontsize=9)

for ax, (j, label) in zip(axes[1:], [(6, 'frac_sup (suppressed groups, 30-300 um)'),
                                    (7, 'frac_exc (excited groups, 30-300 um)')]):
    bins = np.linspace(0, 0.6, 31)
    for g, m in groups:
        v = Rall[m, j]
        v = v[np.isfinite(v)]
        ax.hist(v, bins=bins, density=True, histtype='step', linewidth=2,
                color=colors[g], label=g)
    ax.set_yscale('log')
    ax.set_xlabel(label)
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_photostim_profiles_summary.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

# Scatter of shape features, within-session z (log-density background)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for col, (fx, fy) in enumerate([(5, 6), (5, 7), (6, 7)]):
    for row, coh in enumerate(['E-only', 'pan-neuronal']):
        ax = axes[row, col]
        m = (cohort == coh) & np.isfinite(Zall[:, fx]) & np.isfinite(Zall[:, fy])
        from matplotlib.colors import LogNorm
        ax.hist2d(Zall[m & ~pso, fx], Zall[m & ~pso, fy], bins=70,
                  range=[[-5, 8], [-4, 8]], cmap='Greys', norm=LogNorm())
        ax.scatter(Zall[m & pso, fx], Zall[m & pso, fy], s=7, color='#e74c3c',
                   label=f'photostim outliers (n={np.sum(m & pso)})')
        ax.set_xlabel(f'{FEATURES[fx]} (within-session z)')
        ax.set_ylabel(f'{FEATURES[fy]} (within-session z)')
        ax.set_title(f'{coh}: {np.sum(m)} cells')
        ax.legend(fontsize=9, loc='upper left')
plt.suptitle('Photostim response-shape features (grey: log density of all cells, '
             'red: photostim outliers)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_photostim_profiles_scatter.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)
print("Figures saved.")
