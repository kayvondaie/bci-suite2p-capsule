#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
Pairwise-correlation profiles of cells in E-only vs pan-neuronal mice.

Per session: BCI-epoch df_closedloop -> detrend (minus 30 s moving average),
z-score, full cell x cell correlation matrix. Per cell, features of its
correlation profile (with all other cells):
    mean_r  : mean r (population coupling)
    sd_r    : SD of r
    skew_r  : skew of r
    p95_r   : 95th percentile of r (strongest partners)
    pc1     : loading on the session's first PC of the correlation matrix
Features are z-scored within session (removes cohort-level differences in
correlation scale from indicator / frame rate / state).

Novelty: Mahalanobis distance of each cell from the pooled E-only cloud;
threshold = E-only 99th percentile. Excess fraction above threshold in the
pan-neuronal mice = candidate population not present in E-only mice.

Photostim outliers (mean non-target amp > mean + 2 SD, averaged before/after
maps) are tracked in both datasets.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.stats import skew
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

DETREND_SEC = 30
N_SD_OUTLIER = 2
FEATURES = ['mean_r', 'sd_r', 'skew_r', 'p95_r', 'pc1']
print("Setup complete!")

#%% ============================================================================
# CELL 2: Per-session correlation profiles
# ============================================================================
def preprocess(F, fs):
    F = np.asarray(F, dtype=float)
    F = np.where(np.isfinite(F), F, np.nanmean(F, axis=1, keepdims=True))
    F = F - uniform_filter1d(F, size=max(3, int(DETREND_SEC * fs)), axis=1, mode='nearest')
    sd = F.std(axis=1, keepdims=True)
    sd[sd == 0] = 1
    return (F - F.mean(axis=1, keepdims=True)) / sd


def session_profiles(folder, mouse):
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw'])
    Z = preprocess(data['df_closedloop'], 1.0 / float(data['dt_si']))
    n = Z.shape[0]
    C = Z @ Z.T / Z.shape[1]
    np.fill_diagonal(C, np.nan)
    off = C[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    Cz = np.nan_to_num(C)
    evals, evecs = np.linalg.eigh(Cz)
    pc1 = evecs[:, -1]
    pc1 = pc1 * np.sign(np.sum(pc1))
    feats = np.column_stack([off.mean(axis=1), off.std(axis=1), skew(off, axis=1),
                             np.percentile(off, 95, axis=1), pc1])

    out = np.zeros(n, dtype=bool)
    try:
        AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
        if len(AMP) == 2 and AMP[0].shape == AMP[1].shape and AMP[0].shape[0] == n:
            a = (AMP[0] + AMP[1]) / 2
            a = a.copy()
            a[stimDist < 30] = np.nan
            m = np.nanmean(a, axis=1)
            out = m > np.nanmean(m) + N_SD_OUTLIER * np.nanstd(m)
    except Exception as e:
        print(f"  (no photostim outliers: {e})")
    return feats, out, Z.shape[1]


cells = []   # one row per cell
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
                feats, out, T = session_profiles(folder, mouse)
            except Exception as e:
                print(f"{cohort} {mouse} {session}: skipped ({e})")
                continue
            raw = feats.copy()
            fz = (feats - np.nanmean(feats, axis=0)) / np.nanstd(feats, axis=0)
            for i in range(len(fz)):
                cells.append({'cohort': cohort, 'mouse': mouse, 'session': session,
                              'z': fz[i], 'raw': raw[i], 'photostim_outlier': bool(out[i])})
            print(f"{cohort:12s} {mouse:7s} {session}: {len(fz)} cells, {T} frames, "
                  f"{out.sum()} photostim outliers, raw mean r {np.nanmedian(raw[:, 0]):.3f}")

np.save(os.path.join(RESULTS_DIR, 'pairwise_correlation_profiles.npy'), cells,
        allow_pickle=True)
print(f"\nSaved {len(cells)} cells.")

#%% ============================================================================
# CELL 3: Novelty relative to the E-only cloud
# ============================================================================
Zall = np.array([c['z'] for c in cells])
cohort = np.array([c['cohort'] for c in cells])
pso = np.array([c['photostim_outlier'] for c in cells])
ok = np.all(np.isfinite(Zall), axis=1)

E = Zall[ok & (cohort == 'E-only')]
mu, cov = E.mean(axis=0), np.cov(E.T)
icov = np.linalg.inv(cov)
d = np.full(len(Zall), np.nan)
d[ok] = np.sqrt(np.einsum('ij,jk,ik->i', Zall[ok] - mu, icov, Zall[ok] - mu))
thr = np.nanpercentile(d[cohort == 'E-only'], 99)


def frac_above(mask):
    v = d[mask & ok]
    return np.mean(v > thr), len(v)


print(f"\nMahalanobis threshold (E-only 99th pct) = {thr:.2f}")
for name, mask in [
    ('E-only, all cells', cohort == 'E-only'),
    ('E-only, photostim outliers', (cohort == 'E-only') & pso),
    ('E-only, non-outliers', (cohort == 'E-only') & ~pso),
    ('pan, all cells', cohort == 'pan-neuronal'),
    ('pan, photostim outliers', (cohort == 'pan-neuronal') & pso),
    ('pan, non-outliers', (cohort == 'pan-neuronal') & ~pso),
]:
    f, n = frac_above(mask)
    print(f"  {name:28s} n={n:6d}  fraction above threshold = {f:.3f}")

print("\nPer-feature medians (within-session z): E-only / pan non-outliers / pan photostim outliers")
for j, fn in enumerate(FEATURES):
    a = np.nanmedian(Zall[(cohort == 'E-only'), j])
    b = np.nanmedian(Zall[(cohort == 'pan-neuronal') & ~pso, j])
    c = np.nanmedian(Zall[(cohort == 'pan-neuronal') & pso, j])
    print(f"  {fn:7s} {a:+.2f} / {b:+.2f} / {c:+.2f}")

print("\nPer-session fraction above threshold (pan, non-outliers):")
for s in sorted({(c['mouse'], c['session']) for c in cells if c['cohort'] == 'pan-neuronal'}):
    m = np.array([(c['mouse'], c['session']) == s for c in cells]) & ~pso
    f, n = frac_above(m)
    print(f"  {s[0]:7s} {s[1]}  {f:.3f} (n={n})")

#%% ============================================================================
# CELL 4: Figures
# ============================================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
pairs = [(0, 1), (0, 3), (0, 2)]
for col, (fx, fy) in enumerate(pairs):
    for row, coh in enumerate(['E-only', 'pan-neuronal']):
        ax = axes[row, col]
        m = ok & (cohort == coh)
        ax.hist2d(Zall[m & ~pso, fx], Zall[m & ~pso, fy], bins=80,
                  range=[[-4, 8], [-4, 8]], cmap='Greys', cmin=1)
        ax.scatter(Zall[m & pso, fx], Zall[m & pso, fy], s=6, color='#e74c3c',
                   label=f'photostim outliers (n={np.sum(m & pso)})')
        ax.set_xlabel(f'{FEATURES[fx]} (within-session z)')
        ax.set_ylabel(f'{FEATURES[fy]} (within-session z)')
        ax.set_title(f'{coh}: {np.sum(m)} cells')
        ax.legend(fontsize=9, loc='upper left')
plt.suptitle('Correlation-profile features (grey: all cells, red: photostim outliers)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_corr_profiles_scatter.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
bins = np.linspace(0, np.nanpercentile(d, 99.9), 80)
for name, mask, color in [
    ('E-only', cohort == 'E-only', '0.4'),
    ('pan, non-outliers', (cohort == 'pan-neuronal') & ~pso, '#3498db'),
    ('pan, photostim outliers', (cohort == 'pan-neuronal') & pso, '#e74c3c'),
    ('E-only, photostim outliers', (cohort == 'E-only') & pso, '#e67e22'),
]:
    v = d[mask & ok]
    axes[0].hist(v, bins=bins, density=True, histtype='step', linewidth=2, color=color,
                 label=f'{name} ({np.mean(v > thr):.1%} > thr)')
axes[0].axvline(thr, color='k', ls='--', label='E-only 99th pct')
axes[0].set_xlabel('Mahalanobis distance from E-only cloud')
axes[0].set_ylabel('Density')
axes[0].set_yscale('log')
axes[0].legend(fontsize=9)

for name, mask, color in [('E-only', cohort == 'E-only', '0.4'),
                          ('pan, non-outliers', (cohort == 'pan-neuronal') & ~pso, '#3498db'),
                          ('pan, photostim outliers', (cohort == 'pan-neuronal') & pso, '#e74c3c')]:
    axes[1].hist(Zall[mask & ok, 0], bins=np.linspace(-4, 8, 80), density=True,
                 histtype='step', linewidth=2, color=color, label=name)
axes[1].set_xlabel('mean_r (population coupling, within-session z)')
axes[1].set_ylabel('Density')
axes[1].set_yscale('log')
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_corr_profiles_novelty.png'), dpi=130,
            bbox_inches='tight')
plt.close(fig)
print("Figures saved.")
