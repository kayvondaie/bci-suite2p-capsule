"""Spont dF/F sanity per session: baseline F0 (10th pct) distribution, max dF/F, and
cells with near-zero / negative baselines. Learning sessions: suite2p spont F.npy
(iscell cells); water sessions: data_main spont_pre / spont_post."""
import glob
import os
import sys

import h5py
import numpy as np

CO_DATA = '/root/capsule/data'
LEARN = [('BCI105', '012925'), ('BCI105', '012725'), ('BCI105', '013125'), ('BCI103', '012325'),
         ('BCI93', '020325')]
ROOTS = [f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal',
         f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025']
PAIRS = [('suite2p_spont_pre', 'suite2p_spont_post'), ('suite2p_spont', 'suite2p_spont2')]


def report(tag, epochs):
    for name, F in epochs:
        F0 = np.nanpercentile(F, 10, axis=1)
        dff = (F - F0[:, None]) / np.where(F0 > 0, F0, np.nan)[:, None]
        mx = np.nanmax(np.abs(dff), axis=1)
        q = np.percentile(F0, [0, 1, 5, 50])
        print(f'{tag:22s} {name:16s} cells={F.shape[0]:4d} | F0 min/1%/5%/median = '
              f'{q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}/{q[3]:.1f} | F0<=0: {np.sum(F0 <= 0)} | '
              f'max|dF/F| median={np.nanmedian(mx):.2f}, top3={np.sort(mx[np.isfinite(mx)])[-3:].round(1)} '
              f'(F0 of top: {F0[np.argsort(np.nan_to_num(mx))[-3:]].round(1)})')


for m, s in LEARN:
    for r in ROOTS:
        folder = os.path.join(r, m, s, 'pophys')
        if os.path.isdir(folder):
            break
    for pair in PAIRS:
        dirs = [os.path.join(folder, n, 'plane0') for n in pair]
        if all(os.path.exists(os.path.join(d, 'F.npy')) for d in dirs):
            eps = []
            for n, d in zip(pair, dirs):
                F = np.load(os.path.join(d, 'F.npy'), allow_pickle=True)
                isc = np.load(os.path.join(d, 'iscell.npy'), allow_pickle=True)
                eps.append((n, np.asarray(F[np.asarray(isc)[:, 0] == 1], dtype=float)))
            report(f'learning {m} {s}', eps)
            break
for d in sorted(glob.glob(f'{CO_DATA}/control_water_photostim/*_20*'))[:3]:
    with h5py.File(glob.glob(f'{d}/data_main_*.h5')[0], 'r') as f:
        report(f'water {os.path.basename(d)}', [(k, np.asarray(f[k][:], dtype=float))
                                                 for k in ['spont_pre', 'spont_post']])
