"""Print spont_pre / spont_post size and duration for each control session."""
import glob
import os

import h5py
import numpy as np

ROOT = '/root/capsule/data/control_water_photostim'

for d in sorted(glob.glob(f'{ROOT}/*_20*')):
    sh = np.load(f'{d}/suite2p_photostim_single/plane0/siHeader.npy',
                 allow_pickle=True).item()
    fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
    out = [os.path.basename(d), f'fs={fs:.1f} Hz']
    with h5py.File(glob.glob(f'{d}/data_main_*.h5')[0], 'r') as f:
        for k in ['spont_pre', 'spont_post']:
            if k in f:
                n_cells, n_frames = f[k].shape
                out.append(f'{k}: {n_cells} x {n_frames} ({n_frames / fs / 60:.1f} min)')
            else:
                out.append(f'{k}: MISSING')
    print(' | '.join(out))
