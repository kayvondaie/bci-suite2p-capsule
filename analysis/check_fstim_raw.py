"""Which photostim h5s carry raw per-repeat fluorescence (Fstim_raw) and centroids."""
import glob
import os

import h5py

CO_DATA = '/root/capsule/data'
paths = (sorted(glob.glob(f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal/*/*/pophys/data_photostim*.h5')) +
         sorted(glob.glob(f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025/BCI10[35]/*/pophys/data_photostim*.h5')) +
         sorted(glob.glob(f'{CO_DATA}/control_water_photostim/*/data_photostim*.h5')))
for p in paths:
    with h5py.File(p, 'r') as f:
        keys = {k: f[k].shape for k in ['Fstim', 'Fstim_raw', 'FstimRaw', 'Ftrace', 'seq', 'centroidX']
                if k in f}
    print(p.replace(CO_DATA + '/', ''), keys)
