"""List spont-related keys and shapes in data_main for learning and control sessions."""
import glob
import os

import h5py

CO_DATA = '/root/capsule/data'
paths = (sorted(glob.glob(f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal/*/*/pophys/data_main*.h5')) +
         sorted(glob.glob(f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025/BCI10[35]/*/pophys/data_main*.h5')) +
         sorted(glob.glob(f'{CO_DATA}/control_water_photostim/*/data_main_*.h5')))
for p in paths:
    with h5py.File(p, 'r') as f:
        keys = {k: f[k].shape for k in f.keys()
                if 'spont' in k or k in ('F', 'Fraw', 'df_closedloop')}
    print(p.replace(CO_DATA + '/', ''), keys)
