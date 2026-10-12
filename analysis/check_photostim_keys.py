"""List keys/shapes in data_photostim*.h5 for one learning and one control session."""
import glob

import h5py

CO_DATA = '/root/capsule/data'
paths = [
    f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal/BCI88/011025/pophys/data_photostim.h5',
    f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025/BCI103/012325/pophys/data_photostim.h5',
] + sorted(glob.glob(f'{CO_DATA}/control_water_photostim/731015_2025-01-27/data_photostim7*.h5'))
for p in paths:
    with h5py.File(p, 'r') as f:
        print(p.replace(CO_DATA + '/', ''))
        for k in f.keys():
            obj = f[k]
            print('   ', k, getattr(obj, 'shape', '(group)'))
