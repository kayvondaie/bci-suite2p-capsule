"""For each learning session: which suite2p spont folders exist, their F.npy shape,
and whether the ROI count matches suite2p_BCI (i.e. the same ROI set)."""
import glob
import os

import numpy as np

CO_DATA = '/root/capsule/data'
folders = (sorted(glob.glob(f'{CO_DATA}/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal/*/*/pophys')) +
           sorted(glob.glob(f'{CO_DATA}/BCI_three_factor_2025/three_factor_data_2025/BCI10[35]/*/pophys')))
for d in folders:
    tag = '/'.join(d.split('/')[-3:-1])
    out = []
    ref = None
    for s2p in sorted(glob.glob(d + '/suite2p_*')):
        name = os.path.basename(s2p)
        fpath = os.path.join(s2p, 'plane0', 'F.npy')
        if os.path.exists(fpath):
            shp = np.load(fpath, mmap_mode='r').shape
            if name == 'suite2p_BCI':
                ref = shp[0]
            out.append(f'{name}:{shp}')
        else:
            out.append(f'{name}:noF')
    print(f'{tag:16s} BCI ROIs={ref} | ' + ' '.join(o for o in out if 'BCI' not in o and 'photostim' not in o))
