"""Per-session scale of held-out 3-factor predictions (and actual dW), to see which
sessions dominate a pooled z-score. Reads three_factor_control_vs_learning<TAG>.npy."""
import os
import numpy as np
from scipy import stats as sps

tag = os.environ.get('TAG', '')
res = np.load(f'/root/capsule/results/three_factor_control_vs_learning{tag}.npy', allow_pickle=True)
for r in res:
    if r['n_groups'] < 10:
        continue
    yp, yt = r['Y_pred_all'], r['Y_test_all']
    rp, pp = sps.pearsonr(yp, yt)
    print(f"{r['condition']:15s} {r['mouse']:7s} {r['session']} n={len(yp):6d} wins={r['n_windows']:2d} | "
          f"pred SD={yp.std():.3f} max|pred|={np.abs(yp).max():7.2f} | "
          f"actual SD={yt.std():.3f} max|actual|={np.abs(yt).max():6.1f} | pooled-fold r={rp:+.4f} p={pp:.2g}")
