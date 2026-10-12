"""Per-session significance using the convention of
three_factor_variance_explained_figure.py: Spearman over the concatenated
out-of-fold predictions. Reads three_factor_control_vs_learning.npy."""
import numpy as np
from scipy.stats import spearmanr

res = np.load('/root/capsule/results/three_factor_control_vs_learning.npy',
              allow_pickle=True).tolist()
MIN_GROUPS = 10
for cond in ['control', 'learning', 'learning_pseudo']:
    rows = [r for r in res if r['condition'] == cond and r['n_groups'] >= MIN_GROUPS]
    out = []
    for r in rows:
        rho, p = spearmanr(r['Y_pred_all'], r['Y_test_all'])
        out.append((r['mouse'], r['session'], rho, p))
    n_sig = sum(p < 0.05 for *_, p in out)
    print(f"\n{cond}: {n_sig}/{len(out)} sessions p<0.05 "
          f"(Spearman, concatenated out-of-fold), median rho {np.median([o[2] for o in out]):+.4f}")
    for m, s, rho, p in out:
        print(f"  {m:7s} {s}  rho={rho:+.4f}  p={p:.2e} {'*' if p < 0.05 else ''}")
