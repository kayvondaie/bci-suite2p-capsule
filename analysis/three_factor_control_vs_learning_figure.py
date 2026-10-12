#%% ============================================================================
# Paper-style panel: 3-factor MLR, BCI learning vs water control (E->E pairs)
# ============================================================================
# Same layout and conventions as CELL 9 of three_factor_variance_explained_figure.py:
#   a: pooled out-of-fold predictions vs actual dW, pf.mean_bin_plot, 5 bins,
#      SEM error bars. Learning black, control gray. Predictions are converted
#      to rank-based normal scores within each session before pooling: control
#      predictions are heavy-tailed (a few pairs get large predictions), so a
#      plain z-score compresses most pairs toward 0. Rank scores put every
#      session on the same quantile axis, consistent with the Spearman p in b.
#   b: per-session -log10(p), Spearman over concatenated out-of-fold predictions.
# Reads three_factor_control_vs_learning.npy (three_factor_control_vs_learning.py).
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, pearsonr, rankdata, spearmanr
import plotting_functions as pf

RESULTS_DIR = '/root/capsule/results'
MIN_GROUPS = 10

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 8,
    'svg.fonttype': 'none',
})

res = np.load(os.path.join(RESULTS_DIR, 'three_factor_control_vs_learning.npy'),
              allow_pickle=True).tolist()
res = [r for r in res if r['n_groups'] >= MIN_GROUPS]


def rank_z(x):
    """Rank-based normal scores: rank -> standard-normal quantile."""
    return norm.ppf((rankdata(x) - 0.5) / len(x))


def session_logp(sessions):
    p = np.array([spearmanr(r['Y_pred_all'], r['Y_test_all'])[1] for r in sessions])
    return -np.log10(np.clip(p, 1e-300, 1.0)), p


for lv, lv_label in [('learning', 'Learning'), ('learning_pseudo', 'Learning (pseudo)')]:
    ctrl = [r for r in res if r['condition'] == 'control']
    lrn = [r for r in res if r['condition'] == lv]

    FIG_W = 3
    FIG_H = 1.5
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H),
                             gridspec_kw={'left': 0.17, 'right': 0.97,
                                          'bottom': 0.27, 'top': 0.92,
                                          'wspace': 0.55})

    # --- Panel A: pooled binned plots, z-scored predictions ---
    ax = axes[0]
    plt.sca(ax)
    pooled_r = {}
    for group, color, name in [(ctrl, '0.5', 'control'), (lrn, 'k', lv)]:
        Yp = np.concatenate([rank_z(r['Y_pred_all']) for r in group])
        Yt = np.concatenate([r['Y_test_all'] for r in group])
        pf.mean_bin_plot(Yp, Yt, col=5, color=color)
        pooled_r[name] = pearsonr(Yp, Yt)[0]
    ax.axhline(0, color='k', ls='-', alpha=0.2, linewidth=0.5)
    ax.set_xlabel('Predicted $\\Delta W$ (z)')
    ax.set_ylabel('Actual $\\Delta W$ (z)')
    ax.text(0.98, 0.15, lv_label, color='k', transform=ax.transAxes,
            ha='right', va='bottom')
    ax.text(0.98, 0.03, 'Control', color='0.5', transform=ax.transAxes,
            ha='right', va='bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # --- Panel B: per-session -log10(p) ---
    ax = axes[1]
    lp_c, p_c = session_logp(ctrl)
    lp_l, p_l = session_logp(lrn)
    rng = np.random.default_rng(42)
    jitter = 0.08
    ax.scatter(0 + rng.uniform(-jitter, jitter, len(lp_c)), lp_c, s=15,
               color='0.5', zorder=5, edgecolor='none')
    ax.scatter(1 + rng.uniform(-jitter, jitter, len(lp_l)), lp_l, s=15,
               color='k', zorder=5, edgecolor='none')
    ax.axhline(-np.log10(0.05), color='k', ls=':', alpha=0.4, linewidth=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Control', lv_label.split(' ')[0]])
    ax.get_xticklabels()[0].set_color('0.5')
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel('$-\\log_{10}(p)$')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    print(f"{lv}: pooled r control {pooled_r['control']:+.4f}, {lv} {pooled_r[lv]:+.4f}; "
          f"sig control {np.sum(p_c < 0.05)}/{len(p_c)}, {lv} {np.sum(p_l < 0.05)}/{len(p_l)}")

    fname = f'three_factor_control_vs_{lv}_overlay'
    fig.savefig(os.path.join(RESULTS_DIR, f'{fname}.png'), dpi=300)
    fig.savefig(os.path.join(RESULTS_DIR, f'{fname}.svg'))
    plt.close(fig)
    print(f"Saved {fname}")
