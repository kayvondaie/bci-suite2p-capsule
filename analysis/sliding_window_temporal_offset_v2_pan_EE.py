#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
sliding_window_temporal_offset_v2.py (Hebbian index slope vs behavior, per epoch)
on the pan-neuronal mice (BCI88, BCI93, BCI107), with three pair selections:
  all         : v2 as is
  rest        : outlier postsynaptic cells removed, outliers from the before map
                (as in sliding_window_temporal_offset_v3_by_group.py)
  strict_EE   : outliers removed as stimulated targets AND postsynaptic cells,
                outliers from the average of the before and after maps
  strict_EE_broad : strict_EE, plus cells whose frac_exc (fraction of stim
                groups at 30-300 um exciting the cell, photostim_response_profiles.py)
                exceeds the E-only 99th percentile, removed as targets and
                postsynaptic cells
Outlier = mean non-target (>30 um) amp > mean + 2 SD across cells.
Two dev2 coactivity flavours:
  dev2_lag                : pre(t) * (post(t) - epoch-specific baseline)
                            (sliding_window_temporal_offset.py)
  dev2_fulltrial_baseline : pre(t) * (post(t) - full-trial baseline)
                            (sliding_window_temporal_offset_v2.py)
Baseline = mean over the first N_BASELINE trials. Everything else (epochs,
windows, RPE, slope fit, within-session Spearman, Wilcoxon) follows v2.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon
import data_dict_create_module_test as ddct
from BCI_data_helpers import *

DATA_ROOT = '/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal'
RESULTS_DIR = '/root/capsule/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12})
print("Setup complete!")

#%% ============================================================================
# CELL 2: Configuration
# ============================================================================
mice = ['BCI88', 'BCI93', 'BCI107']
EXCLUDE = {('BCI88', '013125'), ('BCI107', '013125')}   # blown-up dW sessions

WIN_SIZE = 10
WIN_STEP = 5
tau_elig = 10
N_BASELINE = 20
N_SD_OUTLIER = 2

dist_target_lt = 10
dist_nontarg_min = 30
dist_nontarg_max = 1000
amp0_thr = 0.1
amp1_thr = 0.1

SELECTIONS = ['all', 'rest', 'strict_EE', 'strict_EE_broad']

# E-only 99th percentile of frac_exc (from photostim_response_profiles.npy)
_prof = np.load(os.path.join(RESULTS_DIR, 'photostim_response_profiles.npy'),
                allow_pickle=True).tolist()
_fe = np.array([c['raw'][7] for c in _prof if c['cohort'] == 'E-only'])
FRAC_EXC_THR = np.nanpercentile(_fe, 99)
print(f"frac_exc threshold (E-only 99th pct) = {FRAC_EXC_THR:.3f}")


def frac_exc(A, D):
    """Fraction of groups at 30-300 um exciting each cell (> 2 x far-field robust SD),
    as in photostim_response_profiles.py."""
    out = np.full(A.shape[0], np.nan)
    for i in range(A.shape[0]):
        a, d = A[i], D[i]
        ok = np.isfinite(a) & np.isfinite(d) & (d > 30) & (d < 1000)
        if ok.sum() < 10:
            continue
        a, d = a[ok], d[ok]
        ref = a[d >= 400]
        test = a[d < 300]
        if len(ref) >= 5 and len(test) >= 5:
            noise = 1.4826 * np.median(np.abs(ref - np.median(ref)))
            if noise > 0:
                out[i] = np.mean(test / noise > 2)
    return out
CC_MODES = ['dev2_lag', 'dev2_fulltrial_baseline']
EPOCH_ORDER = ['pre', 'go_cue', 'late', 'reward']
n_epochs = len(EPOCH_ORDER)

all_results = {(sel, mode): [] for sel in SELECTIONS for mode in CC_MODES}

#%% ============================================================================
# CELL 3: Main loop
# ============================================================================
def outlier_mask(amp, stimDist):
    a = amp.copy()
    a[stimDist < 30] = np.nan
    m = np.nanmean(a, axis=1)
    return m > np.nanmean(m) + N_SD_OUTLIER * np.nanstd(m), np.isfinite(m)


def build_pairs(AMP, stimDist, sel):
    """v2 pair selection with optional outlier exclusion."""
    if sel == 'rest':
        out_post, finite = outlier_mask(AMP[0], stimDist)
        out_targ = np.zeros_like(out_post)
    elif sel == 'strict_EE':
        out_post, finite = outlier_mask((AMP[0] + AMP[1]) / 2, stimDist)
        out_targ = out_post
    elif sel == 'strict_EE_broad':
        out_post, finite = outlier_mask((AMP[0] + AMP[1]) / 2, stimDist)
        fe = frac_exc((AMP[0] + AMP[1]) / 2, stimDist)
        out_post = out_post | (fe > FRAC_EXC_THR)
        out_targ = out_post
    else:
        out_post = np.zeros(AMP[0].shape[0], dtype=bool)
        out_targ = out_post
        finite = np.ones_like(out_post)
    dw_list, cl_list, nt_list = [], [], []
    for gi in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, gi] < dist_target_lt) &
                      (AMP[0][:, gi] > amp0_thr) & (AMP[1][:, gi] > amp1_thr))[0]
        if cl.size == 0 or np.any(out_targ[cl]):
            continue
        nontarg = np.where((stimDist[:, gi] > dist_nontarg_min) &
                           (stimDist[:, gi] < dist_nontarg_max) &
                           ~out_post & finite)[0]
        if nontarg.size == 0:
            continue
        dw_list.append(AMP[1][nontarg, gi] - AMP[0][nontarg, gi])
        cl_list.append(cl)
        nt_list.append(nontarg)
    return dw_list, cl_list, nt_list, int(np.sum(out_post))


for mouse in mice:
    sessions = sorted(d for d in os.listdir(os.path.join(DATA_ROOT, mouse))
                      if os.path.isdir(os.path.join(DATA_ROOT, mouse, d, 'pophys')))
    for session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        try:
            folder = os.path.join(DATA_ROOT, mouse, session, 'pophys') + '/'
            print(f"\n--- {mouse} {session} ---")
            bci_keys = ['df_closedloop', 'F', 'mouse', 'session', 'conditioned_neuron',
                        'dt_si', 'step_time', 'reward_time', 'BCI_thresholds']
            data = ddct.load_hdf5(folder, bci_keys, ['stimDist', 'favg_raw'])

            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si = data['dt_si']
            F = data['F']                       # (timepoints, neurons, trials)
            trl, n_neurons, n_frames = F.shape[2], F.shape[1], F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]

            data['step_time'] = parse_hdf5_array_string(data['step_time'], trl)
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']],
                          dtype=float)
            hit = np.isfinite(rt)
            rt_filled = rt.copy()
            rt_filled[~np.isfinite(rt_filled)] = 30.0
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            # ---- Epoch activity (lag 0), as in v2 ----
            F_nan = F.copy()
            F_nan[np.isnan(F_nan)] = 0
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            epoch_act = {
                'pre': np.nanmean(F_nan[ts_pre[0]:ts_pre[-1] + 1], axis=0),
                'go_cue': np.nanmean(F_nan[ts_go[0]:ts_go[-1] + 1], axis=0),
                'late': np.zeros((n_neurons, trl)),
                'reward': np.zeros((n_neurons, trl)),
            }
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    idx = get_indices_around_steps(tsta, rewards, pre=20, post=1)
                    idx = idx[idx < n_frames]
                    if len(idx) > 0:
                        epoch_act['late'][:, ti] = np.nanmean(F_nan[idx, :, ti], axis=0)
                    idx = get_indices_around_steps(tsta, rewards, pre=1, post=10)
                    idx = idx[idx < n_frames]
                    if len(idx) > 0:
                        epoch_act['reward'][:, ti] = np.nanmean(F_nan[idx, :, ti], axis=0)
            fulltrial_act = np.nanmean(F_nan, axis=0)
            bl_trials = np.arange(min(N_BASELINE, trl))
            bl_epoch = {ep: np.nanmean(epoch_act[ep][:, bl_trials], axis=1) for ep in EPOCH_ORDER}
            bl_full = np.nanmean(fulltrial_act[:, bl_trials], axis=1)

            win_beh = {k: np.full(n_wins, np.nan) for k in ['hit', 'rpe', 'rt', 'hit_rpe']}
            for wi, ws in enumerate(win_starts):
                t = np.arange(ws, ws + WIN_SIZE)
                win_beh['hit'][wi] = np.nanmean(hit[t])
                win_beh['rt'][wi] = np.nanmean(rt_filled[t])
                win_beh['rpe'][wi] = np.nanmean(rt_rpe[t])
                win_beh['hit_rpe'][wi] = np.nanmean(hit_rpe[t])

            for sel in SELECTIONS:
                dw_list, cl_list, nt_list, n_out = build_pairs(AMP, stimDist, sel)
                if len(dw_list) == 0:
                    print(f"  {sel}: no valid pairs")
                    continue
                Y_T = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
                all_nt = np.concatenate(nt_list)
                n_pairs = len(Y_T)
                cl_weights = np.zeros((n_pairs, n_neurons))
                off = 0
                for cl, nt in zip(cl_list, nt_list):
                    cl_weights[off:off + len(nt), cl] = 1.0 / len(cl)
                    off += len(nt)

                for mode in CC_MODES:
                    hi_with_int = np.full((n_wins, n_epochs), np.nan)
                    for ei, ep in enumerate(EPOCH_ORDER):
                        bl = bl_epoch[ep] if mode == 'dev2_lag' else bl_full
                        for wi, ws in enumerate(win_starts):
                            t = np.arange(ws, ws + WIN_SIZE)
                            pre_act = cl_weights @ epoch_act[ep][:, t]
                            post_dev = epoch_act[ep][all_nt][:, t] - bl[all_nt, None]
                            cc = np.sum(pre_act * post_dev, axis=1)
                            if np.std(cc) == 0:
                                continue
                            A = np.column_stack([np.ones(n_pairs), cc])
                            hi_with_int[wi, ei] = np.linalg.lstsq(A, Y_T, rcond=None)[0][1]
                    all_results[sel, mode].append({
                        'mouse': mouse, 'session': session, 'n_pairs': n_pairs,
                        'n_groups': len(dw_list), 'n_outliers': n_out,
                        'n_windows': n_wins, 'hi_with_int': hi_with_int,
                        **{f'win_{k}': v for k, v in win_beh.items()},
                    })
                print(f"  {sel:9s}: {len(dw_list)} groups, {n_pairs} pairs, "
                      f"{n_out} outliers excluded, {n_wins} windows")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

#%% ============================================================================
# CELL 4: Save
# ============================================================================
np.save(os.path.join(RESULTS_DIR, 'sliding_window_temporal_offset_v2_pan_EE.npy'),
        {f'{k[0]}|{k[1]}': v for k, v in all_results.items()}, allow_pickle=True)
print("Saved.")

#%% ============================================================================
# CELL 5: Within-session correlations and heatmaps (as in v2 CELL 6-7)
# ============================================================================
beh = [('win_hit', 'Hit rate'), ('win_rpe', 'RPE'), ('win_rt', 'Reaction time'),
       ('win_hit_rpe', 'Hit RPE')]
epoch_labels = ['Pre', 'Go cue', 'Late', 'Reward']


def corr_matrix(results):
    cs = np.full((len(results), len(beh), n_epochs), np.nan)
    for si, s in enumerate(results):
        for bi, (bkey, _) in enumerate(beh):
            bvar = s[bkey]
            if np.sum(np.isfinite(bvar)) < 5 or np.std(bvar[np.isfinite(bvar)]) == 0:
                continue
            for ei in range(n_epochs):
                slope = s['hi_with_int'][:, ei]
                ok = np.isfinite(bvar) & np.isfinite(slope)
                if np.sum(ok) >= 5 and np.std(slope[ok]) > 0:
                    cs[si, bi, ei] = spearmanr(bvar[ok], slope[ok])[0]
    mean = np.full((len(beh), n_epochs), np.nan)
    pval = np.full((len(beh), n_epochs), np.nan)
    for bi in range(len(beh)):
        for ei in range(n_epochs):
            v = cs[:, bi, ei]
            v = v[np.isfinite(v)]
            if len(v) < 3:
                continue
            mean[bi, ei] = np.mean(v)
            try:
                pval[bi, ei] = wilcoxon(v).pvalue
            except Exception:
                pval[bi, ei] = 1.0
    return mean, pval


fig, axes = plt.subplots(len(CC_MODES), len(SELECTIONS),
                         figsize=(5.2 * len(SELECTIONS), 3.6 * len(CC_MODES)))
for row, mode in enumerate(CC_MODES):
    for col, sel in enumerate(SELECTIONS):
        res = all_results[sel, mode]
        mean, pval = corr_matrix(res)
        ax = axes[row, col]
        im = ax.imshow(mean, cmap='coolwarm', vmin=-0.11, vmax=0.11, aspect='auto',
                       interpolation='nearest')
        for bi in range(len(beh)):
            for ei in range(n_epochs):
                if np.isnan(mean[bi, ei]):
                    continue
                p = pval[bi, ei]
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                ax.text(ei, bi, f'{mean[bi, ei]:+.3f}' + (f'\n{sig}' if sig else ''),
                        ha='center', va='center', fontsize=9,
                        fontweight='bold' if sig else 'normal')
        ax.set_xticks(range(n_epochs))
        ax.set_xticklabels(epoch_labels, rotation=30, ha='right')
        ax.set_yticks(range(len(beh)))
        ax.set_yticklabels([b[1] for b in beh] if col == 0 else [])
        ax.set_title(f'{sel} | {mode}\n(n={len(res)} sessions)', fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.8, label='Mean rho')
        print(f"\n{sel} | {mode} (n={len(res)})")
        for bi, (_, blabel) in enumerate(beh):
            print(f"  {blabel:14s} " + '  '.join(
                f"{mean[bi, ei]:+.3f}{'*' if pval[bi, ei] < 0.05 else ' '}"
                f"(p={pval[bi, ei]:.3f})" for ei in range(n_epochs)))
fig.suptitle('HI slope vs behavior, pan-neuronal mice (BCI88, BCI93, BCI107)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_temporal_offset_v2_pan_EE.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
