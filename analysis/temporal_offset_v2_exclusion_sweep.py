#%% ============================================================================
# CELL 1: Imports and Setup
# ============================================================================
"""
HI slope vs behavior (sliding_window_temporal_offset_v2, dev2_lag) in the
pan-neuronal mice, sweeping how many putative inhibitory cells are excluded.

Cells are ranked within each session by mean non-target (>30 um) photostim
response, averaged over the before and after maps. The top X% are removed both
as stimulated targets and as postsynaptic cells, for X in EXCLUDE_PCTS.

Outcome-independent reference level: the pan-neuronal exclusion % at which the
remaining cells' mean non-target response distribution best matches the E-only
mice (BCI102-109, BCI_three_factor_2025), by KS distance on within-session
z-scored values. Chosen from photostim data only, before looking at behavior.
"""
import sys, os
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path or sys.path[0] != _THIS_DIR:
    sys.path.insert(0, _THIS_DIR)

import glob
import traceback
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, wilcoxon, ks_2samp
import data_dict_create_module_test as ddct
from BCI_data_helpers import *

PAN_ROOT = '/root/capsule/data/BCI_ai230_slc17a7_riboGcamp8s/ai230_pan_neuronal'
EONLY_ROOT = '/root/capsule/data/BCI_three_factor_2025/three_factor_data_2025'
RESULTS_DIR = '/root/capsule/results'
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 12})

PAN_MICE = ['BCI88', 'BCI93', 'BCI107']
EONLY_MICE = ['BCI102', 'BCI103', 'BCI104', 'BCI105', 'BCI106', 'BCI109']
EXCLUDE = {('BCI88', '013125'), ('BCI107', '013125')}
EXCLUDE_PCTS = [0, 2, 5, 10, 15, 20, 25, 30]

WIN_SIZE, WIN_STEP, tau_elig, N_BASELINE = 10, 5, 10, 20
dist_target_lt, dist_nontarg_min, dist_nontarg_max = 10, 30, 1000
amp0_thr = amp1_thr = 0.1
EPOCH_ORDER = ['pre', 'go_cue', 'late', 'reward']
n_epochs = len(EPOCH_ORDER)

# E-only reference (user's panel, dev2, mean rho)
EONLY_REF = {('RPE', 'pre'): 0.107, ('RPE', 'go_cue'): 0.086,
             ('RT', 'pre'): -0.092, ('RT', 'go_cue'): -0.051}
print("Setup complete!")


def sessions_of(root, mouse):
    d = os.path.join(root, mouse)
    if not os.path.isdir(d):
        return []
    return sorted(s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s, 'pophys')))


def mean_nontarget_amp(AMP, stimDist):
    a = (AMP[0] + AMP[1]) / 2
    a = a.copy()
    a[stimDist < 30] = np.nan
    return np.nanmean(a, axis=1)


def zs(x):
    x = x[np.isfinite(x)]
    return (x - x.mean()) / x.std()


#%% ============================================================================
# CELL 2: E-only reference distribution of mean non-target response
# ============================================================================
eonly_z = []
for mouse in EONLY_MICE:
    for session in sessions_of(EONLY_ROOT, mouse):
        folder = os.path.join(EONLY_ROOT, mouse, session, 'pophys') + '/'
        try:
            data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            if len(AMP) < 2 or AMP[0].shape != AMP[1].shape:
                continue
            eonly_z.append(zs(mean_nontarget_amp(AMP, stimDist)))
        except Exception as e:
            print(f"  E-only {mouse} {session} skipped: {e}")
eonly_z = np.concatenate(eonly_z)
print(f"E-only reference: {len(eonly_z)} cells; skew-ish upper tail "
      f"P99={np.percentile(eonly_z, 99):.2f}, max={eonly_z.max():.2f}")

#%% ============================================================================
# CELL 3: Pan-neuronal sessions — HI slope per exclusion level
# ============================================================================
results = {pct: [] for pct in EXCLUDE_PCTS}
pan_amp = []   # per-session mean non-target amp, for distribution matching

for mouse in PAN_MICE:
    for session in sessions_of(PAN_ROOT, mouse):
        if (mouse, session) in EXCLUDE:
            continue
        try:
            folder = os.path.join(PAN_ROOT, mouse, session, 'pophys') + '/'
            print(f"\n--- {mouse} {session} ---")
            data = ddct.load_hdf5(folder, ['F', 'dt_si', 'step_time', 'reward_time'],
                                  ['stimDist', 'favg_raw'])
            AMP, stimDist = compute_amp_from_photostim(mouse, data, folder)
            dt_si, F = data['dt_si'], data['F']
            trl, n_neurons, n_frames = F.shape[2], F.shape[1], F.shape[0]
            tsta = np.arange(0, 12, dt_si)
            tsta = tsta - tsta[int(2 / dt_si)]
            data['reward_time'] = parse_hdf5_array_string(data['reward_time'], trl)
            rt = np.array([x[0] if len(x) > 0 else np.nan for x in data['reward_time']],
                          dtype=float)
            hit = np.isfinite(rt)
            rt_filled = np.where(np.isfinite(rt), rt, 30.0)
            rt_rpe = -compute_rpe(rt_filled, baseline=2.0, tau=tau_elig, fill_value=10.0)
            hit_rpe = compute_rpe(hit.astype(float), baseline=1.0, tau=tau_elig, fill_value=0.0)

            win_starts = np.arange(0, trl - WIN_SIZE + 1, WIN_STEP)
            n_wins = len(win_starts)
            if n_wins < 5:
                print(f"  Only {n_wins} windows, skipping.")
                continue

            F_nan = np.nan_to_num(F, nan=0.0)
            ts_go = np.where((tsta > 0) & (tsta < 2))[0]
            ts_pre = np.where((tsta > -10) & (tsta < 0))[0]
            act = {'pre': F_nan[ts_pre[0]:ts_pre[-1] + 1].mean(axis=0),
                   'go_cue': F_nan[ts_go[0]:ts_go[-1] + 1].mean(axis=0),
                   'late': np.zeros((n_neurons, trl)), 'reward': np.zeros((n_neurons, trl))}
            for ti in range(trl):
                rewards = data['reward_time'][ti]
                if len(rewards) > 0:
                    for ep, pre, post in [('late', 20, 1), ('reward', 1, 10)]:
                        idx = get_indices_around_steps(tsta, rewards, pre=pre, post=post)
                        idx = idx[idx < n_frames]
                        if len(idx) > 0:
                            act[ep][:, ti] = F_nan[idx, :, ti].mean(axis=0)
            bl_trials = np.arange(min(N_BASELINE, trl))
            bl = {ep: act[ep][:, bl_trials].mean(axis=1) for ep in EPOCH_ORDER}

            beh = {k: np.full(n_wins, np.nan) for k in ['hit', 'rpe', 'rt', 'hit_rpe']}
            for wi, ws in enumerate(win_starts):
                t = slice(ws, ws + WIN_SIZE)
                beh['hit'][wi] = hit[t].mean()
                beh['rt'][wi] = rt_filled[t].mean()
                beh['rpe'][wi] = rt_rpe[t].mean()
                beh['hit_rpe'][wi] = hit_rpe[t].mean()

            m_amp = mean_nontarget_amp(AMP, stimDist)
            pan_amp.append(m_amp)
            finite = np.isfinite(m_amp)

            for pct in EXCLUDE_PCTS:
                if pct == 0:
                    excl = np.zeros(n_neurons, dtype=bool)
                else:
                    thr = np.nanpercentile(m_amp, 100 - pct)
                    excl = m_amp > thr
                dw_list, cl_list, nt_list = [], [], []
                for gi in range(stimDist.shape[1]):
                    cl = np.where((stimDist[:, gi] < dist_target_lt) &
                                  (AMP[0][:, gi] > amp0_thr) & (AMP[1][:, gi] > amp1_thr))[0]
                    if cl.size == 0 or np.any(excl[cl]):
                        continue
                    nt = np.where((stimDist[:, gi] > dist_nontarg_min) &
                                  (stimDist[:, gi] < dist_nontarg_max) & ~excl & finite)[0]
                    if nt.size == 0:
                        continue
                    dw_list.append(AMP[1][nt, gi] - AMP[0][nt, gi])
                    cl_list.append(cl)
                    nt_list.append(nt)
                if not dw_list:
                    continue
                Y = np.nan_to_num(np.concatenate(dw_list), nan=0.0)
                all_nt = np.concatenate(nt_list)
                W = np.zeros((len(Y), n_neurons))
                off = 0
                for cl, nt in zip(cl_list, nt_list):
                    W[off:off + len(nt), cl] = 1.0 / len(cl)
                    off += len(nt)
                hi = np.full((n_wins, n_epochs), np.nan)
                for ei, ep in enumerate(EPOCH_ORDER):
                    pre_all = W @ act[ep]
                    post_dev_all = act[ep][all_nt] - bl[ep][all_nt, None]
                    for wi, ws in enumerate(win_starts):
                        t = slice(ws, ws + WIN_SIZE)
                        cc = np.sum(pre_all[:, t] * post_dev_all[:, t], axis=1)
                        if np.std(cc) == 0:
                            continue
                        A = np.column_stack([np.ones(len(Y)), cc])
                        hi[wi, ei] = np.linalg.lstsq(A, Y, rcond=None)[0][1]
                results[pct].append({'mouse': mouse, 'session': session,
                                     'n_groups': len(dw_list), 'n_pairs': len(Y),
                                     'hi': hi, **beh})
            print(f"  {n_neurons} cells, {n_wins} windows; groups kept at "
                  + ', '.join(f"{p}%:{results[p][-1]['n_groups']}" for p in EXCLUDE_PCTS
                              if results[p] and results[p][-1]['session'] == session))
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()

#%% ============================================================================
# CELL 4: Distribution-matched exclusion level (photostim data only)
# ============================================================================
ks_by_pct = {}
for pct in EXCLUDE_PCTS:
    kept = []
    for m in pan_amp:
        z = (m - np.nanmean(m)) / np.nanstd(m)    # z-score on all cells, as for E-only
        zf = z[np.isfinite(z)]
        if pct > 0:
            zf = zf[zf <= np.percentile(zf, 100 - pct)]
        kept.append(zf)
    ks_by_pct[pct] = ks_2samp(np.concatenate(kept), eonly_z).statistic
matched_pct = min(ks_by_pct, key=ks_by_pct.get)
print("\nKS distance (pan remaining vs E-only), by exclusion %:")
for pct in EXCLUDE_PCTS:
    print(f"  {pct:3d}%: {ks_by_pct[pct]:.4f}{'  <- matched' if pct == matched_pct else ''}")

#%% ============================================================================
# CELL 5: Correlations vs exclusion level
# ============================================================================
BEH = [('hit', 'Hit rate'), ('rpe', 'RPE'), ('rt', 'RT'), ('hit_rpe', 'Hit RPE')]


def mean_rho(res, bkey, ei):
    v = []
    for s in res:
        b, h = s[bkey], s['hi'][:, ei]
        ok = np.isfinite(b) & np.isfinite(h)
        if ok.sum() >= 5 and np.std(b[ok]) > 0 and np.std(h[ok]) > 0:
            v.append(spearmanr(b[ok], h[ok])[0])
    v = np.array(v)
    try:
        p = wilcoxon(v).pvalue
    except Exception:
        p = np.nan
    return v.mean(), v.std() / np.sqrt(len(v)), p, len(v)


table = {}
print("\nMean rho (HI slope vs behavior), pan-neuronal mice, dev2_lag")
for bkey, blabel in BEH:
    for ei, ep in enumerate(EPOCH_ORDER):
        row = [mean_rho(results[pct], bkey, ei) for pct in EXCLUDE_PCTS]
        table[bkey, ep] = row
        print(f"  {blabel:8s} {ep:7s} " + '  '.join(
            f"{p}%:{m:+.3f}{'*' if pv < 0.05 else ' '}" for p, (m, se, pv, n) in
            zip(EXCLUDE_PCTS, row)))

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for ax, (bkey, blabel, ep) in zip(axes.flat, [('rpe', 'RPE', 'pre'), ('rpe', 'RPE', 'go_cue'),
                                               ('rt', 'RT', 'pre'), ('rt', 'RT', 'go_cue')]):
    row = table[bkey, ep]
    m = np.array([r[0] for r in row])
    se = np.array([r[1] for r in row])
    ax.errorbar(EXCLUDE_PCTS, m, yerr=se, fmt='o-', color='k', capsize=3, linewidth=2)
    for x, (mm, _, pv, _) in zip(EXCLUDE_PCTS, row):
        if pv < 0.05:
            ax.text(x, mm, '*', ha='center', va='bottom', fontsize=14)
    ax.axhline(0, color='k', alpha=0.3)
    ref = EONLY_REF.get(('RPE' if bkey == 'rpe' else 'RT', ep))
    if ref is not None:
        ax.axhline(ref, color='#e74c3c', ls='--', label='E-only mice')
    ax.axvline(matched_pct, color='#3498db', ls=':', label=f'distribution-matched ({matched_pct}%)')
    ax.set_title(f'{blabel} x {ep}')
    ax.set_ylabel('Mean rho (HI slope vs behavior)')
    ax.legend(fontsize=9)
for ax in axes[1]:
    ax.set_xlabel('% of cells excluded (highest mean non-target response)')
fig.suptitle('Pan-neuronal mice: HI-behavior correlation vs putative-inhibitory exclusion '
             f'(n={len(results[0])} sessions)', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_temporal_offset_v2_exclusion_sweep.png'),
            dpi=150, bbox_inches='tight')
plt.close(fig)
print("Figure saved.")
