"""Per-session global dW offsets, learning vs water, all mice (grouped
ribo8 / cyto8 / cyto6). Per session:
  nt_dW       mean AMP1 - AMP0 over ordinary non-target pairs (30-1000 um;
              not SST outlier, not garbage), and far pairs only (> 300 um)
  tgt_dAMP    mean over groups of the mean AMP1 - AMP0 of the group's targets
              (< 10 um, AMP > 0.1 both maps, not outlier / garbage)
  tgt_ratio   median over groups of mean AMP1 / mean AMP0 of the targets
  nt_W0       mean AMP0 over the same non-target pairs (baseline level)
Garbage: raw photostim F <= 10% of the session median (where available) or
per-cell SD of AMP over groups > 10x the session median.
Summary: learning vs water session means per group; across sessions,
r(nt_dW, tgt_dAMP). Writes /root/capsule/results/session_offsets.json."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw_dw_dcorr_ribo.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session(')])

GROUP = {'BCI103': 'cyto6', 'BCI105': 'cyto8'}


def load_amp(mouse, sess, cond):
    if cond == 'water':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, sess)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
    else:
        folder = learning_folder(mouse, sess)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    return folder, AMP, D


rows = []
jobs = [(m, s, 'learning') for m, s in LEARNING_SESSIONS if (m, s) not in EXCLUDE] + \
       [(m, s, 'water') for m, s in CONTROL_SESSIONS]
for mouse, sess, cond in jobs:
    try:
        folder, AMP, D = load_amp(mouse, sess, cond)
    except Exception as e:
        print(f'{cond} {mouse} {sess}: skipped ({e})'); continue
    n, G = D.shape
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    bad = garbage(folder, AMP, D, n)
    dW, dWf, W0, tg, tr = [], [], [], [], []
    for g in range(G):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out & ~bad)[0]
        if cl.size == 0:
            continue
        tg.append(np.mean(AMP[1][cl, g] - AMP[0][cl, g]))
        tr.append(np.mean(AMP[1][cl, g]) / np.mean(AMP[0][cl, g]))
        sel = (D[:, g] > dist_nontarg_min) & (D[:, g] < dist_nontarg_max) & ~is_out & ~bad & \
              np.isfinite(AMP[0][:, g]) & np.isfinite(AMP[1][:, g])
        i = np.where(sel)[0]
        dW += list(AMP[1][i, g] - AMP[0][i, g]); W0 += list(AMP[0][i, g])
        dWf += list((AMP[1][i, g] - AMP[0][i, g])[D[i, g] > 300])
    if not dW:
        print(f'{cond} {mouse} {sess}: no pairs'); continue
    dW = np.array(dW); se = dW.std() / np.sqrt(dW.size)
    r = dict(mouse=mouse, session=sess, cond=cond, group=GROUP.get(mouse, 'ribo8'), n_pairs=int(dW.size),
             nt_dW=float(dW.mean()), nt_dW_se=float(se), nt_dW_far=float(np.mean(dWf)) if dWf else np.nan,
             nt_W0=float(np.mean(W0)), tgt_dAMP=float(np.mean(tg)), tgt_ratio=float(np.median(tr)), n_groups=len(tg))
    rows.append(r)
    print(f"{cond:8s} {mouse} {sess} [{r['group']}]: nt dW {r['nt_dW']:+.4f} (se {se:.4f}) far {r['nt_dW_far']:+.4f} | "
          f"nt W0 {r['nt_W0']:+.4f} | target dAMP {r['tgt_dAMP']:+.3f}, ratio {r['tgt_ratio']:.2f} ({len(tg)} groups)", flush=True)

out = {'sessions': rows}
print()
for grp in ['ribo8', 'cyto8', 'cyto6', 'all']:
    R = [r for r in rows if grp == 'all' or r['group'] == grp]
    for cond in ['learning', 'water']:
        v = np.array([r['nt_dW'] for r in R if r['cond'] == cond]); t = np.array([r['tgt_dAMP'] for r in R if r['cond'] == cond])
        if v.size:
            print(f'{grp:6s} {cond:8s}: {v.size:2d} sess | nt dW mean {v.mean():+.4f} (sd {v.std():.4f}, '
                  f'{np.sum(v > 0)}/{v.size} > 0) | target dAMP mean {t.mean():+.3f} (sd {t.std():.3f})')
    x = np.array([r['tgt_dAMP'] for r in R]); y = np.array([r['nt_dW'] for r in R])
    if x.size > 3:
        rr, pp = sps.pearsonr(x, y)
        b = np.polyfit(x, y, 1)
        out[f'{grp}/r_tgt'] = dict(r=float(rr), p=float(pp), slope=float(b[0]), n=int(x.size))
        print(f'{grp:6s} across sessions r(nt dW, target dAMP) = {rr:+.3f} (p={pp:.2g}, n={x.size}); slope {b[0]:+.4f}\n')
with open('/root/capsule/results/session_offsets.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
