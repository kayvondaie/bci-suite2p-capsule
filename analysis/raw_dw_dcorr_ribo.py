"""Raw (unnormalised) dW vs dCorrelation, learning vs water, ribo-GCaMP8s mice
(BCI88, BCI93, BCI107). Spont-vs-spont only (fair comparison).
Per pair (postsynaptic cell i, photostim group g):
  dW  = AMP1 - AMP0 (compute_amp_from_photostim, raw dF/F units)
  dC  = C_post - C_pre, C = mean Pearson correlation of i with g's targets in
        spont_post / spont_pre (data-dict dF/F, each epoch capped at SPONT_MAX_MIN
        min nearest the intervening period, 1 s bins)
Targets: < 10 um, AMP > 0.1 both maps, not SST-outlier, not garbage.
Postsynaptic populations (30-1000 um from the group's targets):
  nt   ordinary non-targets (not SST outlier, not garbage)
  sst  putative SST outliers only (mean non-target AMP > mean + 2 SD; no opsin)
Garbage cells: raw photostim F <= 10% of session median (where the suite2p
folder is available) or per-cell SD of AMP over groups > 10x session median.
Reports per condition x population: pooled pf.mean_bin_plot data (5
equal-count bins of dC), pooled r and OLS slope, sessions individually
significant, dW / dC spreads; learning - water slope by pair bootstrap.
Writes /root/capsule/results/raw_dw_dcorr_ribo.json."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])
import json
from scipy import stats as sps

RIBO = ['BCI88', 'BCI93', 'BCI107']
SPONT_MIN = float(os.environ.get('SPONT_MIN', 3.5))
N_BOOT = int(os.environ.get('N_BOOT', 2000))
rng = np.random.default_rng(0)


def corr_mat(x):
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).mean(axis=1)); sd[sd == 0] = np.inf
    x = x / sd[:, None]
    return (x @ x.T) / x.shape[1]


def bin1s(x, fs):
    n = max(1, int(round(fs))); m = x.shape[1] // n
    return x[:, :m * n].reshape(x.shape[0], m, n).mean(axis=2)


def cap(F, fs, which):
    n = int(round(SPONT_MIN * 60 * fs))
    return F[:, -n:] if which == 0 else F[:, :n]


def load(mouse, sess, cond):
    if cond == 'water':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, sess)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
        AMP, D = compute_amp_from_photostim(mouse, data, folder)
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).item()
        fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            sp = [np.asarray(f[k][:], dtype=float) for k in ['spont_pre', 'spont_post']]
    else:
        folder = learning_folder(mouse, sess)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
        AMP, D = compute_amp_from_photostim(mouse, data, folder)
        fs = 1.0 / float(data['dt_si'])
        sp = load_learning_spont(folder)
    if len(sp) < 2:
        raise FileNotFoundError('need spont pre and post')
    C = [corr_mat(bin1s(np.nan_to_num(spont_dff(cap(F, fs, e))), fs)) for e, F in enumerate(sp)]
    return folder, AMP, D, C


def garbage(folder, AMP, D, n):
    bad = np.zeros(n, bool)
    for sub in ['suite2p_photostim_single', 'suite2p_photostim_single2']:
        p = os.path.join(folder, sub, 'plane0', 'F.npy')
        ic = os.path.join(folder, sub, 'plane0', 'iscell.npy')
        if os.path.exists(p) and os.path.exists(ic):
            icm = np.asarray(np.load(ic, allow_pickle=True))[:, 0] == 1
            mF = np.nanmean(np.asarray(np.load(p, allow_pickle=True)[icm], float), axis=1)
            if mF.size == n:
                bad |= ~(mF > 0.1 * np.nanmedian(mF))
    for A in AMP:
        X = np.where(D > 30, A, np.nan)
        sd = np.nanstd(X, axis=1)
        bad |= ~(sd <= 10 * np.nanmedian(sd))
    return bad


def session(mouse, sess, cond):
    folder, AMP, D, C = load(mouse, sess, cond)
    n, G = D.shape
    if C[0].shape[0] != n or C[1].shape[0] != n:
        raise RuntimeError(f'spont {C[0].shape[0]} cells vs photostim {n}')
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    bad = garbage(folder, AMP, D, n)
    rows = {pop: {k: [] for k in ['dW', 'dC', 'W0', 'C0']} for pop in ['nt', 'sst']}
    for g in range(G):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out & ~bad)[0]
        if cl.size == 0:
            continue
        inband = (D[:, g] > dist_nontarg_min) & (D[:, g] < dist_nontarg_max) & ~bad & \
                 np.isfinite(AMP[0][:, g]) & np.isfinite(AMP[1][:, g])
        for pop, sel in [('nt', inband & ~is_out), ('sst', inband & is_out)]:
            i = np.where(sel)[0]
            if i.size == 0:
                continue
            c0 = C[0][np.ix_(cl, i)].mean(axis=0); c1 = C[1][np.ix_(cl, i)].mean(axis=0)
            rows[pop]['dW'] += list(AMP[1][i, g] - AMP[0][i, g]); rows[pop]['W0'] += list(AMP[0][i, g])
            rows[pop]['dC'] += list(c1 - c0); rows[pop]['C0'] += list(c0)
    info = dict(mouse=mouse, session=sess, cond=cond, n_cells=n, n_sst=int(is_out.sum()), n_bad=int(bad.sum()))
    return {p: {k: np.asarray(v, float) for k, v in r.items()} for p, r in rows.items()}, info


def meanbin(x, y):
    e = np.quantile(x, np.linspace(0, 1, 6))
    b = np.clip(np.digitize(x, e[1:-1]), 0, 4)
    return dict(bx=[float(x[b == k].mean()) for k in range(5)], by=[float(y[b == k].mean()) for k in range(5)],
                be=[float(y[b == k].std() / np.sqrt(np.sum(b == k))) for k in range(5)])


def slope(x, y):
    xm = x - x.mean()
    return float((xm @ (y - y.mean())) / (xm @ xm))


pool = {(c, p): [] for c in ['learning', 'water'] for p in ['nt', 'sst']}
infos = []
jobs = [(m, s, 'learning') for m, s in LEARNING_SESSIONS if m in RIBO and (m, s) not in EXCLUDE] + \
       [(m, s, 'water') for m, s in CONTROL_SESSIONS if m in RIBO]
for mouse, sess, cond in jobs:
    try:
        R, info = session(mouse, sess, cond)
    except Exception as e:
        print(f'{cond} {mouse} {sess}: skipped ({e})'); continue
    infos.append(info)
    for p in R:
        if len(R[p]['dW']) >= 10:
            R[p]['sid'] = f'{mouse}{sess}'
            pool[(cond, p)].append(R[p])
    print(f"{cond} {mouse} {sess}: n={info['n_cells']} sst={info['n_sst']} garbage={info['n_bad']} | "
          f"nt pairs {len(R['nt']['dW'])}, sst pairs {len(R['sst']['dW'])}", flush=True)

out = {'sessions': infos}
for p in ['nt', 'sst']:
    print(f'\n######## postsynaptic = {p}')
    cat = {}
    for c in ['learning', 'water']:
        S = pool[(c, p)]
        if not S:
            continue
        x = np.concatenate([s['dC'] for s in S]); y = np.concatenate([s['dW'] for s in S])
        ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
        cat[c] = (x, y)
        rr, pp = sps.pearsonr(x, y)
        nsig = sum(1 for s in S if np.isfinite(s['dC']).sum() > 10 and
                   sps.pearsonr(s['dC'][np.isfinite(s['dC'])], s['dW'][np.isfinite(s['dC'])])[1] < 0.05 and
                   sps.pearsonr(s['dC'][np.isfinite(s['dC'])], s['dW'][np.isfinite(s['dC'])])[0] > 0)
        res = dict(meanbin(x, y), r=float(rr), p=float(pp), slope=slope(x, y), n=int(x.size), n_sess=len(S),
                   n_sig_pos=int(nsig), dW_mean=float(y.mean()), dW_sd=float(y.std()), dC_mean=float(x.mean()),
                   dC_sd=float(x.std()))
        out[f'{p}/{c}'] = res
        print(f"   {c:8s}: n={x.size} ({len(S)} sess) r={rr:+.4f} p={pp:.2g} slope={res['slope']:+.4f} | "
              f"sig+ {nsig}/{len(S)} | dW mean {y.mean():+.4f} sd {y.std():.4f} | dC mean {x.mean():+.4f} sd {x.std():.4f}")
    if len(cat) == 2:
        bs = []
        for _ in range(N_BOOT):
            v = []
            for c in ['learning', 'water']:
                x, y = cat[c]; k = rng.integers(0, x.size, x.size)
                v.append(slope(x[k], y[k]))
            bs.append(v[0] - v[1])
        bs = np.array(bs); d = out[f'{p}/learning']['slope'] - out[f'{p}/water']['slope']
        pv = 2 * min(np.mean(bs <= 0), np.mean(bs >= 0))
        out[f'{p}/diff'] = dict(slope_diff=float(d), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], p=float(pv))
        print(f'   learning - water slope: {d:+.4f} [{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}], pair-bootstrap p={pv:.3g}')
with open('/root/capsule/results/raw_dw_dcorr_ribo.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
