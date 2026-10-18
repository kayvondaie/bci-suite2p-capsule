"""Does spont correlation predict connectivity W? All four pairings of spont epoch
(pre / post) with photostim map (W0 = before, W1 = after), learning vs water.
  coactivity A_ig : Pearson correlation (frame level, spont dF/F = F/std - 1,
                    epoch capped at SPONT_MAX_MIN) of non-target i with each
                    responsive target of group g, averaged over targets (3-factor
                    target set: < 10 um, AMP > 0.1 in both maps, non-outlier)
  W               : AMP of non-target i to group g in map 0 or map 1
  pairs           : non-targets 30-1000 um, putative SST outliers excluded
Per pairing: pooled Pearson r of within-session z-scored A and W, mean_bin_plot
curves, sessions individually significant, learning - water by pair bootstrap.
Config / loaders taken from three_factor_control_vs_learning.py (executed up to
session_fit). Prints JSON."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

import json
from scipy import stats as sps

N_BOOT = 1000
COL = 5


def spont_epochs(mouse, session, condition):
    """List [pre, post] of capped dF/F (cells x frames) and fs."""
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        sh = np.load(folder + 'suite2p_photostim_single/plane0/siHeader.npy', allow_pickle=True).item()
        fs = float(sh['metadata']['hRoiManager']['scanFrameRate'])
        with h5py.File(glob.glob(folder + 'data_main_*.h5')[0], 'r') as f:
            raw = [np.asarray(f[k][:], dtype=float) for k in ['spont_pre', 'spont_post']]
    else:
        folder = learning_folder(mouse, session)
        fs = 1.0 / float(ddct.load_hdf5(folder, ['dt_si'], [])['dt_si'])
        raw = load_learning_spont(folder)
        if len(raw) < 2:
            raise RuntimeError('no spont pre/post pair')
    return [np.nan_to_num(spont_dff(cap_epoch(F, fs, ei)), nan=0.0) for ei, F in enumerate(raw[:2])]


def amp_maps(mouse, session, condition):
    if condition == 'control':
        folder = os.path.join(CONTROL_ROOT, control_tag(mouse, session)) + '/'
        data = {}
        for pkey in ['photostim', 'photostim2']:
            with h5py.File(folder + f'{pkey}_compact.h5', 'r') as f:
                data[pkey] = {k: f[k][:] for k in ['stimDist', 'favg_raw']}
    else:
        folder = learning_folder(mouse, session)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
    return compute_amp_from_photostim(mouse, data, folder)


def corr(x):
    x = x - x.mean(axis=1, keepdims=True)
    sd = np.sqrt((x ** 2).mean(axis=1))
    sd[sd == 0] = np.inf
    x = x / sd[:, None]
    return (x @ x.T) / x.shape[1]


def z(v):
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def session_pairs(mouse, session, condition):
    AMP, stimDist = amp_maps(mouse, session, condition)
    eps = spont_epochs(mouse, session, condition)
    n = AMP[0].shape[0]
    if eps[0].shape[0] != n:
        raise RuntimeError(f'spont has {eps[0].shape[0]} cells, photostim {n}')
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[stimDist < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    C = [corr(e) for e in eps]
    A = {'pre': [], 'post': []}
    W = {'W0': [], 'W1': []}
    for g in range(stimDist.shape[1]):
        cl = np.where((stimDist[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        if cl.size == 0:
            continue
        nt = np.where((stimDist[:, g] > dist_nontarg_min) & (stimDist[:, g] < dist_nontarg_max) &
                      ~is_out & np.isfinite(AMP[0][:, g]) & np.isfinite(AMP[1][:, g]))[0]
        if nt.size == 0:
            continue
        for k, Ck in zip(['pre', 'post'], C):
            A[k].append(Ck[np.ix_(nt, cl)].mean(axis=1))
        W['W0'].append(AMP[0][nt, g])
        W['W1'].append(AMP[1][nt, g])
    if not W['W0']:
        raise RuntimeError('no valid groups')
    out = {k: z(np.concatenate(v)) for k, v in {**A, **W}.items()}
    out['n_groups'] = len(W['W0'])
    return out


def mean_bin(xx, yy):
    b = np.argsort(xx)
    x, y = xx[b], yy[b]
    row = len(x) // COL
    x, y = x[:row * COL].reshape(COL, row), y[:row * COL].reshape(COL, row)
    return x.mean(1).tolist(), y.mean(1).tolist(), (y.std(1) / np.sqrt(row)).tolist()


PAIRINGS = [('pre', 'W0'), ('pre', 'W1'), ('post', 'W0'), ('post', 'W1')]
data = {}
for condition, sessions in [('control', CONTROL_SESSIONS), ('learning_spont', LEARNING_SESSIONS)]:
    data[condition] = []
    for mouse, session in sessions:
        if (mouse, session) in EXCLUDE:
            continue
        try:
            s = session_pairs(mouse, session, condition)
            if s['n_groups'] < MIN_GROUPS:
                print(f'{condition} {mouse} {session}: only {s["n_groups"]} groups, skipped')
                continue
            data[condition].append((mouse, session, s))
            print(f'{condition} {mouse} {session}: {len(s["W0"])} pairs | ' + ' '.join(
                f'{a}-{w} r={np.corrcoef(s[a], s[w])[0, 1]:+.3f}' for a, w in PAIRINGS))
        except Exception as e:
            print(f'{condition} {mouse} {session}: FAILED {e}')

out = {}
rng = np.random.default_rng(0)
print('\n=== pooled r (within-session z-scored), coactivity vs W ===')
for a, w in PAIRINGS:
    xs = {}
    for cond in ['control', 'learning_spont']:
        S = data[cond]
        x = np.concatenate([s[a] for _, _, s in S])
        y = np.concatenate([s[w] for _, _, s in S])
        r, p = sps.pearsonr(x, y)
        nsig = sum(sps.pearsonr(s[a], s[w])[1] < 0.05 and np.corrcoef(s[a], s[w])[0, 1] > 0 for _, _, s in S)
        bx, by, be = mean_bin(x, y)
        out[f'{cond}/{a}-{w}'] = dict(r=float(r), p=float(p), n=int(len(x)), n_sess=len(S), n_sig=int(nsig),
                                      bx=bx, by=by, be=be)
        xs[cond] = (x, y)
    (xc, yc), (xl, yl) = xs['control'], xs['learning_spont']
    obs = np.corrcoef(xl, yl)[0, 1] - np.corrcoef(xc, yc)[0, 1]
    d = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(xc), len(xc))
        j = rng.integers(0, len(xl), len(xl))
        d.append(np.corrcoef(xl[j], yl[j])[0, 1] - np.corrcoef(xc[i], yc[i])[0, 1])
    d = np.array(d)
    ci = np.percentile(d, [2.5, 97.5])
    pb = max(2 * min(np.mean(d <= 0), np.mean(d >= 0)), 1 / N_BOOT)
    out[f'diff/{a}-{w}'] = dict(diff=float(obs), ci=ci.tolist(), p=float(pb))
    oc, ol = out[f'control/{a}-{w}'], out[f'learning_spont/{a}-{w}']
    print(f'  spont_{a:4s} vs {w}: water r={oc["r"]:+.4f} ({oc["n_sig"]}/{oc["n_sess"]} sig) | '
          f'learning r={ol["r"]:+.4f} ({ol["n_sig"]}/{ol["n_sess"]} sig) | '
          f'diff={obs:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] p={pb:.3g}')
print('JSON' + json.dumps(out))
