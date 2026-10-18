"""Neuropil coefficient test for photostim connectivity (learning sessions).
The data-dict Fstim is built from raw suite2p F.npy (no neuropil subtraction).
Here per-repeat responses are rebuilt from the photostim suite2p folders as
    R_c = dF/F0 - c * dFneu/F0      (F0 = raw-F baseline of the repeat)
so c = 0 reproduces Fstim (checked) and c scales neuropil removal without
changing units. Also R_neu = dFneu/F0 (the ROI's neuropil alone).
Windows / seq alignment as repeat_responses_paper (paper-1 windows; the seq
offset found at c = 0 is reused for every c). Targets: < 10 um, AMP > 0.1 both
maps, non-outlier; non-targets 30-1000 um.
Reports per indicator (s6 = BCI103, s8 = others):
  - robust positive connections (p < 0.01 both maps) at each c, vs chance
  - for connections robust at c = 0: mean W_c / W_0, fraction still robust at
    each c, fraction whose neuropil alone is robust, r(W_0, W_neu)
  - mean W_neu of robust pairs vs all pairs in the same distance band
Writes /root/capsule/results/neuropil_coef.json."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

import json
from scipy import stats as sps

COEFS = [0.0, 0.35, 0.7, 1.0]
P_ROB = 0.01
BANDS = [(30, 50), (50, 100), (100, 300), (300, 1000)]
PS_DIRS = {'photostim': 'suite2p_photostim_single', 'photostim2': 'suite2p_photostim_single2'}
MICE = os.environ.get('MICE', '').split(',') if os.environ.get('MICE') else None


def cut_trials(X, F, ops, sh):
    """stimDist_single_cell trial cut; returns (dX/F0) frames x cells x trials,
    F0 = pre-window mean of raw F. Artifact frames blanked as in the data dict."""
    rate = float(sh['metadata']['hRoiManager']['scanVolumeRate'])
    k = round(rate / 16)
    timepts, pre, post = 69 * k, 5 * k, 20 * k
    trip = np.where(np.std(F, axis=0) < 10)[0]
    trip = np.unique(np.concatenate((trip, trip + 1)))
    trip[trip > F.shape[1] - 1] = F.shape[1] - 1
    F = F.copy(); X = X.copy()
    F[:, trip] = np.nan; X[:, trip] = np.nan
    fpf = ops['frames_per_file']
    out = np.full((timepts, F.shape[0], len(fpf)), np.nan, np.float32)
    strt = 0
    for ti in range(len(fpf)):
        ind = np.arange(strt, strt + fpf[ti]); strt = ind[-1] + 1
        ind = np.concatenate((np.arange(ind[0] - pre, ind[0]), ind, np.arange(ind[-1] + 1, ind[-1] + post)))
        ind = np.clip(ind, 0, F.shape[1] - 1)
        a = X[:, ind].T; f = F[:, ind].T
        f0 = np.mean(f[:pre], axis=0); x0 = np.mean(a[:pre], axis=0)
        a = (a - x0) / f0
        n = min(a.shape[0], timepts)
        out[:n, :, ti] = a[:n]
    return out


def responses(Fs, path, dt):
    """Per-repeat paper-window responses from a (frames x cells x trials) array,
    artifact timing from the h5 favg_raw."""
    with h5py.File(path, 'r') as f:
        favg_raw = f['favg_raw'][:]
    art = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
    art = art - np.nanmean(art[0:4])
    art = np.where(art > 0.5)[0]; art = art[art < 40]
    n_after = max(1, int(round(0.3 / dt))); n_before = max(1, int(round(0.2 / dt)))
    pre = (max(0, int(art[0] - 2 - n_before)), int(art[0] - 2))
    post = (int(art[-1] + 2), int(art[-1] + 2 + n_after))
    with np.errstate(all='ignore'):
        return np.nanmean(Fs[post[0]:post[1]], axis=0) - np.nanmean(Fs[pre[0]:pre[1]], axis=0)


def align_seq(resp, seq, D, n_groups, off=None):
    nearest = np.argmin(D, axis=0)
    def apply(o):
        if o > 0:
            s_, r_ = seq[:-o], resp[:, o:]
        elif o < 0:
            s_, r_ = seq[-o:], resp[:, :o]
        else:
            s_, r_ = seq, resp
        n = min(len(s_), r_.shape[1])
        return s_[:n], r_[:, :n]
    if off is None:
        best, off = -np.inf, 0
        for o in range(-10, 11):
            s_, r_ = apply(o)
            v = np.nanmean([np.nanmean(r_[nearest[g], s_ == g]) for g in range(n_groups) if np.any(s_ == g)])
            if v > best:
                best, off = v, o
    s_, r_ = apply(off)
    r_ = r_.copy()
    for t in range(1, len(s_)):
        if 0 <= s_[t - 1] < n_groups:
            r_[D[:, s_[t - 1]] < PS_TARGET_UM, t] = np.nan
    return s_, r_, off


def tstat(R, s, g, nt):
    X = R[np.ix_(nt, np.where(s == g)[0])]
    n = np.sum(np.isfinite(X), axis=1)
    with np.errstate(all='ignore'):
        mu = np.nanmean(X, axis=1); sd = np.nanstd(X, axis=1, ddof=1)
        t = mu / (sd / np.sqrt(n))
    p = 2 * sps.t.sf(np.abs(t), np.maximum(n - 1, 1))
    p[n < 5] = np.nan
    return mu, t, p


def session(mouse, sess):
    folder = learning_folder(mouse, sess)
    data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    dt = float(data['dt_si'])
    n_cells, n_groups = D.shape
    maps = []
    for key, sub in PS_DIRS.items():
        d = os.path.join(folder, sub, 'plane0')
        if not os.path.exists(os.path.join(d, 'Fneu.npy')):
            raise FileNotFoundError(f'{sub} Fneu')
        ic = np.asarray(np.load(os.path.join(d, 'iscell.npy'), allow_pickle=True))[:, 0] == 1
        F = np.asarray(np.load(os.path.join(d, 'F.npy'), allow_pickle=True)[ic], float)
        N = np.asarray(np.load(os.path.join(d, 'Fneu.npy'), allow_pickle=True)[ic], float)
        if F.shape[0] != n_cells:
            raise RuntimeError(f'{sub}: {F.shape[0]} cells vs {n_cells}')
        ops = np.load(os.path.join(d, 'ops.npy'), allow_pickle=True).item()
        sh = np.load(os.path.join(d, 'siHeader.npy'), allow_pickle=True).item()
        path = ps_file(folder, key)
        with h5py.File(path, 'r') as f:
            seq = np.asarray(f['seq'][:]).astype(int).ravel() - 1
            Fh = f['Fstim'][:, :, :min(50, f['Fstim'].shape[2])]
        SF = cut_trials(F, F, ops, sh)
        # check against the saved Fstim (allowing a trial offset)
        best = (np.inf, 0)
        for o in range(0, 11):
            if SF.shape[2] - o < Fh.shape[2]:
                break
            dlt = np.nanmean(np.abs(SF[:Fh.shape[0], :, o:o + Fh.shape[2]] - Fh))
            best = min(best, (dlt, o))
        rF = responses(SF, path, dt)[:, best[1]:]
        del SF
        rN = responses(cut_trials(N, F, ops, sh), path, dt)[:, best[1]:]
        s_, rF_, off = align_seq(rF, seq, D, n_groups)
        _, rN_, _ = align_seq(rN, seq, D, n_groups, off)
        maps.append(dict(s=s_, rF=rF_, rN=rN_, check=float(best[0]), trial_off=best[1]))
        print(f'   {key}: Fstim reproduction mean |diff| {best[0]:.2g} (trial offset {best[1]})', flush=True)
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ~is_out[:, None]
    rows = {k: [] for k in ['dist'] + [f'{q}{m}_{c}' for q in ['mu', 't', 'p'] for m in range(2) for c in COEFS + ['neu']]}
    for g in range(n_groups):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        nt = np.where(NT[:, g])[0]
        if cl.size == 0 or nt.size == 0:
            continue
        rows['dist'] += list(D[nt, g])
        for m, M in enumerate(maps):
            for c in COEFS + ['neu']:
                R = M['rN'] if c == 'neu' else M['rF'] - c * M['rN']
                mu, t, p = tstat(R, M['s'], g, nt)
                rows[f'mu{m}_{c}'] += list(mu); rows[f't{m}_{c}'] += list(t); rows[f'p{m}_{c}'] += list(p)
    T = {k: np.asarray(v, float) for k, v in rows.items()}
    return T, [M['check'] for M in maps]


def robust(T, c):
    return (T[f'p0_{c}'] < P_ROB) & (T[f'p1_{c}'] < P_ROB) & (T[f't0_{c}'] > 0) & (T[f't1_{c}'] > 0)


pool = {'s6': [], 's8': []}
checks = []
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        print(f'{mouse} {sess}', flush=True)
        T, ch = session(mouse, sess)
        checks.append((mouse, sess, ch))
        pool['s6' if mouse == 'BCI103' else 's8'].append(T)
    except Exception as e:
        print(f'{mouse} {sess}: skipped ({e})')

out = {'checks': checks}
for ind, S in pool.items():
    if not S:
        continue
    cat = {k: np.concatenate([s[k] for s in S]) for k in S[0]}
    ok = np.isfinite(cat['p0_0.0']) & np.isfinite(cat['p1_0.0'])
    cat = {k: v[ok] for k, v in cat.items()}
    n = len(cat['dist']); exp = n * (P_ROB / 2) ** 2
    print(f'\n######## {ind}: {len(S)} sessions, {n} pairs, chance robust {exp:.1f}')
    rob0 = robust(cat, 0.0)
    for c in COEFS + ['neu']:
        rb = robust(cat, c)
        both = rb & rob0
        out[f'{ind}/n_rob/{c}'] = int(rb.sum())
        print(f'   c={str(c):5s} robust {rb.sum():5d} | of the {rob0.sum()} robust at c=0: still robust {both.sum()} '
              f'({100 * both.sum() / max(rob0.sum(), 1):.0f}%)')
    for c in COEFS[1:] + ['neu']:
        ratio = np.array([(cat[f'mu0_{c}'][rob0] + cat[f'mu1_{c}'][rob0]) / (cat['mu0_0.0'][rob0] + cat['mu1_0.0'][rob0])])
        out[f'{ind}/ratio/{c}'] = float(np.nanmedian(ratio))
        print(f'   robust@0: median W_{c} / W_0 = {np.nanmedian(ratio):+.2f}')
    w0 = (cat['mu0_0.0'] + cat['mu1_0.0']) / 2; wn = (cat['mu0_neu'] + cat['mu1_neu']) / 2
    rr = np.corrcoef(w0[rob0], wn[rob0])[0, 1]
    out[f'{ind}/r_w0_wneu_robust'] = float(rr)
    print(f'   robust@0: r(W_0, W_neu) = {rr:+.2f}')
    for lo, hi in BANDS:
        b = (cat['dist'] >= lo) & (cat['dist'] < hi)
        rbb = b & rob0
        if rbb.sum() < 3:
            continue
        res = dict(n_all=int(b.sum()), n_rob=int(rbb.sum()),
                   w0_all=float(np.nanmean(w0[b])), w0_rob=float(np.nanmean(w0[rbb])),
                   wneu_all=float(np.nanmean(wn[b])), wneu_rob=float(np.nanmean(wn[rbb])),
                   w07_rob=float(np.nanmean(((cat['mu0_0.7'] + cat['mu1_0.7']) / 2)[rbb])))
        out[f'{ind}/band/{lo}-{hi}'] = res
        print(f'   {lo}-{hi} um: robust n={rbb.sum()}: W_0 {res["w0_rob"]:.3f}, W_0.7 {res["w07_rob"]:.3f}, '
              f'W_neu {res["wneu_rob"]:.3f} | all pairs n={b.sum()}: W_0 {res["w0_all"]:.4f}, W_neu {res["wneu_all"]:.4f}')
print('\nFstim reproduction checks (mean |diff| per map):')
for m_, s_, ch in checks:
    print(f'   {m_} {s_}: ' + ', '.join(f'{c:.2g}' for c in ch))
with open('/root/capsule/results/neuropil_coef.json', 'w') as fo:
    json.dump(out, fo, default=float)
print('DONE')
