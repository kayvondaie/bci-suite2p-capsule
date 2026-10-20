"""Distributions of per-connection p-values (non-target pairs, per-repeat
t-test of each group's repeats vs 0), learning sessions, per map, for W built
from raw F (c = 0, as the data-dict Fstim) and neuropil-subtracted (c = 0.7).
Per indicator (s6 = BCI103, s8 = others) and distance band:
  hist     two-sided p histogram (20 bins), separately for t > 0 and t < 0
  pi0      Storey estimate of the null fraction, 2 * P(p > 0.5)
  tail     fraction of pairs with p < 1e-3 / 1e-4, by sign
  logp     histogram of -log10 p (0..10) by sign
Per session: median repeats per group, median per-repeat SD of non-target
responses (dF/F0), pi0. MICE env var. Writes /root/capsule/results/pvalue_dist<tag>.json."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
from scipy import stats as sps
from neuropil_reps import cut_trials, responses, align_seq, PS_DIRS

MICE = os.environ.get('MICE', '').split(',') if os.environ.get('MICE') else None
COEFS = [0.0, 0.7]
BANDS = [(30, 50), (50, 100), (100, 300), (300, 1000), (30, 1000)]
PB = np.linspace(0, 1, 21)
LB = np.linspace(0, 10, 41)


def maps_FN(folder, D, dt):
    n_cells, n_groups = D.shape
    out = []
    for key, sub in PS_DIRS.items():
        d = os.path.join(folder, sub, 'plane0')
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
        best = (np.inf, 0)
        for o in range(0, 11):
            if SF.shape[2] - o < Fh.shape[2]:
                break
            best = min(best, (np.nanmean(np.abs(SF[:Fh.shape[0], :, o:o + Fh.shape[2]] - Fh)), o))
        rF = responses(SF, path, dt)[:, best[1]:]
        del SF
        rN = responses(cut_trials(N, F, ops, sh), path, dt)[:, best[1]:]
        s_, rF_, off = align_seq(rF, seq, D, n_groups)
        _, rN_, _ = align_seq(rN, seq, D, n_groups, off)
        out.append((rF_, rN_, s_))
    return out


def session(mouse, sess):
    folder = learning_folder(mouse, sess)
    data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    dt = float(data['dt_si'])
    n_cells, n_groups = D.shape
    M = maps_FN(folder, D, dt)
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ~is_out[:, None]
    rec = {c: {'t': [], 'p': [], 'dist': []} for c in COEFS}
    nrep, sds = [], []
    for g in range(n_groups):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        nt = np.where(NT[:, g])[0]
        if cl.size == 0 or nt.size == 0:
            continue
        for rF, rN, s in M:
            k = np.where(s == g)[0]
            nrep.append(k.size)
            for c in COEFS:
                X = (rF - c * rN)[np.ix_(nt, k)]
                n = np.sum(np.isfinite(X), axis=1)
                with np.errstate(all='ignore'):
                    mu = np.nanmean(X, axis=1); sd = np.nanstd(X, axis=1, ddof=1)
                    t = mu / (sd / np.sqrt(n))
                p = 2 * sps.t.sf(np.abs(t), np.maximum(n - 1, 1))
                ok = (n >= 5) & np.isfinite(t)
                rec[c]['t'] += list(t[ok]); rec[c]['p'] += list(p[ok]); rec[c]['dist'] += list(D[nt[ok], g])
                if c == 0.0:
                    sds.append(np.nanmedian(sd))
    R = {c: {k: np.asarray(v, float) for k, v in rec[c].items()} for c in COEFS}
    info = dict(mouse=mouse, session=sess, n_rep=float(np.median(nrep)), sd=float(np.nanmedian(sds)),
                pi0_c0=float(min(1, 2 * np.mean(R[0.0]['p'] > 0.5))), pi0_c07=float(min(1, 2 * np.mean(R[0.7]['p'] > 0.5))),
                n_pairs=int(len(R[0.0]['p'])))
    return R, info


GROUP = {'BCI103': 'cyto6', 'BCI105': 'cyto8'}   # others: ribo-GCaMP8s
pool = {'ribo8': [], 'cyto8': [], 'cyto6': []}
infos = []
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        R, info = session(mouse, sess)
    except Exception as e:
        print(f'{mouse} {sess}: skipped ({e})'); continue
    infos.append(info)
    pool[GROUP.get(mouse, 'ribo8')].append(R)
    print(f"{mouse} {sess}: {info['n_pairs']} pair-maps, median repeats/group {info['n_rep']:.0f}, "
          f"median repeat SD {info['sd']:.3f}, pi0 c0 {info['pi0_c0']:.3f} c0.7 {info['pi0_c07']:.3f}", flush=True)

out = {'sessions': infos}
for ind, S in pool.items():
    if not S:
        continue
    for c in COEFS:
        t = np.concatenate([s[c]['t'] for s in S]); p = np.concatenate([s[c]['p'] for s in S])
        dd = np.concatenate([s[c]['dist'] for s in S])
        for lo, hi in BANDS:
            b = (dd >= lo) & (dd < hi)
            res = dict(n=int(b.sum()), pi0=float(min(1, 2 * np.mean(p[b] > 0.5))))
            for sg, nm in [(1, 'pos'), (-1, 'neg')]:
                m = b & (np.sign(t) == sg)
                res[f'hist_{nm}'] = (np.histogram(p[m], PB)[0] / max(b.sum(), 1)).tolist()
                res[f'logp_{nm}'] = (np.histogram(np.clip(-np.log10(p[m]), 0, 9.99), LB)[0] / max(b.sum(), 1)).tolist()
                res[f'tail3_{nm}'] = float(np.mean(m & (p < 1e-3)) / max(np.mean(b), 1e-12))
                res[f'tail4_{nm}'] = float(np.mean(m & (p < 1e-4)) / max(np.mean(b), 1e-12))
            out[f'{ind}/{c:g}/{lo}-{hi}'] = res
            print(f"{ind} c={c:g} {lo}-{hi} um: n={res['n']}, pi0={res['pi0']:.3f}, p<1e-3 +{100 * res['tail3_pos']:.3f}% "
                  f"-{100 * res['tail3_neg']:.3f}% (null 0.05% each), p<1e-4 +{100 * res['tail4_pos']:.4f}% -{100 * res['tail4_neg']:.4f}%")
tag = ('_' + '-'.join(MICE)) if MICE else ''
with open(f'/root/capsule/results/pvalue_dist{tag}.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
