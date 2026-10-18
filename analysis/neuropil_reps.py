"""Per-repeat photostim responses with neuropil subtraction,
R_c = dF/F0 - c * dFneu/F0 (F0 = raw-F baseline), from the photostim suite2p
folders; c = 0 reproduces the data-dict Fstim. Helpers copied from
neuropil_coef_test.py. reps_neuropil returns [(r, s)] per map like
repeat_responses_paper."""
import os
import h5py
import numpy as np

PS_DIRS = {'photostim': 'suite2p_photostim_single', 'photostim2': 'suite2p_photostim_single2'}
PS_TARGET_UM = 20


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



def reps_neuropil(folder, D, dt, c, ps_file):
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
        out.append((rF_ - c * rN_, s_))
    return out
