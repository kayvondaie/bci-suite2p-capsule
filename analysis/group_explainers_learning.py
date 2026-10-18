"""What makes some photostim groups move the network more than others?
Learning sessions. Group-level response G_m(g) = mean AMP of g's non-targets
(30-1000 um, outliers excluded) in map m (0 = before, 1 = after).
Candidate group features (same for both maps unless noted):
  ring_resp_m     mean AMP of cells 10-30 um from the sites (near-field / off-
                  target activation; neither target nor non-target), map m
  artifact_m      stim-artifact amplitude in the raw trial-averaged trace of
                  group g (proxy for delivered laser power), map m
  slm_deflect     mean distance of g's target cells from the group's galvo
                  centre (SLM deflection; diffraction efficiency falls with it)
  fov_dist        distance of g's target centroid from the FOV centre
  site_spread     mean pairwise distance between g's target cells
  local_density   number of ROIs within 100 um of g's target cells
  local_bright    mean log raw F of ROIs within 100 um of g's targets
  n_resp_tgt      number of responsive targets (< 10 um, AMP > 0.1 both maps)
  tgt_drive_m     summed AMP of responsive targets, map m
  tgt_hub         mean BCI-epoch correlation of g's targets with all other cells
Reports: stability of G across maps, then per feature pooled r with G (within-
session z), sessions individually significant, and group-level CV unique R2.
MICE env var restricts mice. Prints JSON."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

import json
from scipy import stats as sps
from sklearn.model_selection import KFold

MICE = os.environ.get('MICE', '').split(',') if os.environ.get('MICE') else None


def z(v):
    v = np.asarray(v, dtype=float)
    s = np.nanstd(v)
    return (v - np.nanmean(v)) / (s if s > 0 else 1.0)


def artifact_amp(favg_raw):
    art = np.nanmean(np.nanmean(favg_raw, axis=2), axis=1)
    art = art - np.nanmean(art[0:4])
    fr = np.where(art > 0.5)[0]
    fr = fr[fr < 40]
    if fr.size == 0:
        return np.full(favg_raw.shape[2], np.nan)
    pre = slice(max(0, fr[0] - 4), max(1, fr[0] - 1))
    return np.nanmean(favg_raw[fr], axis=(0, 1)) - np.nanmean(favg_raw[pre], axis=(0, 1))


def session_groups(mouse, session):
    folder = learning_folder(mouse, session)
    data = ddct.load_hdf5(folder, ['df_closedloop', 'dt_si'], ['stimDist', 'favg_raw', 'slmDist',
                                                              'centroidX', 'centroidY'])
    AMP, D = compute_amp_from_photostim(mouse, data, folder)
    k_um = np.nanmedian(D / np.where(data['photostim2']['stimDist'] == 0, np.nan,
                                     data['photostim2']['stimDist']))
    slm = np.asarray(data['photostim']['slmDist'], dtype=float) * k_um
    xy = np.column_stack([np.ravel(data['photostim']['centroidX']),
                          np.ravel(data['photostim']['centroidY'])]) * k_um
    centre = (xy.max(axis=0) + xy.min(axis=0)) / 2
    dcc = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    act = np.nan_to_num(np.asarray(data['df_closedloop'], dtype=float))
    a = act - act.mean(1, keepdims=True)
    sd = np.sqrt((a ** 2).mean(1)); sd[sd == 0] = np.inf
    a = a / sd[:, None]
    Cb = (a @ a.T) / a.shape[1]
    np.fill_diagonal(Cb, np.nan)
    try:
        raw = load_learning_spont(folder)
        bright = np.log(np.clip(np.nanmean(raw[0], axis=1), 1e-3, None))
    except FileNotFoundError:
        bright = np.full(AMP[0].shape[0], np.nan)
    art = [artifact_amp(data['photostim']['favg_raw']), artifact_amp(data['photostim2']['favg_raw'])]
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    rows = []
    for g in range(D.shape[1]):
        cl = np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                      (AMP[1][:, g] > amp1_thr) & ~is_out)[0]
        if cl.size == 0:
            continue
        tg_all = np.where(D[:, g] < dist_target_lt)[0]
        nt = np.where((D[:, g] > dist_nontarg_min) & (D[:, g] < dist_nontarg_max) & ~is_out)[0]
        ring = np.where((D[:, g] >= 10) & (D[:, g] <= 30))[0]
        near = np.where(np.min(dcc[:, tg_all], axis=1) < 100)[0]
        r = dict(g=g, n_resp_tgt=cl.size,
                 slm_deflect=np.nanmean(slm[tg_all, g]),
                 fov_dist=np.linalg.norm(xy[tg_all].mean(0) - centre),
                 site_spread=(np.nanmean(dcc[np.ix_(tg_all, tg_all)][np.triu_indices(tg_all.size, 1)])
                              if tg_all.size > 1 else np.nan),
                 local_density=near.size, local_bright=np.nanmean(bright[near]),
                 tgt_hub=np.nanmean(Cb[cl]))
        for m in range(2):
            r[f'G{m}'] = np.nanmean(AMP[m][nt, g])
            r[f'ring_resp{m}'] = np.nanmean(AMP[m][ring, g]) if ring.size else np.nan
            r[f'artifact{m}'] = art[m][g]
            r[f'tgt_drive{m}'] = np.nansum(AMP[m][cl, g])
        rows.append(r)
    return {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0]}


FEATS = ['ring_resp', 'artifact', 'tgt_drive', 'n_resp_tgt', 'slm_deflect', 'fov_dist',
         'site_spread', 'local_density', 'local_bright', 'tgt_hub']

sessions = []
for mouse, session in LEARNING_SESSIONS:
    if (mouse, session) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        T = session_groups(mouse, session)
        if len(T['g']) < MIN_GROUPS:
            print(f'{mouse} {session}: only {len(T["g"])} groups, skipped')
            continue
        sessions.append((mouse, session, T))
        print(f"{mouse} {session}: {len(T['g'])} groups | r(G0,G1)={np.corrcoef(T['G0'], T['G1'])[0, 1]:+.3f}")
    except Exception as e:
        print(f'{mouse} {session}: FAILED {e}')
        traceback.print_exc()

out = {}
g01 = np.array([np.corrcoef(T['G0'], T['G1'])[0, 1] for _, _, T in sessions])
print(f'\n{len(sessions)} sessions; stability of group effect r(G0, G1): median {np.median(g01):+.3f}, '
      f'>0 in {np.sum(g01 > 0)}/{len(g01)}')
out['stability'] = g01.tolist()


def colname(f, m):
    return f'{f}{m}' if f in ('ring_resp', 'artifact', 'tgt_drive') else f


for m in range(2):
    print(f'\n######## G{m} ########')
    for f in FEATS:
        xs, ys, rs = [], [], []
        for _, _, T in sessions:
            x, y = T[colname(f, m)], T[f'G{m}']
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 5 or np.std(x[ok]) == 0:
                continue
            xs.append(z(x[ok])); ys.append(z(y[ok]))
            rs.append(sps.pearsonr(x[ok], y[ok]))
        if not xs:
            continue
        rr, pp = sps.pearsonr(np.concatenate(xs), np.concatenate(ys))
        npos = sum(a > 0 and b < .05 for a, b in rs); nneg = sum(a < 0 and b < .05 for a, b in rs)
        out[f'G{m}/{f}'] = dict(r=float(rr), p=float(pp), n_pos=int(npos), n_neg=int(nneg), n=len(rs))
        print(f'  {f:14s} pooled r={rr:+.3f} p={pp:.2g} | sessions +{npos}/{len(rs)} -{nneg}/{len(rs)}')
    # multivariate, pooled over sessions (features z-scored within session), CV over groups
    X, y = [], []
    for _, _, T in sessions:
        M = np.column_stack([z(T[colname(f, m)]) for f in FEATS])
        M = np.where(np.isfinite(M), M, 0.0)
        X.append(M); y.append(z(T[f'G{m}']))
    X, y = np.vstack(X), np.concatenate(y)

    def cvr2(Xs):
        yp = np.zeros_like(y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xs):
            A = np.column_stack([np.ones(len(tr)), Xs[tr]])
            b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
            yp[te] = np.column_stack([np.ones(len(te)), Xs[te]]) @ b
        return 1 - np.sum((y - yp) ** 2) / np.sum((y - y.mean()) ** 2)

    full = cvr2(X)
    out[f'G{m}/cv_full'] = float(full)
    print(f'  multivariate CV R2 (all features, pooled groups) = {full:.3f}')
    for j, f in enumerate(FEATS):
        u = full - cvr2(np.delete(X, j, axis=1))
        out[f'G{m}/unique/{f}'] = float(u)
        print(f'     unique R2 {f:14s} {u:+.4f}')
print('JSON' + json.dumps(out))
