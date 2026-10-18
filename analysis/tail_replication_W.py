"""Do individually significant connections replicate, and what distinguishes
them? Learning sessions, per-repeat responses (paper-1 windows).
Per pair (non-target i, group g), map m: t-test of g's repeats against 0
(p_m), plus odd / even repeat halves of map 0.
  1  tail replication: pairs with p0 < P_SEL (each sign) -> fraction with the
     same sign and p1 < 0.05 in map 1 (null 0.025); same for odd -> even
     within map 0. Excess = observed / expected count at P_SEL.
  2  robust pairs (p0 < P_ROB and p1 < P_ROB, same sign) vs null pairs
     (p0 > 0.5 and p1 > 0.5): distance distribution, and feature differences
     after matching on distance bin (feature z-scored within session; robust
     minus the mean of null pairs in the same bin).
Features: corr_bci, corr_bci_1s, nc_bci, dcorr_bci, corr_spont, corr_spont_1s,
row (cell's mean W to other groups, both maps), tgt_resp (g's responsive-target
summed AMP), bright (log mean raw F of i, spont). MICE env var. Prints JSON."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cm_noise_W_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def residual(')])

P_SEL = 0.001
P_ROB = 0.01


def tstats(r, s, g, nt, sel=None):
    k = np.where(s == g)[0]
    if sel is not None:
        k = k[sel(k)]
    R = r[np.ix_(nt, k)]
    n = np.sum(np.isfinite(R), axis=1)
    with np.errstate(all='ignore'):
        mu = np.nanmean(R, axis=1); sd = np.nanstd(R, axis=1, ddof=1)
        t = mu / (sd / np.sqrt(n))
    p = 2 * sps.t.sf(np.abs(t), np.maximum(n - 1, 1))
    p[n < 5] = np.nan
    return mu, t, p


def tail_session(mouse, sess):
    folder = learning_folder(mouse, sess)
    AMP, D, dt, reps, acts, F = load(mouse, sess)
    n_cells, n_groups = D.shape
    amp_avg = (AMP[0] + AMP[1]) / 2
    amp_avg[D < 30] = np.nan
    mnt = np.nanmean(amp_avg, axis=1)
    is_out = mnt > np.nanmean(mnt) + N_SD_OUTLIER * np.nanstd(mnt)
    if mouse not in PAN_NEURONAL_MICE:
        is_out[:] = False
    ok_cell = ~is_out
    NT = (D > dist_nontarg_min) & (D < dist_nontarg_max) & ok_cell[:, None]
    TG = [np.where((D[:, g] < dist_target_lt) & (AMP[0][:, g] > amp0_thr) &
                   (AMP[1][:, g] > amp1_thr) & ok_cell)[0] for g in range(n_groups)]
    fs = 1.0 / dt
    C = {}
    a = acts['bci']
    C['corr_bci'] = corr_mat(a)
    C['corr_bci_1s'] = corr_mat(bin_frames(a, max(1, int(round(fs)))))
    C['dcorr_bci'] = corr_mat(np.clip(np.diff(a, axis=1), 0, None))
    with np.errstate(all='ignore'):
        R = F - np.nanmean(F, axis=2, keepdims=True)
    C['nc_bci'] = corr_mat(np.nan_to_num(np.transpose(R, (1, 0, 2)).reshape(F.shape[1], -1)))
    bright = np.full(n_cells, np.nan)
    if 'spont' in acts and acts['spont'].shape[0] == n_cells:
        C['corr_spont'] = corr_mat(acts['spont'])
        C['corr_spont_1s'] = corr_mat(bin_frames(acts['spont'], max(1, int(round(fs)))))
        try:
            bright = np.log(np.clip(np.nanmean(load_learning_spont(folder)[0], axis=1), 1e-3, None))
        except Exception:
            pass
    for k, Ck in list(C.items()):
        if Ck.shape[0] != n_cells:
            del C[k]
    odd = lambda k: np.arange(k.size) % 2 == 0
    even = lambda k: np.arange(k.size) % 2 == 1
    rows = {k: [] for k in ['g', 'dist', 'mu0', 'mu1', 't0', 't1', 'p0', 'p1', 'to', 'te', 'po', 'pe',
                            'row', 'tgt_resp', 'bright'] + list(C)}
    Wm = [group_means(reps[m][0], reps[m][1], n_groups) for m in range(2)]
    for g in range(n_groups):
        cl = TG[g]
        if cl.size == 0:
            continue
        nt = np.where(NT[:, g])[0]
        if nt.size == 0:
            continue
        r0, s0 = reps[0]; r1, s1 = reps[1]
        mu0, t0, p0 = tstats(r0, s0, g, nt)
        mu1, t1, p1 = tstats(r1, s1, g, nt)
        _, to, po = tstats(r0, s0, g, nt, odd)
        _, te, pe = tstats(r0, s0, g, nt, even)
        for k, v in zip(['mu0', 'mu1', 't0', 't1', 'p0', 'p1', 'to', 'te', 'po', 'pe'],
                        [mu0, mu1, t0, t1, p0, p1, to, te, po, pe]):
            rows[k] += list(v)
        rows['g'] += [g] * nt.size; rows['dist'] += list(D[nt, g])
        rowv = []
        for m in range(2):
            X = np.where(NT, Wm[m], np.nan)
            rowv.append((np.nansum(X[nt], axis=1) - np.nan_to_num(X[nt, g])) /
                        np.maximum(np.sum(np.isfinite(X[nt]), axis=1) - 1, 1))
        rows['row'] += list((rowv[0] + rowv[1]) / 2)
        rows['tgt_resp'] += [np.nansum(AMP[0][cl, g] + AMP[1][cl, g]) / 2] * nt.size
        rows['bright'] += list(bright[nt])
        for k, Ck in C.items():
            rows[k] += list(Ck[np.ix_(cl, nt)].mean(axis=0))
    return {k: np.asarray(v, dtype=float) for k, v in rows.items()}


def replication(pa, ta, pb, tb):
    out = {}
    for sgn, nm in [(1, 'pos'), (-1, 'neg')]:
        sel = (pa < P_SEL) & (np.sign(ta) == sgn) & np.isfinite(pb)
        n_valid = np.sum(np.isfinite(pa) & np.isfinite(pb))
        rep = np.sum(sel & (pb < 0.05) & (np.sign(tb) == sgn))
        out[nm] = dict(n_sel=int(sel.sum()), expected_sel=float(n_valid * P_SEL / 2), n_rep=int(rep),
                       frac_rep=float(rep / sel.sum()) if sel.sum() else np.nan)
    return out


FEATS = ['corr_bci', 'corr_bci_1s', 'nc_bci', 'dcorr_bci', 'corr_spont', 'corr_spont_1s',
         'row', 'tgt_resp', 'bright']
results = []
pool = {f: {'pos': [], 'neg': []} for f in FEATS}
pool_dist = {'pos': [], 'neg': [], 'null': [], 'all': []}
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        T = tail_session(mouse, sess)
        if len(np.unique(T['g'])) < MIN_GROUPS:
            print(f'{mouse} {sess}: too few groups, skipped'); continue
        res = dict(mouse=mouse, session=sess, n=len(T['g']),
                   across=replication(T['p0'], T['t0'], T['p1'], T['t1']),
                   within=replication(T['po'], T['to'], T['pe'], T['te']))
        rob = {'pos': (T['p0'] < P_ROB) & (T['p1'] < P_ROB) & (T['t0'] > 0) & (T['t1'] > 0),
               'neg': (T['p0'] < P_ROB) & (T['p1'] < P_ROB) & (T['t0'] < 0) & (T['t1'] < 0)}
        null = (T['p0'] > 0.5) & (T['p1'] > 0.5)
        n_valid = np.sum(np.isfinite(T['p0']) & np.isfinite(T['p1']))
        res['robust'] = {k: int(v.sum()) for k, v in rob.items()}
        res['robust_expected'] = float(n_valid * (P_ROB / 2) ** 2)
        res['r_mu'] = rel(T['mu0'], T['mu1'])
        dbin = np.clip(np.digitize(T['dist'], DIST_EDGES[1:-1]), 0, len(DIST_EDGES) - 2)
        pool_dist['all'] += list(T['dist']); pool_dist['null'] += list(T['dist'][null])
        for k in rob:
            pool_dist[k] += list(T['dist'][rob[k]])
        for f in FEATS:
            if f not in T or not np.isfinite(T[f]).any():
                continue
            x = z(T[f])
            for k in rob:
                idx = np.where(rob[k] & np.isfinite(x))[0]
                for i in idx:
                    nb = null & (dbin == dbin[i]) & np.isfinite(x)
                    if nb.sum() >= 20:
                        pool[f][k].append((x[i] - np.mean(x[nb]), f'{mouse}{sess}'))
        results.append(res)
        a, w = res['across'], res['within']
        print(f"{mouse} {sess}: n={res['n']} r(mu0,mu1)={res['r_mu']:+.3f} | across: +{a['pos']['n_sel']} sel "
              f"(exp {a['pos']['expected_sel']:.0f}) rep {a['pos']['frac_rep']:.2f}; -{a['neg']['n_sel']} rep "
              f"{a['neg']['frac_rep']:.2f} | within: + rep {w['pos']['frac_rep']:.2f}, - rep {w['neg']['frac_rep']:.2f} | "
              f"robust +{res['robust']['pos']} -{res['robust']['neg']} (exp {res['robust_expected']:.1f} each)", flush=True)
    except FileNotFoundError as e:
        print(f'{mouse} {sess}: skipped ({e})')
    except Exception as e:
        print(f'{mouse} {sess}: FAILED {e}')
        traceback.print_exc()

out = {}
print(f'\n{len(results)} sessions')
for kind in ['across', 'within']:
    for sgn in ['pos', 'neg']:
        ns = sum(r[kind][sgn]['n_sel'] for r in results); ne = sum(r[kind][sgn]['expected_sel'] for r in results)
        nr = sum(r[kind][sgn]['n_rep'] for r in results)
        bt = sps.binomtest(nr, ns, 0.025, alternative='greater').pvalue if ns else np.nan
        out[f'{kind}/{sgn}'] = dict(n_sel=ns, expected_sel=ne, n_rep=nr, frac=nr / ns if ns else np.nan, p=bt)
        print(f'{kind:6s} {sgn}: selected {ns} (expected under null {ne:.0f}), replicated {nr} '
              f'({100 * nr / max(ns, 1):.1f}%, null 2.5%, binomial p={bt:.2g})')
nrp = sum(r['robust']['pos'] for r in results); nrn = sum(r['robust']['neg'] for r in results)
nre = sum(r['robust_expected'] for r in results)
print(f'robust pairs (p<{P_ROB} both maps, same sign): +{nrp}  -{nrn}  (expected each {nre:.1f})')
out['robust'] = dict(pos=nrp, neg=nrn, expected=nre)
for k in ['pos', 'neg', 'null', 'all']:
    d = np.array(pool_dist[k])
    if d.size:
        print(f'distance {k:4s}: n={d.size} median {np.median(d):.0f} um, <50 um {100 * np.mean(d < 50):.1f}%, '
              f'<100 um {100 * np.mean(d < 100):.1f}%')
        out[f'dist/{k}'] = dict(n=int(d.size), median=float(np.median(d)), lt50=float(np.mean(d < 50)),
                                lt100=float(np.mean(d < 100)))
print('features, robust minus distance-matched null (within-session z): mean, t-test p, n')
for f in FEATS:
    for k in ['pos', 'neg']:
        v = np.array([a for a, _ in pool[f][k]])
        if v.size < 3:
            continue
        tt = sps.ttest_1samp(v, 0)
        out[f'feat/{f}/{k}'] = dict(mean=float(v.mean()), p=float(tt.pvalue), n=int(v.size))
        print(f'   {f:14s} {k}: {v.mean():+.3f} z  p={tt.pvalue:.2g}  n={v.size}')
print('JSON' + json.dumps(out, default=float))
