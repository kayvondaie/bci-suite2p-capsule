"""Compact plot data for the W reliability / robust-connection figures
(learning sessions). Reuses tail_session from tail_replication_W.py. Writes
/root/capsule/results/tail_plot_data.json:
  examples   per example session: mu0, mu1 (z within session) for all robust
             pairs + a random subsample, robust flag, distance
  dist       per indicator (s6 = BCI103, s8 = others) and distance bin:
             n pairs, n robust positive
  corr_prob  per indicator, distance band and within-session correlation
             quintile (spont / bci, 1 s bins and frame): n, n robust
  meanbin    pf.mean_bin_plot style: 5 equal-count bins of correlation (z)
             vs Wbar = (W0 + W1) / 2 (z within session), pooled, mean, SEM,
             r, p; for all pairs, pairs < 100 um, and robust pairs"""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tail_replication_W.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('FEATS = [')])

EXAMPLES = {('BCI93', '020325'), ('BCI103', '012325')}
DBIN = [30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]
BANDS = [(30, 50), (50, 100), (100, 300), (300, 1000)]
CFEATS = ['corr_spont_1s', 'corr_bci_1s', 'corr_spont', 'corr_bci']
rng = np.random.default_rng(0)
out = {'examples': {}, 'dist': {}, 'corr_prob': {}, 'meanbin': {}}
pool = {'s6': [], 's8': []}
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE:
        continue
    try:
        T = tail_session(mouse, sess)
        if len(np.unique(T['g'])) < MIN_GROUPS:
            continue
    except Exception as e:
        print(f'{mouse} {sess}: {e}')
        continue
    ok = np.isfinite(T['p0']) & np.isfinite(T['p1'])
    T = {k: v[ok] for k, v in T.items()}
    rob = (T['p0'] < P_ROB) & (T['p1'] < P_ROB) & (T['t0'] > 0) & (T['t1'] > 0)
    ind = 's6' if mouse == 'BCI103' else 's8'
    if (mouse, sess) in EXAMPLES:
        idx = np.union1d(np.where(rob)[0], rng.choice(len(rob), min(2500, len(rob)), replace=False))
        out['examples'][f'{mouse} {sess}'] = dict(
            mu0=np.round(z(T['mu0'])[idx], 3).tolist(), mu1=np.round(z(T['mu1'])[idx], 3).tolist(),
            robust=rob[idx].astype(int).tolist(), dist=np.round(T['dist'][idx]).tolist(),
            r_all=rel(T['mu0'], T['mu1']), n=int(len(rob)), n_rob=int(rob.sum()))
    d = dict(dist=T['dist'], rob=rob, wbar=z((T['mu0'] + T['mu1']) / 2))
    for f in CFEATS:
        d[f] = z(T[f]) if f in T and np.isfinite(T[f]).any() else np.full(len(rob), np.nan)
    pool[ind].append(d)
    print(f'{mouse} {sess}: {len(rob)} pairs, {rob.sum()} robust', flush=True)

for ind, S in pool.items():
    if not S:
        continue
    cat = {k: np.concatenate([s[k] for s in S]) for k in S[0]}
    db = np.digitize(cat['dist'], DBIN) - 1
    out['dist'][ind] = dict(edges=DBIN, n=[int(np.sum(db == b)) for b in range(len(DBIN) - 1)],
                            n_rob=[int(np.sum(cat['rob'][db == b])) for b in range(len(DBIN) - 1)],
                            n_sess=len(S))
    for f in CFEATS:
        for lo, hi in BANDS:
            key = f'{ind}/{f}/{lo}-{hi}'
            ns, nr, centers = [], [], []
            # quintiles within session and band
            q = []
            for s in S:
                m = (s['dist'] >= lo) & (s['dist'] < hi) & np.isfinite(s[f])
                if m.sum() < 50:
                    continue
                edges = np.quantile(s[f][m], [0.2, 0.4, 0.6, 0.8])
                q.append((np.digitize(s[f][m], edges), s['rob'][m], s[f][m]))
            if not q:
                continue
            qq = np.concatenate([a for a, _, _ in q]); rr = np.concatenate([b for _, b, _ in q])
            xx = np.concatenate([c for _, _, c in q])
            out['corr_prob'][key] = dict(n=[int(np.sum(qq == k)) for k in range(5)],
                                         n_rob=[int(np.sum(rr[qq == k])) for k in range(5)],
                                         x=[float(np.mean(xx[qq == k])) for k in range(5)])
        for sel_name in ['all', 'lt100', 'robust']:
            m = np.isfinite(cat[f])
            if sel_name == 'lt100':
                m &= cat['dist'] < 100
            if sel_name == 'robust':
                m &= cat['rob']
            x, y = cat[f][m], cat['wbar'][m]
            if x.size < 20:
                continue
            e = np.quantile(x, np.linspace(0, 1, 6))
            b = np.clip(np.digitize(x, e[1:-1]), 0, 4)
            rr_, pp_ = sps.pearsonr(x, y)
            out['meanbin'][f'{ind}/{f}/{sel_name}'] = dict(
                bx=[float(x[b == k].mean()) for k in range(5)], by=[float(y[b == k].mean()) for k in range(5)],
                be=[float(y[b == k].std() / np.sqrt(np.sum(b == k))) for k in range(5)],
                r=float(rr_), p=float(pp_), n=int(x.size))
with open('/root/capsule/results/tail_plot_data.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
