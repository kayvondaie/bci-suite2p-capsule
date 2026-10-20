"""Which garbage-cell criterion fires, per session (learning sessions):
  lowF  raw F in either photostim epoch < 10% of the session median
  sd    per-repeat response SD > median + 10 MAD in either map
Also the distribution of raw F of flagged vs kept cells, and how many flagged
cells would be non-target pairs. MICE env var."""
import os
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'three_factor_control_vs_learning.py')
with open(_SRC) as _f:
    _code = _f.read()
exec(_code[:_code.index('def session_fit(')])

MICE = os.environ.get('MICE', '').split(',') if os.environ.get('MICE') else None
for mouse, sess in LEARNING_SESSIONS:
    if (mouse, sess) in EXCLUDE or (MICE is not None and mouse not in MICE):
        continue
    try:
        folder = learning_folder(mouse, sess)
        data = ddct.load_hdf5(folder, ['dt_si'], ['stimDist', 'favg_raw'])
        AMP, D = compute_amp_from_photostim(mouse, data, folder)
        dt = float(data['dt_si']); n = D.shape[0]
        lowF = np.zeros(n, bool); sdb = np.zeros(n, bool); mFs = []
        for sub, key in [('suite2p_photostim_single', 'photostim'), ('suite2p_photostim_single2', 'photostim2')]:
            d = os.path.join(folder, sub, 'plane0')
            ic = np.asarray(np.load(os.path.join(d, 'iscell.npy'), allow_pickle=True))[:, 0] == 1
            mF = np.nanmean(np.asarray(np.load(os.path.join(d, 'F.npy'), allow_pickle=True)[ic], float), axis=1)
            if mF.size == n:
                lowF |= ~(mF > 0.1 * np.nanmedian(mF)); mFs.append(mF)
            r, _ = repeat_responses_paper(ps_file(folder, key), dt, D)
            sd = np.nanstd(r, axis=1); med = np.nanmedian(sd); mad = np.nanmedian(np.abs(sd - med))
            sdb |= ~(sd <= med + 10 * mad)
        mF = np.mean(mFs, axis=0) if mFs else np.full(n, np.nan)
        q = np.nanpercentile(mF, [5, 50, 95])
        print(f'{mouse} {sess}: n={n} | lowF {lowF.sum()} | sd {sdb.sum()} | both {(lowF & sdb).sum()} | '
              f'raw F 5/50/95%: {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f} | median raw F of sd-flagged {np.nanmedian(mF[sdb]) if sdb.any() else np.nan:.0f}, '
              f'n with F<=0: {int(np.sum(mF <= 0))}', flush=True)
    except Exception as e:
        print(f'{mouse} {sess}: {e}')
print('DONE')
