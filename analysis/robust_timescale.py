"""Timescale sweep of the W-vs-correlation relation among robust connections.
Same features as robust_neg_decomp.py (corr_pair, corr_spec, corr_tgt,
corr_nt) computed with correlation bin widths BIN_LIST (s; 0 = single frame),
epoch CORR_EP. Robust connections cross-validated (W0 for pairs selected
p < P_SEL positive in map 1, W1 for pairs selected in map 0). Data loads are
cached per session so every bin width uses identical pairs.
Prints pooled r (within-session z), p and sessions significant (+) per
indicator x bin x feature x map; writes /root/capsule/results/robust_timescale_<ep>.json."""
import os
_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'robust_neg_decomp.py')
with open(_P) as _f:
    _src = _f.read()
_src = _src.replace("nb = max(1, int(round(fs)))", "nb = max(1, int(round(fs * BIN_S[0])))")
BIN_S = [1.0]
exec(_src[:_src.rindex("pool = {'s6'")])

BIN_LIST = [float(b) for b in os.environ.get('BIN_LIST', '0,0.1,0.25,0.5,1,2,5').split(',')]
CF = ['corr_pair', 'corr_spec', 'corr_tgt', 'corr_nt']

_cache = {}


def _memo(name, fn):
    def w(*a, **k):
        key = (name, repr(a), repr(sorted(k.items())))
        if key not in _cache:
            _cache[key] = fn(*a, **k)
        return _cache[key]
    return w


ddct.load_hdf5 = _memo('h5', ddct.load_hdf5)
repeat_responses_paper = _memo('rep', repeat_responses_paper)
load_learning_spont = _memo('spont', load_learning_spont)
compute_amp_from_photostim = _memo('amp', compute_amp_from_photostim)

pool = {(ind, b): [] for ind in ['s6', 's8'] for b in BIN_LIST}
for si, (mouse, sess) in enumerate(LEARNING_SESSIONS):
    if (mouse, sess) in EXCLUDE:
        continue
    _cache.clear()
    ind = 's6' if mouse == 'BCI103' else 's8'
    try:
        for b in BIN_LIST:
            BIN_S[0] = b
            T = table(mouse, sess)
            if len(np.unique(T['g'])) < MIN_GROUPS:
                raise RuntimeError('too few groups')
            T['sid'] = np.full(len(T['g']), si, float)
            pool[(ind, b)].append(T)
        print(f'{mouse} {sess}: ok', flush=True)
    except Exception as e:
        for b in BIN_LIST:   # keep pairs identical across bins
            if pool[(ind, b)] and pool[(ind, b)][-1]['sid'][0] == si:
                pool[(ind, b)].pop()
        print(f'{mouse} {sess}: {e}')

out = {}
for ind in ['s6', 's8']:
    print(f'\n######## {ind}, epoch {CORR_EP}: pooled r with W (W0 | W1), cross-selected')
    print('   bin(s)  ' + '  '.join(f'{k:>22s}' for k in CF))
    for b in BIN_LIST:
        S = pool[(ind, b)]
        if not S:
            continue
        cat = {k: np.concatenate([s[k] for s in S]) for k in S[0]}
        cells = []
        for k in CF:
            vals = []
            for m in range(2):
                s_ = (cat[f'p{1 - m}'] < P_SEL) & (cat[f't{1 - m}'] > 0)
                res = meanbin(cat[k][s_], cat[f'W{m}'][s_], cat['sid'][s_])
                out[f'{ind}/{b}/W{m}/{k}'] = res
                vals.append(res)
            cells.append(f"{vals[0]['r']:+.3f}{'*' if vals[0]['p'] < .05 else ' '}|{vals[1]['r']:+.3f}{'*' if vals[1]['p'] < .05 else ' '}"
                         f" ({vals[0]['n_sig']}+{vals[1]['n_sig']})")
        print(f'   {b:6g}  ' + '  '.join(f'{c:>22s}' for c in cells))
print('(* p<0.05 pooled; parentheses: sessions individually significant positive, W0+W1)')
with open(f'/root/capsule/results/robust_timescale_{CORR_EP}.json', 'w') as fo:
    json.dump(out, fo)
print('DONE')
