"""Build results/summary.md and results/summary.csv from results/*.csv (incl. paired Wilcoxon tests vs. base)."""
import csv, glob, os, sys
import numpy as np
from scipy import stats

rd = sys.argv[1] if len(sys.argv) > 1 else 'results'
METRICS = ['acc', 'nmi', 'ari', 'f1_macro']
DS_NAME = {'arxiv': 'ogbn-arxiv', 'products': 'ogbn-products', 'reddit': 'Reddit'}


def read(f):
    return list(csv.DictReader(open(f, encoding='utf-8-sig').read().strip().splitlines()))


def per_run(f):
    return np.array([[float(r[m]) for m in METRICS] for r in read(f) if r['runs'].isdigit()])


rows, files = [], {}
for f in sorted(glob.glob(os.path.join(rd, '*.csv'))):
    name = os.path.basename(f)[:-4]
    if name.startswith('summary'): continue
    d = {r['runs']: r for r in read(f)}
    if 'mean' not in d: continue
    if name.startswith('large_'): grp, rest = 'large', name[6:]
    elif name.startswith('final_'): grp, rest = 'final', name[6:]
    else: grp, rest = 'ablation', name
    ds, _, var = rest.partition('_')
    r = dict(group=grp, dataset=DS_NAME.get(ds, ds), variant=var or 'base',
             runs=sum(1 for k in d if k.isdigit()))
    for m in METRICS:
        r[m] = float(d['mean'][m]); r[m + '_std'] = float(d['std'][m])
    rows.append(r)
    files[(grp, r['dataset'], r['variant'])] = f
order = {'large': 0, 'final': 1, 'ablation': 2}
rows.sort(key=lambda r: (order[r['group']], r['dataset'], r['variant'] != 'base', r['variant']))

with open(os.path.join(rd, 'summary.csv'), 'w', newline='') as fo:
    w = csv.DictWriter(fo, list(rows[0].keys())); w.writeheader(); w.writerows(rows)

title = {'large': 'Large datasets', 'final': 'Final comparison (small datasets)', 'ablation': 'Early ablation'}
out = ['# MAGI results summary', '',
       'Values are mean ± std (%) over runs; best ACC per dataset in each section is in bold.', '']
for g in ['large', 'final', 'ablation']:
    sub = [r for r in rows if r['group'] == g]
    out += [f'## {title[g]}', '', '| Dataset | Variant | Runs | ACC | NMI | ARI | F1-macro |', '|---|---|---|---|---|---|---|']
    best = {}
    for r in sub: best[r['dataset']] = max(best.get(r['dataset'], 0), r['acc'])
    for r in sub:
        cell = lambda m: f"{r[m]*100:.2f} ± {r[m+'_std']*100:.2f}"
        acc = cell('acc')
        if g != 'large' and r['acc'] == best[r['dataset']]: acc = f'**{acc}**'
        out.append(f"| {r['dataset']} | {r['variant']} | {r['runs']} | {acc} | {cell('nmi')} | {cell('ari')} | {cell('f1_macro')} |")
    out.append('')

# Paired tests: every variant vs. the base of the same group/dataset (same seeds 0..n-1)
out += ['## Paired significance tests (Wilcoxon signed-rank vs. base, same seeds)', '',
        'Δ = variant − base in percentage points; wins = runs where variant > base; **p < 0.05**.',
        'With n=5 the smallest attainable two-sided p is 0.0625, with n=3 it is 0.25, so those rows cannot reach significance.', '',
        '| Section | Dataset | Variant | n | ΔACC (wins) | p ACC | ΔNMI | p NMI | ΔARI | p ARI | ΔF1 | p F1 |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
for r in rows:
    if r['group'] == 'large' or r['variant'] == 'base': continue
    bf = files.get((r['group'], r['dataset'], 'base'))
    if not bf: continue
    x, b = per_run(files[(r['group'], r['dataset'], r['variant'])]), per_run(bf)
    n = min(len(x), len(b))
    cells = []
    for i in range(len(METRICS)):
        d = x[:n, i] - b[:n, i]
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        ps = f'**{p:.3f}**' if p < 0.05 else f'{p:.3f}'
        delta = f'{d.mean()*100:+.2f}' + (f' ({int((d > 0).sum())}/{n})' if i == 0 else '')
        cells += [delta, ps]
    out.append(f"| {r['group']} | {r['dataset']} | {r['variant']} | {n} | " + ' | '.join(cells) + ' |')
out.append('')

skipped = [os.path.basename(f) for f in sorted(glob.glob(os.path.join(rd, '*.csv')))
           if 'summary' not in os.path.basename(f) and 'mean' not in open(f).read()]
if skipped:
    out.append('Skipped (header only, superseded by final_ versions): ' + ', '.join(skipped))
open(os.path.join(rd, 'summary.md'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('wrote', os.path.join(rd, 'summary.md'))
