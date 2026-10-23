"""Paired comparison (same seeds) of a config against the baseline from results/*.csv."""
import sys, csv, numpy as np
from scipy import stats

def load(path):
    with open(path, encoding='utf-8-sig') as f:
        rows = [r for r in csv.reader(f) if r and r[0].isdigit()]
    return np.array([[float(v) for v in r[1:5]] for r in rows])  # acc, nmi, ari, f1_macro

base_path = sys.argv[1]
base = load(base_path)
for path in sys.argv[2:]:
    x = load(path)
    n = min(len(x), len(base))
    out = []
    for i, m in enumerate(['ACC', 'NMI', 'ARI', 'F1']):
        d = x[:n, i] - base[:n, i]
        p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        out.append(f'{m} {d.mean():+.4f} (win {int((d > 0).sum())}/{n}, p={p:.3f})')
    print(f'{path.split("/")[-1]:32s} ' + ' | '.join(out))
