"""Evaluate graph-smoothed embeddings Z_k = A_hat^k Z (A_hat = D^-1/2 (A+I) D^-1/2) for clustering."""
import argparse, glob, numpy as np, torch, torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor
from magi.utils import clustering

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--emb_dir', type=str, default='emb')
parser.add_argument('--ks', type=str, default='0,1,2,3')
args = parser.parse_args()

n_clusters = {'Cora': 7, 'Citeseer': 6, 'Photo': 8, 'Computers': 10}[args.dataset]
data = (Planetoid if args.dataset in ['Cora', 'Citeseer'] else Amazon)('data/', args.dataset)[0]
N = data.num_nodes
ei = to_undirected(add_remaining_self_loops(data.edge_index)[0])
adj = SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(N, N)).fill_value(1.)
dinv = adj.sum(dim=1).pow(-0.5)
a_hat = adj.mul(dinv.view(-1, 1)).mul(dinv.view(1, -1))

ks = list(map(int, args.ks.split(',')))
res = {k: [] for k in ks}
for f in sorted(glob.glob(f'{args.emb_dir}/{args.dataset}_seed*.pt')):
    d = torch.load(f)
    z, y = d['z'], d['y']
    for k in ks:
        zk = z
        for _ in range(k):
            zk = a_hat @ zk
        zk = F.normalize(zk, p=2, dim=1)
        acc, nmi, ari, f1, _ = clustering(zk.numpy(), n_clusters, y.numpy(), spectral_clustering=True)
        res[k].append((acc, nmi, ari, f1))
        print(f'{f} k={k} ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f}', flush=True)
for k in ks:
    r = np.array(res[k])
    print(f'{args.dataset} k={k} n={len(r)} | ' + ' '.join(
        f'{m}={r[:, i].mean():.4f}±{r[:, i].std():.4f}' for i, m in enumerate(['ACC', 'NMI', 'ARI', 'F1'])))
