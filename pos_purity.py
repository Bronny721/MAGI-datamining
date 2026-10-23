"""Diagnostic: label purity of MAGI positive pairs under the original vs. degree-corrected null model."""
import sys, torch
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor
from magi.utils import get_sim, get_mask, setup_seed

params = {'Cora': (100, 2), 'Citeseer': (100, 3), 'Photo': (100, 3), 'Computers': (100, 3)}
for name in sys.argv[1:]:
    setup_seed(0)
    data = (Planetoid if name in ['Cora', 'Citeseer'] else Amazon)('data/', name)[0]
    N = data.num_nodes
    ei = to_undirected(add_remaining_self_loops(data.edge_index)[0])
    adj = SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(N, N)); adj.fill_value_(1.)
    _, ab = get_sim(torch.arange(N), adj, wt=params[name][0], wl=params[name][1])
    deg = adj.storage.rowcount().float()
    y = data.y
    for label, m in [('MAGI (row mean)', get_mask(ab)),
                     ('degree-corrected b=0.5', get_mask(ab, deg=deg, beta=0.5)),
                     ('degree-corrected b=1', get_mask(ab, deg=deg, beta=1.0))]:
        r, c = m.storage.row(), m.storage.col()
        purity = (y[r] == y[c]).float().mean().item()
        hub = deg[c].mean().item()
        print(f'{name:9s} {label:24s} pairs={r.numel():8d} purity={purity:.4f} mean_deg_of_positives={hub:.1f}')
