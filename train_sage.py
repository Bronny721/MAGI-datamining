import csv
import random
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_undirected, add_remaining_self_loops
from torch_sparse import SparseTensor

# PyTorch >= 2.6 defaults torch.load(weights_only=True); ogb's dataset loader needs these PyG classes allowlisted
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

from magi.model import Model, Encoder
from magi.neighbor_sampler import NeighborSampler
from magi.utils import setup_seed, get_mask, clustering, prune_positives, expand_positives


parser = argparse.ArgumentParser()
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--max_duration', type=int,
                    default=60, help='max duration time')
parser.add_argument('--kmeans_device', type=str,
                    default='cpu', help='kmeans device, cuda or cpu')
parser.add_argument('--kmeans_batch', type=int, default=-1,
                    help='batch size of kmeans on GPU, -1 means full batch')
parser.add_argument('--batchsize', type=int, default=2048, help='')
parser.add_argument('--test_batch', type=int, default=10000, help='seed nodes per inference batch')

# dataset para
parser.add_argument('--dataset', type=str, default='ogbn-arxiv')

# model para
parser.add_argument('--hidden_channels', type=str, default='512,256')
parser.add_argument('--size', type=str, default='10,10', help='')
parser.add_argument('--projection', type=str, default='')
parser.add_argument('--tau', type=float, default=0.5, help='temperature')
parser.add_argument('--ns', type=float, default=0.5)

# sample para
parser.add_argument('--wt', type=int, default=20)
parser.add_argument('--wl', type=int, default=4)
parser.add_argument('--n', type=int, default=2048)

# learning para
parser.add_argument('--dropout', type=float, default=0, help='')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--wd', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)

# engineering / improvement para
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--legacy_sampler', action='store_true', help='use the original per-node python sampler')
parser.add_argument('--pos_weight', type=str, default='uniform', help='uniform (MAGI) or count')
parser.add_argument('--deg_neg', action='store_true', help='degree-weighted negatives (modularity null model)')
parser.add_argument('--deg_beta', type=float, default=1.0, help='negative weight = (deg / mean deg) ** beta')
parser.add_argument('--pos_deg', action='store_true', help='degree-corrected null model for positive selection')
parser.add_argument('--pos_beta', type=float, default=1.0, help='exponent of degree in the positive null model')
parser.add_argument('--sp_warmup', type=int, default=0, help='epoch to start self-paced positive denoising (0: off)')
parser.add_argument('--sp_drop', type=float, default=0.1, help='fraction of positives dropped per anchor')
parser.add_argument('--sp_interval', type=int, default=10, help='epochs between positive re-ranking')
parser.add_argument('--pos_thresh', type=str, default='mean', help='mean (official code) or all (Eq. 9, every visited node)')
parser.add_argument('--se_warmup', type=int, default=0, help='epoch to start embedding-guided positive expansion (0: off)')
parser.add_argument('--se_tau', type=float, default=0.95, help='embedding cosine similarity threshold for expansion')
parser.add_argument('--se_interval', type=int, default=10, help='epochs between positive expansion updates')
parser.add_argument('--result', type=str, default='', help='csv file to save results')
args = parser.parse_args()


def train():
    ts = time.time()
    randint = random.randint(1, 1000000)
    setup_seed(randint)
    if args.verbose:
        print('random seed : ', randint, '\n', args)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M']:
        path = './data/OGB/'
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(root=path, name=args.dataset)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
        y = y[:, 0]
    elif args.dataset == 'Reddit':
        path = './data/Reddit/'
        dataset = Reddit(root=path)
        x, edge_index, y = dataset[0].x, dataset[0].edge_index, dataset[0].y
    else:
        raise RuntimeError(f"Unknown dataset {args.dataset}")

    edge_index = to_undirected(add_remaining_self_loops(edge_index)[0])

    N, E, num_features = x.shape[0], edge_index.shape[-1], x.shape[-1]
    print(f"Loading {args.dataset} is over, num_nodes: {N: d}, num_edges: {E: d}, "
          f"num_feats: {num_features: d}, time costs: {time.time()-ts: .2f}")

    adj = SparseTensor(row=edge_index[0],
                       col=edge_index[1], sparse_sizes=(N, N))
    adj.fill_value_(1.)
    raw_deg = adj.storage.rowcount().float()
    deg = raw_deg
    deg = (deg / deg.mean()) ** args.deg_beta

    hidden = list(map(int, args.hidden_channels.split(',')))
    if args.projection == '':
        projection = None
    else:
        projection = list(map(int, args.projection.split(',')))
    size = list(map(int, args.size.split(',')))
    assert len(hidden) == len(size)

    train_loader = NeighborSampler(edge_index, adj,
                                   is_train=True,
                                   node_idx=None,
                                   wt=args.wt,
                                   wl=args.wl,
                                   sizes=size,
                                   batch_size=args.batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   legacy_sampling=args.legacy_sampler,
                                   num_workers=args.num_workers)

    test_loader = NeighborSampler(edge_index, adj,
                                  is_train=False,
                                  node_idx=None,
                                  sizes=size,
                                  batch_size=args.test_batch,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=args.num_workers)

    encoder = Encoder(num_features, hidden_channels=hidden,
                      dropout=args.dropout, ns=args.ns).to(device)
    model = Model(
        encoder, in_channels=hidden[-1], project_hidden=projection, tau=args.tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd)

    dataset2n_clusters = {'ogbn-arxiv': 40, 'Reddit': 41,
                          'ogbn-products': 47, 'ogbn-papers100M': 172}
    n_clusters = dataset2n_clusters[args.dataset]

    x = x.to(device)
    print(f"Start training")

    ts_train = time.time()
    stop_pos = False
    for epoch in range(1, args.epochs):
        model.train()
        total_loss = total_examples = 0

        for (batch_size, n_id, adjs), adj_batch, batch in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]

            adj_ = get_mask(adj_batch, deg=raw_deg[batch] if args.pos_deg else None, beta=args.pos_beta,
                            thresh=args.pos_thresh)
            optimizer.zero_grad()
            out = model(x[n_id].to(device), adjs=adjs)
            out = F.normalize(out, p=2, dim=1)
            if args.sp_warmup > 0 and epoch >= args.sp_warmup:
                adj_ = prune_positives(adj_, out.detach(), args.sp_drop)
            if args.se_warmup > 0 and epoch >= args.se_warmup:
                adj_ = expand_positives(adj_batch, adj_, out.detach(), args.se_tau)
            neg_weight = deg[batch] if args.deg_neg else None
            loss = model.loss(out, adj_, pos_weight=args.pos_weight, neg_weight=neg_weight)

            loss.backward()
            optimizer.step()
            total_loss += float(loss)
            total_examples += batch_size

            if args.verbose:
                print(f'(T) | Epoch {epoch:02d}, loss: {loss:.4f}, '
                      f'train_time_cost: {time.time() - ts_train:.2f}, examples: {batch_size:d}, '
                      f'nodes: {out.size(0):d}, gpu_peak: {torch.cuda.max_memory_allocated() / 2**30:.2f}G')

            train_time_cost = time.time() - ts_train
            if train_time_cost // 60 >= args.max_duration:
                print(
                    "*********************** Maximum training time is exceeded ***********************")
                stop_pos = True
                break
        if stop_pos:
            break

    print(f'Finish training, training time cost: {time.time() - ts_train:.2f}')

    del optimizer
    torch.cuda.empty_cache()
    with torch.no_grad():
        model.eval()
        z = []
        for count, ((batch_size, n_id, adjs), _, batch) in enumerate(tqdm(test_loader)):
            if len(hidden) == 1:
                adjs = [adjs]
            adjs = [adj.to(device) for adj in adjs]
            out = model(x[n_id].to(device), adjs=adjs)
            z.append(out.detach().cpu().float())
        z = torch.cat(z, dim=0)
        z = F.normalize(z, p=2, dim=1)

    ts_clustering = time.time()
    print(f'Start clustering, num_clusters: {n_clusters: d}')
    acc, nmi, ari, f1_macro, f1_micro = clustering(z, n_clusters, y.numpy(), kmeans_device=args.kmeans_device,
                                                   batch_size=args.kmeans_batch, tol=1e-4, device=device, spectral_clustering=False)

    print(f'Finish clustering, acc: {acc:.4f}, nmi: {nmi:.4f}, ari: {ari:.4f}, f1_macro: {f1_macro:.4f}, '
          f'f1_micro: {f1_micro:.4f}, clustering time cost: {time.time() - ts_clustering:.2f}')
    return acc, nmi, ari, f1_macro, f1_micro


def run(runs=1, result=None):
    if result:
        with open(result, 'w', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(
                ['runs', 'acc', 'nmi', 'ari', 'f1_macro', 'f1_micro'])

    ACC, NMI, ARI, F1_MA, F1_MI = [], [], [], [], []
    for i in range(runs):
        print(f'\n----------------------runs {i+1: d} start')
        acc, nmi, adjscore, f1_macro, f1_micro = train()
        print(f'\n----------------------runs {i + 1: d} over')
        if result:
            with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
                writer = csv.writer(f_w)
                writer.writerow([i+1, acc, nmi, adjscore, f1_macro, f1_micro])

        ACC.append(acc)
        NMI.append(nmi)
        ARI.append(adjscore)
        F1_MA.append(f1_macro)
        F1_MI.append(f1_micro)

    ACC = np.array(ACC)
    NMI = np.array(NMI)
    ARI = np.array(ARI)
    F1_MA = np.array(F1_MA)
    F1_MI = np.array(F1_MI)
    if result:
        with open(result, 'a', encoding='utf-8-sig', newline='') as f_w:
            writer = csv.writer(f_w)
            writer.writerow(['mean', ACC.mean(), NMI.mean(),
                            ARI.mean(), F1_MA.mean(), F1_MI.mean()])
            writer.writerow(['std', ACC.std(), NMI.std(),
                            ARI.std(), F1_MA.std(), F1_MI.std()])


if __name__ == '__main__':
    run(args.runs, args.result or None)
