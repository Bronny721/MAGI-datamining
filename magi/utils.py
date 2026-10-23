import random
import torch
import numpy as np
from tqdm import tqdm
from torch_sparse import SparseTensor
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
try:
    from sklearnex import patch_sklearn
except:
    def patch_sklearn(): return

from magi.clustering_metric import clustering_metrics
from magi.batch_kmeans_cuda import kmeans


def get_sim(batch, adj, wt=20, wl=3):
    rowptr, col, _ = adj.csr()
    batch_size = batch.shape[0]
    batch_repeat = batch.repeat(wt)
    rw = adj.random_walk(batch_repeat, wl)[:, 1:]

    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    rw = rw.t().reshape(-1, batch_size).t()

    row, col, val = [], [], []
    for i in range(batch.shape[0]):
        rw_nodes, rw_times = torch.unique(rw[i], return_counts=True)
        row += [batch[i].item()] * rw_nodes.shape[0]
        col += rw_nodes.tolist()
        val += rw_times.tolist()

    unique_nodes = list(set(row + col))
    subg2g = dict(zip(unique_nodes, list(range(len(unique_nodes)))))

    row = [subg2g[x] for x in row]
    col = [subg2g[x] for x in col]
    idx = torch.tensor([subg2g[x] for x in batch.tolist()])

    adj_ = SparseTensor(row=torch.LongTensor(row), col=torch.LongTensor(col), value=torch.tensor(val),
                        sparse_sizes=(len(unique_nodes), len(unique_nodes)))

    adj_batch, _ = adj_.saint_subgraph(idx)
    adj_batch = adj_batch.set_diag(0.)
    # src, dst = dict_r[idx[adj_batch.storage.row()[3].item()].item()], dict_r[idx[adj_batch.storage.col()[3].item()].item()]
    return batch, adj_batch


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def clustering(feature, n_clusters, true_labels, kmeans_device='cpu', batch_size=100000, tol=1e-4, device=torch.device('cuda:0'), spectral_clustering=False):
    if spectral_clustering:
        if isinstance(feature, torch.Tensor):
            feature = feature.numpy()
        print("spectral clustering on cpu...")
        patch_sklearn()
        Cluster = SpectralClustering(
            n_clusters=n_clusters, affinity='precomputed', random_state=0)
        f_adj = np.matmul(feature, np.transpose(feature))
        predict_labels = Cluster.fit_predict(f_adj)
    else:
        if kmeans_device == 'cuda':
            if isinstance(feature, np.ndarray):
                feature = torch.tensor(feature)
            print("kmeans on gpu...")
            predict_labels, _ = kmeans(
                X=feature, num_clusters=n_clusters, batch_size=batch_size, tol=tol, device=device)
            predict_labels = predict_labels.numpy()
        else:
            if isinstance(feature, torch.Tensor):
                feature = feature.numpy()
            print("kmeans on cpu...")
            patch_sklearn()
            Cluster = KMeans(n_clusters=n_clusters, max_iter=10000, n_init=20)
            predict_labels = Cluster.fit_predict(feature)

    cm = clustering_metrics(true_labels, predict_labels)
    acc, nmi, adjscore, fms, f1_macro, f1_micro = cm.evaluationClusterModelFromLabel(
        tqdm)
    return acc, nmi, adjscore, f1_macro, f1_micro


def get_mask(adj, deg=None, beta=1.0, thresh='mean'):
    """Select positive pairs whose random-walk co-occurrence exceeds the row's expected count.

    deg=None: the original MAGI null model, expected count = row mean.
    deg given ([n] node degrees aligned with adj columns): degree-corrected null model,
    expected count of v = row mean * d_v^beta / (row mean of d^beta), so hub nodes that
    are visited often only because of their degree are not taken as positives.
    """
    if thresh == 'all':
        # Eq. (9) of the paper with |B| = batch size: every visited node is a positive
        value = adj.storage.value()
        keep = value > 0
        return SparseTensor(row=adj.storage.row()[keep], col=adj.storage.col()[keep],
                            value=value[keep], sparse_sizes=(adj.size(0), adj.size(1)))
    batch_mean = adj.mean(dim=1)
    row_idx = torch.LongTensor(adj.storage.row())
    mean = batch_mean[row_idx]
    if deg is not None:
        col_idx = adj.storage.col()
        d = deg.float()[col_idx] ** beta
        row_d = torch.zeros(adj.size(0)).scatter_reduce_(0, row_idx, d, reduce='mean', include_self=False)
        mean = mean * d / row_d[row_idx]
    mask = (adj.storage.value() - mean) > - 1e-10
    row, col, val = adj.storage.row()[mask], adj.storage.col()[
        mask], adj.storage.value()[mask]
    adj_ = SparseTensor(row=row, col=col, value=val,
                        sparse_sizes=(adj.size(0), adj.size(1)))
    return adj_


@torch.no_grad()
def prune_positives(mask, z, q):
    """Self-paced positive denoising: for every anchor, drop the fraction q of its
    structural positives that have the lowest similarity under the current embeddings z."""
    row, col, val = mask.coo()
    row, col = row.to(z.device), col.to(z.device)
    sim = (z[row] * z[col]).sum(1)
    # sort by anchor, then by similarity (descending); cosine sim is in [-1, 1]
    order = torch.argsort(row.double() * 4 - sim.double())
    row_sorted = row[order]
    cnt = torch.bincount(row_sorted, minlength=mask.size(0))
    start = torch.cumsum(cnt, 0) - cnt
    rank = torch.arange(row_sorted.numel(), device=z.device) - start[row_sorted]
    keep = torch.empty_like(rank, dtype=torch.bool)
    keep[order] = rank < torch.ceil(cnt[row_sorted] * (1 - q))
    keep = keep.cpu()
    return SparseTensor(row=row.cpu()[keep], col=col.cpu()[keep], value=val[keep],
                        sparse_sizes=mask.sizes())


@torch.no_grad()
def expand_positives(adj, mask, z, threshold):
    """Embedding-guided positive expansion: add visited-but-below-mean pairs of the
    random-walk co-occurrence matrix adj whose current embedding similarity exceeds threshold."""
    row, col, val = adj.coo()
    n = adj.size(1)
    mrow, mcol, _ = mask.coo()
    is_pos = torch.isin(row * n + col, mrow * n + mcol)
    cand = (~is_pos) & (row != col) & (val > 0)
    sim = (z[row.to(z.device)] * z[col.to(z.device)]).sum(1).cpu()
    keep = is_pos | (cand & (sim > threshold))
    return SparseTensor(row=row[keep], col=col[keep], value=val[keep], sparse_sizes=mask.sizes())


def scale(z: torch.Tensor):
    zmax = z.max(dim=1, keepdim=True)[0]
    zmin = z.min(dim=1, keepdim=True)[0]
    z_std = (z - zmin) / ((zmax - zmin) + 1e-20)
    z_scaled = z_std
    return z_scaled
