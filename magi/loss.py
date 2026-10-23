import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self, temperature=0.07, scale_by_temperature=True, scale_by_weight=False):
        super(Loss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature
        self.scale_by_weight = scale_by_weight

    def forward(self, out, mask, pos_weight='uniform', neg_weight=None, fn_labels=None):
        """Modularity-guided contrastive loss.

        pos_weight: 'uniform' (original MAGI) averages log-probs over positives;
                    'count' weights each positive pair by its random-walk co-occurrence count.
        neg_weight: optional [B] node weights in the denominator (e.g. degree / mean degree),
                    i.e. the d_u d_v / 2m null-model term of modularity.
        fn_labels:  optional [B] pseudo cluster labels; non-positive pairs in the same cluster
                    are treated as false negatives and removed from the denominator.
        """
        device = out.device
        row, col, val = mask.storage.row(), mask.storage.col(), mask.storage.value()
        row, col, val = row.to(device), col.to(device), val.to(device)
        batch_size = out.shape[0]

        # compute logits; only the [B, B] similarity and one logsumexp buffer are materialized
        dot = torch.matmul(out, out.T)
        dot = torch.div(dot, self.temperature)
        pos_logits = dot[row, col]

        dot.fill_diagonal_(float('-inf'))
        if fn_labels is not None:
            fn_labels = fn_labels.to(device)
            same = fn_labels.view(-1, 1) == fn_labels.view(1, -1)
            same[row, col] = False
            dot = dot.masked_fill(same, float('-inf'))
        if neg_weight is not None:
            dot = dot + torch.log(neg_weight.to(device)).view(1, -1)
        log_probs = pos_logits - torch.logsumexp(dot, dim=1)[row]

        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        if pos_weight == 'uniform':
            w = torch.ones_like(log_probs)
        elif pos_weight == 'count':
            w = val.float()
        else:
            raise ValueError(f"Unknown pos_weight {pos_weight}")
        num = torch.zeros(batch_size, device=device).scatter_add_(0, row, w * log_probs)
        den = torch.zeros(batch_size, device=device).scatter_add_(0, row, w)
        has_pos = den > 0
        loss = -num[has_pos] / den[has_pos]

        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss
