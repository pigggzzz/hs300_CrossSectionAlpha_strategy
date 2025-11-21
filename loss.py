# loss.py
# 自定义损失函数集合：目标是把训练目标（MSE）与金融目标（IC / 排序）结合起来。
# 包含：
#  - pearson correlation loss (maximize IC)
#  - pairwise ranking (hinge) loss (优化排序)
#  - composite loss combining MSE + corr + ranking
#
# 设计理由：
# - 在横截面预测中，排序和 IC 通常比绝对误差更直接决定交易 PnL（long top short bottom）。
# - 但单纯优化 rank 可能忽视残差大小（影响回撤），所以用混合 Loss。
# - 提供权重参数方便在验证集上以经济指标（IC / top-N return）微调权重。
import torch
import torch.nn as nn
import torch.nn.functional as F


def pearson_corr(preds, targets, eps=1e-8):
    """
    Compute Pearson correlation coefficient between preds and targets for a cross-section.
    Returns correlation scalar (torch tensor).
    """
    p = preds.view(-1)
    t = targets.view(-1)
    p_mean = torch.mean(p)
    t_mean = torch.mean(t)
    p_cent = p - p_mean
    t_cent = t - t_mean
    cov = torch.mean(p_cent * t_cent)
    p_std = torch.sqrt(torch.mean(p_cent ** 2) + eps)
    t_std = torch.sqrt(torch.mean(t_cent ** 2) + eps)
    corr = cov / (p_std * t_std + eps)
    return corr


def pearson_loss(preds, targets):
    """
    Loss term that is 1 - corr so minimization maximizes correlation (IC).
    """
    return 1.0 - pearson_corr(preds, targets)


def pairwise_hinge_loss(preds, targets, margin=1e-4):
    """
    Pairwise hinge ranking loss.
    For all pairs (i, j) where target_i > target_j, enforce pred_i > pred_j + margin.
    Loss = mean( max(0, margin - (pred_i - pred_j)) )
    Complexity O(N^2) per cross-section - OK for moderate N.
    """
    p = preds.view(-1)
    t = targets.view(-1)
    N = p.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=preds.device)
    # pairwise difference
    diff_t = t.unsqueeze(1) - t.unsqueeze(0)  # (N, N)
    pos_mask = diff_t > 0  # bool mask where i > j
    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)
    diff_p = p.unsqueeze(1) - p.unsqueeze(0)  # (N, N)
    loss_mat = F.relu(margin - diff_p) * pos_mask.float()
    # average only over positive pairs
    loss = loss_mat.sum() / (pos_mask.sum().float() + 1e-8)
    return loss


class CompositeCrossSectionLoss(nn.Module):
    """
    Composite loss = alpha * MSE + beta * (1 - Corr) + gamma * PairwiseRank
    Return: (total_loss, dict_of_terms)
    Use case: during training use total_loss.backward(); inspector can log term values.
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        """
        preds, targets: tensors (N,1) or (N,)
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        loss_mse = self.mse(preds, targets)
        loss_corr = pearson_loss(preds, targets)
        loss_rank = pairwise_hinge_loss(preds, targets)
        total = self.alpha * loss_mse + self.beta * loss_corr + self.gamma * loss_rank
        return total, {"mse": loss_mse.item(), "corr": loss_corr.item(), "rank": loss_rank.item()}


if __name__ == "__main__":
    # quick sanity test
    import torch
    preds = torch.randn(50, 1)
    targets = torch.randn(50, 1)
    loss_fn = CompositeCrossSectionLoss(alpha=1.0, beta=1.0, gamma=0.5)
    total, terms = loss_fn(preds, targets)
    print("total", total.item(), terms)
