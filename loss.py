# loss.py
# =============================================================
# 自定义横截面预测损失函数（Cross-Section Loss）
#
# 设计目标：
#   - 量化横截面预测（cross-sectional stock return prediction）本质是“排序问题”
#   - 交易收益来自同日股票之间的相对排名，而不是预测绝对误差
#   - 因此，需要把 **经济目标 (IC / 排序)** 纳入训练，而非单独依赖 MSE
#
# 本文件包含三个损失：
#   (1) Pearson correlation loss       —— 最大化横截面相关系数 IC（收益最关键指标）
#   (2) Pairwise hinge ranking loss    —— 直接优化排序（对应 long top / short bottom）
#   (3) CompositeCrossSectionLoss      —— 组合 MSE、IC、Ranking（最符合金融任务）
#
# 设计背后逻辑：
#   - 经典 MSE 会迫使模型拟合“绝对值”，但选股策略关心排序、分组收益
#   - IC (Information Coefficient) 直接衡量信号的横截面质量，是行业标准指标
#   - Pairwise rank 能强制模型“让高收益股票得分更高”
#
#   => 综合三者，可以让模型兼顾：
#         · 稳定性（由 MSE 控制）
#         · 截面相关性（由 IC 控制）
#         · 排序/选股目标（由 Ranking 控制）
# =============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------
# 1) 横截面 Pearson 相关系数（IC）
# -------------------------------------------------------------
def pearson_corr(preds, targets, eps=1e-8):
    """
    计算横截面 Pearson 相关系数（IC）。

    输入：
      preds  : (N,)  当天 N 只股票的预测值
      targets: (N,)  当天 N 只股票的真实 next-return

    为什么要 IC：
      - IC 是量化选股最经典指标
      - 在 long-short 策略中，IC 越高，组合收益越稳健
      - 直接最大化 IC 会让模型“更懂排序，而不是拟合绝对数值”

    返回：
      单个标量相关系数（torch tensor）
    """
    p = preds.view(-1)
    t = targets.view(-1)

    p_mean = torch.mean(p)
    t_mean = torch.mean(t)

    # 去中心化
    p_cent = p - p_mean
    t_cent = t - t_mean

    cov = torch.mean(p_cent * t_cent)
    p_std = torch.sqrt(torch.mean(p_cent ** 2) + eps)
    t_std = torch.sqrt(torch.mean(t_cent ** 2) + eps)

    return cov / (p_std * t_std + eps)


def pearson_loss(preds, targets):
    """
    用 1 - corr 表示损失，使得：
          minimize(L)  <=>  maximize(IC)

    —— 这是横截面金融任务中最直观、最常用的 IC 训练方式。
    """
    return 1.0 - pearson_corr(preds, targets)


# -------------------------------------------------------------
# 2) Pairwise hinge ranking loss（排序损失）
# -------------------------------------------------------------
def pairwise_hinge_loss(preds, targets, margin=1e-4):
    """
    排序损失：强制 “收益更高的股票预测分数也更高”。

    对所有满足 target_i > target_j 的股票对 (i, j)，
    强制模型满足：
         pred_i > pred_j + margin

    损失形式：
         loss = mean( max(0, margin - (pred_i - pred_j)) )

    金融意义：
      - 直接提升 Top-N / 分组收益
      - 强化“相对排序”而不是“绝对误差”
      - 对应 learning-to-rank（RankNet/Hinge Rank）思想

    复杂度 O(N^2)，但每天股票数 ~300，可接受。
    """
    p = preds.view(-1)
    t = targets.view(-1)
    N = p.shape[0]

    if N <= 1:
        return torch.tensor(0.0, device=preds.device)

    diff_t = t.unsqueeze(1) - t.unsqueeze(0)
    pos_mask = diff_t > 0     # i 的收益比 j 高

    if pos_mask.sum() == 0:
        return torch.tensor(0.0, device=preds.device)

    diff_p = p.unsqueeze(1) - p.unsqueeze(0)
    loss_mat = F.relu(margin - diff_p) * pos_mask.float()

    return loss_mat.sum() / (pos_mask.sum().float() + 1e-8)


# -------------------------------------------------------------
# 3) 组合损失 —— MSE + (1−IC) + Rank (核心)
# -------------------------------------------------------------
class CompositeCrossSectionLoss(nn.Module):
    """
    Composite loss = α * MSE + β * (1 - Corr) + γ * PairwiseRank

    三个权重的作用：
      α：控制数值稳定性（避免发散；保持两者在同一数量级）
      β：提升横截面相关性 IC，是量化最关键收益指标
      γ：强化相对排序，直接影响 Top-N 回测收益

    训练时建议：
      - 多观察验证集的 IC / top-N return 来调参
      - 若更关注收益，可提升 gamma 或 beta
      - 若模型不稳定，可提升 alpha
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)

        loss_mse = self.mse(preds, targets)
        loss_corr = pearson_loss(preds, targets)
        loss_rank = pairwise_hinge_loss(preds, targets)

        total = (
            self.alpha * loss_mse +
            self.beta * loss_corr +
            self.gamma * loss_rank
        )

        return total, {
            "mse": loss_mse.item(),
            "corr": loss_corr.item(),
            "rank": loss_rank.item()
        }


# quick sanity test
if __name__ == "__main__":
    preds = torch.randn(50, 1)
    targets = torch.randn(50, 1)
    loss_fn = CompositeCrossSectionLoss(alpha=1.0, beta=1.0, gamma=0.5)
    total, terms = loss_fn(preds, targets)
    print("total", total.item(), terms)
