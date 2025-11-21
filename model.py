# model.py
# PyTorch model for cross-sectional multi-factor daily return prediction.
# Design目标：
# - 适配每天变化的 stock_num（即 variable-length first-dim）
# - 同时兼顾 per-stock 表征（MLP encoder）和截面间交互（轻量 self-attention）
# - 保持模块化，方便你替换 encoder / attention / head
#
# 组件说明（设计思路）：
# 1) StockEncoder (MLP)：对每只股票的 factor 向量进行编码 -> 得到 embedding。
#    理由：因子是表格型特征，MLP 是稳定的基线；LayerNorm/Dropout 提升训练稳定性。
# 2) Cross-stock MultiHeadSelfAttention：可选模块，用来学习同一天内股票间的交互（如流动性竞争、行业关联）。
#    理由：横截面里股票并非完全独立，attention 可以自动学习“哪些股票对某只股票的预测更重要”。
# 3) Residual gating（可学习的门控）：平衡本地特征与跨股票信息的权重（避免 attention 覆盖掉原始信息）。
# 4) Head：把最终 embedding 映射到标量预测（次日收益）。
#
# 输入/输出约定：
# - forward 接受 x: tensor (stock_num, feature_dim)
# - 返回 preds: tensor (stock_num, 1), attn_weights (num_heads, stock_num, stock_num) 或 None
#
# 扩展点（你可以微调）：
# - 替换 encoder 为 group-wise encoder（按行业/市值区间），采用 embedding + MLP
# - 去掉 attention（性能更好，计算量更小）
# - 把 attention 换成 pooling（mean/quantile）作为低成本替代
# - 支持 batch of days （可在上层循环中构造 batch）
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StockEncoder(nn.Module):
    """
    Per-stock MLP encoder.
    Input: (stock_num, feature_dim)
    Output: (stock_num, hidden_dim)
    """
    def __init__(self, feature_dim, hidden_dims=(128, 64), dropout=0.1, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        # x: (N, D)
        return self.net(x)  # (N, out_dim)


class MultiHeadSelfAttention(nn.Module):
    """
    Lightweight Multi-Head Self-Attention across stocks in the same day.
    Input: x (N, E)
    Output: out (N, E), attn_weights (H, N, N)
    Note: Designed for moderate N (hundreds). For very large N consider sparse attention or pooling.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, mask=None):
        # x: (N, E)
        N, E = x.shape
        qkv = self.qkv(x).view(N, 3, self.num_heads, self.head_dim)  # (N, 3, H, D)
        q = qkv[:, 0].permute(1, 0, 2)  # (H, N, D)
        k = qkv[:, 1].permute(1, 0, 2)  # (H, N, D)
        v = qkv[:, 2].permute(1, 0, 2)  # (H, N, D)

        # attention scores (H, N, N)
        # einsum to compute per-head q @ k^T
        scores = torch.einsum('hnd,hdm->hnm', q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # mask: (N, N) boolean, True means mask out
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (H, N, N)
        attn = self.dropout(attn)
        out = torch.einsum('hnm,hmd->hnd', attn, v)  # (H, N, D)
        out = out.permute(1, 0, 2).contiguous().view(N, E)  # (N, E)
        out = self.out_proj(out)
        return out, attn  # out: (N, E)


class CrossSectionModel(nn.Module):
    """
    Full cross-sectional model:
      encoder -> to_embed -> cross-attention -> gated residual -> head
    forward:
      x: (stock_num, feature_dim)
      mask: optional (stock_num, stock_num) boolean to mask attention entries (e.g., for missing)
    returns:
      preds: (stock_num, 1)
      attn: (num_heads, stock_num, stock_num) or None
    """
    def __init__(self,
                 feature_dim,
                 encoder_hidden=(128, 64),
                 embed_dim=64,
                 attn_heads=4,
                 head_hidden=32,
                 attn_dropout=0.1,
                 encoder_dropout=0.1,
                 use_attention=True):
        super().__init__()
        self.encoder = StockEncoder(feature_dim, hidden_dims=encoder_hidden, dropout=encoder_dropout)
        last_enc = self.encoder.out_dim
        if last_enc != embed_dim:
            self.proj = nn.Linear(last_enc, embed_dim)
        else:
            self.proj = nn.Identity()

        self.use_attention = use_attention
        if use_attention:
            self.attn = MultiHeadSelfAttention(embed_dim, num_heads=attn_heads, dropout=attn_dropout)
        else:
            self.attn = None

        # small MLP head that maps embedding to scalar
        self.head = nn.Sequential(
            nn.Linear(embed_dim, head_hidden),
            nn.ReLU(),
            nn.LayerNorm(head_hidden),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, 1)
        )

        # learnable gate controlling weight between local and cross info
        self.gate_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, mask=None):
        """
        x: (N, D)
        mask: optional (N, N) boolean for attention masking
        returns: preds (N, 1), attn_weights or None
        """
        z = self.encoder(x)  # (N, enc_dim)
        z = self.proj(z)     # (N, embed_dim)

        if self.use_attention:
            cross, attn_weights = self.attn(z, mask=mask)  # (N, E), (H, N, N)
        else:
            cross, attn_weights = torch.zeros_like(z), None

        gate = torch.sigmoid(self.gate_param)
        # combine local and cross info
        h = gate * z + (1.0 - gate) * cross
        preds = self.head(h)  # (N, 1)
        return preds, attn_weights


if __name__ == "__main__":
    # quick shape test with simulated day
    feature_dim = 30
    stock_num = 120
    x = torch.randn(stock_num, feature_dim)
    model = CrossSectionModel(feature_dim, use_attention=True)
    preds, attn = model(x)
    print("preds", preds.shape)
    if attn is not None:
        print("attn", attn.shape)
