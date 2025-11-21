# model.py
# ----------------------------------------------------------------------
# PyTorch Model for Cross-Sectional Daily Stock Return Prediction
#
# 设计目标（面向横截面因子模型 Cross-sectional Prediction）：
# 1) 能处理每日股票数量 N 不固定（不同交易日上市/停牌不同）
# 2) 特征为 panel/tabular 结构，适合 MLP 编码
# 3) 利用轻量级 self-attention 捕捉股票之间的横截面关系
# 4) 最终输出 next-day return 的 cross-section 排序分数
#
# 模块组成（核心设计逻辑）：
# ----------------------------------------------------------------------
# (1) StockEncoder（股票级 MLP 编码）
#     输入：每只股票的因子向量（表格数据）
#     输出：embedding（低维表征）
#     作用：提供稳健的 per-stock 表示，是横截面任务的基础。
#
# (2) MultiHeadSelfAttentionBlock（可选）
#     输入：当天所有股票的 embedding (N, E)
#     输出：融合横截面信息后的新 embedding
#     作用：让模型捕捉股票之间的相关性、行业聚集效应、风险共振等结构性关系。
#
# (3) Residual + Gating
#     将原始 encoder 表征 和 attention 输出做加权融合。
#     原因：部分任务 attention 不一定比 MLP 强，gate 可以动态调节权重。
#
# (4) SmallHead（回归头）
#     作用：将 embedding 映射到 scalar（次日收益预测）。
#
# 训练/推理接口：
#     forward(x) —— 输入 (N, D)，输出 (N, 1) 和 attention matrix（如果开启）
#
# ----------------------------------------------------------------------
# 这个模型是“横截面因子模型”的标准结构：MLP 基座 + optional attention，
# 能兼顾稳定训练和横截面排序能力。
# ----------------------------------------------------------------------


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class StockEncoder(nn.Module):
    """
    Per-stock MLP encoder with final LayerNorm.
    Input: (N, D) -> Output: (N, enc_dim)
    """
    def __init__(self, feature_dim, hidden_dims=(128, 64), dropout=0.1, activation=nn.ReLU):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.net = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(in_dim)
        self.out_dim = in_dim

    def forward(self, x):
        # x: (N, D)
        z = self.net(x)
        z = self.ln(z)
        return z  # (N, out_dim)


class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention block with residual, layernorm and FFN (Transformer-style).
    Input: x (N, E)
    Output: out (N, E), attn_weights (H, N, N)
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1, ffn_hidden=None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # LayerNorms for pre/post
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # FFN
        if ffn_hidden is None:
            ffn_hidden = max(embed_dim * 4, 128)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (N, E)
        N, E = x.shape
        # self-attention (pre-norm)
        qkv = self.qkv(self.ln1(x)).view(N, 3, self.num_heads, self.head_dim)  # (N,3,H,D)
        q = qkv[:, 0].permute(1, 0, 2)  # (H,N,D)
        k = qkv[:, 1].permute(1, 0, 2)
        v = qkv[:, 2].permute(1, 0, 2)

        # scores (H,N,N)
        scores = torch.einsum('hnd,hdm->hnm', q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            # mask: (N, N) boolean -> True = mask out
            scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = torch.softmax(scores, dim=-1)  # (H,N,N)
        attn = self.attn_dropout(attn)
        out = torch.einsum('hnm,hmd->hnd', attn, v)  # (H,N,D)
        out = out.permute(1, 0, 2).contiguous().view(N, E)  # (N,E)
        out = self.out_proj(out)

        # residual + second LN + FFN
        x = x + out
        x = x + self.ffn(self.ln2(x))
        # return current representation and attn weights
        return x, attn


class SmallHead(nn.Module):
    """
    Small head mapping embedding -> scalar with residual bottleneck.
    """
    def __init__(self, embed_dim, hidden=32, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = self.ln(h)
        h = self.dropout(h)
        out = self.fc2(h)
        return out


class CrossSectionModel(nn.Module):
    """
    encoder -> (optional) cross-attn-block -> head
    Keeps same forward signature: x (N, D) -> (preds (N,1), attn or None)
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
            # one attention block (can be stacked externally if needed)
            self.attn_block = MultiHeadSelfAttentionBlock(embed_dim, num_heads=attn_heads, dropout=attn_dropout)
        else:
            self.attn_block = None

        self.head = SmallHead(embed_dim, hidden=head_hidden, dropout=0.1)

        # gate param keeps ability to weight encoder vs attn output if we want (kept for compatibility)
        self.gate_param = nn.Parameter(torch.tensor(0.5))

        # initialize weights
        self.init_weights()

    def init_weights(self):
        # Xavier init for linear layers & zero bias
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        x: (N, D)
        returns preds (N,1), attn_weights or None
        """
        if x.shape[0] == 0:
            # guard: empty day
            return torch.zeros((0,1), device=x.device), None

        z = self.encoder(x)  # (N, enc_dim)
        z = self.proj(z)     # (N, embed_dim)

        attn_weights = None
        if self.use_attention and self.attn_block is not None:
            z_attn, attn_weights = self.attn_block(z, mask=mask)  # (N,E)
            # combine with simple gating (learned scalar)
            gate = torch.sigmoid(self.gate_param)
            z = gate * z + (1.0 - gate) * z_attn
        # else z unchanged

        preds = self.head(z)  # (N,1)
        return preds, attn_weights


if __name__ == "__main__":
    # sanity run
    feature_dim = 11
    stock_num = 120
    x = torch.randn(stock_num, feature_dim)
    model = CrossSectionModel(feature_dim, use_attention=True)
    preds, attn = model(x)
    print("preds", preds.shape)
    if attn is not None:
        print("attn", attn.shape)