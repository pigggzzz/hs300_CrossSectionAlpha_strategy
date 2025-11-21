# train_real.py
# 使用真实股票面板数据训练横截面预测模型（CrossSectionModel）

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import CrossSectionModel
from loss import CompositeCrossSectionLoss
from metrics import pearson_ic, topn_long_short_return, direction_accuracy
from dataset import PanelDailyDataset


def collate_identity(batch):
    """DataLoader 保持 batch=[ (X, y, meta) ]"""
    return batch


def main():

    # =====================================================
    # 1. 特征列
    # =====================================================
    feature_cols = [
        "open", "high", "low", "close",
        "volume", "amount",
        "return_t", "amplitude", "log_amount",
        "return_5", "vol_5",
    ]

    csv_path = "hs300_panel.csv"
    full_ds = PanelDailyDataset(csv_path, feature_cols)
    total_days = len(full_ds)
    print(f"[INFO] 加载成功：{total_days} 个交易日")

    # =====================================================
    # 2. 训练 / 验证切分
    # =====================================================
    split = int(total_days * 0.8)
    train_ds = full_ds.data_list[:split]
    val_ds = full_ds.data_list[split:]

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_identity)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_identity)

    # =====================================================
    # 3. 模型
    # =====================================================
    feature_dim = len(feature_cols)
    model = CrossSectionModel(feature_dim, use_attention=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = CompositeCrossSectionLoss(alpha=1.0, beta=1.0, gamma=0.2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备：{device}")
    model.to(device)

    # =====================================================
    # 4. TensorBoard
    # =====================================================
    logdir = "./runs_real/"
    os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(logdir)

    # =====================================================
    # 5. 训练
    # =====================================================
    epochs = 30
    global_step = 0

    for epoch in range(epochs):

        model.train()
        train_losses = []

        for batch in train_loader:
            X, y, meta = batch[0]

            # 确保 tensor
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32)

            X = X.to(device)
            y = y.to(device)

            preds, attn = model(X)

            loss, loss_terms = loss_fn(preds, y)

            optimizer.zero_grad()
            loss.backward()

            # --- 梯度裁剪：减少梯度爆炸 ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            optimizer.step()

            train_losses.append(loss.item())

            writer.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # =====================================================
        # 验证
        # =====================================================
        model.eval()
        ic_list, topn_list, diracc_list = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                X, y, meta = batch[0]

                if not isinstance(X, torch.Tensor):
                    X = torch.tensor(X, dtype=torch.float32)
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y, dtype=torch.float32)

                X = X.to(device)
                y = y.to(device)

                preds, _ = model(X)

                ic = pearson_ic(preds, y)
                ls = topn_long_short_return(preds, y, top_pct=0.1)[2]
                da = direction_accuracy(preds, y)

                # 避免 nan 污染平均数
                if not np.isnan(ic):
                    ic_list.append(ic)
                if not np.isnan(ls):
                    topn_list.append(ls)
                if not np.isnan(da):
                    diracc_list.append(da)

        avg_train_loss = np.mean(train_losses)
        avg_ic = np.mean(ic_list) if len(ic_list) > 0 else np.nan
        avg_topn = np.mean(topn_list) if len(topn_list) > 0 else np.nan
        avg_diracc = np.mean(diracc_list) if len(diracc_list) > 0 else np.nan

        # 写入日志
        writer.add_scalar("val/IC", avg_ic, epoch)
        writer.add_scalar("val/topN_ls", avg_topn, epoch)
        writer.add_scalar("val/direction_acc", avg_diracc, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={avg_train_loss:.6f} | "
            f"val_IC={avg_ic:.4f} | "
            f"val_topN={avg_topn:.4f} | "
            f"val_diracc={avg_diracc:.4f}"
        )

    writer.close()
    print("\n训练完成！打开 TensorBoard 查看训练过程：\n")
    print("    tensorboard --logdir=runs_real\n")


if __name__ == "__main__":
    main()
