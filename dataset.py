# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def winsorize_series(x, lower=0.01, upper=0.99):
    """对单日某个因子做极值处理"""
    lo = np.nanpercentile(x, lower * 100)
    hi = np.nanpercentile(x, upper * 100)
    return np.clip(x, lo, hi)


def zscore_series(x):
    mean = np.nanmean(x)
    std = np.nanstd(x)
    if std == 0 or np.isnan(std):
        return np.zeros_like(x)
    return (x - mean) / std


class PanelDailyDataset(Dataset):
    """
    读取多股票面板 CSV，并对每天做横截面预处理（winsorize + zscore）
    """

    def __init__(self, csv_path, feature_cols):
        df = pd.read_csv(csv_path)

        # 按日期分组
        grouped = df.groupby("date")
        self.data_list = []

        for date, g in grouped:

            # ----- 横截面预处理 -----
            X = g[feature_cols].copy().values

            # 对每一个因子按日做 winsorize + zscore
            for j in range(X.shape[1]):
                col = X[:, j]
                col = winsorize_series(col, 0.01, 0.99)
                col = zscore_series(col)
                X[:, j] = col

            y = g["target_next_ret"].values.reshape(-1, 1)
            symbols = g["symbol"].values

            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            meta = {"symbols": symbols}

            self.data_list.append((X, y, meta))

        print(f"[PanelDailyDataset] 有效交易日数 = {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
