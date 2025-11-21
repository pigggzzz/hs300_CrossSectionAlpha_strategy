import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def winsorize_series(x, lower=0.01, upper=0.99):
    """对单个横截面因子做极值处理"""
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.zeros_like(x)

    lo = np.percentile(valid, lower * 100)
    hi = np.percentile(valid, upper * 100)

    out = np.clip(x, lo, hi)
    # 极值处理后仍保留 NaN
    return out


def zscore_series(x):
    """横截面 zscore 标准化"""
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.zeros_like(x)

    mean = valid.mean()
    std = valid.std()
    if std == 0 or np.isnan(std):
        return np.zeros_like(x)

    out = (x - mean) / std
    # 统一把 NaN 替换成 0
    out = np.where(np.isnan(out), 0.0, out)
    return out


class PanelDailyDataset(Dataset):

    def __init__(self, csv_path, feature_cols):
        df = pd.read_csv(csv_path)

        # 把空字段（空字符串）全部识别成 NaN
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # 按日期分组
        grouped = df.groupby("date")
        self.data_list = []

        for date, g in grouped:
            X = g[feature_cols].astype(float).values

            # 对每个因子做 winsorize + zscore
            for j in range(X.shape[1]):
                col = X[:, j]
                col = winsorize_series(col)
                col = zscore_series(col)
                X[:, j] = col

            y = g["target_next_ret"].astype(float).values.reshape(-1, 1)
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
