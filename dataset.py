#dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


def winsorize_series(x, lower=0.01, upper=0.99):
    """
    对单个横截面因子做 winsorize 极值处理：
    - 在横截面（同一天 N 支股票）上计算分位数
    - 将过高 / 过低的值拉回边界
    目的：减少极端噪声（如异常成交量、撮合错误），提升训练稳定性
    """
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        # 全 NaN 直接返回全 0（占位）
        return np.zeros_like(x)

    lo = np.percentile(valid, lower * 100)
    hi = np.percentile(valid, upper * 100)
    out = np.clip(x, lo, hi)

    # 注意：原本是 NaN 的位置仍保持 NaN，不直接填补
    return out


def zscore_series(x):
    """
    横截面 z-score 标准化：
        (x - mean) / std

    - 在单日的横截面内做标准化，而不是跨天
    - 保证不同因子在相同尺度，有利于模型训练
    """
    valid = x[~np.isnan(x)]
    if len(valid) == 0:
        return np.zeros_like(x)

    mean = valid.mean()
    std = valid.std()

    # 如果 std=0（极小盘 or 单调列），避免炸掉
    if std == 0 or np.isnan(std):
        return np.zeros_like(x)

    out = (x - mean) / std
    # 横截面的 NaN 全部替换为 0，避免模型接收到 nan
    out = np.where(np.isnan(out), 0.0, out)
    return out


class PanelDailyDataset(Dataset):
    """
    面板数据 Dataset（每日一个样本）：

    每个样本：
        X:  (num_stocks, num_features)  —— 当日所有股票的因子矩阵
        y:  (num_stocks, 1)             —— 当日股票的 next-return 目标
        meta: { "symbols": 股票代码数组 }

    设计思路：
    -----------------------------------------
    - 横截面预测的训练单位是“某一天的所有股票”
    - 因此 dataset 返回的是 “按天切片” 的 panel，而不是“按股票切片”
    - 有利于模型一次性学习股票之间的关系（cross-sectional interaction）
    - 每日内部执行 winsorize + zscore，保持横截面标准化
    """

    def __init__(self, csv_path, feature_cols):
        df = pd.read_csv(csv_path)

        # 将空字符串统一转为 NaN，确保处理一致性
        df = df.replace(r'^\s*$', np.nan, regex=True)

        # 按日期聚合为 panel
        grouped = df.groupby("date")
        self.data_list = []

        for date, g in grouped:
            # 取当天所有股票的特征矩阵
            X = g[feature_cols].astype(float).values

            # 对每个因子列做 winsorize + zscore（横截面标准化）
            for j in range(X.shape[1]):
                col = X[:, j]
                col = winsorize_series(col)
                col = zscore_series(col)
                X[:, j] = col

            # 预测标签（次日收益）
            y = g["target_next_ret"].astype(float).values.reshape(-1, 1)

            # 股票代码（用于可视化或调试）
            symbols = g["symbol"].values

            # 转 tensor
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            meta = {"symbols": symbols}

            self.data_list.append((X, y, meta))

        print(f"[PanelDailyDataset] 有效交易日数 = {len(self.data_list)}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 返回： (X, y, meta)
        return self.data_list[idx]
