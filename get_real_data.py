import akshare as ak
import pandas as pd
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------
# 获取沪深300成分股列表
# ---------------------------------------------------------
def get_hs300_symbols():
    """
    返回沪深300成分股代码列表，如 ['000001', '600000', ...]
    """
    df = ak.index_stock_cons("000300")
    symbols = df["品种代码"].tolist()
    symbols = [s for s in symbols if isinstance(s, str)]
    return symbols


# ---------------------------------------------------------
# 下载单只股票历史行情（使用稳定接口）
# ---------------------------------------------------------
def fetch_stock(symbol):
    """
    使用 AkShare 官方接口（更稳定）
    symbol: '000001' → 自动识别成 'sz000001'
            '600000' → 自动识别成 'sh600000'
    """
    if symbol.startswith("6"):
        ts_code = "sh" + symbol
    else:
        ts_code = "sz" + symbol

    df = ak.stock_zh_a_daily(symbol=ts_code)

    if df is None or df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["symbol"] = symbol
    return df


# ---------------------------------------------------------
# 增加因子特征
# ---------------------------------------------------------
def add_features(df):
    df = df.copy()
    df["return_t"] = df["close"].pct_change()
    df["amplitude"] = (df["high"] - df["low"]) / df["close"]
    df["log_amount"] = np.log1p(df["amount"])
    df["return_5"] = df["close"].pct_change(5)
    df["vol_5"] = df["return_t"].rolling(5).std()
    return df


# ---------------------------------------------------------
# 添加 next-day y
# ---------------------------------------------------------
def add_target(df):
    df = df.copy()
    df["target_next_ret"] = df["close"].shift(-1) / df["close"] - 1
    return df


# ---------------------------------------------------------
# 构建沪深300 panel
# ---------------------------------------------------------
def build_hs300_panel(max_stocks=300):
    symbols = get_hs300_symbols()
    print("沪深300成分股数量：", len(symbols))

    panel_list = []

    for sym in tqdm(symbols[:max_stocks]):
        df = fetch_stock(sym)
        if df is None or df.empty:
            continue

        df = add_features(df)
        df = add_target(df)
        df = df.dropna(subset=["target_next_ret"])

        panel_list.append(df)

    if len(panel_list) == 0:
        raise ValueError("没有成功下载任何股票的数据！")

    panel = pd.concat(panel_list, ignore_index=True)
    return panel


# ---------------------------------------------------------
# 主运行
# ---------------------------------------------------------
if __name__ == "__main__":
    df = build_hs300_panel(max_stocks=300)

    print(df.head())
    print(df.tail())
    print("总行数：", len(df))

    df.to_csv("hs300_panel.csv", index=False)
    print("已保存 hs300_panel.csv")
