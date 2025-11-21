# metrics.py
# Evaluation metrics for single-day cross-sectional prediction.
# Implementations:
# - Pearson IC (Information Coefficient)
# - Spearman IC (rank)
# - top-N long/short return (mean of realized returns for chosen top/bottom groups)
# - direction accuracy (sign accuracy)
# - optional hit rate for top-k stocks
import numpy as np
from scipy.stats import spearmanr


def pearson_ic(preds, targets):
    """
    preds, targets: numpy arrays or torch tensors (N,) or (N,1)
    returns float
    """
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy().reshape(-1)
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy().reshape(-1)
    if preds.size == 0:
        return np.nan
    px = preds - preds.mean()
    ty = targets - targets.mean()
    denom = (px.std(ddof=0) * ty.std(ddof=0))
    if denom == 0:
        return 0.0
    return float((px * ty).mean() / denom)


def spearman_ic(preds, targets):
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy().reshape(-1)
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy().reshape(-1)
    if preds.size == 0:
        return np.nan
    r, _ = spearmanr(preds, targets)
    return float(r)


def topn_long_short_return(preds, targets, top_pct=0.1):
    """
    Compute long-short return: mean(targets[top_k]) - mean(targets[bottom_k])
    returns (long_mean, short_mean, long_short)
    """
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy().reshape(-1)
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy().reshape(-1)
    N = preds.shape[0]
    if N == 0:
        return (np.nan, np.nan, np.nan)
    k = max(1, int(N * top_pct))
    order = np.argsort(preds)  # ascending
    top_idx = order[-k:]
    bot_idx = order[:k]
    long_mean = float(np.mean(targets[top_idx]))
    short_mean = float(np.mean(targets[bot_idx]))
    return long_mean, short_mean, long_mean - short_mean


def direction_accuracy(preds, targets):
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy().reshape(-1)
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy().reshape(-1)
    if preds.size == 0:
        return np.nan
    pred_sign = np.sign(preds)
    true_sign = np.sign(targets)
    mask = true_sign != 0
    if mask.sum() == 0:
        return np.nan
    return float((pred_sign[mask] == true_sign[mask]).sum() / mask.sum())


def topk_hit_rate(preds, targets, k=10):
    """
    Hit rate: fraction of predicted top-k that are actually in realized top-k (by true returns).
    """
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy().reshape(-1)
    if hasattr(targets, "detach"):
        targets = targets.detach().cpu().numpy().reshape(-1)
    N = preds.shape[0]
    if N == 0 or k <= 0:
        return np.nan
    k = min(k, N)
    pred_top = np.argsort(preds)[-k:]
    true_top = np.argsort(targets)[-k:]
    hits = len(set(pred_top).intersection(set(true_top)))
    return hits / k


if __name__ == "__main__":
    import numpy as np
    N = 100
    preds = np.random.randn(N)
    targets = np.random.randn(N)
    print("Pearson IC:", pearson_ic(preds, targets))
    print("Spearman IC:", spearman_ic(preds, targets))
    print("TopN:", topn_long_short_return(preds, targets, 0.1))
    print("Direction acc:", direction_accuracy(preds, targets))
    print("Topk hit:", topk_hit_rate(preds, targets, 10))
