import numpy as np
import pandas as pd
from pathlib import Path
from .metrics import safe_roc_auc, quarterly_perf

def apply_threshold(proba, threshold=0.5, max_position=1):

    pos = np.zeros_like(proba, dtype=int)
    pos[proba > threshold] = 1
    pos[proba < (1 - threshold)] = -1
    pos = np.clip(pos, -max_position, max_position)
    return pos

def walk_forward_pnl(df, proba, threshold=0.5, tc_bps=1.0):

    proba_clean = np.where(np.isnan(proba), 0.5, proba)

    pos = apply_threshold(proba_clean, threshold=threshold)
    pos_shift = np.roll(pos, 1)
    pos_shift[0] = 0
    pnl = pos_shift * df["fwd_ret1"].values


    trades = np.abs(np.diff(pos, prepend=0))
    tc = trades * (tc_bps / 1e4)
    pnl_after_tc = pnl - tc

    out = pd.DataFrame({
        "proba": proba_clean,
        "pos": pos,
        "pnl": pnl_after_tc
    }, index=df.index)
    return out

def buy_and_hold_benchmark(df):
    pnl = df["fwd_ret1"].values.copy()
    return pd.Series(pnl, index=df.index, name="bh_pnl")

def evaluate(df, proba, threshold=0.5, tc_bps=1.0):

    y_true_full = (df["fwd_ret1"] > 0).astype(int).values


    mask = ~np.isnan(proba)
    y_true_auc = y_true_full[mask]
    y_score_auc = proba[mask]
    auc = safe_roc_auc(y_true_auc, y_score_auc)

    pnl_df = walk_forward_pnl(df, proba, threshold, tc_bps)
    bh = buy_and_hold_benchmark(df)
    res = {
        "roc_auc": float(auc),
        "strategy_pnl_sum": float(pnl_df["pnl"].sum()),
        "benchmark_pnl_sum": float(bh.sum()),
        "quarterly_strategy": quarterly_perf(df.index, pnl_df["pnl"]),
        "quarterly_benchmark": quarterly_perf(df.index, bh)
    }
    return res, pnl_df, bh
