import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.data.simulate import simulate_intraday_prices, save_csv
from src.data.load import load_prices_from_csv
from src.features.build_features import build_features
from src.models.train import make_model, time_series_cv, save_model
from src.models.backtest import evaluate


ARTIFACTS = PROJECT_ROOT / "artifacts"
DATA_DIR = ARTIFACTS / "data"
MODELS_DIR = ARTIFACTS / "models"
BACKTEST_DIR = ARTIFACTS / "backtests"
PLOTS_DIR = ARTIFACTS / "plots"


def _load_cfg(path: str) -> dict:
    """Load YAML config with safe defaults if keys are missing."""
    cfg = load_config(path)
    # Minimal defaults in case a section is missing
    cfg.setdefault("data", {}).setdefault("years", 3)
    cfg["data"].setdefault("freq", "5min")
    cfg["data"].setdefault("seed", 42)
    cfg["data"].setdefault("symbol", "SYNTH")

    cfg.setdefault("features", {})
    cfg["features"].setdefault("lags", [1, 2, 3, 5, 10])
    cfg["features"].setdefault("roll_windows", [5, 10, 20])
    cfg["features"].setdefault("realized_vol_window", 20)
    cfg["features"].setdefault("imbalance_window", 10)

    cfg.setdefault("labeling", {})
    # quantile-mode defaults (you can override in config.yaml)
    cfg["labeling"].setdefault("method", "quantile")   # "quantile" | "sign"
    cfg["labeling"].setdefault("horizon", 1)
    cfg["labeling"].setdefault("lower_q", 0.20)
    cfg["labeling"].setdefault("upper_q", 0.80)
    cfg["labeling"].setdefault("threshold", 0.0)       # only used in "sign"

    cfg.setdefault("model", {})
    cfg["model"].setdefault("type", "RandomForestClassifier")
    cfg["model"].setdefault(
        "params",
        {"n_estimators": 200, "max_depth": 8, "random_state": 42, "n_jobs": -1},
    )

    cfg.setdefault("cv", {})
    cfg["cv"].setdefault("n_splits", 5)
    cfg["cv"].setdefault("gap", 0)

    cfg.setdefault("backtest", {})
    cfg["backtest"].setdefault("transaction_cost_bps", 1.0)
    cfg["backtest"].setdefault("decision_threshold", 0.5)
    cfg["backtest"].setdefault("max_position", 1)
    return cfg


def main(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)

    #DATA
    if args.mode == "simulate":
        df = simulate_intraday_prices(
            years=(args.years or cfg["data"]["years"]),
            freq=(args.freq or cfg["data"]["freq"]),
            seed=cfg["data"]["seed"],
            symbol=cfg["data"]["symbol"],
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_csv(df, DATA_DIR / "prices.csv")
        print(f"[data] simulated -> {DATA_DIR / 'prices.csv'}  rows={len(df):,}")
    else:
        if not args.data_path:
            raise SystemExit("--data_path is required when --mode csv")
        df = load_prices_from_csv(args.data_path)
        if df.empty:
            raise SystemExit(f"No rows found in {args.data_path}")
        print(f"[data] loaded CSV: {args.data_path}  rows={len(df):,}")

    #FEATURES
    feats = build_features(
        df,
        lags=tuple(cfg["features"]["lags"]),
        roll_windows=tuple(cfg["features"]["roll_windows"]),
        realized_vol_window=cfg["features"]["realized_vol_window"],
        imbalance_window=cfg["features"]["imbalance_window"],
    )
    if feats.empty:
        raise SystemExit("Feature frame is empty after preprocessing (check data and feature windows).")

    #LABEL + DESIGN MATRIX 
    method = str(cfg["labeling"].get("method", "quantile")).lower()
    horizon = int(cfg["labeling"].get("horizon", 1))


    feats["ret1"] = np.log(feats["close"]).diff(1)
    feats["fwd_ret1"] = feats["ret1"].shift(-horizon)

    if method == "quantile":
        lo_q = float(cfg["labeling"].get("lower_q", 0.20))
        hi_q = float(cfg["labeling"].get("upper_q", 0.80))

        q_lo = feats["fwd_ret1"].quantile(lo_q)
        q_hi = feats["fwd_ret1"].quantile(hi_q)


        y_all = pd.Series(np.nan, index=feats.index, dtype="float64")
        y_all.loc[feats["fwd_ret1"] >= q_hi] = 1.0
        y_all.loc[feats["fwd_ret1"] <= q_lo] = 0.0


        train_idx = y_all.dropna().index
        y = y_all.loc[train_idx].astype(int)

        X = feats.drop(columns=["fwd_ret1"], errors="ignore")
        X = X.loc[train_idx]
        X = (
            X.select_dtypes(include=["number"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        # Align y to X after dropna
        y = y.loc[X.index]

        print(f"[labels] method=quantile lo={lo_q} hi={hi_q} classes={y.value_counts().to_dict()}  train_rows={len(X)}")

    else:  
        thresh = float(cfg["labeling"].get("threshold", 0.0))
        y = (feats["fwd_ret1"] > thresh).astype(int)
        X = feats.drop(columns=["fwd_ret1"], errors="ignore")
        X = (
            X.select_dtypes(include=["number"])
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        y = y.loc[X.index]
        print(f"[labels] method=sign thresh={thresh} classes={y.value_counts().to_dict()}  train_rows={len(X)}")

    if len(X) < 100:
        print(f"[warn] very few training rows after cleaning: {len(X)}")

    # MODEL + TIME-SERIES CV
    model = make_model(cfg["model"]["type"], dict(cfg["model"]["params"]))
    proba_train = time_series_cv(X, y, n_splits=cfg["cv"]["n_splits"], gap=cfg["cv"]["gap"], model=model)


    proba_full = pd.Series(np.nan, index=feats.index, dtype="float64")
    proba_full.loc[X.index] = proba_train

    # EVALUATE 
    res, pnl_df, bh = evaluate(
        feats,                              # evaluate on full timeline
        proba_full.values,                  # np.ndarray incl. NaN at neutral rows
        threshold=cfg["backtest"]["decision_threshold"],
        tc_bps=cfg["backtest"]["transaction_cost_bps"],
    )

    # SAVE ARTIFACTS
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


    model.fit(X, y)
    save_model(model, MODELS_DIR / "model.joblib")

    pnl_df.to_csv(BACKTEST_DIR / "pnl_timeseries.csv")
    bh.to_csv(BACKTEST_DIR / "benchmark_timeseries.csv")
    q_strat = res["quarterly_strategy"]
    q_bench = res["quarterly_benchmark"]
    pd.DataFrame({"strategy": q_strat, "benchmark": q_bench}).to_csv(BACKTEST_DIR / "quarterly_summary.csv")

    with open(BACKTEST_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "roc_auc": res["roc_auc"],
                "strategy_pnl_sum": res["strategy_pnl_sum"],
                "benchmark_pnl_sum": res["benchmark_pnl_sum"],
            },
            f,
            indent=2,
        )


    print("\n=== HFMP Results ===")
    print(f"ROC-AUC: {res['roc_auc']:.4f}" if pd.notna(res["roc_auc"]) else "ROC-AUC: NaN (insufficient class variation)")
    print(f"Strategy PnL (sum): {res['strategy_pnl_sum']:.6f}")
    print(f"Benchmark PnL (sum): {res['benchmark_pnl_sum']:.6f}")

    print("\nQuarterly outperformance (strategy > benchmark):")
    q_merge = pd.concat([q_strat, q_bench], axis=1).dropna()
    q_merge.columns = ["strategy", "benchmark"]
    q_merge["beat_bench"] = q_merge["strategy"] > q_merge["benchmark"]
    print(q_merge)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    ap.add_argument("--mode", choices=["simulate", "csv"], default="simulate")
    ap.add_argument("--data_path", type=str, default=str(DATA_DIR / "prices.csv"))
    ap.add_argument("--years", type=int, default=None)
    ap.add_argument("--freq", type=str, default=None)
    
    args = ap.parse_args()
    main(args)
