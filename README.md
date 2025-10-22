# High-Frequency Market Prediction (HFMP)

A reproducible starter project for building and evaluating a high‑frequency market prediction model on financial time‑series.  
It includes:
- Synthetic data generator that mimics intraday OHLCV and microstructure signals
- Feature engineering (order imbalance, realized volatility, time‑to‑close, lags/rolling stats)
- Leak‑proof **time‑series cross‑validation**
- **Walk‑forward backtesting** over 3 years with quarterly performance attribution
- **ROC‑AUC** model metric and **benchmark** (buy‑and‑hold) comparison
- Clean, modular `src/` with a single `run_pipeline.py` entrypoint

> Use this scaffold with your real data by replacing the loader in `src/data/load.py`.

## Quickstart

```
# 1) Create env (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run full pipeline (generates synthetic data by default)
python scripts/run_pipeline.py --mode simulate --years 3 --freq 5min --model rf
```

Outputs:
- `artifacts/data/` – CSVs of synthetic prices & features
- `artifacts/models/` – trained model and metadata
- `artifacts/backtests/` – per‑fold & quarterly performance
- `artifacts/plots/` – metric charts


## Notes on Realistic Evaluation

- The pipeline uses **TimeSeriesSplit** and **walk‑forward** splits to prevent leakage.
- Feature windows use only past information.
- Backtest evaluates directional predictions converted to positions using decision thresholds.
- Benchmark is a buy‑and‑hold (constant 1) over the same period.

## Extending
- Plug in your data in `src/data/load.py`
- Add alpha features in `src/features/build_features.py`
- Try other models in `src/models/train.py` (XGBoost/LightGBM/etc.)
