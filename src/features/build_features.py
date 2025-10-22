import numpy as np
import pandas as pd

NUM_COLS = ["open", "high", "low", "close", "volume"]


def _ensure_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    for c in NUM_COLS:
        if c in out.columns:
            out[c] = (
                out[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.strip()
            )
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def order_imbalance(df: pd.DataFrame, window: int = 10):
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    bar_imb = ((df["close"] - df["open"]) / rng).clip(-5, 5)
    return bar_imb.rolling(window, min_periods=window).mean().fillna(0)

def realized_vol(df: pd.DataFrame, window: int = 20):
    ret = np.log(df["close"]).diff()
    return (ret.rolling(window).std() * np.sqrt(252)).fillna(0)

def time_to_close_index(df: pd.DataFrame):
    ts = pd.DatetimeIndex(df.index)
    market_close = ts.normalize() + pd.Timedelta(hours=16)
    delta = (market_close - ts)  # TimedeltaIndex
    return delta / pd.Timedelta(minutes=1)  # minutes as float

def add_lags(df: pd.DataFrame, cols, lags=(1, 2, 3, 5, 10)):
    for c in cols:
        for L in lags:
            df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def rolling_stats(df: pd.DataFrame, col: str, windows=(5, 10, 20)):
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}"]  = df[col].rolling(w).std()
    return df


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's RSI
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    loss = down.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = gain / (loss + 1e-12)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

def momentum(series: pd.Series, window: int = 10) -> pd.Series:

    return np.log(series).diff(window)


def build_features(df: pd.DataFrame,
                lags=(1, 2, 3, 5, 10),
                roll_windows=(5, 10, 20),
                realized_vol_window=20,
                imbalance_window=10):

    out = _ensure_numeric_cols(df).dropna(subset=NUM_COLS).copy()
    out = out.sort_index()


    out["imbalance"] = order_imbalance(out, imbalance_window)
    out["rvol"] = realized_vol(out, realized_vol_window)

    try:
        out["ttc_min"] = time_to_close_index(out)
    except Exception:
        out["ttc_min"] = 0.0

    out["ret1"] = np.log(out["close"]).diff(1)
    out["fwd_ret1"] = out["ret1"].shift(-1)


    out["rsi_14"] = rsi(out["close"], period=14)


    out["ema12"] = ema(out["close"], 12)
    out["ema26"] = ema(out["close"], 26)
    out["ema_spread"] = out["ema12"] - out["ema26"]
    out["ema_spread_norm"] = out["ema_spread"] / (out["close"] + 1e-12)


    macd_line, signal_line, hist = macd(out["close"], 12, 26, 9)
    out["macd_line"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist


    out["mom_5"]  = momentum(out["close"], 5)
    out["mom_10"] = momentum(out["close"], 10)
    out["mom_20"] = momentum(out["close"], 20)

    
    out = rolling_stats(out, "rvol", roll_windows)

    
    out = add_lags(out, ["ret1", "imbalance", "rvol", "rsi_14", "ema_spread_norm", "macd_hist"], lags)

    
    return out
