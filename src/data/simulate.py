import numpy as np
import pandas as pd
from pathlib import Path

def simulate_intraday_prices(years=3, freq="5min", seed=42, symbol="SYNTH"):
    rng = np.random.default_rng(seed)

    start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
    end = pd.Timestamp.today().normalize()
    days = pd.bdate_range(start, end, freq="C")
    intraday = pd.date_range("09:30", "16:00", freq=freq).time
    idx = pd.MultiIndex.from_product([days, intraday], names=["date", "time"])


    n = len(idx)
    drift = 0.00002  
    noise = rng.normal(scale=0.0015, size=n)
    season = np.sin(np.linspace(0, 6*np.pi, n)) * 0.0008
    ret = drift + season + noise
    price = 100 * (1 + pd.Series(ret).cumsum().apply(np.exp) * 0)


    price = 100 * np.exp(pd.Series(ret).cumsum())
    price.index = idx

    df = pd.DataFrame(index=pd.to_datetime(
        [pd.Timestamp(d) + pd.Timedelta(str(t)) for d, t in idx.to_list()]
    ))
    df.index.name = "ts"
    df["close"] = price.values

    df["open"] = df["close"].shift(1).fillna(df["close"])
    rng2 = np.random.default_rng(seed+1)
    spread = np.abs(rng2.normal(0.0008, 0.0003, len(df)))
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + spread)
    df["low"]  = df[["open", "close"]].min(axis=1) * (1 - spread)


    t = np.arange(len(df))
    vol_shape = (np.sin(2*np.pi*t/len(intraday)) + 1.2)  
    base_vol = 1e4
    vol_noise = np.maximum(0, rng2.normal(0, 1500, len(df)))
    df["volume"] = (base_vol * vol_shape + vol_noise).astype(int)

    df["symbol"] = symbol
    return df

def save_csv(df: pd.DataFrame, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
