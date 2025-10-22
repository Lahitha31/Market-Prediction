import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf

OUT = Path("artifacts/data/prices.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY", help="Ticker symbol, e.g., AAPL, MSFT, SPY")
    ap.add_argument("--interval", default="5m", help="Valid: 1m, 2m, 5m, 15m, 30m, 60m, 1d, etc.")
    ap.add_argument("--period", default="60d", help="For intraday, Yahoo limits to ~60d (e.g., 30d, 60d).")
    args = ap.parse_args()

    print(f"Downloading {args.symbol} {args.interval} for {args.period} from Yahoo…")
    df = yf.download(args.symbol, interval=args.interval, period=args.period, auto_adjust=False, progress=False)

    if df.empty:
        raise SystemExit("No data returned—try a different symbol/interval/period.")


    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low":  "low",
        "Close":"close",
        "Volume":"volume",
    }).reset_index().rename(columns={"Datetime":"ts", "Date":"ts"})


    if "ts" not in df.columns:

        df = df.rename(columns={df.columns[0]: "ts"})

    df["symbol"] = args.symbol
    df = df[["ts", "open", "high", "low", "close", "volume", "symbol"]].sort_values("ts")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df):,} rows to {OUT}")

if __name__ == "__main__":
    main()
