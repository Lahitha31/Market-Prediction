import pandas as pd
from pathlib import Path

NUM_COLS = ["open", "high", "low", "close", "volume"]

def load_prices_from_csv(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")


    df = pd.read_csv(path, parse_dates=["ts"])


    df.columns = [c.strip().lower() for c in df.columns]


    for c in NUM_COLS:
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                    .str.replace(",", "", regex=False)
                    .str.replace("$", "", regex=False)
                    .str.strip()
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")


    df = df.dropna(subset=["ts"] + NUM_COLS).copy()

    # Index & sort
    df = df.set_index("ts").sort_index()

    return df
