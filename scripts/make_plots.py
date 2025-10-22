# scripts/make_plots.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BT = ROOT / "artifacts" / "backtests"
PLOTS = ROOT / "artifacts" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


pnl_df = pd.read_csv(BT / "pnl_timeseries.csv", parse_dates=["ts"]).set_index("ts")
bh = pd.read_csv(BT / "benchmark_timeseries.csv", parse_dates=["ts"]).set_index("ts")["bh_pnl"]
qsum = pd.read_csv(BT / "quarterly_summary.csv", index_col=0)


eq_strategy = pnl_df["pnl"].cumsum()
eq_bench = bh.cumsum()

plt.figure()
plt.plot(eq_strategy, label="Strategy (cum PnL)")
plt.plot(eq_bench, label="Benchmark (cum PnL)")
plt.title("Equity Curves")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL")
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS / "equity_curves.png")
plt.close()


plt.figure()
qsum[["strategy", "benchmark"]].plot(kind="bar")
plt.title("Quarterly PnL")
plt.xlabel("Quarter")
plt.ylabel("PnL")
plt.tight_layout()
plt.savefig(PLOTS / "quarterly_pnl.png")
plt.close()

print(f"Saved plots to: {PLOTS}")
