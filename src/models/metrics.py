from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

def safe_roc_auc(y_true, y_score):

    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float("nan")

def quarterly_perf(ts_index, pnl_series):

    idx = pd.DatetimeIndex(ts_index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df = pd.DataFrame({"pnl": pnl_series}, index=idx)
    q = df.resample("QE")["pnl"].sum() 
    q.index = q.index.to_period("Q").astype(str)
    return q
