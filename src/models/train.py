import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path

def make_model(model_type: str, params: dict):
    if model_type.lower() in ["rf", "randomforest", "randomforestclassifier"]:
        return RandomForestClassifier(**params)
    elif model_type.lower() in ["logreg", "logisticregression", "lr"]:

        C = params.pop("C", 1.0)
        max_iter = params.pop("max_iter", 200)
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", LogisticRegression(C=C, max_iter=max_iter, n_jobs=params.get("n_jobs", None)))
        ])
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def time_series_cv(X, y, n_splits=5, gap=0, model=None):
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    preds = np.full_like(y, fill_value=np.nan, dtype=float)
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        ytr = y.iloc[train_idx]
        mdl = model
        mdl.fit(Xtr, ytr)
        if hasattr(mdl, "predict_proba"):
            p = mdl.predict_proba(Xte)[:, 1]
        else:
            p = mdl.decision_function(Xte)
            # map to 0..1
            p = (p - p.min()) / (p.max() - p.min() + 1e-9)
        preds[test_idx] = p
    return preds

def save_model(model, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
