"""
Train / predict several models on the same daily split as SARIMA (test 20%).
Used by export_demo_data.py → CSV + JSON for Streamlit (no heavy training in the app).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sarima_model import FACTOR_COLS, aggregate_to_daily, load_fa_data, train_test_split

LAG_TABULAR = 14
LSTM_LOOKBACK = 14
LSTM_EPOCHS = 30
LSTM_BATCH = 16
SEASONAL_M = 7


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    m = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[m], y_pred[m]
    if len(y_true) == 0:
        return {"RMSE": float("nan"), "MAE": float("nan"), "MAPE": float("nan")}
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def build_lag_tabular(
    pm25: pd.Series,
    factors: pd.DataFrame,
    lag: int,
    split_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rows: target at index t uses lags [t-lag, t). Train if t < split_idx else test."""
    pm = pm25.astype(float)
    fac = factors.reindex(pm.index).ffill().bfill()
    vals = pm.values
    rows_X, rows_y, rows_pos = [], [], []
    for t in range(lag, len(pm)):
        lags = vals[t - lag : t]
        if all(c in fac.columns for c in FACTOR_COLS):
            fvec = fac.iloc[t][FACTOR_COLS].values.astype(float)
        else:
            fvec = np.zeros(3, dtype=float)
        rows_X.append(np.concatenate([lags, fvec]))
        rows_y.append(vals[t])
        rows_pos.append(t)
    X = np.asarray(rows_X, dtype=float)
    y = np.asarray(rows_y, dtype=float)
    pos = np.asarray(rows_pos, dtype=int)
    train_mask = pos < split_idx
    test_mask = ~train_mask
    return X, y, train_mask, test_mask


def _lstm_sequences(arr_1d: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(arr_1d) - look_back):
        x.append(arr_1d[i : i + look_back])
        y.append(arr_1d[i + look_back])
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _predict_lstm_univariate_aligned(
    pm_all: pd.Series, split_idx: int, look_back: int, epochs: int, batch_size: int
) -> Optional[np.ndarray]:
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.layers import Dense, LSTM
        from tensorflow.keras.models import Sequential
    except Exception:
        return None

    values = pm_all.values.reshape(-1, 1).astype(float)
    sp = split_idx
    lb = look_back
    train_vals = values[:sp]
    test_concat = values[sp - lb :]

    scaler_y = MinMaxScaler((0, 1))
    train_s = scaler_y.fit_transform(train_vals).ravel()
    test_s = scaler_y.transform(test_concat).ravel()

    x_tr, y_tr = _lstm_sequences(train_s, lb)
    x_te, y_te = _lstm_sequences(test_s, lb)
    if len(x_te) == 0:
        return None
    x_tr = x_tr.reshape(-1, lb, 1)
    x_te = x_te.reshape(-1, lb, 1)

    model = Sequential([LSTM(32, input_shape=(lb, 1)), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_s = model.predict(x_te, verbose=0).ravel()
    pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    return pred


def _predict_hybrid_residual_lstm(
    y_train: pd.Series,
    y_test: pd.Series,
    pred_sarimax_test: np.ndarray,
    insample_train: np.ndarray,
    look_back: int,
    epochs: int,
    batch_size: int,
) -> Optional[np.ndarray]:
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.layers import Dense, LSTM
        from tensorflow.keras.models import Sequential
    except Exception:
        return None

    res_train = np.asarray(y_train.values, dtype=float).ravel() - np.asarray(
        insample_train, dtype=float
    ).ravel()
    if len(res_train) != len(y_train):
        return None

    scaler_res = MinMaxScaler((0, 1))
    res_scaled = scaler_res.fit_transform(res_train.reshape(-1, 1)).ravel()
    rx, ry = _lstm_sequences(res_scaled, look_back)
    if len(rx) < 5:
        return None
    rx = rx.reshape(-1, look_back, 1)

    lstm_res = Sequential([LSTM(32, input_shape=(look_back, 1)), Dense(1)])
    lstm_res.compile(optimizer="adam", loss="mse")
    lstm_res.fit(rx, ry, epochs=epochs, batch_size=batch_size, verbose=0)

    window = list(res_scaled[-look_back:])
    residual_preds_scaled = []
    for _ in range(len(y_test)):
        x_in = np.asarray(window, dtype=float).reshape(1, look_back, 1)
        r_next = float(lstm_res.predict(x_in, verbose=0).ravel()[0])
        residual_preds_scaled.append(r_next)
        window = window[1:] + [r_next]

    residual_hat = scaler_res.inverse_transform(np.asarray(residual_preds_scaled).reshape(-1, 1)).ravel()
    return pred_sarimax_test + residual_hat


def run_multi_model_comparison(test_ratio: float = 0.2) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Returns:
    - wide DataFrame: date, actual, pred_sarimax, ci_low, ci_high, pred_rf, pred_xgb,
      pred_lstm_uni, pred_hybrid (NaN if model skipped / failed)
    - metrics: model name -> {RMSE, MAE, MAPE}
    """
    df = load_fa_data()
    daily = aggregate_to_daily(df)
    if "PM2.5" not in daily.columns:
        raise ValueError("PM2.5 missing after daily aggregation")
    daily = daily.dropna(subset=["PM2.5"])
    for c in FACTOR_COLS:
        if c in daily.columns:
            daily[c] = daily[c].ffill().bfill()

    n = len(daily)
    split_idx = int(n * (1 - test_ratio))
    y_train, y_test, exog_train, exog_test = train_test_split(daily, test_ratio)
    dates_test = pd.DatetimeIndex(y_test.index)
    actual = np.asarray(y_test.values, dtype=float)

    pred_sarimax = np.full(len(y_test), np.nan)
    ci_low = np.full(len(y_test), np.nan)
    ci_high = np.full(len(y_test), np.nan)
    insample_train: Optional[np.ndarray] = None

    import joblib

    model_path = _project_root() / "data" / "processed" / "sarima_model.joblib"
    try:
        bundle = joblib.load(model_path)
        model = bundle["model"]
        X_te = exog_test.values if exog_test is not None else None
        fc = model.predict(n_periods=len(y_test), X=X_te, return_conf_int=True, alpha=0.05)
        if isinstance(fc, tuple) and len(fc) == 2:
            pred_sarimax = np.asarray(fc[0], dtype=float).ravel()
            conf = np.asarray(fc[1])
            if conf.ndim == 2 and conf.shape[1] >= 2:
                ci_low = conf[:, 0]
                ci_high = conf[:, 1]
        else:
            pred_sarimax = np.asarray(fc, dtype=float).ravel()
        X_tr = exog_train.values if exog_train is not None else None
        try:
            ins = model.predict_in_sample(X=X_tr)
            ins = np.asarray(ins, dtype=float).ravel()
        except Exception:
            ins = np.asarray(model.arima_res_.fittedvalues, dtype=float).ravel()
        if len(ins) != len(y_train):
            ins = ins[-len(y_train) :]
        insample_train = ins
    except Exception:
        last = float(y_train.iloc[-1])
        pred_sarimax = np.full(len(y_test), last)
        ci_low = pred_sarimax - 20
        ci_high = pred_sarimax + 20

    pm_all = daily["PM2.5"]
    fac_all = (
        daily[FACTOR_COLS]
        if all(c in daily.columns for c in FACTOR_COLS)
        else pd.DataFrame(index=daily.index)
    )

    pred_rf = np.full(len(y_test), np.nan)
    pred_xgb = np.full(len(y_test), np.nan)
    try:
        X_tab, y_tab, tr_m, te_m = build_lag_tabular(pm_all, fac_all, LAG_TABULAR, split_idx)
        rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
        )
        rf.fit(X_tab[tr_m], y_tab[tr_m])
        pred_rf_all = rf.predict(X_tab[te_m])
        pad = max(0, LAG_TABULAR - split_idx)
        if pad > 0:
            pred_rf_all = np.concatenate([np.full(pad, np.nan), pred_rf_all])
        pred_rf = pred_rf_all[: len(y_test)]

        from xgboost import XGBRegressor

        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(X_tab[tr_m], y_tab[tr_m])
        pred_xgb_all = xgb.predict(X_tab[te_m])
        if pad > 0:
            pred_xgb_all = np.concatenate([np.full(pad, np.nan), pred_xgb_all])
        pred_xgb = pred_xgb_all[: len(y_test)]
    except Exception:
        pass

    pred_lstm_uni = np.full(len(y_test), np.nan)
    z = _predict_lstm_univariate_aligned(
        pm_all, split_idx, LSTM_LOOKBACK, LSTM_EPOCHS, LSTM_BATCH
    )
    if z is not None and len(z) >= len(y_test):
        pred_lstm_uni = z[: len(y_test)]
    elif z is not None:
        pred_lstm_uni[: len(z)] = z

    pred_hybrid = np.full(len(y_test), np.nan)
    if insample_train is not None:
        h = _predict_hybrid_residual_lstm(
            y_train,
            y_test,
            pred_sarimax,
            insample_train,
            LSTM_LOOKBACK,
            LSTM_EPOCHS,
            LSTM_BATCH,
        )
        if h is not None:
            pred_hybrid = h

    out = pd.DataFrame(
        {
            "date": dates_test,
            "actual": actual,
            "pred_sarimax": pred_sarimax,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "pred_rf": pred_rf,
            "pred_xgb": pred_xgb,
            "pred_lstm_uni": pred_lstm_uni,
            "pred_hybrid": pred_hybrid,
        }
    )

    metrics: dict[str, dict[str, float]] = {}
    named = {
        "SARIMAX": pred_sarimax,
        "RandomForest (lag+FA)": pred_rf,
        "XGBoost (lag+FA)": pred_xgb,
        "LSTM (univariate)": pred_lstm_uni,
        "Hybrid SARIMAX+LSTM(residual)": pred_hybrid,
    }
    for name, pred in named.items():
        metrics[name] = compute_metrics(actual, pred)

    return out, metrics
