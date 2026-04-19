"""
LSTM utilities for PM2.5 forecasting.

This module provides:
- sequence builder for univariate time series
- LSTM training on daily PM2.5
- test-set prediction and simple metrics
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.models import Sequential
except Exception:
    Sequential = None
    Dense = None
    LSTM = None


def has_tensorflow() -> bool:
    """Return True if TensorFlow/Keras is available."""
    return Sequential is not None and Dense is not None and LSTM is not None


def build_lstm_sequences(values: np.ndarray, look_back: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Convert 1D array into supervised learning sequences."""
    x_data, y_data = [], []
    for i in range(len(values) - look_back):
        x_data.append(values[i : i + look_back])
        y_data.append(values[i + look_back])
    return np.array(x_data), np.array(y_data)


def train_lstm_on_daily_pm25(
    pm25_series: pd.Series,
    look_back: int = 14,
    epochs: int = 50,
    batch_size: int = 16,
) -> Optional[dict]:
    """
    Train LSTM on daily PM2.5 and predict on test split (last 20%).

    Returns dict:
    - dates: DatetimeIndex of test targets
    - actual: inverse-scaled true values
    - predicted: inverse-scaled predictions
    - mse: mean squared error on test
    """
    if not has_tensorflow():
        return None
    if pm25_series is None or len(pm25_series) <= look_back + 10:
        return None

    values = pm25_series.values.reshape(-1, 1).astype(float)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values).flatten()

    x_all, y_all = build_lstm_sequences(scaled, look_back=look_back)
    if len(x_all) < 20:
        return None

    x_all = x_all.reshape((x_all.shape[0], x_all.shape[1], 1))
    split = int(len(x_all) * 0.8)
    x_train, y_train = x_all[:split], y_all[:split]
    x_test, y_test = x_all[split:], y_all[split:]

    model = Sequential(
        [
            LSTM(32, input_shape=(look_back, 1)),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    pred_scaled = model.predict(x_test, verbose=0).flatten()
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    dates = pm25_series.index[look_back + split :]
    mse = float(np.mean((y_true - pred) ** 2))
    return {"dates": dates, "actual": y_true, "predicted": pred, "mse": mse}
