import sys
import os
import pytest
import pandas as pd
import numpy as np

from src.getMultiplePredictions import (
    prepare_features,
    train_model,
    predict_future,
    optimize_sl_tp,
    mark_best_trade
)

@pytest.fixture
def sample_df():
    # Erstelle einen kleinen DataFrame mit allen nötigen Spalten
    data = {
        'timestamp': ['2025-06-17 12:00:00', '2025-06-17 12:01:00', '2025-06-17 12:02:00', '2025-06-17 12:03:00'],
        'open': [100, 101, 102, 103],
        'high': [101, 102, 103, 104],
        'low': [99, 100, 101, 102],
        'close': [100.5, 101.5, 102.5, 103.5],
        'volume': [10, 12, 11, 13],
        'SMA_20': [100, 100.5, 101, 101.5],
        'SMA_50': [99, 99.5, 100, 100.5],
        'ATR': [1, 1, 1, 1],
        'Doji': [0, 1, 0, 1],
        'Golden_Cross': [1, 0, 1, 0],
        'Volume_Change': [0, 2, -1, 2],
        'Avg_Volume': [10, 11, 11, 12]
    }
    return pd.DataFrame(data)

def test_prepare_features(sample_df):
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']
    df = prepare_features(sample_df.copy(), feature_cols)
    assert 'target' in df.columns
    assert df['Doji'].isin([0, 1]).all()
    assert df['Golden_Cross'].isin([0, 1]).all()
    assert not df['target'].isnull().any()

def test_train_model_and_predict_future(sample_df):
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']
    df = prepare_features(sample_df.copy(), feature_cols)
    X = df[feature_cols]
    y = df['target']
    model = train_model(X, y)
    future_predictions, future_timestamps = predict_future(model, X, df, steps_ahead=2)
    assert len(future_predictions) == 2
    assert len(future_timestamps) == 2
    assert all(isinstance(ts, str) for ts in future_timestamps)

def test_optimize_sl_tp(sample_df):
    tp, sl, avg_atr = optimize_sl_tp(sample_df.copy())
    assert tp in [0.5, 1.0, 1.5, 2.0]
    assert sl in [0.5, 1.0, 1.5, 2.0]
    assert avg_atr > 0

def test_mark_best_trade(sample_df):
    # Simuliere future DataFrame und predictions
    future = pd.DataFrame({
        'timestamp': ['2025-06-17 12:04:00', '2025-06-17 12:05:00'],
        'predicted_close': [104, 105],
        'trade_direction': [None, None],
        'take_profit': [None, None],
        'stop_loss': [None, None],
        'top_trade': [False, False]
    })
    future_predictions = [0.01, -0.01]
    avg_atr = 1
    TP_MULTIPLIER = 1.0
    SL_MULTIPLIER = 1.0
    result = mark_best_trade(future.copy(), future_predictions, avg_atr, TP_MULTIPLIER, SL_MULTIPLIER)
    assert 'trade_direction' in result.columns
    assert 'top_trade' in result.columns
    assert result['top_trade'].isin([True, False]).all()