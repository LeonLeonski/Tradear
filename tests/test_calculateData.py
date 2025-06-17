import pytest
import pandas as pd
from src import calculateData

def test_calculate_basic_metrics():
    df = pd.DataFrame({
        'open': [1, 2],
        'high': [2, 3],
        'low': [0, 1],
        'close': [1.5, 2.5],
        'volume': [100, 200]
    })
    result = calculateData.calculate_basic_metrics(df.copy())
    assert 'Range' in result.columns
    assert result['Range'].iloc[0] == 2
    assert result['Range'].iloc[1] == 2
    assert 'Change' in result.columns
    assert 'Pct_Change' in result.columns

def test_calculate_sma():
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    sma = calculateData.calculate_sma(df, 3)
    assert len(sma) == 5
    assert sma.iloc[2] == pytest.approx((1+2+3)/3)

def test_identify_doji():
    df = pd.DataFrame({
        'open': [1, 2],
        'high': [2, 3],
        'low': [0, 1],
        'close': [1.05, 2.05]
    })
    result = calculateData.identify_doji(df.copy())
    assert 'Doji' in result.columns
    assert result['Doji'].dtype == bool

def test_calculate_atr():
    df = pd.DataFrame({
        'high': [2, 3, 4],
        'low': [1, 2, 3],
        'close': [1.5, 2.5, 3.5]
    })
    result = calculateData.calculate_atr(df.copy(), period=2)
    assert 'ATR' in result.columns

def test_calculate_golden_cross_signal():
    df = pd.DataFrame({'close': [1, 2, 3, 4, 5]})
    result = calculateData.calculate_golden_cross_signal(df.copy(), short_period=2, long_period=3)
    assert 'SMA_20' in result.columns
    assert 'SMA_50' in result.columns
    assert 'Golden_Cross' in result.columns

def test_calculate_volume_change():
    df = pd.DataFrame({'volume': [100, 200, 150]})
    result = calculateData.calculate_volume_change(df.copy())
    assert 'Volume_Change' in result.columns
    assert result['Volume_Change'].iloc[1] == 100

def test_calculate_avg_volume():
    df = pd.DataFrame({'volume': [10, 20, 30, 40, 50]})
    result = calculateData.calculate_avg_volume(df.copy(), period=3)
    assert 'Avg_Volume' in result.columns
    assert result['Avg_Volume'].iloc[2] == pytest.approx((10+20+30)/3)

def test_fill_missing_with_string():
    df = pd.DataFrame({'a': [1, None, 3], 'b': [None, 2, 3]})
    result = calculateData.fill_missing_with_string(df.copy(), fill_value="TEST")
    assert (result == "TEST").any().any() 