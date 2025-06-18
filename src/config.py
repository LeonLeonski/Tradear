# Binance API
BINANCE_LIMIT = 1440  # Anzahl der Kerzen (z.B. 1440 für 24h bei 1m-Intervall)

# Random Forest Modell
RANDOM_FOREST_ESTIMATORS = 100  # Anzahl der Bäume im RandomForest

# Trade-Optimierung
TP_OPTIONS = [0.5, 1.0, 1.5, 2.0]  # Take-Profit-Multiplikatoren
SL_OPTIONS = [0.5, 1.0, 1.5, 2.0]  # Stop-Loss-Multiplikatoren

# Statistische Signifikanz
SIGNIFICANCE_LEVEL = 0.01  # Schwellenwert für P

# Aktive Features für ML-Modelle
FEATURE_COLS = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross',
    'Volume_Change', 'Avg_Volume',
    'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Width', 'Rolling_Volatility', 'Momentum'
]