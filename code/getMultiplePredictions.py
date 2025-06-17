import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json

# Daten laden und vorbereiten
with open('./data/calculated_bitcoin_data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df.replace("Unzureichende Daten", np.nan, inplace=True)

feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']
df = df.dropna(subset=feature_cols)
df['close_next'] = df['close'].shift(-1)
df['target'] = (df['close_next'] - df['close']) / df['close']
df = df.dropna(subset=['target'])
df['Doji'] = df['Doji'].astype(int)
df['Golden_Cross'] = df['Golden_Cross'].astype(int)

X = df[feature_cols]
y = df['target']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

steps_ahead = 10
last_timestamp = pd.to_datetime(df.iloc[-1]['timestamp']).floor('min')
last_features = X.iloc[-1].copy()
future_predictions = []
future_timestamps = []

for step in range(steps_ahead):
    next_timestamp = last_timestamp + pd.Timedelta(minutes=step + 1)
    future_timestamps.append(next_timestamp.strftime('%Y-%m-%d %H:%M:%S'))

    pred_change = model.predict([last_features])[0]
    future_predictions.append(pred_change)

    last_close = last_features['close']
    predicted_close = last_close * (1 + pred_change)

    new_features = last_features.copy()
    new_features['open'] = last_close
    new_features['close'] = predicted_close
    new_features['high'] = max(new_features['open'], predicted_close) * 1.001
    new_features['low'] = min(new_features['open'], predicted_close) * 0.999
    new_features['volume'] = 0

    last_features = new_features

historical = df[['timestamp', 'close']].copy()

future = pd.DataFrame({
    'timestamp': future_timestamps,
    'predicted_close': [np.nan] * len(future_timestamps),
    'trade_direction': [None] * len(future_timestamps),
    'take_profit': [None] * len(future_timestamps),
    'stop_loss': [None] * len(future_timestamps),
    'top_trade': [False] * len(future_timestamps)
})

for i, p in enumerate(future_predictions):
    if i == 0:
        future.loc[i, 'predicted_close'] = historical['close'].iloc[-1] * (1 + p)
    else:
        future.loc[i, 'predicted_close'] = future.loc[i-1, 'predicted_close'] * (1 + p)

# SL/TP-Optimierung (24h zurück)
recent = df.dropna(subset=["open", "high", "low", "close"]).copy()
recent["timestamp"] = pd.to_datetime(recent["timestamp"])
recent = recent.sort_values("timestamp").iloc[-1440:]
recent["TR"] = recent.apply(lambda row: max(row["high"] - row["low"],
                                             abs(row["high"] - row["close"]),
                                             abs(row["low"] - row["close"])), axis=1)
avg_atr = recent["TR"].mean()

tp_options = [0.5, 1.0, 1.5, 2.0]
sl_options = [0.5, 1.0, 1.5, 2.0]
best_result = {"Win_Rate": -1}

for tp in tp_options:
    for sl in sl_options:
        wins, losses = 0, 0
        for i in range(len(recent) - 10):
            entry = recent.iloc[i]
            tp_price = entry["close"] + tp * avg_atr
            sl_price = entry["close"] - sl * avg_atr
            for j in range(1, 11):
                next_candle = recent.iloc[i + j]
                if next_candle["high"] >= tp_price:
                    wins += 1
                    break
                elif next_candle["low"] <= sl_price:
                    losses += 1
                    break
        total = wins + losses
        win_rate = wins / total if total > 0 else 0
        if win_rate > best_result["Win_Rate"]:
            best_result = {
                "TP_MULTIPLIER": tp,
                "SL_MULTIPLIER": sl,
                "Win_Rate": win_rate
            }

TP_MULTIPLIER = best_result["TP_MULTIPLIER"]
SL_MULTIPLIER = best_result["SL_MULTIPLIER"]

# Nur den besten Trade markieren
lookahead = 10
best_idx = None
best_strength = -1

for i, p in enumerate(future_predictions):
    predicted_close = future.loc[i, 'predicted_close']
    threshold = 0.01 * avg_atr / predicted_close

    if abs(p) < threshold:
        continue

    direction = "long" if p > 0 else "short"
    tp = predicted_close + TP_MULTIPLIER * avg_atr if direction == "long" else predicted_close - TP_MULTIPLIER * avg_atr
    sl = predicted_close - SL_MULTIPLIER * avg_atr if direction == "long" else predicted_close + SL_MULTIPLIER * avg_atr

    will_hit = False
    for j in range(1, lookahead + 1):
        if i + j < len(future):
            test_price = future.loc[i + j, 'predicted_close']
            if direction == "long" and (test_price >= tp or test_price <= sl):
                will_hit = True
                break
            elif direction == "short" and (test_price <= tp or test_price >= sl):
                will_hit = True
                break

    if will_hit:
        strength = abs(tp - sl)
        if strength > best_strength:
            best_idx = i
            best_strength = strength
            best_values = {
                "direction": direction,
                "tp": tp,
                "sl": sl
            }

# Nur für den besten Trade setzen
if best_idx is not None:
    future.loc[best_idx, 'trade_direction'] = best_values["direction"]
    future.loc[best_idx, 'take_profit'] = best_values["tp"]
    future.loc[best_idx, 'stop_loss'] = best_values["sl"]
    future.loc[best_idx, 'top_trade'] = True

# Kombinieren & Speichern
combined = []
for i, row in historical.iterrows():
    combined.append({
        'timestamp': row['timestamp'],
        'actual_close': row['close'],
        'predicted_close': None,
        'take_profit': None,
        'stop_loss': None,
        'trade_direction': None,
        'top_trade': False
    })

for i, row in future.iterrows():
    combined.append({
        'timestamp': row['timestamp'],
        'actual_close': None,
        'predicted_close': row['predicted_close'],
        'take_profit': row['take_profit'],
        'stop_loss': row['stop_loss'],
        'trade_direction': row['trade_direction'],
        'top_trade': row['top_trade']
    })

with open('./data/combined_predictions.json', 'w') as f:
    json.dump(combined, f, indent=2)

print("Vorhersagen gespeichert. Top-Trade markiert.")
