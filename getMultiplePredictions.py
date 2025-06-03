import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json

# Daten laden und vorbereiten
with open('calculated_bitcoin_data.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

# "Unzureichende Daten" durch np.nan ersetzen
df.replace("Unzureichende Daten", np.nan, inplace=True)

feature_cols = ['open', 'high', 'low', 'close', 'volume',
                'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']

# Entferne Zeilen mit fehlenden Werten in Features
df = df.dropna(subset=feature_cols)

# Label erzeugen: prozentuale Kursänderung für nächste Periode
df['close_next'] = df['close'].shift(-1)
df['target'] = (df['close_next'] - df['close']) / df['close']

# Letzte Zeile hat kein Label, deshalb entfernen
df = df.dropna(subset=['target'])

# Konvertiere boolsche Spalten zu int
df['Doji'] = df['Doji'].astype(int)
df['Golden_Cross'] = df['Golden_Cross'].astype(int)

X = df[feature_cols]
y = df['target']

# Modell trainieren auf den gesamten Daten
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Mehrstufige Vorhersage iterativ für die nächsten 10 Zeitpunkte
steps_ahead = 10

# Letzten Zeitstempel auf volle Minute abrunden (wichtig)
last_timestamp = pd.to_datetime(df.iloc[-1]['timestamp']).floor('min')

# Start mit letztem bekannten Feature-Set
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
    new_features['volume'] = 0  # Platzhalter

    last_features = new_features

# Historische Daten vorbereiten
historical = df[['timestamp', 'close']].copy()

# Zukunftsdaten vorbereiten (korrekte Länge!)
future = pd.DataFrame({
    'timestamp': future_timestamps,
    'predicted_close': [np.nan] * len(future_timestamps)
})

for i, p in enumerate(future_predictions):
    if i == 0:
        future.loc[i, 'predicted_close'] = historical['close'].iloc[-1] * (1 + p)
    else:
        future.loc[i, 'predicted_close'] = future.loc[i-1, 'predicted_close'] * (1 + p)

# ATR-basiertes SL/TP-Backtesting auf den letzten 1440 Minuten

# Bereinige ATR
df["ATR"] = pd.to_numeric(df["ATR"], errors="coerce")

# Letzte 1440 Minuten (24h)
recent = df.dropna(subset=["open", "high", "low", "close"]).copy()
recent["timestamp"] = pd.to_datetime(recent["timestamp"])
recent = recent.sort_values("timestamp").iloc[-1440:]

# Berechne durchschnittlichen True Range als Fallback falls ATR fehlt
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

# Optimal ermittelte Werte verwenden
TP_MULTIPLIER = best_result["TP_MULTIPLIER"]
SL_MULTIPLIER = best_result["SL_MULTIPLIER"]

print(f"Optimale TP/SL-Multiplikatoren gewählt: TP={TP_MULTIPLIER}, SL={SL_MULTIPLIER}, Win-Rate={best_result['Win_Rate']:.2%}")


# Take-Profit und Stop-Loss für jede Vorhersage berechnen
future['take_profit'] = future['predicted_close'] + TP_MULTIPLIER * avg_atr
future['stop_loss'] = future['predicted_close'] - SL_MULTIPLIER * avg_atr

# Kombiniere historische und prognostizierte Daten (inkl. Take-Profit/Stop-Loss)
combined = []

for i, row in historical.iterrows():
    combined.append({
        'timestamp': row['timestamp'],
        'actual_close': row['close'],
        'predicted_close': None,
        'take_profit': None,
        'stop_loss': None
    })

for i, row in future.iterrows():
    combined.append({
        'timestamp': row['timestamp'],
        'actual_close': None,
        'predicted_close': row['predicted_close'],
        'take_profit': row['take_profit'],
        'stop_loss': row['stop_loss']
    })

# Speichere die kombinierten Daten in eine JSON-Datei
with open('combined_predictions.json', 'w') as f:
    json.dump(combined, f, indent=2)

print("Minütliche, mehrstufige Vorhersagen mit Take-Profit/Stop-Loss in 'combined_predictions.json' gespeichert.")
