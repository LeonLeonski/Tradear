import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import json
import sys
import logging
from src import config  # Importiere die Parameter aus config.py

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("tradear.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.replace("Unzureichende Daten", np.nan, inplace=True)
        return df
    except FileNotFoundError:
        logging.error(f"Fehler: '{filepath}' nicht gefunden.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Fehler beim Parsen der JSON-Datei: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unerwarteter Fehler beim Laden der Daten: {e}")
        sys.exit(1)

def prepare_features(df, feature_cols):
    try:
        df = df.dropna(subset=feature_cols)
        df['close_next'] = df['close'].shift(-1)
        df['target'] = (df['close_next'] - df['close']) / df['close']
        df = df.dropna(subset=['target'])
        df['Doji'] = df['Doji'].astype(int)
        df['Golden_Cross'] = df['Golden_Cross'].astype(int)
        return df
    except Exception as e:
        logging.error(f"Fehler bei der Feature-Vorbereitung: {e}")
        sys.exit(1)

def train_model(X, y):
    try:
        model = RandomForestRegressor(
            n_estimators=config.RANDOM_FOREST_ESTIMATORS, random_state=42
        )
        model.fit(X, y)
        return model
    except Exception as e:
        logging.error(f"Fehler beim Trainieren des Modells: {e}")
        sys.exit(1)

def predict_future(model, X, df, steps_ahead=10):
    try:
        last_timestamp = pd.to_datetime(df.iloc[-1]['timestamp']).floor('min')
        last_features = X.iloc[-1].copy()
        future_predictions = []
        future_timestamps = []
        for step in range(steps_ahead):
            next_timestamp = last_timestamp + pd.Timedelta(minutes=step + 1)
            future_timestamps.append(next_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            pred_change = model.predict(pd.DataFrame([last_features]))[0]
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
        return future_predictions, future_timestamps
    except Exception as e:
        logging.error(f"Fehler bei der Zukunftsprognose: {e}")
        sys.exit(1)

def optimize_sl_tp(df):
    try:
        recent = df.dropna(subset=["open", "high", "low", "close"]).copy()
        recent["timestamp"] = pd.to_datetime(recent["timestamp"])
        recent = recent.sort_values("timestamp").iloc[-1440:]
        recent["TR"] = recent.apply(lambda row: max(row["high"] - row["low"],
                                                    abs(row["high"] - row["close"]),
                                                    abs(row["low"] - row["close"])), axis=1)
        avg_atr = recent["TR"].mean()
        tp_options = config.TP_OPTIONS
        sl_options = config.SL_OPTIONS
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
        return best_result["TP_MULTIPLIER"], best_result["SL_MULTIPLIER"], avg_atr
    except Exception as e:
        logging.error(f"Fehler bei der SL/TP-Optimierung: {e}")
        sys.exit(1)

def mark_best_trade(future, future_predictions, avg_atr, TP_MULTIPLIER, SL_MULTIPLIER, lookahead=10):
    best_idx = None
    best_strength = -1
    best_values = {}
    try:
        for i, p in enumerate(future_predictions):
            predicted_close = future.loc[i, 'predicted_close']
            threshold = 0.01 * avg_atr / predicted_close if predicted_close else 0
            if abs(p) < threshold:
                continue
            direction = "long" if p > 0 else "short"
            tp = predicted_close + TP_MULTIPLIER * avg_atr if direction == "long" else predicted_close - TP_MULTIPLIER * avg_atr
            sl = predicted_close - SL_MULTIPLIER * avg_atr if direction == "long" else predicted_close + SL_MULTIPLIER * avg_atr
            will_hit = False
            # Wir starten bei j=2, um Lookahead Bias zu vermeiden:
            # TP/SL dürfen nicht in der ersten Minute nach Einstieg getroffen werden.
            for j in range(2, lookahead + 1):
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
        if best_idx is not None:
            future.loc[best_idx, 'trade_direction'] = best_values["direction"]
            future.loc[best_idx, 'take_profit'] = best_values["tp"]
            future.loc[best_idx, 'stop_loss'] = best_values["sl"]
            future.loc[best_idx, 'top_trade'] = True
    except Exception as e:
        logging.error(f"Fehler beim Markieren des besten Trades: {e}")
        sys.exit(1)
    return future

def save_predictions(historical, future, filename):
    combined = []
    try:
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
        with open(filename, 'w') as f:
            json.dump(combined, f, indent=2)
        logging.info("Vorhersagen gespeichert. Top-Trade markiert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der kombinierten Ergebnisse: {e}")
        sys.exit(1)

def main():
    setup_logging()
    pd.set_option('future.no_silent_downcasting', True)
    feature_cols = ['open', 'high', 'low', 'close', 'volume',
                    'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']
    df = load_data('./data/calculated_bitcoin_data.json')
    df = prepare_features(df, feature_cols)
    X = df[feature_cols]
    y = df['target']
    model = train_model(X, y)
    # Steps ahead kann als Parameter übergeben werden, z.B. config.BINANCE_LIMIT oder separat
    future_predictions, future_timestamps = predict_future(model, X, df, steps_ahead=10)
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
    TP_MULTIPLIER, SL_MULTIPLIER, avg_atr = optimize_sl_tp(df)
    future = mark_best_trade(future, future_predictions, avg_atr, TP_MULTIPLIER, SL_MULTIPLIER)
    save_predictions(historical, future, './data/combined_predictions.json')

if __name__ == "__main__":
    main()
