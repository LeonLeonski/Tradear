import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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

def prepare_features(df, feature_cols, steps_ahead=1):
    try:
        df = df.dropna(subset=feature_cols)
        df['close_next'] = df['close'].shift(-steps_ahead)
        df['target_class'] = (df['close_next'] > df['close']).astype(int)
        df = df.dropna(subset=['target_class'])
        df['Doji'] = df['Doji'].astype(int)
        df['Golden_Cross'] = df['Golden_Cross'].astype(int)
        return df
    except Exception as e:
        logging.error(f"Fehler bei der Feature-Vorbereitung: {e}")
        sys.exit(1)

def train_model(X, y):
    try:
        # Begrenze Modellkomplexität
        model = RandomForestClassifier(
            n_estimators=50,  # z.B. weniger Bäume für geringere Komplexität
            max_depth=5,      # maximale Tiefe begrenzen
            random_state=42
        )
        # Cross-Validation
        scores = cross_val_score(model, X, y, cv=5)
        print("Cross-Validation Score:", np.mean(scores))
        # Modell auf allen Daten trainieren
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
        future_probabilities = []
        future_timestamps = []
        X_future = []
        for step in range(steps_ahead):
            next_timestamp = last_timestamp + pd.Timedelta(minutes=step + 1)
            future_timestamps.append(next_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
            X_future.append(last_features.values)
            pred_proba = model.predict_proba(pd.DataFrame([last_features]))[0]
            pred_class = int(pred_proba[1] > 0.5)
            future_predictions.append(pred_class)
            future_probabilities.append(pred_proba[1])
            last_close = last_features['close']
            # Simuliere Kursentwicklung: Steigt/fällt um einen kleinen Prozentsatz (z.B. 0.1%)
            if pred_class == 1:
                predicted_close = last_close * 1.001
            else:
                predicted_close = last_close * 0.999
            new_features = last_features.copy()
            new_features['open'] = last_close
            new_features['close'] = predicted_close
            new_features['high'] = max(new_features['open'], predicted_close) * 1.001
            new_features['low'] = min(new_features['open'], predicted_close) * 0.999
            new_features['volume'] = 0
            last_features = new_features
        X_future = np.array(X_future)
        probas = model.predict_proba(X_future)
        confidences = probas[:, 1]  # Wahrscheinlichkeit für Klasse 'steigt'
        return future_predictions, future_probabilities, future_timestamps, confidences
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

def mark_best_trades(future, future_predictions, future_probabilities, avg_atr, TP_MULTIPLIER, SL_MULTIPLIER, df, lookahead=10):
    """
    Markiert nur Trades, die ein ausreichend starkes Signal haben,
    TP/SL nicht in der ersten Minute treffen und lässt zwischen Trades einen Mindestabstand.
    Zusätzlich: Trendfilter (nur Long bei SMA_20 > SMA_50, nur Short bei SMA_20 < SMA_50)
    und Mindest-CRV.
    Loggt abgelehnte Trades inkl. Confidence und CRV.
    """
    min_confidence = 0.75  # Mindestwahrscheinlichkeit für Trade
    min_tp_sl_distance = 0.2 * avg_atr  # Mindestabstand zwischen TP und SL
    i = 0
    while i < len(future_predictions):
        pred_class = future_predictions[i]
        confidence = future_probabilities[i]
        predicted_close = future.loc[i, 'predicted_close']

        # Trendfilter: Hole SMA_20 und SMA_50 aus den historischen Daten (letzter Wert)
        sma_20 = df['SMA_20'].iloc[-1]
        sma_50 = df['SMA_50'].iloc[-1]

        # Trendfilter anwenden
        if pred_class == 1:  # Long
            if not (sma_20 > sma_50 and confidence > min_confidence):
                crv = float('nan')
                logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv if not np.isnan(crv) else 'n/a'} (Long, Trendfilter/Confidence)")
                i += 1
                continue
        elif pred_class == 0:  # Short
            if not (sma_20 < sma_50 and confidence > min_confidence):
                crv = float('nan')
                logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv if not np.isnan(crv) else 'n/a'} (Short, Trendfilter/Confidence)")
                i += 1
                continue

        direction = "long" if pred_class == 1 else "short"
        tp = predicted_close + TP_MULTIPLIER * avg_atr if direction == "long" else predicted_close - TP_MULTIPLIER * avg_atr
        sl = predicted_close - SL_MULTIPLIER * avg_atr if direction == "long" else predicted_close + SL_MULTIPLIER * avg_atr

        # CRV-Bedingung
        crv = None
        if direction == "long":
            crv = (tp - predicted_close) / (predicted_close - sl) if (predicted_close - sl) != 0 else 0
            if crv < 1.5 or tp <= predicted_close:
                logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv:.2f} (Long, CRV)")
                i += 1
                continue
        elif direction == "short":
            crv = (predicted_close - tp) / (sl - predicted_close) if (sl - predicted_close) != 0 else 0
            if crv < 1.5 or tp >= predicted_close:
                logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv:.2f} (Short, CRV)")
                i += 1
                continue

        if abs(tp - sl) < min_tp_sl_distance:
            logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv:.2f} (TP/SL zu nah beieinander)")
            i += 1
            continue
        will_hit = False
        # Starte bei j=2, um Lookahead Bias zu vermeiden
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
            future.loc[i, 'trade_direction'] = direction
            future.loc[i, 'take_profit'] = tp
            future.loc[i, 'stop_loss'] = sl
            future.loc[i, 'top_trade'] = True
            i += lookahead  # Mindestabstand: nach Trade für lookahead Minuten keine neuen Trades
        else:
            logging.info(f"Trade abgelehnt: Confidence={confidence:.2f}, CRV={crv:.2f} (Kein TP/SL Hit im Lookahead)")
            i += 1
    return future

def save_predictions(historical, future, filename, tp_multiplier, sl_multiplier, avg_atr):
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
        # Metadaten anhängen
        out = {
            "predictions": combined,
            "tp_multiplier": tp_multiplier,
            "sl_multiplier": sl_multiplier,
            "avg_atr": avg_atr
        }
        with open(filename, 'w') as f:
            json.dump(out, f, indent=2)
        logging.info("Vorhersagen gespeichert. Top-Trades markiert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der kombinierten Ergebnisse: {e}")
        sys.exit(1)

def main():
    setup_logging()
    pd.set_option('future.no_silent_downcasting', True)
    feature_cols = config.FEATURE_COLS
    steps_ahead = 10
    df = load_data('./data/calculated_bitcoin_data.json')
    df = prepare_features(df, feature_cols, steps_ahead=steps_ahead)
    X = df[feature_cols]
    y = df['target_class']
    model = train_model(X, y)
    future_predictions, future_probabilities, future_timestamps, confidences = predict_future(model, X, df, steps_ahead=steps_ahead)
    historical = df[['timestamp', 'close']].copy()
    future = pd.DataFrame({
        'timestamp': future_timestamps,
        'predicted_close': [np.nan] * len(future_timestamps),
        'trade_direction': [None] * len(future_timestamps),
        'take_profit': [None] * len(future_timestamps),
        'stop_loss': [None] * len(future_timestamps),
        'top_trade': [False] * len(future_timestamps),
        'confidence': confidences
    })
    # Optional: Nur Trades mit hoher Confidence erlauben (z. B. > 0.75)
    # future = future[future['confidence'] > 0.75]

    # Simuliere die Kursentwicklung wie in predict_future
    last_close = historical['close'].iloc[-1]
    for i, pred_class in enumerate(future_predictions):
        if i == 0:
            if pred_class == 1:
                future.loc[i, 'predicted_close'] = last_close * 1.001
            else:
                future.loc[i, 'predicted_close'] = last_close * 0.999
        else:
            prev_close = future.loc[i-1, 'predicted_close']
            if pred_class == 1:
                future.loc[i, 'predicted_close'] = prev_close * 1.001
            else:
                future.loc[i, 'predicted_close'] = prev_close * 0.999
    TP_MULTIPLIER, SL_MULTIPLIER, avg_atr = optimize_sl_tp(df)
    future = mark_best_trades(future, future_predictions, future_probabilities, avg_atr, TP_MULTIPLIER, SL_MULTIPLIER, df)
    save_predictions(historical, future, './data/combined_predictions.json', TP_MULTIPLIER, SL_MULTIPLIER, avg_atr)

if __name__ == "__main__":
    main()
