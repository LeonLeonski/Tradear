import pandas as pd
import logging
import os
import json
import math

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("tradear.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def load_existing_backtests(output_json_path):
    """Lädt bestehende Backtest-Top-Trades, falls vorhanden."""
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception:
                return []
    else:
        # Datei existiert nicht, also erstelle sie leer
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        return []
    return []

def save_backtests(backtests, output_json_path):
    """Speichert alle Backtest-Top-Trades."""
    # Konvertiere alle Timestamps zu Strings
    for trade in backtests:
        if "timestamp" in trade and not isinstance(trade["timestamp"], str):
            trade["timestamp"] = str(trade["timestamp"])
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(clean_for_json(backtests), f, indent=2, ensure_ascii=False)

def trade_already_tracked(existing, trade):
    """Prüft, ob ein Top-Trade schon im Backtest-Log ist (z.B. anhand Timestamp und Richtung)."""
    for t in existing:
        if (
            t.get("timestamp") == trade.get("timestamp")
            and t.get("trade_direction") == trade.get("trade_direction")
            and t.get("take_profit") == trade.get("take_profit")
            and t.get("stop_loss") == trade.get("stop_loss")
        ):
            return True
    return False

def update_trade_result(trade, candles, lookahead=10):
    """Überprüft, ob der Trade beendet wurde und aktualisiert trade_result."""
    # Finde Index der Einstiegs-Kerze
    entry_idx = candles.index[candles['timestamp'] == trade['timestamp']]
    if len(entry_idx) == 0:
        trade["trade_result"] = "not_found"
        return trade
    entry_idx = entry_idx[0]

    direction = trade['trade_direction']
    tp = trade['take_profit']
    sl = trade['stop_loss']
    trade_result = trade.get("trade_result", "open")
    for i in range(1, lookahead + 1):
        if entry_idx + i >= len(candles):
            break
        high = candles.iloc[entry_idx + i]['high']
        low = candles.iloc[entry_idx + i]['low']
        if direction == "long":
            if high >= tp:
                trade_result = "win"
                break
            if low <= sl:
                trade_result = "loss"
                break
        elif direction == "short":
            if low <= tp:
                trade_result = "win"
                break
            if high >= sl:
                trade_result = "loss"
                break
    trade["trade_result"] = trade_result
    return trade

def backtest_top_trade(trades_json_path, candles_json_path, output_json_path, lookahead=10):
    """
    Hängt den aktuellen Top-Trade an die Backtest-Datei an (falls neu)
    und prüft für alle bisherigen Top-Trades, ob sie beendet wurden.
    """
    # Lade aktuelle Trades und Candles
    with open(trades_json_path, "r", encoding="utf-8") as f:
        trades = pd.read_json(f)
    with open(candles_json_path, "r", encoding="utf-8") as f:
        candles = pd.read_json(f)
    if "Date" in candles.columns:
        candles["timestamp"] = candles["Date"]

    # Lade bisherige Backtests (erstellt Datei falls nicht vorhanden)
    existing_backtests = load_existing_backtests(output_json_path)

    # Finde aktuellen Top-Trade
    top_trades = trades[trades.get("top_trade", False) == True]
    if not top_trades.empty:
        top_trade = top_trades.iloc[0].to_dict()
        # Prüfe, ob dieser Trade schon getrackt wird
        if not trade_already_tracked(existing_backtests, top_trade):
            # Setze trade_result initial auf "open"
            top_trade["trade_result"] = "open"
            existing_backtests.append(top_trade)

    # Aktualisiere alle bisherigen Top-Trades
    updated_backtests = []
    for trade in existing_backtests:
        updated_trade = update_trade_result(trade, candles, lookahead=lookahead)
        updated_backtests.append(updated_trade)

    # Speichern
    save_backtests(updated_backtests, output_json_path)
    logging.info(f"Backtest-Top-Trades aktualisiert und gespeichert in {output_json_path}")

def clean_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    return obj

if __name__ == "__main__":
    setup_logging()
    backtest_top_trade(
        trades_json_path="./data/combined_predictions.json",
        candles_json_path="./data/calculated_bitcoin_data.json",
        output_json_path="./data/backtest_results.json",
        lookahead=10
    )