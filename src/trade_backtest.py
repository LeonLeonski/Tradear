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
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "trades" in data:
                    return data["trades"]
                if isinstance(data, list):
                    return data
                return []
            except Exception:
                return []
    else:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump({"trades": [], "portfolio": {}}, f, indent=2, ensure_ascii=False)
        return []
    return []

def clean_for_json(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    return obj

def save_backtests(backtests, portfolio, output_json_path):
    for trade in backtests:
        if "timestamp" in trade and not isinstance(trade["timestamp"], str):
            trade["timestamp"] = str(trade["timestamp"])
    out = {
        "trades": clean_for_json(backtests),
        "portfolio": portfolio
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

def trade_already_tracked(existing, trade):
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
    entry_idx = candles.index[candles['timestamp'] == trade['timestamp']]
    if len(entry_idx) == 0:
        trade["trade_result"] = "not_found"
        trade["status"] = "closed"
        trade["exit_price"] = None
        trade["pnl"] = 0.0
        return trade
    entry_idx = entry_idx[0]

    direction = trade['trade_direction']
    tp = trade['take_profit']
    sl = trade['stop_loss']
    trade_result = trade.get("trade_result", "open")
    status = "open"
    exit_price = None
    for i in range(1, lookahead + 1):
        if entry_idx + i >= len(candles):
            break
        high = candles.iloc[entry_idx + i]['high']
        low = candles.iloc[entry_idx + i]['low']
        if direction == "long":
            if high >= tp:
                trade_result = "win"
                status = "closed"
                exit_price = tp
                break
            if low <= sl:
                trade_result = "loss"
                status = "closed"
                exit_price = sl
                break
        elif direction == "short":
            if low <= tp:
                trade_result = "win"
                status = "closed"
                exit_price = tp
                break
            if high >= sl:
                trade_result = "loss"
                status = "closed"
                exit_price = sl
                break
    trade["trade_result"] = trade_result
    trade["status"] = status
    trade["exit_price"] = exit_price
    trade["pnl"] = calculate_pnl(trade)
    return trade

def calculate_pnl(trade):
    """
    Berechnet den Profit & Loss für einen abgeschlossenen Trade.
    Für SHORT: PnL = (Entry - Exit) / Entry * Position_Size
    Für LONG:  PnL = (Exit - Entry) / Entry * Position_Size
    """
    if trade.get('status') != 'closed':
        return 0.0
    entry = trade.get('entry_price')
    exit = trade.get('exit_price')
    size = trade.get('position_size', 0)
    if entry is None or exit is None or size == 0:
        return 0.0
    if trade.get('trade_direction') == 'short':
        return (entry - exit) / entry * size
    else:
        return (exit - entry) / entry * size

def update_portfolio(backtests, initial_balance=0):
    balance = initial_balance
    closed_trades = 0
    win = 0
    loss = 0
    for trade in backtests:
        if trade.get("status") == "closed":
            if "pnl" not in trade:
                trade["pnl"] = calculate_pnl(trade)
            balance += trade["pnl"]
            closed_trades += 1
            if trade["trade_result"] == "win":
                win += 1
            elif trade["trade_result"] == "loss":
                loss += 1
    return {
        "balance": balance,
        "closed_trades": closed_trades,
        "win": win,
        "loss": loss,
        "winrate": (win / closed_trades * 100) if closed_trades else 0
    }

def backtest_top_trade(trades_json_path, candles_json_path, output_json_path, lookahead=10):
    with open(trades_json_path, "r", encoding="utf-8") as f:
        trades = pd.read_json(f)
    with open(candles_json_path, "r", encoding="utf-8") as f:
        candles = pd.read_json(f)
    if "Date" in candles.columns:
        candles["timestamp"] = candles["Date"]

    existing_backtests = load_existing_backtests(output_json_path)

    # Finde aktuellen Top-Trade
    top_trades = trades[trades.get("top_trade", False) == True]
    if not top_trades.empty:
        top_trade = top_trades.iloc[0].to_dict()
        if not trade_already_tracked(existing_backtests, top_trade):
            top_trade["trade_result"] = "open"
            top_trade["status"] = "open"
            top_trade["position_size"] = 1000  # Beispiel: 1000 USD pro Trade
            # Hole den Einstiegspreis (close) aus den Candles zum Trade-Zeitpunkt
            entry_row = candles[candles['timestamp'] == top_trade['timestamp']]
            if not entry_row.empty:
                top_trade["entry_price"] = float(entry_row.iloc[0]['close'])
            else:
                top_trade["entry_price"] = None
            existing_backtests.append(top_trade)

    # Aktualisiere alle bisherigen Top-Trades
    updated_backtests = []
    for trade in existing_backtests:
        updated_trade = update_trade_result(trade, candles, lookahead=lookahead)
        updated_backtests.append(updated_trade)

    # Portfolio berechnen
    portfolio = update_portfolio(updated_backtests, initial_balance=0)

    # Speichern
    save_backtests(updated_backtests, portfolio, output_json_path)
    logging.info(f"Paper-Trading-Backtest und Portfolio gespeichert in {output_json_path}")

if __name__ == "__main__":
    setup_logging()
    backtest_top_trade(
        trades_json_path="./data/combined_predictions.json",
        candles_json_path="./data/calculated_bitcoin_data.json",
        output_json_path="./data/backtest_results.json",
        lookahead=10
    )