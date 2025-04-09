import requests
import time

def fetch_bitcoin_prices():
    # Example API endpoint (Binance API for Bitcoin/USDT pair)
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",  # 1-minute interval
        "limit": 1         # Fetch the latest data point
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract relevant information
        for candle in data:
            timestamp = int(candle[0])
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])

            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(timestamp / 1000))}")
            print(f"Open: {open_price}, High: {high_price}, Low: {low_price}, Close: {close_price}, Volume: {volume}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    fetch_bitcoin_prices()