import xml.etree.ElementTree as ET
import pandas as pd
import json

# 1. XML-Datei einlesen und in eine Liste von Dicts umwandeln
tree = ET.parse('bc_data.xml')
root = tree.getroot()

data = []
for entry in root.findall('Entry'):
    timestamp = entry.find('Timestamp').text
    open_price = float(entry.find('Open').text)
    high_price = float(entry.find('High').text)
    low_price = float(entry.find('Low').text)
    close_price = float(entry.find('Close').text)
    volume = float(entry.find('Volume').text)
    data.append({
        'timestamp': timestamp,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close_price,
        'volume': volume
    })

# 2. In DataFrame umwandeln und Datum als Index setzen
df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['timestamp'])
df.set_index('Date', inplace=True)

# 3. Berechnungen (aus calculatePrice.py übernommen)

def calculate_basic_metrics(df):
    df['Range'] = df['high'] - df['low']
    df['Change'] = df['close'].diff()
    df['Pct_Change'] = df['close'].pct_change() * 100
    return df

def calculate_sma(df, period):
    return df['close'].rolling(window=period).mean()

def identify_doji(df, body_threshold=0.1, range_threshold=0):
    df['Doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < body_threshold) & ((df['high'] - df['low']) > range_threshold)
    return df

def calculate_atr(df, period=14):
    df['HL'] = df['high'] - df['low']
    df['HCp'] = abs(df['high'] - df['close'].shift(1))
    df['LCp'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['HL', 'HCp', 'LCp']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    return df

def calculate_golden_cross_signal(df, short_period=20, long_period=50):
    df['SMA_20'] = calculate_sma(df, short_period)
    df['SMA_50'] = calculate_sma(df, long_period)
    df['Golden_Cross'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    return df

def calculate_volume_change(df):
    df['Volume_Change'] = df['volume'].diff()
    return df

def calculate_avg_volume(df, period):
    df['Avg_Volume'] = df['volume'].rolling(window=period).mean()
    return df

# 4. Alle Berechnungen ausführen
df = calculate_basic_metrics(df)
df = identify_doji(df)
df = calculate_atr(df)
df = calculate_golden_cross_signal(df)
df = calculate_volume_change(df)
df = calculate_avg_volume(df, 20)

# 5. Ersetzen von NaN-Werten durch "Unzureichende Daten"
df.fillna("Unzureichende Daten", inplace=True)

# 6. Für JSON: Index zurücksetzen und Datum formatieren
result_df = df.reset_index()
result_df['timestamp'] = result_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')

# Wähle nur relevante Spalten für die Ausgabe
columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']

result_data = result_df[columns_to_save]

# 7. In JSON speichern
with open('calculated_bitcoin_data.json', 'w') as outfile:
    json.dump(result_data.to_dict(orient='records'), outfile, indent=2)

print("Berechnungen abgeschlossen und in 'calculated_bitcoin_data.json' gespeichert.")
