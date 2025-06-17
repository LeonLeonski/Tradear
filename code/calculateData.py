import xml.etree.ElementTree as ET
import pandas as pd
import json
import scipy.stats as stats
import sys
import os

# Fehlerbehandlung für das Einlesen der XML-Datei
try:
    if not os.path.exists('./data/bc_data.xml'):
        raise FileNotFoundError("Die Datei './data/bc_data.xml' wurde nicht gefunden.")
    tree = ET.parse('./data/bc_data.xml')
    root = tree.getroot()
except FileNotFoundError as fnf_err:
    print(f"Fehler: {fnf_err}")
    sys.exit(1)
except ET.ParseError as parse_err:
    print(f"Fehler beim Parsen der XML-Datei: {parse_err}")
    sys.exit(1)
except Exception as e:
    print(f"Unerwarteter Fehler beim Einlesen der XML-Datei: {e}")
    sys.exit(1)

data = []
for entry in root.findall('Entry'):
    try:
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
    except Exception as e:
        print(f"Fehler beim Verarbeiten eines Eintrags: {e}")
        continue

# Fehlerbehandlung für leere Daten
if not data:
    print("Keine gültigen Daten in der XML-Datei gefunden.")
    sys.exit(1)

# 2. In DataFrame umwandeln und Datum als Index setzen
try:
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['timestamp'])
    df.set_index('Date', inplace=True)
except Exception as e:
    print(f"Fehler beim Erstellen des DataFrames: {e}")
    sys.exit(1)

# 3. Berechnungen (aus calculatePrice.py übernommen)

def calculate_basic_metrics(df):
    try:
        df['Range'] = df['high'] - df['low']
        df['Change'] = df['close'].diff()
        df['Pct_Change'] = df['close'].pct_change() * 100
    except Exception as e:
        print(f"Fehler bei calculate_basic_metrics: {e}")
    return df

def calculate_sma(df, period):
    try:
        return df['close'].rolling(window=period).mean()
    except Exception as e:
        print(f"Fehler bei calculate_sma: {e}")
        return pd.Series([None]*len(df))

def identify_doji(df, body_threshold=0.1, range_threshold=0):
    try:
        df['Doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < body_threshold) & ((df['high'] - df['low']) > range_threshold)
    except Exception as e:
        print(f"Fehler bei identify_doji: {e}")
        df['Doji'] = False
    return df

def calculate_atr(df, period=14):
    try:
        df['HL'] = df['high'] - df['low']
        df['HCp'] = abs(df['high'] - df['close'].shift(1))
        df['LCp'] = abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['HL', 'HCp', 'LCp']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=period).mean()
    except Exception as e:
        print(f"Fehler bei calculate_atr: {e}")
    return df

def calculate_golden_cross_signal(df, short_period=20, long_period=50):
    try:
        df['SMA_20'] = calculate_sma(df, short_period)
        df['SMA_50'] = calculate_sma(df, long_period)
        df['Golden_Cross'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    except Exception as e:
        print(f"Fehler bei calculate_golden_cross_signal: {e}")
        df['Golden_Cross'] = False
    return df

def calculate_volume_change(df):
    try:
        df['Volume_Change'] = df['volume'].diff()
    except Exception as e:
        print(f"Fehler bei calculate_volume_change: {e}")
    return df

def calculate_avg_volume(df, period):
    try:
        df['Avg_Volume'] = df['volume'].rolling(window=period).mean()
    except Exception as e:
        print(f"Fehler bei calculate_avg_volume: {e}")
    return df

def calculate_correlation_and_pvalue(df):
    """
    Berechnet die Pearson-Korrelation und den P-Wert zwischen Schlusskurs und Volumen.
    """
    try:
        df_clean = df.dropna(subset=['close', 'volume'])
        if len(df_clean) > 1:
            correlation, p_value = stats.pearsonr(df_clean['close'], df_clean['volume'])
            print(f"Pearson-Korrelation (Close vs. Volume): {correlation:.4f}")
            print(f"P-Wert: {p_value:.4f}")
            return correlation, p_value
        else:
            print("Nicht genug Datenpunkte für die Korrelationsberechnung.")
            return None, None
    except Exception as e:
        print(f"Fehler bei calculate_correlation_and_pvalue: {e}")
        return None, None

# 4. Alle Berechnungen ausführen
try:
    df = calculate_basic_metrics(df)
    df = identify_doji(df)
    df = calculate_atr(df)
    df = calculate_golden_cross_signal(df)
    df = calculate_volume_change(df)
    df = calculate_avg_volume(df, 20)
    correlation, p_value = calculate_correlation_and_pvalue(df)
except Exception as e:
    print(f"Fehler bei der Berechnung der Kennzahlen: {e}")
    sys.exit(1)

# 5. Ersetzen von NaN-Werten durch "Unzureichende Daten"
try:
    for col in df.columns:
        df[col] = df[col].fillna("Unzureichende Daten")
        df[col] = df[[col]].infer_objects(copy=False)[col]
except Exception as e:
    print(f"Fehler beim Ersetzen von NaN-Werten: {e}")

# 6. Für JSON: Index zurücksetzen und Datum formatieren
try:
    result_df = df.reset_index()
    result_df['timestamp'] = result_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
except Exception as e:
    print(f"Fehler beim Formatieren des DataFrames für JSON: {e}")
    sys.exit(1)

# Wähle nur relevante Spalten für die Ausgabe
columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                   'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']

try:
    result_data = result_df[columns_to_save]
except Exception as e:
    print(f"Fehler beim Auswählen der Spalten für die Ausgabe: {e}")
    sys.exit(1)

# 7. In JSON speichern
try:
    with open('./data/calculated_bitcoin_data.json', 'w') as outfile:
        json.dump(result_data.to_dict(orient='records'), outfile, indent=2)
    print("Berechnungen abgeschlossen und in 'calculated_bitcoin_data.json' gespeichert.")
except Exception as e:
    print(f"Fehler beim Speichern der Ergebnisse in die JSON-Datei: {e}")
