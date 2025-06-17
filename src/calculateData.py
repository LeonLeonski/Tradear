import xml.etree.ElementTree as ET
import pandas as pd
import json
import scipy.stats as stats
import sys
import os
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("tradear.log", encoding="utf-8"),
            logging.StreamHandler()
        ]
    )

def load_xml_data(filepath):
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Die Datei '{filepath}' wurde nicht gefunden.")
        tree = ET.parse(filepath)
        root = tree.getroot()
        return root
    except FileNotFoundError as fnf_err:
        logging.error(f"Fehler: {fnf_err}")
        sys.exit(1)
    except ET.ParseError as parse_err:
        logging.error(f"Fehler beim Parsen der XML-Datei: {parse_err}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unerwarteter Fehler beim Einlesen der XML-Datei: {e}")
        sys.exit(1)

def xml_to_dataframe(root):
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
            logging.warning(f"Fehler beim Verarbeiten eines Eintrags: {e}")
            continue
    if not data:
        logging.error("Keine gültigen Daten in der XML-Datei gefunden.")
        sys.exit(1)
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['timestamp'])
    df.set_index('Date', inplace=True)
    return df

def calculate_basic_metrics(df):
    try:
        df['Range'] = df['high'] - df['low']
        df['Change'] = df['close'].diff()
        df['Pct_Change'] = df['close'].pct_change() * 100
    except Exception as e:
        logging.warning(f"Fehler bei calculate_basic_metrics: {e}")
    return df

def calculate_sma(df, period):
    try:
        return df['close'].rolling(window=period).mean()
    except Exception as e:
        logging.warning(f"Fehler bei calculate_sma: {e}")
        return pd.Series([None]*len(df))

def identify_doji(df, body_threshold=0.1, range_threshold=0):
    try:
        df['Doji'] = ((abs(df['close'] - df['open']) / (df['high'] - df['low'])) < body_threshold) & ((df['high'] - df['low']) > range_threshold)
    except Exception as e:
        logging.warning(f"Fehler bei identify_doji: {e}")
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
        logging.warning(f"Fehler bei calculate_atr: {e}")
    return df

def calculate_golden_cross_signal(df, short_period=20, long_period=50):
    try:
        df['SMA_20'] = calculate_sma(df, short_period)
        df['SMA_50'] = calculate_sma(df, long_period)
        df['Golden_Cross'] = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
    except Exception as e:
        logging.warning(f"Fehler bei calculate_golden_cross_signal: {e}")
        df['Golden_Cross'] = False
    return df

def calculate_volume_change(df):
    try:
        df['Volume_Change'] = df['volume'].diff()
    except Exception as e:
        logging.warning(f"Fehler bei calculate_volume_change: {e}")
    return df

def calculate_avg_volume(df, period):
    try:
        df['Avg_Volume'] = df['volume'].rolling(window=period).mean()
    except Exception as e:
        logging.warning(f"Fehler bei calculate_avg_volume: {e}")
    return df

def calculate_correlation_and_pvalue(df):
    try:
        df_clean = df.dropna(subset=['close', 'volume'])
        if len(df_clean) > 1:
            correlation, p_value = stats.pearsonr(df_clean['close'], df_clean['volume'])
            logging.info(f"Pearson-Korrelation (Close vs. Volume): {correlation:.4f}")
            logging.info(f"P-Wert: {p_value:.4f}")
            return correlation, p_value
        else:
            logging.info("Nicht genug Datenpunkte für die Korrelationsberechnung.")
            return None, None
    except Exception as e:
        logging.warning(f"Fehler bei calculate_correlation_and_pvalue: {e}")
        return None, None

def fill_missing_with_string(df, fill_value="Unzureichende Daten"):
    try:
        for col in df.columns:
            df[col] = df[col].fillna(fill_value)
            df[col] = df[[col]].infer_objects(copy=False)[col]
    except Exception as e:
        logging.warning(f"Fehler beim Ersetzen von NaN-Werten: {e}")
    return df

def save_to_json(df, columns_to_save, filename):
    try:
        result_df = df.reset_index()
        result_df['timestamp'] = result_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        result_data = result_df[columns_to_save]
        with open(filename, 'w') as outfile:
            json.dump(result_data.to_dict(orient='records'), outfile, indent=2)
        logging.info(f"Berechnungen abgeschlossen und in '{filename}' gespeichert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Ergebnisse in die JSON-Datei: {e}")

def main():
    setup_logging()
    root = load_xml_data('./data/bc_data.xml')
    df = xml_to_dataframe(root)
    df = calculate_basic_metrics(df)
    df = identify_doji(df)
    df = calculate_atr(df)
    df = calculate_golden_cross_signal(df)
    df = calculate_volume_change(df)
    df = calculate_avg_volume(df, 20)
    calculate_correlation_and_pvalue(df)
    df = fill_missing_with_string(df)
    columns_to_save = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'SMA_20', 'SMA_50', 'ATR', 'Doji', 'Golden_Cross', 'Volume_Change', 'Avg_Volume']
    save_to_json(df, columns_to_save, './data/calculated_bitcoin_data.json')

if __name__ == "__main__":
    main()
