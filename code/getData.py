import requests
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta, timezone
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

def fetch_binance_klines(symbol='BTCUSDT', interval='1m', limit=1440):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Fehler beim Abrufen der Binance-Daten: {e}")
        sys.exit(1)

def timestamp_to_str(ms):
    try:
        dt = datetime.fromtimestamp(ms / 1000, timezone.utc) + timedelta(hours=2)
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logging.warning(f"Fehler beim Umwandeln des Timestamps: {e}")
        return "Unbekannt"

def save_to_xml(data, filename='./data/bc_data.xml'):
    try:
        root = ET.Element("BitcoinData")

        for candle in data:
            try:
                open_time = candle[0]
                open_price = candle[1]
                high_price = candle[2]
                low_price = candle[3]
                close_price = candle[4]
                volume = candle[5]

                entry = ET.SubElement(root, "Entry")
                ET.SubElement(entry, "Timestamp").text = timestamp_to_str(open_time)
                ET.SubElement(entry, "Open").text = str(open_price)
                ET.SubElement(entry, "High").text = str(high_price)
                ET.SubElement(entry, "Low").text = str(low_price)
                ET.SubElement(entry, "Close").text = str(close_price)
                ET.SubElement(entry, "Volume").text = str(volume)
            except Exception as e:
                logging.warning(f"Fehler beim Verarbeiten eines Candles: {e}")
                continue

        xml_str = ET.tostring(root, encoding='utf-8')
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding='utf-8')

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'wb') as f:
            f.write(pretty_xml)

        logging.info(f"Datei '{filename}' mit {len(data)} Einträgen gespeichert.")
    except Exception as e:
        logging.error(f"Fehler beim Speichern der XML-Datei: {e}")
        sys.exit(1)

def main():
    setup_logging()
    try:
        logging.info("Starte Datenabruf von Binance...")
        klines = fetch_binance_klines(limit=1440)
        if not klines or not isinstance(klines, list):
            logging.error("Keine gültigen Daten von der Binance-API erhalten.")
            sys.exit(1)
        save_to_xml(klines)
    except Exception as e:
        logging.critical(f"Unerwarteter Fehler im Hauptprogramm: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
