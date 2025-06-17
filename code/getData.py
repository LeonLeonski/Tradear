import requests
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta

def fetch_binance_klines(symbol='BTCUSDT', interval='1m', limit=1440):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def timestamp_to_str(ms):
    # ms Timestamp in format YYYY-MM-DD HH:MM:SS umwandeln
    dt = datetime.utcfromtimestamp(ms / 1000) + timedelta(hours=2)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

def save_to_xml(data, filename='./data/bc_data.xml'):
    root = ET.Element("BitcoinData")

    for candle in data:
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

    xml_str = ET.tostring(root, encoding='utf-8')
    pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ", encoding='utf-8')

    with open(filename, 'wb') as f:
        f.write(pretty_xml)

    print(f"Datei '{filename}' mit {len(data)} Einträgen gespeichert.")

def main():
    # 1440 Einträge = 24h bei 1-Minuten-Intervall
    klines = fetch_binance_klines(limit=1440)
    save_to_xml(klines)

if __name__ == "__main__":
    main()
