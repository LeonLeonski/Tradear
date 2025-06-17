import pytest
from src import getData
import os

def test_timestamp_to_str_valid():
    # 1.1.2024 00:00:00 UTC + 2h = 01:00:00
    ms = 1704067200000  # entspricht 2024-01-01 00:00:00 UTC
    result = getData.timestamp_to_str(ms)
    assert result == "2024-01-01 02:00:00"

def test_timestamp_to_str_invalid():
    # Übergib einen ungültigen Wert
    result = getData.timestamp_to_str("not_a_timestamp")
    assert result == "Unbekannt"

def test_fetch_binance_klines(monkeypatch):
    # Simuliere eine erfolgreiche Antwort
    class DummyResponse:
        def raise_for_status(self): pass
        def json(self): return [{"dummy": "data"}]
    def dummy_get(*args, **kwargs): return DummyResponse()
    monkeypatch.setattr("requests.get", dummy_get)
    result = getData.fetch_binance_klines(limit=1)
    assert isinstance(result, list) or isinstance(result, dict)

def test_save_to_xml(tmp_path):
    # Schreibe eine kleine Testdatei
    test_data = [
        [1704067200000, 1, 2, 0.5, 1.5, 100]
    ]
    filename = tmp_path / "test.xml"
    getData.save_to_xml(test_data, str(filename))
    assert os.path.exists(filename)
    with open(filename, "rb") as f:
        content = f.read()
        assert b"<BitcoinData>" in content
        assert b"<Entry>" in content