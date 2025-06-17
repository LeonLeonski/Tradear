# Tradear 📈
Scientific-Programming semester project  

## 1. Quick start

```bash
# 1 – clone & enter the project
git clone https://github.com/LeonLeonski/Tradear.git
cd Tradear

# 2 - install dependencies
python -m pip install -r requirements.txt

# 3 – run the data-to-dashboard pipeline
python main.py
```

---

## 2. Tests ausführen

```bash
# pytest installieren (falls noch nicht vorhanden)
pip install pytest

# Tests ausführen (im Projektordner)
python -m pytest
```

---

## 3. Konfiguration anpassen (`config.py`)

Die wichtigsten Parameter des Projekts sind zentral in der Datei [`src/config.py`](src/config.py) abgelegt.  
Hier kannst du z.B. die Anzahl der geladenen Binance-Kerzen, Modellparameter und Optimierungsoptionen anpassen:

```python
# src/config.py (Beispiel)
BINANCE_LIMIT = 1440  # Anzahl der Kerzen (z.B. 1440 für 24h)
RANDOM_FOREST_ESTIMATORS = 100  # Anzahl Bäume im RandomForest
TP_OPTIONS = [0.5, 1.0, 1.5, 2.0]  # Take-Profit-Multiplikatoren
SL_OPTIONS = [0.5, 1.0, 1.5, 2.0]  # Stop-Loss-Multiplikatoren
SIGNIFICANCE_LEVEL = 0.05  # Schwellenwert für P-Wert
```

**So nutzt du die Konfiguration:**  
Alle Hauptskripte importieren diese Parameter automatisch.  
Wenn du z.B. mehr oder weniger Daten laden, andere Modellgrößen oder andere Optimierungsoptionen testen willst, passe einfach die Werte in `config.py` an und starte das Projekt wie gewohnt.

---

# Roadmap & To-Do-Liste

## To-Dos (Kurzfristig)

- [X] Fehlerbehandlung in allen Skripten verbessern (z.B. try/except für Dateioperationen)
- [X] Code modularisieren: Wiederverwendbare Funktionen für Indikatoren und Datenverarbeitung auslagern
- [X] Logging statt print-Statements verwenden
- [X] Unit Tests für zentrale Funktionen schreiben
- [X] Konfigurationsdatei für Parameter (API-Keys, Schwellenwerte, Limits) einführen
- [X] `.gitignore`-Datei korrigieren (`combined_predictions.json` statt `combinded_predictions.json`)
- [X] Dokumentation der einzelnen Skripte und deren Zusammenspiel ergänzen
- [X] Deprecated Nachrichten entfernen

## Roadmap (Mittelfristig)

1. **Zentrales Hauptskript**
   - [X] Ein zentrales Skript erstellen, das die Einzelschritte orchestriert (z.B. `main.py`)
   - [X] Einzelne Logik in Modulen/Funktionen belassen

2. **Effizientere Datenbeschaffung**
   - [ ] Mechanismus implementieren, der nur neue Daten von der API holt (letztes Datum prüfen)
   - [ ] Vermeidung von doppelten Datenabrufen

3. **Datenhaltung verbessern**
   - [ ] Umstieg von Dateispeicherung auf eine Datenbank (z.B. SQLite oder PostgreSQL)
   - [ ] Datenbankstruktur entwerfen und Migration der bestehenden Daten

4. **Frontend/Backend-Trennung**
   - [ ] Backend-API für Datenbereitstellung erstellen (statt direktem Dateizugriff im Frontend)
   - [ ] Zeitzonenhandling und Datenformatierung im Backend zentralisieren

5. **Wallet-Anbindung (Langfristig)**
   - [ ] Paper-Trading-Modus implementieren und testen
   - [ ] Sichere Anbindung eines Wallets für automatische Transaktionen
   - [ ] Fehlerbehandlung, Logging und Sicherheitsmaßnahmen für Live-Trading

---

## Skripte und Zusammenspiel (Kurzüberblick)

- **getData.py**  
  Holt aktuelle Bitcoin-Daten von Binance und speichert sie als XML.

- **calculateData.py**  
  Liest die XML-Daten, berechnet Indikatoren und speichert das Ergebnis als JSON.

- **getMultiplePredictions.py**  
  Nutzt die berechneten Daten, erstellt Kursprognosen und speichert alles als kombiniertes JSON.

- **main.py**  
  Startet die drei Skripte automatisch in der richtigen Reihenfolge.

**Ablauf:**  
`main.py` → `getData.py` → `calculateData.py` → `getMultiplePredictions.py`

Alle Zwischenergebnisse werden als Datei gespeichert und von den nächsten Schritten genutzt.

*Diese Liste dient als Übersicht für die nächsten Entwicklungsschritte und kann bei Bedarf erweitert oder angepasst werden.*