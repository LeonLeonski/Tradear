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

# Roadmap & To-Do-Liste

## To-Dos (Kurzfristig)

- [ ] Fehlerbehandlung in allen Skripten verbessern (z.B. try/except für Dateioperationen)
- [ ] Code modularisieren: Wiederverwendbare Funktionen für Indikatoren und Datenverarbeitung auslagern
- [ ] Logging statt print-Statements verwenden
- [ ] Unit Tests für zentrale Funktionen schreiben
- [ ] Konfigurationsdatei für Parameter (API-Keys, Schwellenwerte, Limits) einführen
- [ ] `.gitignore`-Datei korrigieren (`combined_predictions.json` statt `combinded_predictions.json`)
- [ ] Dokumentation der einzelnen Skripte und deren Zusammenspiel ergänzen

## Roadmap (Mittelfristig)

1. **Zentrales Hauptskript**
   - [ ] Ein zentrales Skript erstellen, das die Einzelschritte orchestriert (z.B. `main.py`)
   - [ ] Einzelne Logik in Modulen/Funktionen belassen

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

*Diese Liste dient als Übersicht für die nächsten Entwicklungsschritte und kann bei Bedarf erweitert oder