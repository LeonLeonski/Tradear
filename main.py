import subprocess
import sys
import os

# Liste der Skripte, die nacheinander ausgeführt werden sollen
scripts = [
    "code/getData.py",
    "code/calculateData.py",
    "code/getMultiplePredictions.py"
]

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"Starte {script_name} ...")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Fehler beim Ausführen von {script_name}.")
        sys.exit(result.returncode)
    print(f"{script_name} erfolgreich abgeschlossen.\n")

if __name__ == "__main__":
    for script in scripts:
        run_script(script)
    print("Alle Skripte wurden erfolgreich ausgeführt.")