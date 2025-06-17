import subprocess
import sys

# Liste der Module, die nacheinander ausgeführt werden sollen
modules = [
    "src.getData",
    "src.calculateData",
    "src.getMultiplePredictions"
]

def run_module(module_name):
    print(f"Starte {module_name} ...")
    result = subprocess.run([sys.executable, "-m", module_name])
    if result.returncode != 0:
        print(f"Fehler beim Ausführen von {module_name}.")
        sys.exit(result.returncode)
    print(f"{module_name} erfolgreich abgeschlossen.\n")

if __name__ == "__main__":
    for module in modules:
        run_module(module)
    print("Alle Skripte wurden erfolgreich ausgeführt.")