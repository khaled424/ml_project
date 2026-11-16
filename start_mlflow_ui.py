import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = PROJECT_DIR / "venv" / "bin" / "python"

if not PYTHON_BIN.is_file():
    print(
        f"Python de l'environnement virtuel introuvable : {PYTHON_BIN}",
        file=sys.stderr,
    )
    print(
        "Créez l'environnement puis installez requirements.txt.",
        file=sys.stderr,
    )
    raise SystemExit(1)

command = [
    str(PYTHON_BIN),
    "-m",
    "mlflow",
    "ui",
    "--backend-store-uri",
    f"sqlite:///{PROJECT_DIR / 'mlflow.db'}",
    "--host",
    "127.0.0.1",
    "--port",
    "5000",
]
os.execv(str(PYTHON_BIN), command)
