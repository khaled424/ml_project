import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable
MAIN_PY = PROJECT_DIR / "main.py"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"


@task(name="update-project")
def update_project() -> None:
    """Récupère les derniers changements depuis GitHub."""
    subprocess.run(
        ["git", "pull", "--ff-only", "origin", "main"],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="install-dependencies")
def install_dependencies() -> None:
    """Installe les dépendances déclarées par le projet."""
    logger = get_run_logger()
    logger.info("Installation des dépendances du projet...")
    subprocess.run(
        [PYTHON_BIN, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="check-code-quality")
def check_code_quality() -> None:
    """Vérifie les scripts Python du projet avec Flake8."""
    logger = get_run_logger()
    logger.info("Vérification de la qualité du code avec Flake8...")
    subprocess.run(
        [
            PYTHON_BIN,
            "-m",
            "flake8",
            "--max-line-length=120",
            "model_pipeline.py",
            "main.py",
            "pipeline_prefect.py",
        ],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="run-unit-tests")
def run_unit_tests() -> None:
    """Exécute les tests unitaires avec pytest."""
    logger = get_run_logger()
    logger.info("Exécution des tests unitaires...")
    subprocess.run(
        [PYTHON_BIN, "-m", "pytest", "-v"],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="prepare-data")
def prepare_data_task() -> None:
    """Prépare les données et sauvegarde les ensembles train/test."""
    logger = get_run_logger()
    logger.info("Préparation des données...")
    subprocess.run(
        [PYTHON_BIN, str(MAIN_PY), "--prepare"],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="train-model")
def train_model_task() -> None:
    """Entraîne le modèle à partir des données préparées."""
    logger = get_run_logger()
    logger.info("Entraînement et sauvegarde du modèle...")
    subprocess.run(
        [PYTHON_BIN, str(MAIN_PY), "--train"],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="evaluate-model")
def evaluate_model_task() -> None:
    """Évalue le modèle sauvegardé sur les données de test."""
    logger = get_run_logger()
    logger.info("Évaluation des performances du modèle...")
    subprocess.run(
        [PYTHON_BIN, str(MAIN_PY), "--evaluate"],
        cwd=PROJECT_DIR,
        check=True,
    )


@flow(name="code", log_prints=True)
def flow_code() -> None:
    """Installe les dépendances puis vérifie le code et les tests."""
    update_project()
    install_dependencies()
    check_code_quality()
    run_unit_tests()


@flow(name="train", log_prints=True)
def flow_train() -> None:
    """Prépare les données, puis entraîne le modèle."""
    update_project()
    prepare_data_task()
    train_model_task()


@flow(name="evaluate", log_prints=True)
def flow_evaluate() -> None:
    """Évalue les données préparées avec le modèle sauvegardé."""
    update_project()
    evaluate_model_task()


@flow(name="all", log_prints=True)
def flow_all() -> None:
    """Exécute les contrôles, puis tout le pipeline ML."""
    flow_code()
    prepare_data_task()
    train_model_task()
    evaluate_model_task()


def main() -> None:
    """Sélectionne et exécute un flow depuis la ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect.")
    parser.add_argument(
        "--flow",
        choices=["code", "train", "evaluate", "all"],
        default="all",
        help="Flow à exécuter.",
    )
    args = parser.parse_args()

    flows = {
        "code": flow_code,
        "train": flow_train,
        "evaluate": flow_evaluate,
        "all": flow_all,
    }
    flows[args.flow]()


if __name__ == "__main__":
    main()
