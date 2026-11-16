import argparse
import subprocess
import sys
from pathlib import Path

from prefect import flow, get_run_logger, task

PROJECT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = sys.executable
MAIN_PY = PROJECT_DIR / "main.py"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements.txt"
DOCKER_IMAGE_NAME = "khaled_4bi6_mlops"
DOCKERHUB_USERNAME = "khaledgannounhahahaha"
DOCKER_IMAGE_TAG = "1.0"


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
            "app.py",
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


@task(name="start-fastapi-server")
def start_api_task() -> None:
    """Lance Uvicorn au premier plan afin que Prefect suive le serveur."""
    logger = get_run_logger()
    logger.info("Démarrage de l'API FastAPI sur http://127.0.0.1:8000...")
    subprocess.run(
        [
            PYTHON_BIN,
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="build-docker-image")
def build_docker_image_task() -> None:
    """Construit l'image Docker depuis la racine du projet."""
    logger = get_run_logger()
    image = f"{DOCKER_IMAGE_NAME}:latest"
    logger.info("Construction de l'image Docker %s...", image)
    subprocess.run(
        ["docker", "build", "-t", image, "."],
        cwd=PROJECT_DIR,
        check=True,
    )


@task(name="push-docker-image")
def push_docker_image_task() -> None:
    """Tague l'image puis la publie sur Docker Hub."""
    logger = get_run_logger()
    local_image = f"{DOCKER_IMAGE_NAME}:latest"
    remote_image = (
        f"{DOCKERHUB_USERNAME}/{DOCKER_IMAGE_NAME}:{DOCKER_IMAGE_TAG}"
    )

    logger.info("Tag de l'image vers %s...", remote_image)
    subprocess.run(
        ["docker", "tag", local_image, remote_image],
        cwd=PROJECT_DIR,
        check=True,
    )
    logger.info("Publication de l'image sur Docker Hub...")
    subprocess.run(
        ["docker", "push", remote_image],
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


@flow(name="api", log_prints=True)
def flow_api() -> None:
    """Lance le serveur FastAPI et le garde actif pendant le flow."""
    update_project()
    start_api_task()


@flow(name="all", log_prints=True)
def flow_all() -> None:
    """Exécute les contrôles, puis tout le pipeline ML."""
    flow_code()
    prepare_data_task()
    train_model_task()
    evaluate_model_task()


@flow(name="cd-docker-pipeline", log_prints=True)
def flow_cd() -> None:
    """Construit et publie l'image Docker de l'API."""
    build_docker_image_task()
    push_docker_image_task()


def main() -> None:
    """Sélectionne et exécute un flow depuis la ligne de commande."""
    parser = argparse.ArgumentParser(description="Pipeline MLOps avec Prefect.")
    parser.add_argument(
        "--flow",
        choices=["code", "train", "evaluate", "api", "all", "cd"],
        default="all",
        help="Flow à exécuter.",
    )
    args = parser.parse_args()

    flows = {
        "code": flow_code,
        "train": flow_train,
        "evaluate": flow_evaluate,
        "api": flow_api,
        "all": flow_all,
        "cd": flow_cd,
    }
    flows[args.flow]()


if __name__ == "__main__":
    main()
