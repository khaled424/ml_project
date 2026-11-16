import argparse
import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model_pipeline import (
    DEFAULT_MODEL_PATH,
    evaluate_model,
    load_model,
    prepare_data,
    run_pipeline,
    train_and_log_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = str(PROJECT_DIR / "data" / "Churn_Modelling.csv")
DEFAULT_PREPARED_DATA_PATH = str(PROJECT_DIR / "artifacts" / "prepared_data.joblib")
PreparedData = Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]


def save_prepared_data(
    prepared_data: PreparedData,
    prepared_data_path: str,
) -> None:
    """Enregistre les ensembles préparés pour les étapes train et evaluate."""
    directory = os.path.dirname(prepared_data_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(prepared_data_path, "wb") as prepared_file:
        joblib.dump(prepared_data, prepared_file)


def load_prepared_data(prepared_data_path: str) -> PreparedData:
    """Charge et valide les ensembles train/test enregistrés par --prepare."""
    if not os.path.exists(prepared_data_path):
        raise FileNotFoundError(
            f"Données préparées introuvables : {prepared_data_path}. "
            "Lancez d'abord main.py --prepare."
        )

    with open(prepared_data_path, "rb") as prepared_file:
        prepared_data = joblib.load(prepared_file)

    if (
        not isinstance(prepared_data, tuple)
        or len(prepared_data) != 4
        or not isinstance(prepared_data[0], pd.DataFrame)
        or not isinstance(prepared_data[1], pd.DataFrame)
        or not isinstance(prepared_data[2], pd.Series)
        or not isinstance(prepared_data[3], pd.Series)
    ):
        raise ValueError("Le fichier préparé ne contient pas les données train/test attendues.")

    return prepared_data


def print_metrics(metrics: dict, model_path: str) -> None:
    """Affiche les principales métriques calculées sur le jeu de test."""
    print("\n=== Évaluation du modèle ===")
    print(f"Accuracy score : {metrics['accuracy'] * 100:.2f}%")
    print(metrics["classification_report"])
    print("\nMatrice de confusion :")
    print(metrics["confusion_matrix"])
    print(f"\nModèle sauvegardé dans : {model_path}")


def main() -> None:
    """Exécute le pipeline complet ou une étape demandée en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Pipeline de prédiction de churn client."
    )
    stages = parser.add_mutually_exclusive_group()
    stages.add_argument(
        "--prepare",
        action="store_true",
        help="Prépare les données et sauvegarde les ensembles train/test.",
    )
    stages.add_argument(
        "--train",
        action="store_true",
        help="Entraîne le modèle à partir des données préparées.",
    )
    stages.add_argument(
        "--evaluate",
        action="store_true",
        help="Évalue le modèle sauvegardé sur le jeu de test.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Chemin du fichier CSV source.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Chemin de sauvegarde ou de lecture du modèle.",
    )
    parser.add_argument(
        "--prepared-data-path",
        type=str,
        default=DEFAULT_PREPARED_DATA_PATH,
        help="Chemin de sauvegarde ou de lecture des données préparées.",
    )
    args = parser.parse_args()

    if args.prepare:
        prepared_data = prepare_data(args.data_path)
        save_prepared_data(prepared_data, args.prepared_data_path)
        X_train, X_test, y_train, y_test = prepared_data
        print("Préparation terminée.")
        print(f"X_train : {X_train.shape}; y_train : {y_train.shape}")
        print(f"X_test  : {X_test.shape}; y_test  : {y_test.shape}")
        print(f"Données préparées sauvegardées dans : {args.prepared_data_path}")
    elif args.train:
        X_train, X_test, y_train, y_test = load_prepared_data(args.prepared_data_path)
        _, metrics = train_and_log_model(
            X_train,
            y_train,
            X_test,
            y_test,
            model_path=args.model_path,
        )
        print_metrics(metrics, args.model_path)
    elif args.evaluate:
        _, X_test, _, y_test = load_prepared_data(args.prepared_data_path)
        model: RandomForestClassifier = load_model(args.model_path)
        metrics = evaluate_model(model, X_test, y_test)
        print_metrics(metrics, args.model_path)
    else:
        _, metrics = run_pipeline(args.data_path, args.model_path)
        print_metrics(metrics, args.model_path)


if __name__ == "__main__":
    main()
