import os
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET_COLUMN = "Exited"
DEFAULT_MODEL_PATH = "/home/khaled/ml_project/classifier.joblib"
MLFLOW_EXPERIMENT_NAME = "Churn_Prediction_Experiment"
MLFLOW_MODEL_NAME = "ChurnRandomForestModel"


def configure_mlflow() -> None:
    """Configure MLflow avec la base SQLite locale du projet.

    La variable ``MLFLOW_TRACKING_URI`` peut remplacer l'URI locale, notamment
    pour isoler les tests. L'interface MLflow doit utiliser la même base.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_db = Path(__file__).resolve().parent / "mlflow.db"
        tracking_uri = f"sqlite:///{tracking_db}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def load_data(data_path: str) -> pd.DataFrame:
    """Charge un fichier CSV et vérifie que son contenu n'est pas vide.

    Args:
        data_path: Chemin vers un fichier CSV existant.

    Returns:
        Les données sous forme de DataFrame pandas.

    Raises:
        TypeError: Si ``data_path`` n'est pas une chaîne.
        FileNotFoundError: Si le chemin n'existe pas.
        ValueError: Si le CSV ne contient aucune ligne ou aucune colonne.
        pandas.errors.ParserError: Si pandas ne peut pas analyser le CSV.
    """
    if not isinstance(data_path, str) or not data_path.strip():
        raise TypeError("data_path doit être une chaîne non vide.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Le fichier de données est introuvable : {data_path}")

    df = pd.read_csv(data_path)
    if df.empty or len(df.columns) == 0:
        raise ValueError(f"Le fichier CSV ne contient pas de données : {data_path}")
    return df


def preprocess_data(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Prépare le dataset et sépare les variables explicatives de la cible.

    Encode ``Gender`` si elle est présente et retire les colonnes d'identification
    ``Surname``, ``Geography``, ``RowNumber`` et ``CustomerId``.

    Args:
        df: Dataset brut contenant une ligne par observation.
        target_column: Nom de la variable cible à prédire.

    Returns:
        Un tuple ``(X, y)`` où ``X`` est un DataFrame numérique sans la cible
        et ``y`` une Series contenant la cible.

    Raises:
        TypeError: Si ``df`` n'est pas un DataFrame ou ``target_column`` n'est
            pas une chaîne.
        ValueError: Si le DataFrame est vide, contient des valeurs manquantes,
            ou si les variables explicatives ne sont pas numériques.
        KeyError: Si la colonne cible est absente.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un DataFrame pandas.")
    if not isinstance(target_column, str) or not target_column.strip():
        raise TypeError("target_column doit être une chaîne non vide.")
    if df.empty:
        raise ValueError("Le DataFrame est vide.")

    data = df.copy()
    if target_column not in data.columns:
        raise KeyError(f"La colonne cible '{target_column}' est absente dans les données.")
    if data.isna().any().any():
        raise ValueError("Le dataset contient des valeurs manquantes; veuillez les traiter avant le prétraitement.")

    if "Gender" in data.columns:
        encoder = LabelEncoder()
        data["Gender"] = encoder.fit_transform(data["Gender"])

    columns_to_drop = [col for col in ["Surname", "Geography", "RowNumber", "CustomerId"] if col in data.columns]
    data = data.drop(columns=columns_to_drop, errors="ignore")

    X = data.drop(columns=[target_column])
    y = data[target_column]
    non_numeric_columns = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Les variables explicatives doivent être numériques : {non_numeric_columns}")
    if X.empty or X.shape[1] == 0:
        raise ValueError("Aucune variable explicative n'est disponible après le prétraitement.")
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Sépare les données en ensembles d'entraînement et de test stratifiés.

    Args:
        X: DataFrame des variables explicatives, sans valeurs manquantes.
        y: Series de la cible, de même longueur que ``X``.
        test_size: Proportion réservée au test, strictement comprise entre 0 et 1.
        random_state: Graine aléatoire pour rendre la séparation reproductible.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` avec lignes alignées.

    Raises:
        TypeError: Si les types d'entrée ne correspondent pas aux types attendus.
        ValueError: Si les données sont vides, incompatibles ou si la proportion
            de test est invalide.
    """
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X doit être un DataFrame et y une Series pandas.")
    if X.empty or y.empty:
        raise ValueError("X et y ne doivent pas être vides.")
    if len(X) != len(y):
        raise ValueError("X et y doivent contenir le même nombre de lignes.")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être strictement compris entre 0 et 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def prepare_data(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Charge, prétraite et sépare les données en ensembles train et test.

    Args:
        data_path: Chemin vers le fichier CSV source.
        test_size: Proportion réservée au jeu de test.
        random_state: Graine aléatoire pour rendre la séparation reproductible.

    Returns:
        ``(X_train, X_test, y_train, y_test)`` prêt pour l'entraînement et
        l'évaluation.

    Raises:
        FileNotFoundError: Si le fichier CSV n'existe pas.
        KeyError: Si la colonne cible ``Exited`` est absente.
        TypeError: Si une entrée ne respecte pas le contrat d'une étape.
        ValueError: Si les données sont invalides ou ne peuvent pas être séparées.
    """
    df = load_data(data_path)
    X, y = preprocess_data(df)
    return split_data(X, y, test_size=test_size, random_state=random_state)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Entraîne une forêt aléatoire sur les données préparées.

    Args:
        X_train: Variables explicatives numériques de l'ensemble d'entraînement.
        y_train: Étiquettes de la cible, alignées avec les lignes de ``X_train``.

    Returns:
        Le classifieur Random Forest ajusté.

    Raises:
        TypeError: Si les entrées ne sont pas un DataFrame et une Series.
        ValueError: Si les données sont vides, non alignées, incomplètes ou
            contiennent des variables non numériques.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train doit être un DataFrame et y_train une Series pandas.")
    if X_train.empty or y_train.empty:
        raise ValueError("Les données d'entraînement ne doivent pas être vides.")
    if X_train.shape[1] == 0:
        raise ValueError("X_train doit contenir au moins une variable explicative.")
    if len(X_train) != len(y_train):
        raise ValueError("X_train et y_train doivent contenir le même nombre de lignes.")
    if X_train.isna().any().any() or y_train.isna().any():
        raise ValueError("Les données d'entraînement contiennent des valeurs manquantes.")
    non_numeric_columns = X_train.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Les variables explicatives doivent être numériques : {non_numeric_columns}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Évalue le modèle et renvoie les métriques et prédictions.

    Args:
        model: Modèle déjà entraîné et compatible avec l'API scikit-learn.
        X_test: Variables explicatives numériques de l'ensemble de test.
        y_test: Vérités terrain alignées avec ``X_test``.

    Returns:
        Dictionnaire comprenant ``accuracy``, ``predictions``,
        ``classification_report`` et ``confusion_matrix``.

    Raises:
        TypeError: Si ``model``, ``X_test`` ou ``y_test`` n'a pas le type attendu.
        ValueError: Si les données de test sont vides ou mal alignées.
    """
    if not isinstance(X_test, pd.DataFrame) or not isinstance(y_test, pd.Series):
        raise TypeError("X_test doit être un DataFrame et y_test une Series pandas.")
    if X_test.empty or y_test.empty:
        raise ValueError("Les données de test ne doivent pas être vides.")
    if len(X_test) != len(y_test):
        raise ValueError("X_test et y_test doivent contenir le même nombre de lignes.")
    if X_test.shape[1] == 0:
        raise ValueError("X_test doit contenir au moins une variable explicative.")
    if X_test.isna().any().any() or y_test.isna().any():
        raise ValueError("Les données de test contiennent des valeurs manquantes.")
    non_numeric_columns = X_test.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(f"Les variables explicatives doivent être numériques : {non_numeric_columns}")
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("model doit être un RandomForestClassifier entraîné.")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "predictions": y_pred,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def train_and_log_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_path: str = DEFAULT_MODEL_PATH,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Entraîne, évalue et trace une expérience MLflow complète.

    Args:
        X_train: Variables explicatives utilisées pour l'entraînement.
        y_train: Cible associée aux lignes de ``X_train``.
        X_test: Variables explicatives réservées à l'évaluation.
        y_test: Vérités terrain associées aux lignes de ``X_test``.
        model_path: Chemin où sauvegarder également le modèle avec joblib.

    Returns:
        Le modèle entraîné et les métriques calculées sur le jeu de test.

    Raises:
        ValueError: Si les données ou le chemin du modèle sont invalides.
        OSError: Si le modèle local ne peut pas être sauvegardé.
        Les erreurs MLflow de configuration, tracking ou registre sont
        propagées à l'appelant.
    """
    configure_mlflow()

    with mlflow.start_run(run_name="RandomForest_Run"):
        model = train_model(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "random_state": model.random_state,
                "test_size": len(X_test) / (len(X_train) + len(X_test)),
                "feature_count": X_train.shape[1],
            }
        )
        mlflow.log_metrics(
            {
                "accuracy": metrics["accuracy"],
                "precision": precision_score(y_test, metrics["predictions"], zero_division=0),
                "recall": recall_score(y_test, metrics["predictions"], zero_division=0),
                "f1_score": f1_score(y_test, metrics["predictions"], zero_division=0),
            }
        )
        mlflow.log_dict(
            {
                "labels": [0, 1],
                "matrix": metrics["confusion_matrix"].tolist(),
            },
            "confusion_matrix.json",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X_train.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        save_model(model, model_path)

    return model, metrics


def save_model(model: RandomForestClassifier, output_path: str = DEFAULT_MODEL_PATH) -> str:
    """Sérialise le modèle entraîné au chemin indiqué avec joblib.

    Args:
        model: Modèle entraîné à sauvegarder.
        output_path: Fichier de destination. Son dossier est créé au besoin.

    Returns:
        Le chemin du fichier créé.

    Raises:
        TypeError: Si le chemin n'est pas une chaîne non vide ou si ``model``
            n'est pas un RandomForestClassifier.
        OSError: Si le fichier ou son dossier ne peut pas être écrit.
    """
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("model doit être un RandomForestClassifier.")
    if not isinstance(output_path, str) or not output_path.strip():
        raise TypeError("output_path doit être une chaîne non vide.")
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "wb") as model_file:
        joblib.dump(model, model_file)

    return output_path


def load_model(model_path: str = DEFAULT_MODEL_PATH) -> RandomForestClassifier:
    """Charge un modèle sérialisé par :func:`save_model`.

    Args:
        model_path: Chemin d'un fichier de modèle existant.

    Returns:
        Le modèle désérialisé.

    Raises:
        TypeError: Si le chemin n'est pas une chaîne non vide.
        FileNotFoundError: Si le fichier n'existe pas.
        OSError: Si le fichier ne peut pas être lu.
        Les erreurs de désérialisation sont propagées si le fichier est
        corrompu ou n'est pas un fichier joblib valide.
    """
    if not isinstance(model_path, str) or not model_path.strip():
        raise TypeError("model_path doit être une chaîne non vide.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le modèle n’existe pas : {model_path}")

    with open(model_path, "rb") as model_file:
        model = joblib.load(model_file)
    if not isinstance(model, RandomForestClassifier):
        raise ValueError("Le fichier chargé ne contient pas un RandomForestClassifier.")
    return model


def run_pipeline(data_path: str, model_path: str = DEFAULT_MODEL_PATH) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Exécute le pipeline complet et sauvegarde le modèle obtenu.

    Args:
        data_path: Chemin du CSV contenant les données brutes.
        model_path: Destination du modèle sérialisé.

    Returns:
        Un tuple composé du modèle entraîné et du dictionnaire de métriques
        produit par :func:`evaluate_model`.

    Raises:
        FileNotFoundError: Si le CSV n'existe pas.
        KeyError: Si la colonne cible est absente.
        TypeError: Si une entrée ne respecte pas le contrat de son étape.
        ValueError: Si les données ne peuvent pas être préparées ou séparées.
        OSError: Si le modèle ne peut pas être sauvegardé.
    """
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    return train_and_log_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_path=model_path,
    )
