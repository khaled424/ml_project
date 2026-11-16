import argparse
import datetime
import os
from pathlib import Path
from typing import Any, Dict

import mlflow
import mlflow.sklearn
from elasticsearch import Elasticsearch
from sklearn.metrics import f1_score, precision_score, recall_score

from model_pipeline import (
    DEFAULT_MODEL_PATH,
    MLFLOW_MODEL_NAME,
    configure_mlflow,
    evaluate_model,
    prepare_data,
    save_model,
    train_model,
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "Churn_Modelling.csv"
DEFAULT_ELASTICSEARCH_URL = "http://127.0.0.1:9200"
DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_INDEX_NAME = "mlflow-metrics"


def ensure_metrics_index(es: Elasticsearch, index_name: str) -> None:
    """Crée l'index de métriques avec son mapping s'il n'existe pas."""
    if es.indices.exists(index=index_name):
        return

    es.indices.create(
        index=index_name,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "run_id": {"type": "keyword"},
                    "experiment_id": {"type": "keyword"},
                    "status": {"type": "keyword"},
                    "metric_test_accuracy": {"type": "float"},
                    "metric_test_precision": {"type": "float"},
                    "metric_test_recall": {"type": "float"},
                    "metric_test_f1_score": {"type": "float"},
                    "param_n_estimators": {"type": "keyword"},
                    "param_random_state": {"type": "keyword"},
                    "param_feature_count": {"type": "keyword"},
                }
            },
        },
    )


def run_monitored_pipeline(
    data_path: Path,
    elasticsearch_url: str,
    mlflow_tracking_uri: str,
    index_name: str,
) -> Dict[str, Any]:
    """Entraîne et évalue le modèle, puis trace les résultats dans MLflow et Elasticsearch."""
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    configure_mlflow()

    es = Elasticsearch([elasticsearch_url])
    if not es.ping():
        raise ConnectionError(f"Elasticsearch ne répond pas à {elasticsearch_url}.")
    ensure_metrics_index(es, index_name)

    X_train, X_test, y_train, y_test = prepare_data(str(data_path))
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    predictions = metrics["predictions"]

    logged_metrics = {
        "test_accuracy": float(metrics["accuracy"]),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_f1_score": float(f1_score(y_test, predictions, zero_division=0)),
    }

    with mlflow.start_run(run_name="Atelier_7_Monitoring") as run:
        mlflow.log_params(
            {
                "n_estimators": model.n_estimators,
                "random_state": model.random_state,
                "test_size": len(X_test) / (len(X_train) + len(X_test)),
                "feature_count": X_train.shape[1],
            }
        )
        mlflow.log_metrics(logged_metrics)
        mlflow.log_dict(
            {"labels": [0, 1], "matrix": metrics["confusion_matrix"].tolist()},
            "confusion_matrix.json",
        )
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=MLFLOW_MODEL_NAME,
            input_example=X_train.head(5),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        save_model(model, str(DEFAULT_MODEL_PATH))

    document = {
        "@timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": "FINISHED",
        "metric_test_accuracy": logged_metrics["test_accuracy"],
        "metric_test_precision": logged_metrics["test_precision"],
        "metric_test_recall": logged_metrics["test_recall"],
        "metric_test_f1_score": logged_metrics["test_f1_score"],
        "param_n_estimators": str(model.n_estimators),
        "param_random_state": str(model.random_state),
        "param_feature_count": str(X_train.shape[1]),
    }
    response = es.index(index=index_name, document=document)
    es.close()

    print(f"Run MLflow : {run.info.run_id}")
    print(f"Accuracy test : {logged_metrics['test_accuracy']:.4f}")
    print(f"Document Elasticsearch : {response['_id']}")
    return {"run_id": run.info.run_id, "metrics": logged_metrics, "document_id": response["_id"]}


def main() -> None:
    """Parse les options d'exécution de l'atelier 7."""
    parser = argparse.ArgumentParser(description="Monitoring MLflow et Elasticsearch.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--elasticsearch-url", default=DEFAULT_ELASTICSEARCH_URL)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--index", default=DEFAULT_INDEX_NAME)
    args = parser.parse_args()

    run_monitored_pipeline(
        data_path=args.data_path,
        elasticsearch_url=args.elasticsearch_url,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        index_name=args.index,
    )


if __name__ == "__main__":
    main()
