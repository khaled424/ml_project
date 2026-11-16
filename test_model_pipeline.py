import os

import pandas as pd
import pytest

from model_pipeline import (
    evaluate_model,
    load_data,
    load_model,
    prepare_data,
    preprocess_data,
    save_model,
    split_data,
    train_model,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            'RowNumber': [1, 2, 3, 4],
            'CustomerId': [101, 102, 103, 104],
            'Surname': ['A', 'B', 'C', 'D'],
            'CreditScore': [600, 700, 650, 710],
            'Geography': ['France', 'Spain', 'France', 'Germany'],
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Age': [30, 45, 40, 50],
            'Tenure': [1, 3, 5, 2],
            'Balance': [1000, 2000, 1500, 2500],
            'NumOfProducts': [1, 2, 2, 1],
            'HasCrCard': [1, 0, 1, 0],
            'IsActiveMember': [1, 0, 1, 1],
            'EstimatedSalary': [10000, 20000, 20000, 30000],
            'Exited': [0, 1, 0, 1],
        }
    )


def test_load_data_valid_file(sample_df, tmp_path):
    csv_path = tmp_path / 'sample.csv'
    sample_df.to_csv(csv_path, index=False)

    df = load_data(str(csv_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 4
    assert 'Exited' in df.columns


def test_prepare_data_creates_train_and_test_sets(sample_df, tmp_path):
    csv_path = tmp_path / 'sample.csv'
    sample_df.to_csv(csv_path, index=False)

    prepared_data = prepare_data(
        str(csv_path),
        test_size=0.5,
        random_state=42,
    )

    assert isinstance(prepared_data, tuple)
    X_train, X_test, y_train, y_test = prepared_data
    assert X_train.shape == (2, 9)
    assert X_test.shape == (2, 9)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_preprocess_data(sample_df):
    X, y = preprocess_data(sample_df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'Exited' not in X.columns
    assert len(X) == len(y)
    assert set(X.columns) == {
        'CreditScore',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary',
    }


def test_preprocess_rejects_missing_target(sample_df):
    with pytest.raises(KeyError, match="Exited"):
        preprocess_data(sample_df.drop(columns=["Exited"]))


def test_preprocess_rejects_missing_values(sample_df):
    sample_df.loc[0, "Age"] = None

    with pytest.raises(ValueError, match="valeurs manquantes"):
        preprocess_data(sample_df)


def test_split_data(sample_df):
    X, y = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) + len(X_test) == len(X)
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0


def test_split_data_rejects_invalid_test_size(sample_df):
    X, y = preprocess_data(sample_df)

    with pytest.raises(ValueError, match="test_size"):
        split_data(X, y, test_size=1)


def test_train_model(sample_df):
    X, y = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)

    model = train_model(X_train, y_train)

    assert hasattr(model, 'predict')
    preds = model.predict(X_test)
    assert len(preds) == len(y_test)


def test_evaluate_model(sample_df):
    X, y = preprocess_data(sample_df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.5, random_state=42)
    model = train_model(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)

    assert 'accuracy' in metrics
    assert 'predictions' in metrics
    assert 'classification_report' in metrics
    assert 'confusion_matrix' in metrics
    assert len(metrics['predictions']) == len(y_test)


def test_save_and_load_model(sample_df, tmp_path):
    X, y = preprocess_data(sample_df)
    X_train, _, y_train, _ = split_data(X, y, test_size=0.5, random_state=42)
    model = train_model(X_train, y_train)

    model_path = tmp_path / 'model.joblib'
    save_model(model, str(model_path))

    assert os.path.exists(model_path)

    loaded_model = load_model(str(model_path))
    assert hasattr(loaded_model, 'predict')


def test_load_data_missing_file():
    with pytest.raises(FileNotFoundError):
        load_data('/tmp/fichier_inexistant.csv')


def test_load_model_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(str(tmp_path / "missing.joblib"))
