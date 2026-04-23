import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow.catboost
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

import mlflow

mlflow.set_tracking_uri("http://mlflow:5000")
experiment_name = "fourthr"
# mlflow.create_experiment(
#     experiment_name, artifact_location="s3://pipeout-database/mlflow-artifacts"
# )
mlflow.set_experiment(experiment_name)


def load_config(CONFIG_PATH):
    """
    Selects the current dataset's config file we are interest in.
    """
    with open(CONFIG_PATH, "r") as f:
        full_config = yaml.safe_load(f)

    try:
        current_dataset = full_config["CURRENT_DATASET"]
        logging.info(f"\nloading current dataset: {current_dataset}")
        if current_dataset not in full_config["DATASETS"]:
            raise ValueError(f"\nDataset {current_dataset} not found!")

        return full_config["DATASETS"][current_dataset]

    except Exception as e:
        logging.exception(
            f"There was an error handling the config cleaning.yaml file {e}"
        )
        raise


def get_config_file():
    try:
        base_dir = Path(__file__).resolve().parent.parent
        path = base_dir / "configs" / "training.yaml"
        return path
    except NameError:  # if it is a jupyter file
        return Path("/training-app/configs/training.yaml")


def splitting(df_base: pd.DataFrame):
    y = df_base["Target_Evaded"]

    ## IMI is leakage!
    ## Periodo_Atual is leakage!

    cols_to_drop = [
        "RGA_Anon",
        "Situação atual",
        "Target_Evaded",
        "Idade_Ingresso",
        "IMI",
        "Periodo_Atual",
    ]
    X = df_base.drop(columns=cols_to_drop)
    cat_features = X.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    for col in cat_features:
        X[col] = X[col].astype(str)

    gss = GroupShuffleSplit(n_splits=2, train_size=0.7, random_state=42)
    train_idx, test_idx = next(gss.split(df_base, groups=df_base["RGA_Anon"]))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


def selecting_active_students(df: pd.DataFrame) -> pd.DataFrame:
    df_ativos = df[
        df["Situação atual"].isin(
            [
                "MATRICULADO NO PERÍODO",
                "AFASTAMENTO POR BLOQUEIO DE MATRICULA",
                "AFASTAMENTO POR TRANCAMENTO DE MATRICULA",
            ]
        )
    ].copy()

    df = df.drop(df_ativos.index)

    df["Target_Evaded"] = np.where(
        df["Situação atual"] == "EXCLUSAO POR CONCLUSAO (FORMADO)",
        0,  # not evaded
        1,  # else, evade
    )
    return df


def model_fitting(X_train, y_train):

    params = {"iterations": 300, "learning_rate": 0.01, "depth": 6, "verbose": 0}

    cat = CatBoostClassifier(**params)
    cat_features = X_train.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    cat.fit(X_train, y_train, cat_features=cat_features)

    mlflow.log_params(params)
    print(type(cat))
    return cat


def results(cat, y_test, X_test):

    y_pred = cat.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Métricas Detalhadas ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["Formado (0)", "Evadido (1)"]
        )
    )
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Formado", "Evadido"]
    )
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão: Evasão Estudantil")
    plt.show()


if __name__ == "__main__":
    with mlflow.start_run() as run:
        startTime = time.time()
        print(f"Experiment ID: {run.info.experiment_id}")
        CONFIG_PATH = get_config_file()
        dfs = load_config(CONFIG_PATH)

        df_base = pd.read_csv(dfs["TRAINING_DATASET"])

        columns_to_drop = [    "Sexo",
                "Raça",
                "Estrutura",
                "Período ingresso",
                "Tipo ingresso",
                "AnoSem"]

        df_base = selecting_active_students(df_base)
        
        df_base.drop(
            columns=columns_to_drop,
            inplace=True,
        )

        X_train, X_test, y_train, y_test = splitting(df_base)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        model = model_fitting(X_train, y_train)
        results(model, y_test, X_test)
        mlflow.catboost.log_model(model, "model")

        totalTime = time.time() - startTime
        print(f"Total training time: {totalTime:.2f} seconds")
