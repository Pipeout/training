import logging
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


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

    cat = CatBoostClassifier(iterations=300, learning_rate=0.01, depth=6, verbose=0)
    cat_features = X_train.select_dtypes(
        include=["object", "bool", "category"]
    ).columns.tolist()

    cat.fit(X_train, y_train, cat_features=cat_features)
    return cat


def results(cat, y_test):
    y_pred = cat.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Métricas Detalhadas ---")
    print(
        classification_report(
            y_test, y_pred, target_names=["Formado (0)", "Evadido (1)"]
        )
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Formado", "Evadido"]
    )
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusão: Evasão Estudantil")
    plt.show()


if __name__ == "__main__":
    startTime = time.time()

    CONFIG_PATH = get_config_file()
    dfs = load_config(CONFIG_PATH)

    df_base = pd.read_csv(dfs["TRAINING_DATASET"])

    df_base = selecting_active_students(df_base)
    df_base.drop(
        columns={
            "Sexo",
            "Raça",
            # "Naturalidade",
            # "Reprovacao_Ponderada_Semestral",
            # "UF Naturalidade",
            #  "Periodo_Atual",
            "Estrutura",
            "Período ingresso",
            "Tipo ingresso",
            "AnoSem",
        },
        inplace=True,
    )

    X_train, X_test, y_train, y_test = splitting(df_base)
    model = model_fitting(X_train, y_train)
    results(model, y_test)
    totalTime = time.time() - startTime
    print(f"Total training time: {totalTime:.2f} seconds")
