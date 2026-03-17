import logging
import os
from pathlib import Path

import joblib

# import joblib
import pandas as pd
import yaml
from scipy import *
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "training.yaml"


def load_config(CONFIG_PATH) :
  """
  Selects the current dataset's config file we are interest in.
  """
  with open(CONFIG_PATH, "r") as f:
    full_config = yaml.safe_load(f)

  try:
    current_dataset = full_config["CURRENT_DATASET"]
    logging.info(f"\nloading current dataset: {current_dataset}")
    if current_dataset not in full_config['DATASETS']:
      raise ValueError(f"\nDataset {current_dataset} not found!")

    return full_config["DATASETS"][current_dataset]

  except Exception as e:
    logging.exception(f"There was an error handling the config cleaning.yaml file {e}")
    raise


if __name__ == "__main__":

  config = load_config(CONFIG_PATH)
  df_to_train = pd.read_csv(config["TRAINING_DATASET"])





  df_to_train_final = pd.get_dummies(df_to_train,
                                    columns=["Sexo",
                                              "Raça",
                                              "Naturalidade",
                                              "UF Naturalidade",
                                              "Tipo ingresso",
                                              "Tipo de demanda"
                                              ],drop_first=True)


  print("Starting training")

  # X = df_to_train_final.drop('dataset', axis=1)
  # not_encoded_y = df_to_train_final['dataset']



  # X_train_troll, X_test_troll, y_train_troll, y_test_troll = train_test_split(X, not_encoded_y, test_size=0.2, random_state=42)


  # le = LabelEncoder()
  # y = le.fit_transform(not_encoded_y)
  # print(list(le.classes_))




  # X = df_to_train_final.drop('dataset', axis=1)

  # # Selecionar as 5 melhores features
  # selector = SelectKBest(score_func=f_regression, k=7)
  # X_new = selector.fit_transform(X, y)

  # # Ver as features selecionadas
  # selected_features = selector.get_feature_names_out(input_features=X.columns)
  # print("As 5 melhores features selecionadas pelo método de filtragem:")
  # print(selected_features)


  # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




  # rf = RandomForestClassifier()
  # rf.fit(X_train, y_train)
  # preds = rf.predict(X_test)




  # # saving the model
  # joblib.dump(rf, "rforest_pkl")


  # accuracy_rf = accuracy_score(y_test, preds)
  # print(f"Random Forest's acurracy: {accuracy_rf*100:.2f}%")

  # print("\nClassification report")
  # print(classification_report(y_test, preds))

  # print("\nConfusion Matrix")
  # cm = confusion_matrix(y_test, preds)
  # fig = px.imshow(cm, text_auto=True).update_layout(title={"text": "Confusion Matrix"}, font=my_font).show()
