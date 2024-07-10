"""Script to create Catboost Models for the dataset"""
import os

import mlflow
import dagshub
import yaml
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv


def config_mlflow():
    """Configure MLflow to log metrics to the Dagshub repository"""
    os.environ["MLFLOW_TRACKING_USERNAME"] = "pedrochitarra"
    mlflow.set_tracking_uri("https://dagshub.com/pedrochitarra/"
                            "indicators-of-heart-disease.mlflow")
    dagshub.init("indicators-of-heart-disease", "pedrochitarra", mlflow=True)
    os.environ["DAGSHUB_TOKEN"] = os.getenv("DAGSHUB_TOKEN")

    mlflow.set_experiment("catboost-model")


def objective(x_train: pd.DataFrame, y_train: pd.Series, params: dict):
    """Objective function for hyperopt optimization.
    Args:
        params (dict): Hyperparameters for the Catboost model
    Returns:
        dict: Dictionary containing the accuracy score and status
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=2506)

    print("x_train", x_train)
    # Load constants/categorical_features from params.yaml
    with open("params.yaml", encoding="utf-8") as file:
        dvc_params = yaml.safe_load(file)

    params["cat_features"] = dvc_params["categorical_features"]
    with mlflow.start_run(run_name='catboost-model'):
        rf = CatBoostClassifier(**params)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_val)
        y_val_list = y_val.copy().values.tolist()
        accuracy = accuracy_score(y_val_list, y_pred)
        f1 = f1_score(y_val_list, y_pred, pos_label="Yes")
        precision = precision_score(y_val_list, y_pred, pos_label="Yes")
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)

    return {'loss': 1-accuracy, 'status': STATUS_OK}


def run_optimization(num_trials: int):
    """Optimize hyperparameters for a Catboost model using Hyperopt.
    Args:
        num_trials (int): The number of parameter evaluations for the
            optimizer to explore
    """
    search_space = {
        'border_count': scope.int(hp.choice(
            'max_depth', [32, 5, 10, 20, 50, 100, 200])),
        'depth': scope.int(hp.quniform('depth', 1, 10, 1)),
        'iterations': scope.int(hp.choice('iterations',
                                          [250, 100, 500, 1000])),
        'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 100, 1),
        'learning_rate': hp.quniform('learning_rate', 0.001, 0.3, 0.01),
        'random_state': 2506
    }
    # Load the training dataset
    x_train = pd.read_parquet("data/processed/heart_train_cleaned.parquet")
    y_train = x_train.pop("HadHeartAttack")

    # For reproducible results
    rstate = np.random.default_rng(2506)

    fmin(
        fn=lambda params: objective(x_train, y_train, params),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


if __name__ == '__main__':
    load_dotenv()
    # Load params.yaml file
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))["modeling"]
    n_trials = params["n_trials"]
    config_mlflow()
    run_optimization(n_trials)
