"""Manage MLflow credentials and experiments"""
import os
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import yaml
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.utils.models.plotoutputs import plot_confusion_matrix


def config_mlflow() -> None:
    """Configure MLflow to log metrics to the Dagshub repository

    Args:
        experiment_name (str): Name of the experiment in MLflow
    """
    load_dotenv()
    os.environ["MLFLOW_TRACKING_USERNAME"] = "pedrochitarra"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri("https://dagshub.com/pedrochitarra/"
                            "indicators-of-heart-disease.mlflow")

    mlflow.autolog()
    np.random.seed(2506)
    random.seed(2506)
    os.environ['PYTHONHASHSEED'] = str(2506)


def create_mlflow_experiment(experiment_name: str) -> None:
    """Create a MLFlow experiment. First set the MLFlow credentials using
    config_mlflow() and then create the experiment."""
    config_mlflow()
    # First create the experiment if it doesn't exist
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(experiment_name)


def register_best_model(model_family: str, loss_function: str) -> None:
    """Register the best model after the optimization process.

    Args:
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
    """
    client = MlflowClient()
    # Select the model with the lowest loss_function
    experiment = client.get_experiment_by_name(f"{model_family}_experiment")
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{loss_function} DESC"])[0]

    # Register the best model
    run_id = best_run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/",
                          f"{model_family}_best_model")


def register_best_experiment(
        x_train: ArrayLike, y_train: ArrayLike,
        model_family: str, loss_function: str,
        best_params: dict) -> str:
    """Register the best experiment found by the optimization process.

    Args:
        params: Hyperparameter dictionary for the given model.

    Returns:
        run_id: The ID of the run in MLflow.
    """
    total_samples_size = x_train.shape[0]
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=2506)

    # Load constants/categorical_features from params.yaml
    with open("params.yaml", encoding="utf-8") as file:
        dvc_params = yaml.safe_load(file)

    best_params["cat_features"] = dvc_params["categorical_features"]

    if model_family == 'catboost':
        model = CatBoostClassifier(**best_params, verbose=False)
        # Train the model
        model.fit(x_train, y_train, eval_set=(x_val, y_val))
    elif model_family == 'xgboost':
        model = XGBClassifier(**best_params)
        # Train the model
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    # Evaluate the model (replace with your desired metrics)
    test_hat = model.predict(x_val)
    train_hat = model.predict(x_train)

    plt.switch_backend("agg")
    with mlflow.start_run() as run:
        # Log params
        mlflow.log_params(best_params)
        # Log metrics
        mlflow.log_metric("accuracy", accuracy_score(y_val, test_hat))
        mlflow.log_metric("f1", f1_score(y_val, test_hat, pos_label="Yes"))
        mlflow.log_metric("precision", precision_score(
            y_val, test_hat, pos_label="Yes"))
        mlflow.log_metric("train_accuracy", accuracy_score(y_train, train_hat))
        mlflow.log_metric("train_f1", f1_score(
            y_train, train_hat, pos_label="Yes"))
        mlflow.log_metric("train_precision", precision_score(
            y_train, train_hat, pos_label="Yes"))
        mlflow.log_param("loss_function", loss_function)
        mlflow.log_param("total_samples_size", total_samples_size)

        # Log the model
        if model_family == "catboost":
            mlflow.catboost.log_model(model, "model")

        # Plot train matrix confusion =========================================
        plot_confusion_matrix(y_train, train_hat, "train")
        mlflow.log_artifact("train_confusion_matrix.png")
        # Plot test matrix confusion ==========================================
        plot_confusion_matrix(y_val, test_hat, "test")
        mlflow.log_artifact("test_confusion_matrix.png")
        # Delete the files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        # Log the dataset and the params.yaml file ============================
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id


def load_model_by_name(model_name: str):
    """
    Loads a pre-trained model from an MLflow server.

    This function connects to an MLflow server using the provided tracking URI,
    username, and password.
    It retrieves the latest version of the model_name model registered on
    the server.
    The function then loads the model using the specified run ID and returns
    the loaded model.

    Args:
        model_name: The name of the model to load.

    Returns:
        loaded_model: The loaded pre-trained model.
    """
    config_mlflow()
    client = mlflow.MlflowClient()
    registered_model = client.get_registered_model(model_name)
    run_id = registered_model.latest_versions[-1].run_id
    logged_model = f"runs:/{run_id}/model"
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model
