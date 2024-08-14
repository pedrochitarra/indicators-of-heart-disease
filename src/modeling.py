"""Script to create Catboost Models for the dataset"""
import yaml
import pandas as pd

from dotenv import load_dotenv

from src.utils.mlflow.manage_mlflow import (create_mlflow_experiment,
                                            register_best_model,
                                            register_best_experiment)
from src.utils.models.hpo import classification_optimization


def main():
    """Main function to run the optimization process. It loads the training
    dataset, optimizes the hyperparameters, registers the best experiment,
    and registers the best model."""
    load_dotenv()
    # Load params.yaml file
    modeling_params = yaml.safe_load(
        open("params.yaml", encoding="utf-8"))["modeling"]
    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]
    create_mlflow_experiment(f"{selected_model_family}_experiment")

    # Load the training dataset
    df_train_heart = pd.read_parquet(
        "data/processed/heart_train_cleaned.parquet")

    print(df_train_heart.columns.to_list())

    x_train_heart = df_train_heart.copy()
    x_train_heart = x_train_heart.drop(columns=["HadHeartAttack"])

    y_train_heart = df_train_heart.copy()["HadHeartAttack"]

    best_classification_params = classification_optimization(
        x_train=x_train_heart, y_train=y_train_heart,
        model_family=selected_model_family,
        loss_function=selected_loss_function,
        objective_function=selected_objective_function,
        num_trials=n_trials, diagnostic=True)
    register_best_experiment(
        x_train=x_train_heart, y_train=y_train_heart,
        model_family=selected_model_family,
        loss_function=selected_loss_function,
        best_params=best_classification_params)
    register_best_model(
        model_family=selected_model_family,
        loss_function=selected_loss_function)


if __name__ == "__main__":
    main()
