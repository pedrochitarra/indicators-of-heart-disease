"""Script to create Catboost Models for the dataset"""
import os
import random

import mlflow
from mlflow.tracking import MlflowClient
from numpy.typing import ArrayLike
import yaml
import pandas as pd
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from dotenv import load_dotenv
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


def config_mlflow(experiment_name: str) -> None:
    """Configure MLflow to log metrics to the Dagshub repository

    Args:
        experiment_name (str): Name of the experiment in MLflow
    """
    load_dotenv()
    os.environ["MLFLOW_TRACKING_USERNAME"] = "pedrochitarra"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN")
    mlflow.set_tracking_uri("https://dagshub.com/pedrochitarra/"
                            "indicators-of-heart-disease.mlflow")

    # First create the experiment if it doesn't exist
    try:
        mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        pass
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    np.random.seed(2506)
    random.seed(2506)
    os.environ['PYTHONHASHSEED'] = str(2506)


def classification_objective(x_train: ArrayLike, y_train: ArrayLike,
                             model_family: str, loss_function: str,
                             params: dict) -> dict:
    """Trainable function for classification models.

    Args:
        x_train (ArrayLike): Training features
        y_train (ArrayLike): Training target
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
        params: Hyperparameter dictionary for the given model.

    Returns:
        A dictionary containing the metrics from training.
    """
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=2506)

    # Load constants/categorical_features from params.yaml
    with open("params.yaml", encoding="utf-8") as file:
        dvc_params = yaml.safe_load(file)

    params["cat_features"] = dvc_params["categorical_features"]

    if model_family == 'catboost':
        # Cast integer params from float to int
        integer_params = ['depth', 'min_data_in_leaf', 'max_bin']
        for param in integer_params:
            if param in params:
                params[param] = int(params[param])

        # Extract nested conditional parameters
        if params['bootstrap_type']['bootstrap_type'] == 'Bayesian':
            bagging_temp = params['bootstrap_type'].get(
                'bagging_temperature')
            params['bagging_temperature'] = bagging_temp

        if params['grow_policy']['grow_policy'] == 'LossGuide':
            max_leaves = params['grow_policy'].get('max_leaves')
            params['max_leaves'] = int(max_leaves)

        params['bootstrap_type'] = params['bootstrap_type'][
            'bootstrap_type']
        params['grow_policy'] = params['grow_policy']['grow_policy']

        # Random_strength cannot be < 0
        params['random_strength'] = max(params['random_strength'], 0)
        # fold_len_multiplier cannot be < 1
        params['fold_len_multiplier'] = max(
            params['fold_len_multiplier'], 1)

        model = CatBoostClassifier(**params, verbose=False)
        # Train the model
        model.fit(x_train, y_train, eval_set=(x_val, y_val))
    elif model_family == 'xgboost':
        model = XGBClassifier(**params)
        # Train the model
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)])

    # Predict on the validation set
    y_pred = model.predict(x_val)
    # Calculate the loss
    if loss_function == 'F1':
        loss = 1 - f1_score(y_val, y_pred, pos_label="Yes")
    elif loss_function == 'Accuracy':
        loss = 1 - accuracy_score(y_val, y_pred)
    elif loss_function == 'Precision':
        loss = 1 - precision_score(y_val, y_pred, pos_label="Yes")

    return {'loss': loss, 'status': STATUS_OK}


def classification_optimization(x_train: ArrayLike, y_train: ArrayLike,
                                model_family: str, loss_function: str,
                                objective_function: str,
                                num_trials: int,
                                diagnostic: bool = False) -> dict:
    """Optimize hyperparameters for a model using Hyperopt.
    Args:
        x_train (ArrayLike): Training features
        y_train (ArrayLike): Training target
        model_family (str): Model family to optimize
        loss_function (str): Loss function to optimize
        objective_function (str): Objective function to optimize
        num_trials (int): Number of trials to run
        diagnostic (bool): Whether to print diagnostic information

    Returns:
        dict: Dictionary containing the best hyperparameters
    """
    if model_family == "random_forest":
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
            'min_samples_split': scope.int(hp.quniform(
                'min_samples_split', 2, 10, 1)),
            'min_samples_leaf': scope.int(hp.quniform(
                'min_samples_leaf', 1, 4, 1)),
            'random_state': 2506
        }

    elif model_family == "catboost":
        # Integer and string parameters, used with hp.choice()
        bootstrap_type = [
            # {'bootstrap_type': 'Poisson'},
            {'bootstrap_type': 'Bayesian',
             'bagging_temperature': hp.loguniform('bagging_temperature',
                                                  np.log(1), np.log(50))},
            {'bootstrap_type': 'Bernoulli'}]
        # Remove 'Armijo' if not using GPU
        leb_list = ['No', 'AnyImprovement']
        grow_policy = [{'grow_policy': 'SymmetricTree'},
                       {'grow_policy': 'Depthwise'},
                       {'grow_policy': 'Lossguide',
                        'max_leaves': hp.quniform('max_leaves', 2, 32, 1)}]

        # Maximum tree depth in CatBoost
        search_space = {
            'depth': hp.quniform('depth', 2, 10, 1),
            # If using CPU just set this to 254
            # 'max_bin' : hp.quniform('max_bin', 1, 32, 1),
            'max_bin': 254,
            'l2_leaf_reg': hp.uniform('l2_leaf_reg', 0, 50),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 1, 50, 1),
            'random_strength': hp.loguniform('random_strength',
                                             np.log(0.005), np.log(5)),
            # Uncomment if using categorical features
            # 'one_hot_max_size' : hp.quniform('one_hot_max_size', 2, 16, 1),
            'bootstrap_type': hp.choice('bootstrap_type', bootstrap_type),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.25),
            'eval_metric': loss_function,
            # 'objective': loss_function.upper(),
            'objective': objective_function,
            # crashes kernel - reason unknown
            # 'score_function' : hp.choice('score_function', score_function),
            'leaf_estimation_backtracking': hp.choice(
                'leaf_estimation_backtracking', leb_list),
            'grow_policy': hp.choice('grow_policy', grow_policy),
            # CPU only
            'colsample_bylevel': hp.quniform(
                'colsample_bylevel', 0.1, 1, 0.01),
            'fold_len_multiplier': hp.loguniform(
                'fold_len_multiplier', np.log(1.01), np.log(2.5)),
            'od_type': 'Iter',
            'od_wait': 25,
            'task_type': 'CPU'
        }

    elif model_family == "xgboost":
        search_space = {
            'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
            'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
            'learning_rate': hp.quniform('learning_rate', 0.001, 0.3, 0.01),
            'random_state': 2506
        }

    # For reproducible results
    rstate = np.random.default_rng(2506)
    trials = Trials()
    best_params = fmin(
        fn=lambda params: classification_objective(
            x_train, y_train, model_family, loss_function, params),
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=trials,
        rstate=rstate
    )

    # Unpack nested dicts first
    if model_family == "catboost":
        best_params['bootstrap_type'] = bootstrap_type[
            best_params['bootstrap_type']]['bootstrap_type']
        best_params['grow_policy'] = grow_policy[
            best_params['grow_policy']]['grow_policy']
        best_params['eval_metric'] = loss_function

        best_params['leaf_estimation_backtracking'] = leb_list[
            best_params['leaf_estimation_backtracking']]

        # Cast floats of integer params to int
        integer_params = ['depth', 'min_data_in_leaf', 'max_bin']
        for param in integer_params:
            if param in best_params:
                best_params[param] = int(best_params[param])
        if 'max_leaves' in best_params:
            best_params['max_leaves'] = int(best_params['max_leaves'])

        print('{' + '\n'.join('{}: {}'.format(k, v)
              for k, v in best_params.items()) + '}')

        if diagnostic:
            for i, trial in enumerate(trials.trials):
                print(f"Trial # {i} result: {trial['result']['loss']}")

    return best_params


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
        cm = confusion_matrix(y_train, train_hat)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Include the number of samples in each cell
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                         color='red')
        plt.savefig("train_confusion_matrix.png")
        mlflow.log_artifact("train_confusion_matrix.png")
        # Plot test matrix confusion ==========================================
        cm = confusion_matrix(y_val, test_hat)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix of the classifier')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Include the number of samples in each cell
        for i in range(2):
            for j in range(2):
                plt.text(j, i, f"{cm[i, j]}", ha='center', va='center',
                         color='red')
        plt.savefig("test_confusion_matrix.png")
        mlflow.log_artifact("test_confusion_matrix.png")
        # Delete the files
        os.remove("train_confusion_matrix.png")
        os.remove("test_confusion_matrix.png")
        # Log the dataset
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("dvc.yaml")
        mlflow.log_artifact("data/processed/heart_train_cleaned.parquet")

    return run.info.run_id


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


if __name__ == '__main__':
    load_dotenv()
    # Load params.yaml file
    modeling_params = yaml.safe_load(
        open("params.yaml", encoding="utf-8"))["modeling"]
    n_trials = modeling_params["n_trials"]
    selected_loss_function = modeling_params["loss_function"]
    selected_model_family = modeling_params["model_family"]
    selected_objective_function = modeling_params["objective_function"]
    config_mlflow(f"{selected_model_family}_experiment")

    # Load the training dataset
    df_train_heart = pd.read_parquet(
        "data/processed/heart_train_cleaned.parquet")

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
