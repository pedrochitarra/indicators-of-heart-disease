"""Modeling Optimization with Hyperopt"""
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from numpy.typing import ArrayLike
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope


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
