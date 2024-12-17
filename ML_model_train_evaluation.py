# ml_model_train_evaluation.py

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

def get_kernel(kernel_choice, params):
    """
    Return the kernel based on the kernel choice.
    """
    if kernel_choice == "RBF":
        from sklearn.gaussian_process.kernels import RBF
        return RBF(length_scale=params.get("length_scale", 1.0))
    elif kernel_choice == "Matern":
        from sklearn.gaussian_process.kernels import Matern
        return Matern(length_scale=params.get("length_scale", 1.0), nu=1.5)
    elif kernel_choice.startswith("DotProduct"):
        from sklearn.gaussian_process.kernels import DotProduct
        sigma_0 = params.get("sigma_0", 1.0)
        return DotProduct(sigma_0=sigma_0)
    else:
        raise ValueError(f"Unsupported kernel choice: {kernel_choice}")

def objective(trial, model_name, X, y, model_hyperparameters):
    """
    Objective function for Optuna hyperparameter optimization.
    """
    config = model_hyperparameters[model_name]
    # Sample hyperparameters
    sampled_params = {param: func(trial) for param, func in config['tunable_params'].items()}
    sampled_params.update(config['fixed_params'])

    # Handle special cases
    if model_name == "GaussianProcessRegressor":
        kernel_choice = sampled_params.pop("kernel")
        kernel = get_kernel(kernel_choice, sampled_params)
        sampled_params['kernel'] = kernel

    # Initialize the model
    try:
        model = config['model_class'](**sampled_params)
    except TypeError as e:
        print(f"Error initializing {model_name} with params {sampled_params}: {e}")
        raise

    # Perform cross-validation
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
        mean_r2 = np.mean(cv_scores)
    except Exception as e:
        print(f"Error during cross-validation for {model_name}: {e}")
        raise

    return mean_r2

def tune_and_train_models(model_hyperparameters, feature_sets, y_train, y_val, n_trials=50, seed=42):
    """
    Tune hyperparameters using Optuna and train models.
    """
    best_models = defaultdict(dict)
    models_test_predictions = defaultdict(dict)
    models_val_predictions = defaultdict(dict)

    feature_types = ['ecfp', 'descriptors']
    subgroup_options = [True, False]

    for model_name in model_hyperparameters.keys():
        print(f"\n=== Tuning {model_name} ===")
        for feature_type in feature_types:
            for subgroup in subgroup_options:
                subgroup_status = 'With Subgroup' if subgroup else 'Without Subgroup'
                key = f"{model_name}_{feature_type}_{subgroup_status}"
                print(f"\n--- Configuration: {feature_type.upper()} Features, {subgroup_status} ---")

                # Select feature set
                if feature_type == 'ecfp':
                    if subgroup:
                        X_train_feat, X_val_feat, X_test_feat = feature_sets['ecfp_with_subgroup']
                    else:
                        X_train_feat, X_val_feat, X_test_feat = feature_sets['ecfp_no_subgroup']
                elif feature_type == 'descriptors':
                    if subgroup:
                        X_train_feat, X_val_feat, X_test_feat = feature_sets['descriptors_with_subgroup']
                    else:
                        X_train_feat, X_val_feat, X_test_feat = feature_sets['descriptors_no_subgroup']
                else:
                    raise ValueError(f"Unsupported feature type: {feature_type}")

                # Combine train and validation for cross-validation
                X_cv = np.vstack((X_train_feat, X_val_feat))
                y_cv = np.hstack((y_train, y_val))

                # Create Optuna study
                study = optuna.create_study(direction='maximize',
                                            sampler=TPESampler(seed=seed),
                                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0))

                # Optimize
                study.optimize(lambda trial: objective(trial, model_name, X_cv, y_cv, model_hyperparameters),
                              n_trials=n_trials, n_jobs=1)

                print(f"\nBest trial for {key}:")
                print(f"  R²: {study.best_trial.value:.4f}")
                print(f"  Params: {study.best_trial.params}")

                # Retrieve best hyperparameters
                best_params = study.best_trial.params.copy()

                # Special handling for GaussianProcessRegressor
                if model_name == "GaussianProcessRegressor":
                    kernel_choice = best_params.pop("kernel")
                    kernel = get_kernel(kernel_choice, best_params)
                    best_params['kernel'] = kernel

                # Initialize the best model with best hyperparameters
                try:
                    model = model_hyperparameters[model_name]['model_class'](**best_params)
                except TypeError as e:
                    print(f"Error initializing {model_name} with params {best_params}: {e}")
                    continue

                # Train the model on training data
                model.fit(X_train_feat, y_train)

                # Store the best model
                best_models[model_name][subgroup_status] = model

                # Predict on validation and test sets
                preds_val = model.predict(X_val_feat)
                preds_test = model.predict(X_test_feat)

                # Store predictions
                models_val_predictions[f"{model_name}_{feature_type}"][subgroup_status] = preds_val
                models_test_predictions[f"{model_name}_{feature_type}"][subgroup_status] = preds_test

    return best_models, models_val_predictions, models_test_predictions
