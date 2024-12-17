# Define the objective function for Optuna
def objective(trial, model_name, X, y):
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
        if kernel_choice == "RBF":
            kernel = GaussianProcessRegressor().kernel_.RBF(length_scale=sampled_params.pop("length_scale", 1.0))
        elif kernel_choice == "Matern":
            kernel = GaussianProcessRegressor().kernel_.Matern(length_scale=sampled_params.pop("length_scale", 1.0), nu=1.5)
        elif kernel_choice.startswith("DotProduct"):
            sigma_0 = sampled_params.pop("sigma_0", 1.0)
            kernel = GaussianProcessRegressor().kernel_.DotProduct(sigma_0=sigma_0)
        else:
            raise ValueError(f"Unsupported kernel choice: {kernel_choice}")
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

# Function to perform hyperparameter tuning and model training
def tune_and_train_models(model_hyperparameters, feature_sets, y_train, y_val, n_trials=50):
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
                                            sampler=TPESampler(seed=SEED),
                                            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0))

                # Optimize
                study.optimize(lambda trial: objective(trial, model_name, X_cv, y_cv),
                              n_trials=n_trials, n_jobs=1)

                print(f"\nBest trial for {key}:")
                print(f"  R²: {study.best_trial.value:.4f}")
                print(f"  Params: {study.best_trial.params}")

                # Retrieve best hyperparameters
                best_params = study.best_trial.params.copy()

                # Special handling for GaussianProcessRegressor
                if model_name == "GaussianProcessRegressor":
                    kernel_choice = best_params.pop("kernel")
                    if kernel_choice == "RBF":
                        kernel = GPR_kernel = GaussianProcessRegressor().kernel_.RBF(length_scale=1.0)
                        best_params['kernel'] = kernel
                    elif kernel_choice == "Matern":
                        kernel = GPR_kernel = GaussianProcessRegressor().kernel_.Matern(length_scale=1.0, nu=1.5)
                        best_params['kernel'] = kernel
                    elif kernel_choice.startswith("DotProduct"):
                        sigma_0 = best_params.pop("sigma_0", 1.0)
                        kernel = GPR_kernel = GaussianProcessRegressor().kernel_.DotProduct(sigma_0=sigma_0)
                        best_params['kernel'] = kernel
                    else:
                        raise ValueError(f"Unsupported kernel choice: {kernel_choice}")

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

# ==============================
# 8. Execute Hyperparameter Tuning and Model Training
# ==============================

# Define target variables
y_train = train_df[target].values
y_val = val_df[target].values
y_test = test_df[target].values

# Perform hyperparameter tuning and train models
best_models, models_val_predictions, models_test_predictions = tune_and_train_models(
    model_hyperparameters=model_hyperparameters,
    feature_sets=feature_sets,
    y_train=y_train,
    y_val=y_val,
    n_trials=50
)

# ==============================
# 9. Evaluation and Visualization
# ==============================

# Collect all predictions into a list for ensemble
def collect_predictions(models_test_predictions):
    """
    Collect predictions from all models for ensembling.
    """
    predictions = defaultdict(list)
    for model_feature, subgroups in models_test_predictions.items():
        for subgroup, preds in subgroups.items():
            key = f"{model_feature}_{subgroup}"
            predictions[key].append(preds)
    return predictions

# Calculate R² and RMSE for each model
metrics_list = []
for model_feature, subgroups in models_test_predictions.items():
    for subgroup, preds in subgroups.items():
        metrics = compute_metrics(y_test, preds)
        metrics['Model'] = model_feature
        metrics['Category'] = subgroup
        metrics_list.append(metrics)

# Create a DataFrame for metrics
metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df[['Model', 'Category', 'R2', 'RMSE', 'MAE']]

# Display metrics table
print("\n=== Model Performance Metrics ===\n")
print(tabulate(metrics_df, headers='keys', tablefmt='pretty', floatfmt=".4f"))
