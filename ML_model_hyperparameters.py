from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

model_hyperparameters = {
    "GaussianProcessRegressor": {
        "model_class": GaussianProcessRegressor,
        "tunable_params": {
            "kernel": lambda trial: trial.suggest_categorical(
                "kernel", ["RBF", "Matern", "DotProduct_1", "DotProduct_2"]
            ),
            "alpha": lambda trial: trial.suggest_float("alpha", 1e-3, 1e-1, log=True),
            "n_restarts_optimizer": lambda trial: trial.suggest_int("n_restarts_optimizer", 0, 10),
            "normalize_y": lambda trial: trial.suggest_categorical("normalize_y", [True, False]),
            "optimizer": lambda trial: trial.suggest_categorical("optimizer", ["fmin_l_bfgs_b", "None"]),
            "copy_X_train": lambda trial: trial.suggest_categorical("copy_X_train", [True, False]),
            "random_state": lambda trial: SEED
        },
        "fixed_params": {}
    },
    "MLPRegressor": {
        "model_class": MLPRegressor,  # Corrected to MLPRegressor
        "tunable_params": {
            "hidden_layer_sizes": lambda trial: trial.suggest_categorical(
                "hidden_layer_sizes",
                [(50,), (100,), (100, 50), (150, 100, 50), (200, 100), (300, 150, 50)]
            ),
            "activation": lambda trial: trial.suggest_categorical("activation", ['relu', 'tanh', 'logistic']),
            "solver": lambda trial: trial.suggest_categorical("solver", ['adam', 'sgd']),
            "alpha": lambda trial: trial.suggest_float("alpha", 1e-4, 1e-2, log=True),
            "learning_rate": lambda trial: trial.suggest_categorical("learning_rate", ['constant', 'adaptive']),
            "learning_rate_init": lambda trial: trial.suggest_float("learning_rate_init", 1e-5, 1e-1, log=True),
            "batch_size": lambda trial: trial.suggest_categorical("batch_size", [16, 32, 64, 128, 'auto']),
            "momentum": lambda trial: trial.suggest_float("momentum", 0.0, 1.0) if trial.params.get("solver") == 'sgd' else 0.9,
            "nesterovs_momentum": lambda trial: trial.suggest_categorical("nesterovs_momentum", [True, False]) if trial.params.get("solver") == 'sgd' else False,
            "beta_1": lambda trial: trial.suggest_float("beta_1", 0.8, 0.999) if trial.params.get("solver") == 'adam' else 0.9,
            "beta_2": lambda trial: trial.suggest_float("beta_2", 0.9, 0.9999) if trial.params.get("solver") == 'adam' else 0.999,
            "power_t": lambda trial: trial.suggest_float("power_t", 0.1, 0.9) if trial.params.get("solver") == 'sgd' else 0.5,
            "early_stopping": lambda trial: trial.suggest_categorical("early_stopping", [True, False]),
            "validation_fraction": lambda trial: trial.suggest_float("validation_fraction", 0.1, 0.3),
            "tol": lambda trial: trial.suggest_float("tol", 1e-5, 1e-2, log=True)
        },
        "fixed_params": {
            'random_state': SEED,
            'max_iter': 1000
        }
    },
    "XGBoost": {
        "model_class": XGBRegressor,  # Corrected to XGBRegressor
        "tunable_params": {
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": lambda trial: trial.suggest_int("max_depth", 3, 20),
            "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": lambda trial: trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": lambda trial: trial.suggest_float("gamma", 0, 1),
            "min_child_weight": lambda trial: trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": lambda trial: trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": lambda trial: trial.suggest_float("reg_lambda", 0, 1)
        },
        "fixed_params": {
            'random_state': SEED,
            'verbosity': 0
        }
    },
    "RandomForest": {
        "model_class": RandomForestRegressor,
        "tunable_params": {
            "n_estimators": lambda trial: trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": lambda trial: trial.suggest_int("max_depth", 5, 50),
            "min_samples_split": lambda trial: trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": lambda trial: trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": lambda trial: trial.suggest_categorical("max_features", ['sqrt', 'log2', None])
        },
        "fixed_params": {
            'random_state': SEED,
            'n_jobs': -1
        }
    },
    "CatBoost": {
        "model_class": CatBoostRegressor,  # Corrected to CatBoostRegressor
        "tunable_params": {
            "iterations": lambda trial: trial.suggest_int("iterations", 100, 1000),
            "depth": lambda trial: trial.suggest_int("depth", 3, 10),
            "learning_rate": lambda trial: trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": lambda trial: trial.suggest_float("l2_leaf_reg", 0, 10),
            "subsample": lambda trial: trial.suggest_float("subsample", 0.5, 1.0),
            "random_strength": lambda trial: trial.suggest_float("random_strength", 0, 1)
        },
        "fixed_params": {
            'logging_level': 'Silent',
            'random_state': SEED
        }
    },
    "SVR": {
        "model_class": SVR,
        "tunable_params": {
            "C": lambda trial: trial.suggest_float("C", 0.1, 100, log=True),
            "epsilon": lambda trial: trial.suggest_float("epsilon", 0.001, 1, log=True),
            "gamma": lambda trial: trial.suggest_categorical("gamma", ['scale', 'auto'] + list(np.logspace(-4, 1, 6)))
        },
        "fixed_params": {}
    },
    "ElasticNet": {
        "model_class": ElasticNet,
        "tunable_params": {
            "alpha": lambda trial: trial.suggest_float("alpha", 0.001, 1),
            "l1_ratio": lambda trial: trial.suggest_float("l1_ratio", 0, 1)
        },
        "fixed_params": {
            'random_state': SEED,
            'max_iter': 10000
        }
    }
}
