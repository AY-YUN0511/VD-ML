import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import KFold
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os

def define_objective(trial, conv_type, num_layers_options, emb_dim_options, drop_ratio_options, lr_options, weight_decay_options, 
                    train_val_df, target_column, device, SEED):
    """
    Define the objective function for Optuna hyperparameter optimization.
    This function performs 5-Fold CV and returns the mean R² across folds.
    """
    # Suggest hyperparameters
    num_layers = trial.suggest_categorical("num_layers", num_layers_options)
    emb_dim = trial.suggest_categorical("emb_dim", emb_dim_options)
    drop_ratio = trial.suggest_float("drop_ratio", *drop_ratio_options)
    lr = trial.suggest_float("lr", *lr_options, log=True)
    weight_decay = trial.suggest_float("weight_decay", *weight_decay_options, log=True)

    # Initialize model
    model = GraphOnlyGNN(
        num_layers=num_layers,
        emb_dim=emb_dim,
        input_dim=2048,  # ECFP has 2048 bits
        conv_type=conv_type,
        drop_ratio=drop_ratio
    ).to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    # Perform 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    val_r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_df)):
        # Split data
        train_fold_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        # Prepare DataLoaders
        train_loader = prepare_dataloader(
            train_fold_df, target_column=target_column, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(SEED)
        )
        val_loader = prepare_dataloader(
            val_fold_df, target_column=target_column, batch_size=32, shuffle=False, generator=torch.Generator().manual_seed(SEED)
        )

        # Training Loop with Early Stopping
        best_val_r2 = -np.inf
        patience = 10
        patience_counter = 0
        num_epochs = 100
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs.squeeze(), batch.y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_losses = []
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(batch)
                    loss = criterion(outputs.squeeze(), batch.y)
                    val_losses.append(loss.item())
                    val_predictions.extend(outputs.squeeze().cpu().numpy())
                    val_targets.extend(batch.y.cpu().numpy())

            # Calculate R²
            val_r2 = r2_score(val_targets, val_predictions)
            val_r2_scores.append(val_r2)

            # Early Stopping Check
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load the best model state
        model.load_state_dict(best_model_state)

        # Evaluate on validation fold
        _, val_r2, _, _ = evaluate_model(model, val_loader, criterion, device)
        val_r2_scores.append(val_r2)

    # Calculate mean R² across folds
    mean_val_r2 = np.mean(val_r2_scores)
    return mean_val_r2

def perform_hyperparameter_tuning(conv_type, num_trials=50, train_val_df=None, target_column='log_Vdss',
                                  device='cpu', SEED=42):
    """
    Perform hyperparameter tuning using Optuna for a given convolution type.
    """
    # Define hyperparameter search space
    num_layers_options = [3, 4, 5, 6]
    emb_dim_options = [64, 128, 256]
    drop_ratio_options = (0.2, 0.5)
    lr_options = (1e-4, 1e-2)
    weight_decay_options = (1e-5, 1e-3)

    # Define the objective function with fixed conv_type
    def objective(trial):
        return define_objective(
            trial=trial,
            conv_type=conv_type,
            num_layers_options=num_layers_options,
            emb_dim_options=emb_dim_options,
            drop_ratio_options=drop_ratio_options,
            lr_options=lr_options,
            weight_decay_options=weight_decay_options,
            train_val_df=train_val_df,
            target_column=target_column,
            device=device,
            SEED=SEED
        )

    # Create Optuna study
    sampler = TPESampler(seed=SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=num_trials, timeout=None)

    print(f"\nBest trial for Conv Type {conv_type}:")
    print(f"  R²: {study.best_trial.value:.4f}")
    print(f"  Params: {study.best_trial.params}")

    return study.best_trial.params

def train_final_model(params, conv_type):
    """
    Train the final model on the entire training set using the best hyperparameters.
    """
    # Initialize model with best hyperparameters
    model = GraphOnlyGNN(
        num_layers=params['num_layers'],
        emb_dim=params['emb_dim'],
        input_dim=2048,  # ECFP has 2048 bits
        conv_type=conv_type,
        drop_ratio=params['drop_ratio']
    ).to(device)

    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    criterion = nn.MSELoss()

    # Training Loop with Early Stopping
    best_val_r2 = -np.inf
    patience = 10
    patience_counter = 0
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train_model(model, optimizer, criterion, train_loader, device)
        val_loss, val_r2, _, _ = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R²: {val_r2:.4f}")

        # Early Stopping
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0
            # Save the best model state
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load the best model state
    model.load_state_dict(best_model_state)

    # Evaluate on Test Set
    test_loss, test_r2, y_test_true, y_test_pred = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Test R²: {test_r2:.4f}")

    # Save the final model
    model_save_path = os.path.join(save_dir, f'final_model_{conv_type}.pt')
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    return test_r2, y_test_true, y_test_pred
