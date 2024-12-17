import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation

def perform_label_propagation(train_df, val_df, test_df, target='log_humanVDssL/kg',
                              features_to_test=None, subgroups=None):
    if features_to_test is None:
        features_to_test = ['log_MW', 'log_HBA', 'log_TPSANO', 'moka_ionState7.4_encoded', 'MoKa.LogP', 'MoKa.LogD7.4']
    if subgroups is None:
        subgroups = [0.5, 1.7]
    
    # Standardization
    scaler = StandardScaler()
    train_df[features_to_test] = scaler.fit_transform(train_df[features_to_test])
    val_df[features_to_test] = scaler.transform(val_df[features_to_test])
    test_df[features_to_test] = scaler.transform(test_df[features_to_test])
    
    # Define subgroups
    train_df['subgroup'] = train_df[target].apply(
        lambda x: 0 if x < subgroups[0] else (1 if subgroups[0] <= x < subgroups[1] else 2)
    )
    val_df['subgroup'] = np.nan
    test_df['subgroup'] = np.nan
    
    # Label Propagation
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.loc[:len(train_df) - 1, 'subgroup'] = train_df['subgroup'].values
    
    X_combined = combined_df[features_to_test]
    y_combined = combined_df['subgroup'].fillna(-1)
    
    label_spread = LabelPropagation(kernel='rbf', gamma=0.9, max_iter=20, n_neighbors=20)
    label_spread.fit(X_combined, y_combined)
    combined_df['spread_subgroup'] = label_spread.predict(X_combined)
    
    # Create binary indicators
    for i, cls in enumerate(label_spread.classes_):
        combined_df[f'subgroup{i+1}'] = (combined_df['spread_subgroup'] == cls).astype(int)
    
    # Split data
    train_len, val_len, test_len = len(train_df), len(val_df), len(test_df)
    train_df['subgroup'] = combined_df.iloc[:train_len]['spread_subgroup'].values
    val_df['subgroup'] = combined_df.iloc[train_len:train_len + val_len]['spread_subgroup'].values
    test_df['subgroup'] = combined_df.iloc[train_len + val_len:train_len + val_len + test_len]['spread_subgroup'].values
    
    for i in range(len(label_spread.classes_)):
        train_df[f'subgroup{i+1}'] = combined_df.iloc[:train_len][f'subgroup{i+1}'].values
        val_df[f'subgroup{i+1}'] = combined_df.iloc[train_len:train_len + val_len][f'subgroup{i+1}'].values
        test_df[f'subgroup{i+1}'] = combined_df.iloc[train_len + val_len:train_len + val_len + test_len][f'subgroup{i+1}'].values
    
    return train_df, val_df, test_df
