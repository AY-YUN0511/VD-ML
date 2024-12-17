# VD-ML

# Enhancing Volume of Distribution Predictions with Subgroup-Based Machine Learning Techniques

![Project Overview](https://github.com/user-attachments/assets/7ee508f3-1aa2-45d9-bbcb-3229aeacea0d)


> This repository contains a comprehensive machine learning (ML) and Graph Neural Network (GNN) pipeline designed to predict molecular properties based on SMILES representations. A standout feature of this pipeline is the integration of subgroup analysis, which enhances model performance by incorporating probabilistic subgroup information derived from the dataset. The pipeline encompasses data preprocessing, feature extraction, model training with hyperparameter tuning, evaluation, and visualization. It leverages various ML models and advanced GNN architectures to achieve robust and accurate predictions.

<br><br>
## Dataset

The dataset used in this project is sourced from the following paper:

> **Lombardo, F., Berellini, G., & Obach, R. S. (2018).** Trend Analysis of a Database of Intravenous Pharmacokinetics Parameters in Humans for 1352 Drug Compounds. *Drug Metab. Dispos.*, **46**, 1466-1477.

This dataset comprises intravenous pharmacokinetic parameters in humans for **1,352 drug compounds**, including features like SMILES representations and various pharmacokinetic measurements. A significant aspect of this dataset is the inclusion of subgroup classifications derived from label propagation techniques, which are instrumental in enhancing model performance.


## Installed Package Versions

Ensure that the following packages are installed with the specified versions for compatibility and optimal performance:

- **PyTorch:** `2.5.1+cu124`
- **Torchvision:** `0.20.1+cu121`
- **Transformers:** `4.46.3`
- **RDKit:** `2024.03.6`
- **DGL:** `1.1.0`
- **DeepChem:** `2.8.0`

## Features

1. **Data Preprocessing:**
   - **Log-transformations** of target variables to normalize distributions.
   - **Label encoding** for categorical features.
   - **Splitting data** into training, validation, and test sets.
   - **Subgroup Analysis:** Assigning probabilistic subgroup features to capture underlying data structures.

2. **Feature Extraction:**
   - **Calculation of molecular descriptors** using RDKit.
   - **Generation of Extended-Connectivity Fingerprints (ECFP)** for molecular graphs.
   - **Scaling and normalization** of features for optimal model performance.
   - **Integration of Subgroup Features:** Combining molecular descriptors and subgroup probabilities to enrich the feature space.

3. **Graph Neural Networks (GNNs):**
   - **Conversion of SMILES strings** to graph representations using RDKit and PyTorch Geometric.
   - **Implementation of various GNN architectures,** including GIN, GAT, GCN, and Transformer-based convolutions.
   - **Training with 5-Fold Cross-Validation** to ensure robust performance.
   - **Enhanced GNNs with Subgroups:** Incorporating subgroup probabilities as additional node or graph-level features to improve learning.

4. **Machine Learning Models:**
   - **Implementation of diverse regression models:** Random Forest, XGBoost, CatBoost, Support Vector Regression (SVR), ElasticNet, and Multi-Layer Perceptron (MLP).
   - **Subgroup Utilization:** Leveraging subgroup probabilities to inform model training and enhance predictive accuracy.
   - **Hyperparameter tuning** using Optuna for optimal model configurations.

5. **Evaluation and Visualization:**
   - **Calculation of key performance metrics:** R², RMSE, and MAE.
   - **Visualization of model performances** through comprehensive plots.
   - **Saving of trained models** and prediction results for future reference.

6. **Ensemble Methods:**
   - **Implementation of weighted average ensemble strategies** to combine predictions from multiple models, enhancing overall accuracy.

---

# License
This project is licensed under the MIT License.
