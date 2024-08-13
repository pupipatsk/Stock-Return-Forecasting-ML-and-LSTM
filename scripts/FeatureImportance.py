# %% [markdown]
# # Feature Importance
#
# - Parameters depends on each model
# - return a dataframe sorted by feature importance
# - Example usages at the last part
#
# Note: feature importance of KNN and NN models can not be done directly (hard to implement and depends on the size of the data)

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

import matplotlib.pyplot as plt

import joblib

import torch
import torch.nn as nn

# import tensorflow as tf
# from innvestigate.utils.keras import checks
# from innvestigate.utils.keras import checks as kchecks
# from innvestigate.utils.keras import backend as kb
# from innvestigate.utils.keras import applications as kapp
# from innvestigate import create_analyzer

# %% [markdown]
# ## Functions for Feature Importance

# %%
def get_feature_importance(model, feature_names=None):
    if hasattr(model, 'coef_'):
        if feature_names is None or len(feature_names) != len(model.coef_[0]):
            feature_names = [f'Feature_{i}' for i in range(len(model.coef_[0]))]

        feature_importance = {feature_names[i]: coef for i, coef in enumerate(model.coef_[0])}
        df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        df = df.abs()
        df = df.sort_values(by='Importance', ascending=False)
        return df

    else:
        print("Warning: Model doesn't have coef_ attribute. Feature importance cannot be extracted.")
        return None


def get_svc_feature_importance(svm_model, feature_names=None):
    if hasattr(svm_model, 'coef_'):
        if feature_names is None or len(feature_names) != len(svm_model.coef_[0]):
            feature_names = [f'Feature_{i}' for i in range(len(svm_model.coef_[0]))]

        feature_importance = {feature_names[i]: coef for i, coef in enumerate(svm_model.coef_[0])}
        df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        df = df.abs()
        df = df.sort_values(by='Importance', ascending=False)
        return df

    elif svm_model.kernel == 'linear':
        print("Warning: Model doesn't have coef_ attribute. Feature importance cannot be extracted.")
        return None

    else:
        print("Warning: This SVM model type doesn't support direct feature importance extraction.")
        return None


def get_svr_feature_importance(model, X, y, scoring='neg_mean_squared_error', feature_names=None):
    results = permutation_importance(model, X, y, scoring=scoring)

    importance = results.importances_mean

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def get_xgboost_feature_importance(model, feature_names):
    importance = model.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df

def get_rf_feature_importance(model, feature_names=None):
    if hasattr(model, 'feature_importances_'):
        if feature_names is None or len(feature_names) != len(model.feature_importances_):
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]

        feature_importance = {feature_names[i]: importance for i, importance in enumerate(model.feature_importances_)}
        df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
        df = df.sort_values(by='Importance', ascending=False)
        return df

    else:
        print("Warning: Model doesn't have feature_importances_ attribute. Feature importance cannot be extracted.")
        return None

def get_lstm_feature_importance(model_file, feature_names, device="cpu"):
    state_dict = torch.load(model_file, map_location=torch.device(device))
    model_state_dict = state_dict['model']

    hidden_size = 256
    input_size = 1
    num_layers = 4
    lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    linear1 = nn.Linear(hidden_size, 64)
    linear2 = nn.Linear(64, 1)

    # Depends on the architecture
    lstm_weight_ih = model_state_dict['layer1.weight_ih_l0'].T  # Shape: (hidden_size*4, input_size)
    lstm_weight_hh = model_state_dict['layer1.weight_hh_l0'].T  # Shape: (hidden_size*4, hidden_size)
    lstm_bias_ih = model_state_dict['layer1.bias_ih_l0']  # Shape: (hidden_size*4,)
    lstm_bias_hh = model_state_dict['layer1.bias_hh_l0']  # Shape: (hidden_size*4,)
    linear1_weight = model_state_dict['layer2.0.weight'].T  # Shape: (64, hidden_size)
    linear1_bias = model_state_dict['layer2.0.bias']  # Shape: (64,)
    linear2_weight = model_state_dict['layer3.weight'].T  # Shape: (1, 64)
    linear2_bias = model_state_dict['layer3.bias']  # Shape: (1,)

    importance1 = np.abs(np.matmul(lstm_weight_ih, np.diag(lstm_weight_hh.flatten())) + lstm_bias_ih + lstm_bias_hh)
    importance2 = np.abs(np.matmul(linear1_weight, linear2_weight.flatten())) + linear1_bias + linear2_bias
    importance = np.concatenate((importance1.flatten(), importance2))

    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    return feature_importance_df


# %% [markdown]
# # Example usages

# %% [markdown]
# ## Linear/Logistic Regression

# %%
# feature_names = ["Open", "High", "Low", "Close", "Volume"]
# feature_importance_df = get_regression_feature_importance(model, feature_names)
# print(feature_importance_df)

# %% [markdown]
# ## SVC

# %%
# feature_names =  ["Open", "High", "Low", "Close", "Volume"]
# feature_importance_df = get_svc_feature_importance(model, feature_names)
# print(feature_importance_df)

# %% [markdown]
# ## SVR

# %%
# model = SVR() something
# permutation_importance_df = get_permutation_importance_svr(model, X, y, feature_names=['Feature1', 'Feature2', ...])
# print(permutation_importance_df)

# %% [markdown]
# ## XGBoost (for both Classifier and Regressor)

# %%
# feature_importance = get_xgboost_feature_importance(xgb_model, feature_names)
# print("Ranked Feature Importance:")
# print(feature_importance)

# %% [markdown]
# ## Random Forest (for both Classifier and Regressor)

# %%
# model = RandomForestClassifier()
# feature_importance_df = get_rf_feature_importance(model, feature_names=['Feature1', 'Feature2', ...])
# print(feature_importance_df)

# %% [markdown]
# ## LSTM

# %%
# model_file_path = 'models/LSTM1.pth.tar'
# feature_names = ["Open", "High", "Low", "Close", "Volume"]

# feature_importance = get_lstm_feature_importance_from_file(model_file_path, feature_names)
# print("Ranked Feature Importance:")
# print(feature_importance)


