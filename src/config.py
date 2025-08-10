"""Configuration settings for the house price prediction project."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Data files
RAW_DATA_PATH = DATA_DIR / "train.csv"
PROCESSED_DATA_PATH = DATA_DIR / "df_model_ready.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# XGBoost parameters
XGBOOST_PARAMS = {
    'n_estimators': 1000,
    'learning_rate': 0.005,
    'max_depth': 2,
    'min_child_weight': 30,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'reg_alpha': 25,
    'reg_lambda': 25,
    'gamma': 10,
    'random_state': RANDOM_STATE,
    'eval_metric': 'rmse'
}

# Feature lists
NONE_FEATURES = {
    'PoolQC': 'No pool',
    'MiscFeature': 'No miscellaneous feature',
    'Alley': 'No alley access',
    'Fence': 'No fence',
    'FireplaceQu': 'No fireplace',
    'GarageType': 'No garage',
    'GarageFinish': 'No garage',
    'GarageQual': 'No garage',
    'GarageCond': 'No garage',
    'BsmtQual': 'No basement',
    'BsmtCond': 'No basement',
    'BsmtExposure': 'No basement',
    'BsmtFinType1': 'No basement',
    'BsmtFinType2': 'No basement',
    'MasVnrType': 'No masonry veneer'
}

ZERO_FEATURES = [
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
    'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'
]

FINAL_FEATURES = [
    'TotalSF', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF',
    'OverallQual', 'OverallCond', 'ExterQual_num', 'KitchenQual_num',
    'HouseAge', 'YearsSinceRemodel',
    'TotalBath', 'GarageCars', 'GarageArea',
    'Fireplaces', 'HasBasement', 'HasGarage', 'LotArea'
]