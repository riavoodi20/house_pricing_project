"""Model training and evaluation functions."""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import *

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handle model training, evaluation, and comparison."""

    def __init__(self, data_path: str = PROCESSED_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}

    def load_data(self) -> None:
        """Load processed dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Model data loaded: {self.df.shape}")
        except FileNotFoundError:
            logger.error(f"Processed data not found: {self.data_path}")
            raise

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables."""
        if self.df is None:
            self.load_data()

        X = self.df[FINAL_FEATURES]
        y = self.df['SalePrice']

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        logger.info(f"Data split - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        return X, y

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models for comparison."""
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, max_depth=10, random_state=RANDOM_STATE
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=6, random_state=RANDOM_STATE
            )
        }

        logger.info(f"Initialized {len(self.models)} models for comparison")
        return self.models

    def evaluate_models(self) -> Dict[str, Dict[str, float]]:
        """Train and evaluate all models."""
        if self.X_train is None:
            self.prepare_features()

        self.initialize_models()

        # Fit scaler for linear regression
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            if name == 'Linear Regression':
                model.fit(X_train_scaled, self.y_train)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                pred = model.predict(self.X_test)

            # Calculate metrics
            r2 = r2_score(self.y_test, pred)
            rmse = np.sqrt(mean_squared_error(self.y_test, pred))
            mae = mean_absolute_error(self.y_test, pred)

            self.results[name] = {
                'R²': r2,
                'RMSE': rmse,
                'MAE': mae
            }

            logger.info(f"{name} - R²: {r2:.3f}, RMSE: ${rmse:,.0f}")

        return self.results

    def train_final_model(self) -> xgb.XGBRegressor:
        """Train the optimized XGBoost model."""
        if self.X_train is None:
            self.prepare_features()

        logger.info("Training final XGBoost model with optimization...")

        # Use all features for final model
        X = self.df.drop(columns=['SalePrice'])
        y = self.df['SalePrice']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        model = xgb.XGBRegressor(**XGBOOST_PARAMS)
        model.fit(X_train, y_train)

        # Evaluate final model
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Log results
        logger.info("Final Model Performance:")
        logger.info(f"  Test R²: {test_r2:.3f}")
        logger.info(f"  Test RMSE: ${test_rmse:,.0f}")
        logger.info(f"  Training R²: {train_r2:.3f}")
        logger.info(f"  Training RMSE: ${train_rmse:,.0f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")

        # Save model
        model_path = MODELS_DIR / 'final_xgboost_model.joblib'
        joblib.dump(model, model_path)
        logger.info(f"Model saved: {model_path}")

        return model