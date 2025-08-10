"""Feature engineering functions."""

import logging
from pandas import DataFrame
from config import FINAL_FEATURES, PROCESSED_DATA_PATH

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handle feature engineering and creation."""

    def __init__(self, df: DataFrame):
        self.df = df.copy()

    def create_age_features(self, current_year: int = 2024) -> DataFrame:
        """Create age-related features."""
        self.df['HouseAge'] = current_year - self.df['YearBuilt']
        self.df['YearsSinceRemodel'] = current_year - self.df['YearRemodAdd']
        self.df['IsRemodeled'] = (self.df['YearRemodAdd'] != self.df['YearBuilt']).astype(int)

        logger.info("Age features created: HouseAge, YearsSinceRemodel, IsRemodeled")
        return self.df

    def create_area_features(self) -> DataFrame:
        """Create combined area features."""
        self.df['TotalSF'] = self.df['1stFlrSF'] + self.df['2ndFlrSF'] + self.df['TotalBsmtSF']

        porch_cols = ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']
        self.df['TotalPorchSF'] = self.df[porch_cols].sum(axis=1)

        logger.info("Area features created: TotalSF, TotalPorchSF")
        return self.df

    def create_bathroom_features(self) -> DataFrame:
        """Create combined bathroom features."""
        self.df['TotalBath'] = (self.df['FullBath'] + self.df['BsmtFullBath'] +
                                0.5 * (self.df['HalfBath'] + self.df['BsmtHalfBath']))

        logger.info("Bathroom features created: TotalBath")
        return self.df

    def create_quality_features(self) -> DataFrame:
        """Convert quality ratings to numerical values."""
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}

        quality_cols = ['ExterQual', 'KitchenQual']
        for col in quality_cols:
            if col in self.df.columns:
                self.df[f'{col}_num'] = self.df[col].map(quality_map)
                logger.info(f"Quality feature created: {col}_num")

        return self.df

    def create_binary_features(self) -> DataFrame:
        """Create binary indicator features."""
        binary_features = {
            'HasGarage': self.df['GarageArea'] > 0,
            'HasBasement': self.df['TotalBsmtSF'] > 0,
            'HasFireplace': self.df['Fireplaces'] > 0
        }

        for feature, condition in binary_features.items():
            self.df[feature] = condition.astype(int)
            logger.info(f"Binary feature created: {feature}")

        return self.df

    def engineer_all_features(self) -> DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Starting feature engineering...")

        self.create_age_features()
        self.create_area_features()
        self.create_bathroom_features()
        self.create_quality_features()
        self.create_binary_features()

        logger.info("Feature engineering completed")
        return self.df

    def prepare_model_dataset(self, save_path: str = PROCESSED_DATA_PATH) -> DataFrame:
        """Prepare final dataset for modeling."""
        # Check if all required features exist
        missing_features = [f for f in FINAL_FEATURES if f not in self.df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Create model-ready dataset
        model_features = FINAL_FEATURES + ['SalePrice']
        df_model = self.df[model_features].copy()

        # Save processed dataset
        df_model.to_csv(save_path, index=False)
        logger.info(f"Model-ready dataset saved: {save_path}")
        logger.info(f"Dataset shape: {df_model.shape}")

        return df_model