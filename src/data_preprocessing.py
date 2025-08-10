"""Data preprocessing and cleaning functions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data loading, cleaning, and preprocessing."""

    def __init__(self, data_path: str = RAW_DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.df_filled = None

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded successfully: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_missing_data(self) -> pd.Series:
        """Analyze and visualize missing data patterns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        missing_data = self.df.isnull().sum()
        missing_percentage = missing_data * 100 / len(self.df)
        missing_values = missing_percentage[missing_percentage > 0].sort_values(ascending=False)

        if len(missing_values) > 0:
            plt.figure(figsize=(12, 10))
            top_20_missing = missing_values.head(20).sort_values(ascending=True)
            plt.barh(top_20_missing.index, top_20_missing.values)
            plt.xlabel('Percentage of Missing Data')
            plt.ylabel('Columns with Missing Values')
            plt.title('Top 20 Columns with Missing Data')
            plt.tight_layout()
            plt.savefig(OUTPUTS_DIR / 'missing_data_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()

        logger.info(f"Missing data analysis complete: {len(missing_values)} columns have missing values")
        return missing_values

    def fill_missing_values(self) -> pd.DataFrame:
        """Fill missing values using domain knowledge."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df_filled = self.df.copy()

        # Fill features where missing = "None" (feature doesn't exist)
        for feature, reason in NONE_FEATURES.items():
            if feature in self.df_filled.columns and self.df_filled[feature].isnull().sum() > 0:
                before = self.df_filled[feature].isnull().sum()
                self.df_filled[feature] = self.df_filled[feature].fillna('None')
                logger.info(f"Filled {before} missing values in {feature} with 'None'")

        # Fill numerical features with 0
        for feature in ZERO_FEATURES:
            if feature in self.df_filled.columns and self.df_filled[feature].isnull().sum() > 0:
                before = self.df_filled[feature].isnull().sum()
                self.df_filled[feature] = self.df_filled[feature].fillna(0)
                logger.info(f"Filled {before} missing values in {feature} with 0")

        # Special case: GarageYrBlt
        if 'GarageYrBlt' in self.df_filled.columns:
            before = self.df_filled['GarageYrBlt'].isnull().sum()
            self.df_filled['GarageYrBlt'] = self.df_filled['GarageYrBlt'].fillna(self.df_filled['YearBuilt'])
            if before > 0:
                logger.info(f"Filled {before} missing values in GarageYrBlt with YearBuilt")

        # Fill categorical with mode
        categorical_cols = self.df_filled.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df_filled[col].isnull().sum() > 0:
                before = self.df_filled[col].isnull().sum()
                mode_val = self.df_filled[col].mode()[0] if len(self.df_filled[col].mode()) > 0 else 'Unknown'
                self.df_filled[col] = self.df_filled[col].fillna(mode_val)
                logger.info(f"Filled {before} missing values in {col} with mode: {mode_val}")

        # Fill numerical with median
        numeric_cols = self.df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df_filled[col].isnull().sum() > 0:
                before = self.df_filled[col].isnull().sum()
                median_val = self.df_filled[col].median()
                self.df_filled[col] = self.df_filled[col].fillna(median_val)
                logger.info(f"Filled {before} missing values in {col} with median: {median_val}")

        logger.info("Missing value imputation completed")
        return self.df_filled

    def create_visualizations(self) -> None:
        """Create exploratory data analysis visualizations."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        plt.figure(figsize=(15, 10))

        # 1. Sale Price Distribution
        plt.subplot(2, 2, 1)
        plt.hist(self.df['SalePrice'], bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        plt.title('Sale Price Distribution')
        plt.xlabel('Price ($)')
        plt.ylabel('Frequency')

        # 2. Price vs Living Area
        plt.subplot(2, 2, 2)
        plt.scatter(self.df['GrLivArea'], self.df['SalePrice'], alpha=0.6, color='green')
        plt.title('Price vs Living Area')
        plt.xlabel('Living Area (sq ft)')
        plt.ylabel('Price ($)')

        # 3. Top 10 Neighborhoods by Price
        plt.subplot(2, 2, 3)
        neighborhood_prices = self.df.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False)
        neighborhood_prices.head(10).plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Top 10 Neighborhoods by Price')
        plt.xticks(rotation=45)

        # 4. Price vs Year Built
        plt.subplot(2, 2, 4)
        plt.scatter(self.df['YearBuilt'], self.df['SalePrice'], alpha=0.6, color='purple')
        plt.title('Price vs Year Built')
        plt.xlabel('Year Built')
        plt.ylabel('Price ($)')

        plt.tight_layout()
        plt.savefig(OUTPUTS_DIR / 'eda_visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Print statistics
        stats = {
            'Mean': self.df['SalePrice'].mean(),
            'Median': self.df['SalePrice'].median(),
            'Std Dev': self.df['SalePrice'].std(),
            'Skewness': self.df['SalePrice'].skew(),
            'Min': self.df['SalePrice'].min(),
            'Max': self.df['SalePrice'].max()
        }

        logger.info("SalePrice Statistics:")
        for stat, value in stats.items():
            if stat == 'Skewness':
                logger.info(f"{stat}: {value:.3f}")
            else:
                logger.info(f"{stat}: ${value:,.0f}")