"""Main script to run the complete house price prediction pipeline."""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from config import RAW_DATA_PATH, OUTPUTS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUTS_DIR / 'pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the complete ML pipeline."""
    try:
        logger.info("=" * 50)
        logger.info("HOUSE PRICE PREDICTION PIPELINE")
        logger.info("=" * 50)

        # Step 1: Data Preprocessing
        logger.info("\n1. DATA PREPROCESSING")
        logger.info("-" * 30)

        preprocessor = DataPreprocessor(RAW_DATA_PATH)
        df = preprocessor.load_data()

        # Analyze missing data
        missing_values = preprocessor.analyze_missing_data()

        # Fill missing values
        df_filled = preprocessor.fill_missing_values()

        # Create visualizations
        preprocessor.create_visualizations()

        # Step 2: Feature Engineering
        logger.info("\n2. FEATURE ENGINEERING")
        logger.info("-" * 30)

        engineer = FeatureEngineer(df_filled)
        df_engineered = engineer.engineer_all_features()
        df_model = engineer.prepare_model_dataset()

        # Step 3: Model Training and Comparison
        logger.info("\n3. MODEL TRAINING & COMPARISON")
        logger.info("-" * 30)

        trainer = ModelTrainer()
        results = trainer.evaluate_models()

        logger.info("\nModel Comparison Results:")
        for name, metrics in results.items():
            logger.info(f"{name:20} - R²: {metrics['R²']:.3f}, RMSE: ${metrics['RMSE']:,.0f}")

        # Step 4: Final Model Training
        logger.info("\n4. FINAL MODEL OPTIMIZATION")
        logger.info("-" * 30)

        final_model = trainer.train_final_model()

        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 50)

        return final_model

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    model = main()