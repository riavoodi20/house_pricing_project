# house_pricing_project

# House Price Prediction with XGBoost

Predicting house prices using the Ames Housing Dataset with machine learning. Achieves strong performance through feature engineering and XGBoost optimization.

## Approach

**Data Processing:**
- Handled missing values with domain knowledge (e.g., 'None' for missing garage = no garage)
- Created age features (HouseAge, YearsSinceRemodel)
- Combined area measurements (TotalSF = 1st + 2nd + Basement)
- Converted quality ratings to numerical scales

**Model Selection:**
Compared Linear Regression, Random Forest, Gradient Boosting, and XGBoost. XGBoost performed best with heavy regularization to prevent overfitting.

**Key Features:**
TotalSF • OverallQual • GrLivArea • HouseAge • ExterQual_num • TotalBath • GarageCars

## Usage

```python
import xgboost as xgb
import pandas as pd

# Load processed data
df = pd.read_csv('data/df_model_ready.csv')
X = df.drop(columns=['SalePrice'])

# Train model
# Ex: 
model = xgb.XGBRegressor(
    n_estimators=1000, learning_rate=0.005, max_depth=2,
    reg_alpha=25, reg_lambda=25, random_state=42
)
model.fit(X_train, y_train)
```

##  Structure
```
├── data/                           # Dataset files
├── models/                         # Saved model files
├── outputs/                        # Results and visualizations
├── config.py                       # Configuration settings
├── data_preprocessing.py           # Data cleaning & EDA
├── feature_engineering.py          # Feature creation
├── model_training.py               # Model training & evaluation
└── main.py                         # Main execution script
```

## Results
- **Model:** XGBoost Regressor
- **Test R²:** 0.818
- **Test RMSE:** $37,314
- **Features:** 17 engineered features from 79 original variables

---
**Dataset:** [Ames Housing (Kaggle)](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) • 1,460 samples • 79 original features
