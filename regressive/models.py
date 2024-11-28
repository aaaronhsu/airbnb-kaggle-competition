# models.py
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

def get_xgb_model(trial=None):
    """Get XGBoost model with parameters"""
    if trial is None:
        # Default parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 2000,
            'min_child_weight': 3,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.5,
            'reg_lambda': 1.5,
            'random_state': 42,
            'callbacks': [xgb.callback.EarlyStopping(rounds=50)]
        }
    else:
        # Optuna optimization parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 2.0, log=True),
            'random_state': 42,
            'callbacks': [xgb.callback.EarlyStopping(rounds=50)]
        }

    return xgb.XGBRegressor(**params)

def get_rf_model():
    """Get Random Forest model with parameters"""
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'random_state': 42,
        'n_jobs': -1
    }
    return RandomForestRegressor(**params)

def get_lasso_model():
    """Get Lasso model"""
    return LassoCV(random_state=42, n_jobs=-1)
