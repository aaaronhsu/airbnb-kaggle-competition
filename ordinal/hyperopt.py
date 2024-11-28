import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import xgboost as xgb
import lightgbm as lgb
from models import get_model
import pickle
from datetime import datetime

def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv('processed_train.csv')
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y

def optimize_xgboost():
    """Optimize XGBoost hyperparameters"""
    print("\nOptimizing XGBoost...")
    
    # Define search space
    search_spaces = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'max_depth': Integer(3, 10),
        'min_child_weight': Integer(1, 7),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'n_estimators': Integer(100, 1000),
        'reg_alpha': Real(0.001, 10, prior='log-uniform'),
        'reg_lambda': Real(0.001, 10, prior='log-uniform'),
    }
    
    # Create base model
    base_model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=6,
        tree_method='hist',
        random_state=42
    )
    
    return base_model, search_spaces

def optimize_lightgbm():
    """Optimize LightGBM hyperparameters"""
    print("\nOptimizing LightGBM...")
    
    # Define search space
    search_spaces = {
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'num_leaves': Integer(20, 100),
        'max_depth': Integer(3, 10),
        'min_child_samples': Integer(1, 50),
        'subsample': Real(0.6, 1.0),
        'colsample_bytree': Real(0.6, 1.0),
        'n_estimators': Integer(100, 1000),
        'reg_alpha': Real(0.001, 10, prior='log-uniform'),
        'reg_lambda': Real(0.001, 10, prior='log-uniform'),
    }
    
    # Create base model
    base_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=6,
        random_state=42
    )
    
    return base_model, search_spaces

def run_bayesian_optimization(X, y, model_type='xgboost'):
    """Run Bayesian optimization for specified model"""
    # Get model and search spaces
    if model_type == 'xgboost':
        base_model, search_spaces = optimize_xgboost()
    elif model_type == 'lightgbm':
        base_model, search_spaces = optimize_lightgbm()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create BayesSearchCV object
    bayes_cv = BayesSearchCV(
        estimator=base_model,
        search_spaces=search_spaces,
        n_iter=50,  # Number of parameter settings that are sampled
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        random_state=42,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    # Fit the optimizer
    print(f"\nRunning Bayesian optimization for {model_type}...")
    bayes_cv.fit(X, y)
    
    # Print results
    print("\nBest parameters found:")
    for param, value in bayes_cv.best_params_.items():
        print(f"{param}: {value}")
    
    print(f"\nBest RMSE: {np.sqrt(-bayes_cv.best_score_):.4f}")
    
    return bayes_cv

def save_results(optimizer, model_type):
    """Save optimization results"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = {
        'best_params': optimizer.best_params_,
        'best_score': optimizer.best_score_,
        'cv_results': optimizer.cv_results_
    }
    
    with open(f'hyperopt_results_{model_type}_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save readable version of best parameters
    with open(f'best_params_{model_type}_{timestamp}.txt', 'w') as f:
        f.write("Best Parameters:\n")
        for param, value in optimizer.best_params_.items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest RMSE: {np.sqrt(-optimizer.best_score_):.4f}")

def main():
    # Load data
    X, y = load_data()
    
    # Run optimization for both models
    for model_type in ['xgboost', 'lightgbm']:
        optimizer = run_bayesian_optimization(X, y, model_type)
        save_results(optimizer, model_type)
        
        print(f"\nOptimization results for {model_type} saved.")

if __name__ == "__main__":
    main()
