# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pickle
from feature_engineering import prepare_data
from models import get_xgb_model, get_rf_model, get_lasso_model

def optimize_weights(predictions, y_true):
    """Optimize ensemble weights using scipy optimize"""
    from scipy.optimize import minimize

    def objective(weights):
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        pred = np.zeros_like(y_true, dtype=float)  # Explicitly set dtype to float
        for (_, p), w in zip(predictions.items(), weights):
            pred += w * p.astype(float)  # Ensure predictions are float
        return np.sqrt(mean_squared_error(y_true, pred))

    initial_weights = np.ones(len(predictions)) / len(predictions)
    bounds = [(0, 1) for _ in range(len(predictions))]

    result = minimize(
        objective,
        initial_weights,
        bounds=bounds,
        constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    )

    return result.x

def train_ensemble_cv(X, y, n_folds=5):
    """Train ensemble of models with cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    ensemble_models = []
    ensemble_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\nTraining fold {fold + 1}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Initialize models
        xgb_model = get_xgb_model()
        rf_model = get_rf_model()

        # Train XGBoost
        print("Training XGBoost...")
        eval_set = [(X_val, y_val)]
        xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=100
        )

        # Train Random Forest
        print("\nTraining Random Forest...")
        rf_model.fit(X_train, y_train)

        # Get predictions for this fold
        fold_predictions = {
            'xgb': xgb_model.predict(X_val).astype(float),
            'rf': rf_model.predict(X_val).astype(float)
        }

        # Calculate individual model RMSEs
        for name, preds in fold_predictions.items():
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            print(f"{name} RMSE: {rmse:.4f}")

        # Find optimal weights
        weights = optimize_weights(fold_predictions, y_val)
        print("\nOptimal weights:", {name: f"{w:.3f}" for name, w in zip(fold_predictions.keys(), weights)})

        # Calculate ensemble prediction
        ensemble_pred = np.zeros_like(y_val, dtype=float)
        for (name, pred), weight in zip(fold_predictions.items(), weights):
            ensemble_pred += weight * pred

        rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
        print(f"Ensemble RMSE: {rmse:.4f}")

        # Store models and weights
        fold_models = {
            'xgb': xgb_model,
            'rf': rf_model
        }
        ensemble_models.append((fold_models, weights))
        ensemble_scores.append(rmse)

    print(f"\nMean Ensemble RMSE: {np.mean(ensemble_scores):.4f} ± {np.std(ensemble_scores):.4f}")
    return ensemble_models, ensemble_scores

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('processed_train.csv')

    # Prepare data with engineered features
    print("Preparing data...")
    X, y = prepare_data(train_df, is_training=True)

    # Train ensemble with cross-validation
    print("Training ensemble...")
    ensemble_models, ensemble_scores = train_ensemble_cv(X, y)

    # Save models and feature names
    print("Saving models...")
    model_data = {
        'ensemble_models': ensemble_models,
        'feature_names': X.columns.tolist(),
        'ensemble_scores': ensemble_scores
    }

    with open('ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print("Training completed!")

if __name__ == "__main__":
    main()
