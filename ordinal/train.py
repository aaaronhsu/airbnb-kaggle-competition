import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from models import get_model, get_ensemble_model
import pickle
from datetime import datetime
import os

def load_data():
    """Load and prepare data for training"""
    print("Loading data...")
    train_df = pd.read_csv('processed_train.csv')

    # Separate features and target
    X = train_df.drop('price', axis=1)
    y = train_df['price'].astype(int)

    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print("\nFeature list:", list(X.columns))
    print("\nClass distribution:")
    print(y.value_counts(normalize=True).sort_index().round(3))

    return X, y

def cross_validate(X, y, model_type='ensemble', n_folds=5):
    """Perform cross-validation for a specific model type"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    cv_results = {
        'rmse': [],
        'accuracy': [],
        'adjacent_accuracy': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/{n_folds}")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Train model
        if model_type == 'ensemble':
            model = get_ensemble_model()
        else:
            model = get_model(model_type)

        model.fit(X_train, y_train)

        # Evaluate
        eval_results = model.evaluate(X_val, y_val)

        # Store results
        cv_results['rmse'].append(eval_results['rmse'])
        cv_results['accuracy'].append(eval_results['accuracy'])
        cv_results['adjacent_accuracy'].append(eval_results['adjacent_accuracy'])

        print(f"Fold {fold} - RMSE: {eval_results['rmse']:.4f}, "
              f"Accuracy: {eval_results['accuracy']:.4f}, "
              f"Adjacent Accuracy: {eval_results['adjacent_accuracy']:.4f}")

    return cv_results

def train_and_save_model(X, y, model_type, results_dir):
    """Train and save a specific model"""
    print(f"\nTraining {model_type.upper()} model...")

    # Train model
    if model_type == 'ensemble':
        model = get_ensemble_model()
    else:
        model = get_model(model_type)

    model.fit(X, y)

    # Save model
    model_filename = f'{results_dir}/{model_type}_model.pkl'
    # In train.py, when saving the model:
    model_data = {
        'model': model,
        'features': X.columns.tolist()  # Save the feature list
    }

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)

    print(f"{model_type.upper()} model saved to: {model_filename}")

    # Perform cross-validation
    print(f"\nPerforming cross-validation for {model_type.upper()}...")
    cv_results = cross_validate(X, y, model_type)

    # Print results
    print(f"\n{model_type.upper()} Cross-validation Results:")
    print(f"RMSE: {np.mean(cv_results['rmse']):.4f} ± {np.std(cv_results['rmse']):.4f}")
    print(f"Accuracy: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
    print(f"Adjacent Accuracy: {np.mean(cv_results['adjacent_accuracy']):.4f} ± {np.std(cv_results['adjacent_accuracy']):.4f}")

    return cv_results

def main():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    X, y = load_data()

    # Models to train
    model_types = ['xgboost', 'lightgbm', 'random_forest', 'ensemble']

    # Train and evaluate all models
    all_results = {}
    for model_type in model_types:
        print(f"\n{'='*50}")
        print(f"Processing {model_type.upper()} model")
        print('='*50)

        cv_results = train_and_save_model(X, y, model_type, results_dir)
        all_results[model_type] = cv_results

    # Save all results
    results_summary = {
        model_type: {
            'rmse_mean': np.mean(results['rmse']),
            'rmse_std': np.std(results['rmse']),
            'accuracy_mean': np.mean(results['accuracy']),
            'accuracy_std': np.std(results['accuracy']),
            'adjacent_accuracy_mean': np.mean(results['adjacent_accuracy']),
            'adjacent_accuracy_std': np.std(results['adjacent_accuracy'])
        }
        for model_type, results in all_results.items()
    }

    # Save results summary
    results_df = pd.DataFrame(results_summary).T
    results_df.to_csv(f'{results_dir}/model_comparison.csv')

    # Print final comparison
    print("\n" + "="*50)
    print("Final Model Comparison")
    print("="*50)
    print("\nRMSE:")
    for model_type in model_types:
        print(f"{model_type.upper()}: {results_summary[model_type]['rmse_mean']:.4f} ± {results_summary[model_type]['rmse_std']:.4f}")

    print("\nAccuracy:")
    for model_type in model_types:
        print(f"{model_type.upper()}: {results_summary[model_type]['accuracy_mean']:.4f} ± {results_summary[model_type]['accuracy_std']:.4f}")

    print("\nAdjacent Accuracy:")
    for model_type in model_types:
        print(f"{model_type.upper()}: {results_summary[model_type]['adjacent_accuracy_mean']:.4f} ± {results_summary[model_type]['adjacent_accuracy_std']:.4f}")

    print(f"\nDetailed results saved to: {results_dir}/model_comparison.csv")

if __name__ == "__main__":
    main()
