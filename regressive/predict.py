# predict.py
import pandas as pd
import numpy as np
import pickle
from feature_engineering import prepare_data

def load_models():
    """Load trained ensemble models"""
    with open('ensemble_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def predict(test_df):
    """Make predictions using ensemble models"""
    # Load models and feature names
    print("Loading models...")
    model_data = load_models()
    ensemble_models = model_data['ensemble_models']
    feature_names = model_data['feature_names']

    # Prepare test data
    print("Preparing test data...")
    X_test, _ = prepare_data(test_df, is_training=False)
    X_test = X_test[feature_names]  # Ensure features match training data

    # Make predictions for each fold
    print("Making predictions...")
    all_predictions = []

    for models, weights in ensemble_models:
        fold_predictions = np.zeros(len(X_test), dtype=float)
        for (name, model), weight in zip(models.items(), weights):
            fold_predictions += weight * model.predict(X_test).astype(float)
        all_predictions.append(fold_predictions)

    # Average predictions across folds
    final_predictions = np.mean(all_predictions, axis=0)

    # Floor predictions and ensure they're non-negative
    final_predictions = np.maximum(np.floor(final_predictions), 0).astype(int)

    # Create submission dataframe
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'price': final_predictions
    })

    return submission_df


def main():
    # Load test data
    print("Loading test data...")
    test_df = pd.read_csv('processed_test.csv')

    # Make predictions
    submission_df = predict(test_df)

    # Save predictions
    output_file = 'predictions.csv'
    submission_df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")

    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(submission_df['price'].describe())

    # Print most common predictions
    print("\nMost common predicted prices:")
    print(submission_df['price'].value_counts().head())

if __name__ == "__main__":
    main()
