import pandas as pd
import numpy as np
from models import get_model, get_ensemble_model
import os
import json
from datetime import datetime
import pickle

def load_trained_model():
    """Load the trained model and its feature list"""
    try:
        print("Loading trained model...")
        with open('trained_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
            
        if isinstance(model_data, dict):
            model = model_data['model']
            training_features = model_data['features']
        else:
            model = model_data
            # Default feature list if not saved with model
            training_features = [
                'neighborhood_price_index',
                'accommodates',
                'beds',
                'bedrooms',
                'borough_Brooklyn',
                'borough_Manhattan',
                'borough_Queens',
                'borough_Bronx',
                'borough_Staten Island',
                'calculated_host_listings_count_entire_homes',
                'calculated_host_listings_count_private_rooms',
                'minimum_nights',
                'premium_amenities_count',
                'instant_bookable',
                'host_listings_count',
                'availability_rate_30',
                'avg_review_score',
                'listing_age_days'
            ]
            
        print("Model loaded successfully!")
        return model, training_features
    except FileNotFoundError:
        raise Exception("No trained model found. Please run train.py first.")

def make_predictions(model, test_df, training_features):
    """Make predictions using the trained model"""
    # Ensure test data has exactly the same features as training data
    print("\nAligning features with training data...")
    
    # Check for missing columns
    missing_cols = set(training_features) - set(test_df.columns)
    if missing_cols:
        print(f"Warning: Missing features in test data: {missing_cols}")
        for col in missing_cols:
            test_df[col] = 0  # Fill with zeros for missing features
    
    # Remove extra columns
    extra_cols = set(test_df.columns) - set(training_features) - {'id'}
    if extra_cols:
        print(f"Warning: Extra features in test data (will be ignored): {extra_cols}")
        test_df = test_df.drop(columns=list(extra_cols))
    
    # Ensure columns are in the same order as training
    X_test = test_df[training_features]
    
    print("\nMaking predictions...")
    pred = model.predict(X_test)
    
    # Print prediction distribution
    unique, counts = np.unique(pred, return_counts=True)
    dist = dict(zip(unique, counts))
    print("\nPrediction distribution:")
    for price_class in sorted(dist.keys()):
        print(f"Class {price_class}: {dist[price_class]} ({dist[price_class]/len(pred):.3f})")
    
    return pred

def create_submission(predictions, test_ids):
    """Create submission file"""
    submission = pd.DataFrame({
        'id': test_ids,
        'price': predictions
    })
    
    # Create submissions directory if it doesn't exist
    os.makedirs('submissions', exist_ok=True)
    
    # Save submission with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'submissions/submission_{timestamp}.csv'
    submission.to_csv(filename, index=False)
    
    return filename

def main():
    try:
        # Load test data
        print("Loading test data...")
        test_df = pd.read_csv('processed_test.csv')
        
        # Load trained model and feature list
        model, training_features = load_trained_model()
        
        # Make predictions
        predictions = make_predictions(model, test_df, training_features)
        
        # Create submission file
        print("\nCreating submission file...")
        submission_file = create_submission(predictions, test_df['id'])
        print(f"Submission saved to: {submission_file}")
        
        # Validate predictions
        print("\nValidating predictions...")
        print(f"Number of predictions: {len(predictions)}")
        print(f"Prediction range: {predictions.min()} to {predictions.max()}")
        print("\nPrediction value counts:")
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        print(pred_counts)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
