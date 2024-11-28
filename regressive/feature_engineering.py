# feature_engineering.py
import pandas as pd
import numpy as np

def add_engineered_features(X):
    """Add engineered features to the dataset"""
    X = X.copy()

    # Price-related ratios (only if we have the required features)
    if all(col in X.columns for col in ['beds', 'bedrooms']):
        X['beds_per_bedroom'] = X['beds'] / X['bedrooms'].replace(0, 1)

    if all(col in X.columns for col in ['accommodates', 'bedrooms']):
        X['accommodates_per_bedroom'] = X['accommodates'] / X['bedrooms'].replace(0, 1)

    # Location-price interactions
    if all(col in X.columns for col in ['neighborhood_price_index', 'premium_amenities_count']):
        X['location_amenities'] = X['neighborhood_price_index'] * X['premium_amenities_count']

    # Availability interactions
    if all(col in X.columns for col in ['availability_rate_30', 'neighborhood_price_index']):
        X['availability_demand'] = (1 - X['availability_rate_30']) * X['neighborhood_price_index']

    if all(col in X.columns for col in ['availability_rate_365', 'neighborhood_price_index']):
        X['availability_pressure'] = (1 - X['availability_rate_365']) * X['neighborhood_price_index']

    # Host experience and listing interaction
    if all(col in X.columns for col in ['host_experience_years', 'avg_review_score']):
        X['host_experience_score'] = X['host_experience_years'] * X['avg_review_score']

    # Location and amenity interactions
    if all(col in X.columns for col in ['neighborhood_price_index', 'premium_amenities_count']):
        X['premium_location_score'] = X['neighborhood_price_index'] * X['premium_amenities_count']

    # Demand indicators
    if all(col in X.columns for col in ['availability_rate_30', 'avg_review_score']):
        X['demand_score'] = (1 - X['availability_rate_30']) * X['avg_review_score']

    # Listing capacity utilization
    if all(col in X.columns for col in ['accommodates', 'bedrooms', 'beds']):
        X['space_efficiency'] = X['accommodates'] / (X['beds'] + X['bedrooms']).replace(0, 1)

    # Review activity
    if all(col in X.columns for col in ['number_of_reviews_ltm', 'number_of_reviews_l30d']):
        X['review_momentum'] = X['number_of_reviews_l30d'] / (X['number_of_reviews_ltm'].replace(0, 1))

    return X

def prepare_data(df, is_training=True):
    """Prepare data for modeling"""
    if is_training:
        X = df.drop('price', axis=1)
        y = df['price']
    else:
        X = df.copy()
        y = None

    # Add engineered features
    X = add_engineered_features(X)

    # Handle any missing values
    X = X.fillna(X.mean())

    # Remove any non-numeric columns that might have been created
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X = X[numeric_cols]

    return X, y
