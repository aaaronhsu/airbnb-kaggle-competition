import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import ast
from haversine import haversine

# Constants
MANHATTAN_CENTER = (40.7829, -73.9654)
ROOM_TYPE_MAPPING = {
    'Entire home/apt': 0,
    'Private room': 1,
    'Shared room': 2,
    'Hotel room': 3
}

def parse_amenities(amenities_str):
    """Safely parse amenities string"""
    try:
        amenities_str = amenities_str.strip()
        if pd.isna(amenities_str):
            return 0
        amenities_list = ast.literal_eval(amenities_str)
        return len(amenities_list)
    except:
        return 0

def process_room_type(x):
    """Process room type with consistent mapping"""
    if 'Shared' in str(x):
        return 2
    elif 'Private' in str(x):
        return 1
    elif 'Entire' in str(x):
        return 0
    else:
        return 3

def process_property_type(x):
    """Process property type with consistent mapping"""
    x_lower = str(x).lower()
    if 'apartment' in x_lower:
        return 'Apartment'
    elif 'house' in x_lower:
        return 'House'
    elif 'condo' in x_lower:
        return 'Condo'
    else:
        return 'Other'

def load_and_preprocess_data(filepath, is_train=True, le_dict=None):
    """Load and preprocess the data"""
    # Read CSV
    df = pd.read_csv(filepath)

    # Create processed dataframe with a copy
    features = ['property_type', 'neighbourhood_group_cleansed',
               'minimum_nights', 'bedrooms', 'accommodates',
               'room_type', 'bathrooms', 'latitude', 'longitude',
               'amenities', 'calculated_host_listings_count']

    df_processed = df[features].copy()

    # Handle missing values
    df_processed.loc[:, 'bedrooms'] = df_processed['bedrooms'].fillna(1)
    df_processed.loc[:, 'bathrooms'] = df_processed['bathrooms'].fillna(1)
    df_processed.loc[:, 'calculated_host_listings_count'] = df_processed['calculated_host_listings_count'].fillna(1)

    # Ensure numeric columns are float
    numeric_columns = ['minimum_nights', 'bedrooms', 'accommodates',
                      'bathrooms', 'latitude', 'longitude',
                      'calculated_host_listings_count']
    for col in numeric_columns:
        df_processed[col] = df_processed[col].astype(float)

    # Feature engineering

    # 1. Distance to Manhattan center
    df_processed.loc[:, 'distance_to_center'] = df_processed.apply(
        lambda row: haversine((row['latitude'], row['longitude']), MANHATTAN_CENTER),
        axis=1
    )

    # 2. Process amenities
    df_processed.loc[:, 'amenity_count'] = df_processed['amenities'].apply(parse_amenities)

    # 3. Borough level statistics
    borough_stats = df_processed.groupby('neighbourhood_group_cleansed').agg({
        'calculated_host_listings_count': 'mean',
        'distance_to_center': 'mean'
    }).reset_index()

    borough_stats.columns = ['neighbourhood_group_cleansed',
                           'borough_avg_listings',
                           'borough_avg_distance']

    df_processed = df_processed.merge(borough_stats, on='neighbourhood_group_cleansed')

    # Handle categorical variables
    # Process room_type
    df_processed['room_type'] = df_processed['room_type'].apply(process_room_type)

    # Process property_type
    df_processed['property_type'] = df_processed['property_type'].apply(process_property_type)

    if is_train:
        le_dict = {}
        # Only need to fit-transform property_type and neighbourhood in training
        for col in ['property_type', 'neighbourhood_group_cleansed']:
            le_dict[col] = LabelEncoder()
            df_processed[col] = le_dict[col].fit_transform(df_processed[col])
    else:
        # Transform using existing label encoders
        for col in ['property_type', 'neighbourhood_group_cleansed']:
            df_processed[col] = le_dict[col].transform(df_processed[col])

    # Drop columns we don't need anymore
    df_processed = df_processed.drop('amenities', axis=1)

    # Ensure all columns are float
    for col in df_processed.columns:
        df_processed[col] = df_processed[col].astype(float)

    if is_train:
        # Process prices and create target variable
        prices = df['price'].apply(lambda x: float(str(x).replace('$', '').replace(',', '')))

        # Remove outliers
        q1 = prices.quantile(0.01)
        q3 = prices.quantile(0.99)
        prices = prices[(prices >= q1) & (prices <= q3)]

        # Create labels using qcut with duplicate handling
        try:
            labels = pd.qcut(prices, q=6, labels=[0,1,2,3,4,5], duplicates='drop')
        except ValueError:
            labels = pd.cut(prices, bins=6, labels=[0,1,2,3,4,5])

        # Ensure df_processed and labels align
        df_processed = df_processed.loc[labels.index]
        labels = labels.astype(int)

        return df_processed, labels, le_dict
    else:
        return df_processed, None, le_dict

def train_model(X, y):
    """Train XGBoost classifier"""
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=6,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=True
    )

    # Print validation results
    y_pred = model.predict(X_val)
    print("\nValidation Results:")
    print(classification_report(y_val, y_pred))

    return model

def main():
    # Load and preprocess training data
    print("Loading and preprocessing training data...")
    X_train, y_train, le_dict = load_and_preprocess_data('train.csv', is_train=True)

    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)

    # Load and preprocess test data
    print("\nLoading and preprocessing test data...")
    test_df = pd.read_csv('test.csv')
    test_ids = test_df['id']
    X_test, _, _ = load_and_preprocess_data('test.csv', is_train=False, le_dict=le_dict)

    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test)

    # Create submission file
    submission = pd.DataFrame({
        'id': test_ids,
        'price': predictions
    })

    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created!")

if __name__ == "__main__":
    main()
