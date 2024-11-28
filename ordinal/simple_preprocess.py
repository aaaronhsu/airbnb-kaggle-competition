import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SimpleAirbnbProcessor:
    def __init__(self):
        self.label_encoders = {}

    def convert_to_numeric(self, df, column):
        """Convert a column to numeric, handling currency and percentage strings"""
        if df[column].dtype == 'object':
            # Remove currency symbols and commas
            df[column] = df[column].str.replace(r'[$,]', '', regex=True)
            # Remove percentage signs
            df[column] = df[column].str.replace('%', '', regex=True)
            # Convert to float
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df

    def encode_categorical(self, series, col_name):
        """Encode categorical column with handling for unseen categories"""
        # Fill NA with 'missing'
        series = series.fillna('missing')

        if col_name not in self.label_encoders:
            # First time encoding (training data)
            self.label_encoders[col_name] = LabelEncoder()
            # Make sure 'missing' is in the classes
            unique_values = list(series.unique()) + ['missing']
            self.label_encoders[col_name].fit(unique_values)

        # Transform values, replacing unseen categories with 'missing'
        known_categories = set(self.label_encoders[col_name].classes_)
        series = series.map(lambda x: 'missing' if x not in known_categories else x)
        return self.label_encoders[col_name].transform(series)

    def process_data(self, filepath, is_test=False):
        """Process data with minimal transformations"""
        print(f"Processing {'test' if is_test else 'training'} data...")

        # Load data
        df = pd.read_csv(filepath)

        # Store ID column for test data
        id_column = None
        if is_test:
            id_column = df['id'].copy()

        # Drop text columns
        columns_to_drop = [
            'name',
            'description',
            'host_verifications',
            'bathrooms_text',
            'amenities',
            'reviews'
        ]

        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        print(f"\nDropped text columns: {[col for col in columns_to_drop if col in df.columns]}")

        # Basic numeric conversions
        numeric_columns = [
            'accommodates',
            'bathrooms',
            'bedrooms',
            'beds',
            'minimum_nights',
            'maximum_nights',
            'number_of_reviews',
            'number_of_reviews_ltm',
            'number_of_reviews_l30d',
            'calculated_host_listings_count',
            'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms',
            'calculated_host_listings_count_shared_rooms',
            'availability_30',
            'availability_60',
            'availability_90',
            'availability_365',
            'review_scores_rating',
            'review_scores_accuracy',
            'review_scores_cleanliness',
            'review_scores_checkin',
            'review_scores_communication',
            'review_scores_location',
            'review_scores_value',
            'reviews_per_month',
            'host_listings_count',
            'host_total_listings_count'
        ]

        # Convert numeric columns
        for col in numeric_columns:
            if col in df.columns:
                df = self.convert_to_numeric(df, col)
                df[col] = df[col].fillna(df[col].median())

        # Convert boolean columns
        boolean_columns = [
            'instant_bookable',
            'has_availability',
            'host_is_superhost',
            'host_has_profile_pic',
            'host_identity_verified'
        ]

        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].map({'t': 1, 'f': 0, True: 1, False: 0}).fillna(0).astype(int)

        # Handle categorical columns with label encoding
        categorical_columns = [
            'neighbourhood_cleansed',
            'neighbourhood_group_cleansed',
            'property_type',
            'room_type',
            'host_response_time'
        ]

        for col in categorical_columns:
            if col in df.columns:
                df[col] = self.encode_categorical(df[col], col)

        # Convert dates to timestamps
        date_columns = ['host_since', 'first_review', 'last_review']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).astype(np.int64) // 10**9
                df[col] = df[col].fillna(df[col].median())

        # Add back ID column for test data
        if is_test and id_column is not None:
            df['id'] = id_column

        print(f"\nProcessed data shape: {df.shape}")
        print("\nFeature types:")
        print(df.dtypes.value_counts())
        print("\nRemaining columns:")
        print(sorted(df.columns.tolist()))

        return df

def main():
    processor = SimpleAirbnbProcessor()

    try:
        # Process training data
        train_df = processor.process_data('train.csv', is_test=False)
        train_df.to_csv('simple_processed_train.csv', index=False)

        # Process test data using same processor (to maintain encodings)
        test_df = processor.process_data('test.csv', is_test=True)
        test_df.to_csv('simple_processed_test.csv', index=False)

        # Print sample of processed data
        print("\nSample of processed training data:")
        print(train_df.head())

        # Print basic statistics
        print("\nBasic statistics of numeric features:")
        print(train_df.describe().round(2))

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
