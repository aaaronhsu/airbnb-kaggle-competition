import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')



class AirbnbDataProcessor:
    def __init__(self):
        # Initialize scalers
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()

        # Property type mapping (simplified categories)
        self.property_type_mapping = {
            'Entire rental unit': 'standard_full',
            'Private room in rental unit': 'private_room',
            'Entire condo': 'premium_full',
            'Entire home': 'premium_full',
            'Private room in home': 'private_room',
            'Entire townhouse': 'premium_full',
            'Entire guest suite': 'standard_full',
            'Room in hotel': 'hotel',
            'Private room in townhouse': 'private_room',
            'Entire loft': 'premium_full',
            'Private room in condo': 'private_room',
            'Shared room in rental unit': 'shared',
            'Room in boutique hotel': 'hotel',
            'Private room in loft': 'private_room'
        }

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        # Radius of earth in kilometers
        r = 6371

        return c * r

    def process_property_types(self, df):
        """Process property types into high-level categories"""
        # First map detailed property types to categories
        property_type_mapping = {
            # Entire Premium Properties
            'Entire villa': 'entire_premium',
            'Entire vacation home': 'entire_premium',
            'Entire cottage': 'entire_premium',
            'Entire bungalow': 'entire_premium',
            'Entire guest suite': 'entire_premium',
            'Entire guesthouse': 'entire_premium',
            'Entire townhouse': 'entire_premium',
            'Entire condo': 'entire_premium',
            'Entire loft': 'entire_premium',

            # Entire Standard Properties
            'Entire rental unit': 'entire_standard',
            'Entire home': 'entire_standard',
            'Entire place': 'entire_standard',
            'Entire serviced apartment': 'entire_standard',

            # Hotel-Style Accommodations
            'Room in hotel': 'hotel_style',
            'Room in boutique hotel': 'hotel_style',
            'Room in aparthotel': 'hotel_style',
            'Room in serviced apartment': 'hotel_style',

            # Private Rooms
            'Private room in rental unit': 'private_room',
            'Private room in home': 'private_room',
            'Private room in townhouse': 'private_room',
            'Private room in condo': 'private_room',
            'Private room in serviced apartment': 'private_room',
            'Private room in guest suite': 'private_room',
            'Private room in loft': 'private_room',
            'Private room in bed and breakfast': 'private_room',
            'Private room in casa particular': 'private_room',
            'Private room in guesthouse': 'private_room',
            'Private room': 'private_room',
            'Private room in resort': 'private_room',
            'Private room in villa': 'private_room',
            'Private room in bungalow': 'private_room',
            'Private room in vacation home': 'private_room',
            'Private room in cottage': 'private_room',

            # Shared/Budget Accommodations
            'Shared room in rental unit': 'shared_budget',
            'Shared room in home': 'shared_budget',
            'Shared room in townhouse': 'shared_budget',
            'Shared room in hostel': 'shared_budget',
            'Shared room in condo': 'shared_budget',
            'Shared room in serviced apartment': 'shared_budget',
            'Shared room': 'shared_budget',
            'Private room in hostel': 'shared_budget',
            'Camper/RV': 'shared_budget',
            'Boat': 'shared_budget',
            'Houseboat': 'shared_budget',
            'Tiny home': 'shared_budget'
        }

        # Map detailed types to categories
        df['property_category'] = df['property_type'].map(property_type_mapping)
        df['property_category'].fillna('other', inplace=True)

        # Create high-level property type features
        df['is_entire_place'] = df['property_category'].isin(['entire_premium', 'entire_standard']).astype(int)
        df['is_private_room'] = (df['property_category'] == 'private_room').astype(int)
        df['is_hotel'] = (df['property_category'] == 'hotel_style').astype(int)
        df['is_shared'] = (df['property_category'] == 'shared_budget').astype(int)

        # Create premium vs standard entire place distinction
        df['is_premium_entire'] = (df['property_category'] == 'entire_premium').astype(int)
        df['is_standard_entire'] = (df['property_category'] == 'entire_standard').astype(int)

        # Calculate some property type statistics if not test data
        if 'price' in df.columns:
            property_stats = df.groupby('property_category')['price'].agg(['mean', 'count'])
            print("\nProperty Type Statistics:")
            print(property_stats.round(2))

        return df


    def create_derived_features(self, df, is_test=False):
        """Create all derived features"""
        # Store neighborhood price mapping from training data
        if not is_test:
            neighborhood_prices = df.groupby('neighbourhood_cleansed')['price'].agg(['mean', 'count'])
            valid_neighborhoods = neighborhood_prices[neighborhood_prices['count'] >= 5]
            self.neighborhood_price_index = valid_neighborhoods['mean'] / valid_neighborhoods['mean'].mean()

        # Apply neighborhood price index
        df['neighborhood_price_index'] = df['neighbourhood_cleansed'].map(
            getattr(self, 'neighborhood_price_index', {})
        )
        df['neighborhood_price_index'].fillna(1.0, inplace=True)

        # Define major NYC landmarks with their coordinates
        LANDMARKS = {
            'times_square': (40.7580, -73.9855),
            'central_park': (40.7829, -73.9654),
            'world_trade': (40.7127, -74.0134),
            'empire_state': (40.7484, -73.9857),
            'brooklyn_bridge': (40.7061, -73.9969),
            'rockefeller_center': (40.7587, -73.9787),
            'wall_street': (40.7068, -74.0090),
            'high_line': (40.7480, -74.0048)
        }

        # Calculate log-transformed distances to all landmarks
        for landmark_name, coords in LANDMARKS.items():
            df[f'log_distance_to_{landmark_name}'] = np.log1p(
                df.apply(
                    lambda row: self.haversine_distance(
                        row['latitude'],
                        row['longitude'],
                        coords[0],
                        coords[1]
                    ),
                    axis=1
                )
            )

        # Create minimum distance to any landmark
        landmark_distance_columns = [f'log_distance_to_{landmark}' for landmark in LANDMARKS.keys()]
        df['log_min_landmark_distance'] = df[landmark_distance_columns].min(axis=1)

        # Create mean distance to all landmarks
        df['log_mean_landmark_distance'] = df[landmark_distance_columns].mean(axis=1)


        # Borough from neighborhood_group_cleansed
        df['borough'] = df['neighbourhood_group_cleansed']

        # Premium amenities count
        premium_amenities = {
            'Pool', 'Hot tub', 'Gym', 'Doorman',
            'Elevator', 'Free parking', 'Washer', 'Dryer'
        }
        df['amenities_list'] = df['amenities'].apply(lambda x: eval(x) if isinstance(x, str) else [])
        df['premium_amenities_count'] = df['amenities_list'].apply(
            lambda x: sum(1 for amenity in x if any(premium in amenity for premium in premium_amenities))
        )

        # Availability rate
        df['availability_rate_30'] = df['availability_30'] / 30

        # Average review score
        review_score_columns = [col for col in df.columns if col.startswith('review_scores_')]
        df['avg_review_score'] = df[review_score_columns].mean(axis=1)

        # Listing age
        df['host_since'] = pd.to_datetime(df['host_since'])
        current_date = pd.Timestamp.now()
        df['listing_age_days'] = (current_date - df['host_since']).dt.total_seconds() / (24*60*60)

        # Property types
        df = self.process_property_types(df)

        return df

    def process_categorical_features(self, df):
        """Process categorical features using appropriate encoding"""
        # Property type encoding
        df['property_type_simplified'] = df['property_type'].map(self.property_type_mapping)
        df['property_type_simplified'].fillna('other', inplace=True)

        # One-hot encoding for nominal categories
        nominal_features = ['property_type_simplified', 'borough']
        for feature in nominal_features:
            if feature in df.columns:
                dummies = pd.get_dummies(df[feature], prefix=feature)
                df = pd.concat([df, dummies], axis=1)
                df.drop(columns=[feature], inplace=True)

        return df

    def process_numeric_features(self, df):
        """Process numeric features with appropriate scaling"""
        # Features for robust scaling (contains outliers)
        robust_scale_features = [
            'minimum_nights',
            'host_listings_count',
            'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms'
        ]

        # Features for standard scaling
        standard_scale_features = [
            'neighborhood_price_index',
            'avg_review_score',
            'availability_rate_30',
            'log_distance_to_times_square'  # Add to standard scaling
        ]

        # Log transform features
        log_transform_features = [
            'listing_age_days',
            'premium_amenities_count'
        ]

        # Apply transformations
        for feature in robust_scale_features:
            if feature in df.columns:
                df[feature] = self.robust_scaler.fit_transform(df[[feature]])

        for feature in standard_scale_features:
            if feature in df.columns:
                df[feature] = self.standard_scaler.fit_transform(df[[feature]])

        for feature in log_transform_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])

        return df

    def handle_missing_values(self, df):
        """Handle missing values with appropriate strategies"""
        # Categorical features: fill with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # Numeric features: fill with median
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)

        return df

    def process_data(self, filepath, is_test=False):
        """Main processing function"""
        # Load data
        df = pd.read_csv(filepath)

        # Store ID column for test data
        id_column = None
        if is_test:
            id_column = df['id'].copy()

        # Handle missing values
        df = self.handle_missing_values(df)

        # Create derived features
        df = self.create_derived_features(df, is_test=is_test)

        # Process categorical features
        df = self.process_categorical_features(df)

        # Process numeric features
        df = self.process_numeric_features(df)

        # Select features to keep
        features_to_keep = [
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
            'listing_age_days',
            'log_mean_landmark_distance',  # Keep this
            'log_min_landmark_distance',   # Keep this
            'is_entire_place',
            'is_private_room',
            'is_hotel',
            'is_shared',
            'is_premium_entire',
            'is_standard_entire'
        ]


        if not is_test:
            features_to_keep.append('price')

        # Keep only selected features
        df = df[features_to_keep]

        # Add back ID column for test data
        if is_test and id_column is not None:
            df['id'] = id_column

        return df


def main():
    processor = AirbnbDataProcessor()

    try:
        # Process training data
        print("Processing training data...")
        train_df = processor.process_data('train.csv', is_test=False)
        train_df.to_csv('processed_train.csv', index=False)

        # Print information about the processed training dataset
        print("\nProcessed training data shape:", train_df.shape)
        print("\nTraining feature distributions:")
        print(train_df.describe().round(2))

        # Print class distribution
        if 'price' in train_df.columns:
            print("\nPrice category distribution:")
            print(train_df['price'].value_counts(normalize=True).round(3))

        # Process test data using the same processor (with learned mappings)
        print("\nProcessing test data...")
        test_df = processor.process_data('test.csv', is_test=True)
        test_df.to_csv('processed_test.csv', index=False)

        # Print information about the processed test dataset
        print("\nProcessed test data shape:", test_df.shape)
        print("\nTest feature distributions:")
        print(test_df.describe().round(2))

        # Verify feature alignment between train and test
        train_features = set(train_df.columns) - {'price'}  # Remove price from comparison
        test_features = set(test_df.columns) - {'id'}  # Remove id from comparison

        if train_features != test_features:
            print("\nWarning: Feature mismatch between train and test sets!")
            print("Features in train but not in test:", train_features - test_features)
            print("Features in test but not in train:", test_features - train_features)
        else:
            print("\nSuccess: Train and test sets have matching features!")
            print("Number of features:", len(train_features))
            print("\nFeature list:")
            print(sorted(train_features))

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
