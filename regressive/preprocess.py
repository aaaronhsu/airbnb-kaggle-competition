import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class AirbnbDataProcessor:
    def __init__(self):
        # Initialize scalers
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()

        # Initialize all the mapping dictionaries
        self.property_type_mapping = {
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

        self.response_time_mapping = {
            'within an hour': 'very_fast',
            'within a few hours': 'fast',
            'within a day': 'moderate',
            'a few days or more': 'slow'
        }

        self.room_type_mapping = {
            'Entire home/apt': 'entire_place',
            'Private room': 'private_room',
            'Hotel room': 'hotel_room',
            'Shared room': 'shared_room'
        }

        self.bathrooms_mapping = {
            # Private full bathrooms
            '1 bath': 'one_bath',
            '1 private bath': 'one_bath',
            '1.5 baths': 'one_plus_bath',
            '2 baths': 'two_bath',
            '2.5 baths': 'two_plus_bath',
            '3 baths': 'three_plus_bath',
            '3.5 baths': 'three_plus_bath',
            '4 baths': 'three_plus_bath',
            '4.5 baths': 'three_plus_bath',
            '5 baths': 'three_plus_bath',
            '5.5 baths': 'three_plus_bath',
            '6 baths': 'three_plus_bath',
            '7 baths': 'three_plus_bath',
            '10.5 baths': 'three_plus_bath',
            '11.5 baths': 'three_plus_bath',

            # Shared bathrooms
            '1 shared bath': 'one_shared',
            '1.5 shared baths': 'one_shared',
            '2 shared baths': 'two_plus_shared',
            '2.5 shared baths': 'two_plus_shared',
            '3 shared baths': 'two_plus_shared',
            '3.5 shared baths': 'two_plus_shared',
            '4 shared baths': 'two_plus_shared',
            '4.5 shared baths': 'two_plus_shared',
            '5 shared baths': 'two_plus_shared',
            '6 shared baths': 'two_plus_shared',

            # Half baths
            'Half-bath': 'half_bath',
            'Private half-bath': 'half_bath',
            'Shared half-bath': 'half_bath',

            # No baths
            '0 baths': 'no_bath',
            '0 shared baths': 'no_bath'
        }

        self.premium_amenities = {
            'Pool', 'Private pool', 'Shared pool',
            'Hot tub', 'Sauna', 'Indoor fireplace',
            'Gym', 'Piano', 'Pool table',
            'City view', 'Water view', 'Park view', 'Beach view', 'Waterfront',
            'Beach access', 'Ski-in/ski-out',
            'Private backyard', 'Patio', 'Balcony', 'Garden',
            'BBQ grill', 'Outdoor dining area',
            "Chef's kitchen", 'Wine cooler', 'Espresso machine',
            'Smart TV', 'Sound system', 'Game console',
            'Dedicated workspace', 'Office',
            'Bathtub', 'Rain shower', 'Multiple shower heads',
            'Concierge service', 'Cleaning service'
        }

        self.boolean_columns = [
            'instant_bookable',
            'host_has_profile_pic',
            'host_identity_verified'
        ]

        self.binary_features = {
            'host_is_superhost': {'t': 1, 'f': 0},
            'has_availability': {'t': 1, 'f': 0},
        }


        self.low_importance_features = [
            'review_score_variance',
            'host_response_rate_cat',
            'availability_ratio',
            'number_of_reviews',
            'review_scores_value',
            'reviews_per_day',
            'host_has_profile_pic',
            'bathroom_category_half_bath',
            'property_type_simplified_other',
            'is_professional_host',
            'bathroom_category_nan',
            'host_listing_count_cat',
            'host_is_superhost',
            'has_availability',
            'superhost_high_rating'
        ]

        self.redundant_features = [
            'availability_30',
            'availability_365',
            'review_scores_rating',
            'review_scores_accuracy',
            'review_scores_value',
            'review_scores_cleanliness',
            'review_scores_communication',
            'host_since',
            'first_review',
            'last_review',
            'last_review_month',
            'last_review_quarter',
            'property_type_simplified_private_room',
            'property_type_simplified_shared_budget',
            'property_type_simplified_entire_standard',
            'host_listing_count_cat',
            'response_time_category'
        ]

        self.important_features = [
            'neighborhood_price_index',
            'room_type_category_private_room',
            'room_type_category_entire_place',
            'accommodates',
            'beds',
            'bedrooms',
            'bathroom_category_one_shared',
            'bathroom_category_one_bath',
            'bathroom_category_two_plus_shared',
            'borough',
            'calculated_host_listings_count_entire_homes',
            'calculated_host_listings_count_private_rooms',
            'premium_amenities_count',
            'instant_bookable',
            'host_listings_count',
            'minimum_nights',
            'availability_rate_30',
            'avg_review_score',
            'listing_age_days'
        ]

    def load_data(self, filepath):
        """Load the data with appropriate date parsing"""
        return pd.read_csv(filepath, parse_dates=['host_since', 'first_review', 'last_review'])

    def parse_amenities(self, amenities_str):
        """Convert amenities string to list"""
        if pd.isna(amenities_str):
            return []
        return amenities_str.replace('[', '').replace(']', '').replace('"', '').split(', ')

    def parse_reviews(self, reviews_str):
        """Convert reviews string to list"""
        if pd.isna(reviews_str):
            return []
        return reviews_str.split("\n---------------------------------\n")

    def process_property_types(self, df):
        """Process property types using mapping"""
        df['property_type_simplified'] = df['property_type'].map(self.property_type_mapping)
        df['property_type_simplified'] = df['property_type_simplified'].fillna('other')
        return df

    def get_neighborhood_category(self, neighborhood, borough):
        """Get neighborhood category with fallback to borough"""
        borough_mapping = {
            'Manhattan': 'manhattan_other',
            'Brooklyn': 'brooklyn_other',
            'Queens': 'queens_other',
            'Bronx': 'bronx_other',
            'Staten Island': 'staten_island_other'
        }
        return borough_mapping.get(borough, 'other')

    def process_neighborhoods(self, df, is_test=False):
        """Process neighborhoods and create neighborhood price index"""
        # First create the basic neighborhood category
        df['neighborhood_category'] = df.apply(
            lambda x: self.get_neighborhood_category(
                x['neighbourhood_cleansed'],
                x['neighbourhood_group_cleansed']
            ),
            axis=1
        )

        if is_test:
            # For test data, use the neighborhood price mapping from training data
            if not hasattr(self, 'neighborhood_price_mapping'):
                raise ValueError("No neighborhood price mapping found. Process training data first.")

            df['neighborhood_price_index'] = (
                df['neighbourhood_cleansed']
                .map(self.neighborhood_price_mapping)
                .fillna(1.0)  # Use citywide average for unknown neighborhoods
            )

            # Use stored bins for consistent categorization
            df['neighborhood_price_category'] = pd.cut(
                df['neighborhood_price_index'],
                bins=self.neighborhood_price_bins,
                labels=self.neighborhood_price_labels,
                include_lowest=True
            )

        else:
            # Calculate neighborhood price index for training data
            neighborhood_prices = df.groupby('neighbourhood_cleansed')['price'].agg([
                'mean',
                'median',
                'count'
            ]).reset_index()

            # Only consider neighborhoods with enough listings
            min_listings = 5
            valid_neighborhoods = neighborhood_prices[neighborhood_prices['count'] >= min_listings].copy()

            # Calculate citywide average price
            citywide_avg = df['price'].mean()

            # Calculate price index
            valid_neighborhoods.loc[:, 'price_index'] = valid_neighborhoods['mean'] / citywide_avg

            # Create mapping dictionary and store it
            self.neighborhood_price_mapping = dict(zip(
                valid_neighborhoods['neighbourhood_cleansed'],
                valid_neighborhoods['price_index']
            ))

            # Apply mapping
            df['neighborhood_price_index'] = (
                df['neighbourhood_cleansed']
                .map(self.neighborhood_price_mapping)
                .fillna(1.0)
            )

            # Create and store bins for price categories
            self.neighborhood_price_bins = [-np.inf] + list(
                df['neighborhood_price_index'].quantile([0.2, 0.4, 0.6, 0.8, 1.0])
            )
            self.neighborhood_price_labels = ['very_low', 'low', 'medium', 'high', 'very_high']

            # Create categorical version
            df['neighborhood_price_category'] = pd.cut(
                df['neighborhood_price_index'],
                bins=self.neighborhood_price_bins,
                labels=self.neighborhood_price_labels,
                include_lowest=True
            )

        # Print statistics
        print("\nNeighborhood Price Index Statistics:")
        stats_df = pd.DataFrame({
            'count': df.groupby('neighborhood_price_category', observed=True)['neighborhood_price_index'].count(),
            'mean_index': df.groupby('neighborhood_price_category', observed=True)['neighborhood_price_index'].mean()
        })

        if not is_test:
            stats_df['mean_price'] = df.groupby('neighborhood_price_category', observed=True)['price'].mean()

        print(stats_df.round(2))

        return df

    def process_categorical_features(self, df):
        """Process all other categorical features"""
        df['response_time_category'] = df['host_response_time'].map(self.response_time_mapping)
        df['room_type_category'] = df['room_type'].map(self.room_type_mapping)
        df['bathroom_category'] = df['bathrooms_text'].map(self.bathrooms_mapping)
        return df

    def process_amenities(self, df):
        """Process amenities and count premium ones"""
        self.premium_amenities_lower = {amenity.lower() for amenity in self.premium_amenities}

        def count_premium_amenities(amenities_list):
            if not amenities_list:
                return 0
            amenities_set = {amenity.lower() for amenity in amenities_list}
            return len(amenities_set.intersection(self.premium_amenities_lower))

        df['premium_amenities_count'] = df['amenities_list'].apply(count_premium_amenities)
        return df

    def process_binary_features(self, df):
        """Process binary features with proper handling of missing values"""
        for feature, mapping in self.binary_features.items():
            if feature in df.columns:
                # Convert to string first to handle any non-standard values
                df[feature] = df[feature].astype(str).str.lower()
                # Map known values and fill unknown with 0
                df[feature] = df[feature].map(mapping).fillna(0)
        return df

    def process_boolean_columns(self, df):
        """Convert boolean columns to 1/0 integers"""
        for column in self.boolean_columns:
            if column in df.columns:
                df[column] = df[column].astype(int)
        return df

    def create_time_features(self, df):
        """Create time-based features"""
        # Current timestamp for relative calculations
        current_time = pd.Timestamp.now()

        # Listing age
        df['listing_age_days'] = (current_time - df['host_since']).dt.total_seconds() / (24*60*60)

        # Review period
        df['review_period_days'] = (df['last_review'] - df['first_review']).dt.total_seconds() / (24*60*60)

        # Last review features
        df['last_review_month'] = df['last_review'].dt.month
        df['last_review_quarter'] = df['last_review'].dt.quarter

        return df

    def process_review_scores(self, df):
        """Process review scores"""
        review_score_cols = [col for col in df.columns if col.startswith('review_scores_')]
        df['avg_review_score'] = df[review_score_cols].mean(axis=1)
        df['review_score_variance'] = df[review_score_cols].var(axis=1)
        df['reviews_per_day'] = df['number_of_reviews'] / df['review_period_days'].replace(0, 1)
        return df

    def process_availability(self, df):
        """Process availability metrics"""
        df['availability_rate_30'] = df['availability_30'] / 30
        df['availability_rate_365'] = df['availability_365'] / 365
        df['availability_ratio'] = df['availability_30'] / df['availability_365'].replace(0, 365)
        return df

    def create_host_features(self, df):
        """Create host-related features"""
        # Host experience
        current_time = pd.Timestamp.now()
        df['host_experience_years'] = (current_time - df['host_since']).dt.total_seconds() / (365.25*24*60*60)

        # Response rate categories
        def categorize_response_rate(rate):
            if pd.isna(rate):
                return 'missing'
            rate = float(rate)
            if rate < 70:
                return 'low'
            elif rate < 85:
                return 'medium'
            elif rate < 95:
                return 'high'
            else:
                return 'very_high'

        def categorize_acceptance_rate(rate):
            if pd.isna(rate):
                return 'missing'
            rate = float(rate)
            if rate < 50:
                return 'low'
            elif rate < 75:
                return 'medium'
            elif rate < 90:
                return 'high'
            else:
                return 'very_high'

        df['host_response_rate_cat'] = df['host_response_rate'].apply(categorize_response_rate)
        df['host_acceptance_rate_cat'] = df['host_acceptance_rate'].apply(categorize_acceptance_rate)

        # Listing count categories
        df['host_listing_count_cat'] = pd.cut(
            df['host_listings_count'],
            bins=[-float('inf'), 1, 3, 10, float('inf')],
            labels=['single', 'few', 'multiple', 'professional']
        )

        # Binary features
        df['is_experienced_host'] = (df['host_experience_years'] > 2).astype(int)
        df['is_professional_host'] = (df['host_listings_count'] > 3).astype(int)

        return df

    def create_location_features(self, df):
        """Create location-based features"""
        df['borough'] = df['neighbourhood_group_cleansed'].map({
            'Manhattan': 1,
            'Brooklyn': 2,
            'Queens': 3,
            'Bronx': 4,
            'Staten Island': 5
        })
        return df

    def create_interaction_features(self, df):
        """Create interaction features"""
        # Premium location based on neighborhood price index
        df['premium_location_amenities'] = (
            (df['neighborhood_price_index'] > df['neighborhood_price_index'].median()) &
            (df['premium_amenities_count'] > 3)
        ).astype(int)

        df['superhost_high_rating'] = (
            (df['host_is_superhost'] == 1) &
            (df['review_scores_rating'] > 4.5)
        ).astype(int)

        return df

    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        df_cleaned = df.copy()

        # Handle specific columns first
        binary_cols = ['host_is_superhost', 'has_availability']
        for col in binary_cols:
            if col in df_cleaned.columns:
                df_cleaned[col] = df_cleaned[col].fillna(0)

        # Convert categorical columns to string type
        categorical_columns = df_cleaned.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            df_cleaned[col] = df_cleaned[col].astype(str)

        # Numeric columns: fill with median
        numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())

        # Categorical columns: fill with mode
        categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])

        return df_cleaned

    def encode_categorical_features(self, df):
        """Encode categorical features"""
        df_encoded = df.copy()

        # Convert categorical columns to string type first
        categorical_columns = df_encoded.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            df_encoded[col] = df_encoded[col].astype(str)

        nominal_features = [
            'property_type_simplified',
            'room_type_category',
            'bathroom_category',
        ]

        ordinal_features = {
            'host_response_rate_cat': ['missing', 'low', 'medium', 'high', 'very_high'],
            'host_acceptance_rate_cat': ['missing', 'low', 'medium', 'high', 'very_high'],
            'host_listing_count_cat': ['single', 'few', 'multiple', 'professional'],
            'response_time_category': ['missing', 'slow', 'moderate', 'fast', 'very_fast'],
            'neighborhood_price_category': ['very_low', 'low', 'medium', 'high', 'very_high']
        }

        # One-hot encoding for nominal features
        for feature in nominal_features:
            if feature in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[feature], prefix=feature)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(columns=[feature], inplace=True)

        # Label encoding for ordinal features
        for feature, ordering in ordinal_features.items():
            if feature in df_encoded.columns:
                if pd.api.types.is_categorical_dtype(df_encoded[feature]):
                    df_encoded[feature] = df_encoded[feature].astype(str)

                mapping = {cat: i for i, cat in enumerate(ordering)}
                df_encoded[feature] = df_encoded[feature].map(mapping)
                df_encoded[feature] = df_encoded[feature].fillna(-1).astype(int)

        return df_encoded

    def select_features(self, df):
        """Apply feature selection to the dataset"""
        # First, drop text and unnecessary columns
        columns_to_drop = [
            'name', 'description', 'reviews', 'reviews_list',
            'amenities', 'amenities_list', 'bathrooms_text',
            'property_type', 'room_type', 'host_response_time',
            'host_verifications', 'latitude', 'longitude',
            'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
            'host_total_listings_count', 'bathrooms',
            'maximum_nights'
        ]

        # Drop initial columns
        df_selected = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

        # Drop low importance and redundant features
        features_to_drop = self.low_importance_features + self.redundant_features
        df_selected = df_selected.drop(columns=[col for col in features_to_drop if col in df_selected.columns], errors='ignore')

        # Verify important features are present
        missing_important = [feat for feat in self.important_features if feat not in df_selected.columns]
        if missing_important:
            print(f"Warning: Missing important features: {missing_important}")

        print(f"\nFeature selection: Reduced features from {df.shape[1]} to {df_selected.shape[1]}")
        print(f"Retained features: {sorted(df_selected.columns.tolist())}")

        return df_selected

    def prepare_final_features(self, df):
        """Prepare final features for modeling"""
        # Apply feature selection before other transformations
        df_final = self.select_features(df)

        # Define categorical columns that should be one-hot encoded
        categorical_columns = [
            'neighborhood_category',
            'neighborhood_price_category',
            'bathroom_category_no_bath',
            'bathroom_category_one_bath',
            'bathroom_category_one_plus_bath',
            'bathroom_category_one_shared',
            'bathroom_category_three_plus_bath',
            'bathroom_category_two_bath',
            'bathroom_category_two_plus_bath',
            'bathroom_category_two_plus_shared',
            'host_acceptance_rate_cat',
            'room_type_category_entire_place',
            'room_type_category_hotel_room',
            'room_type_category_private_room',
            'room_type_category_shared_room',
            'property_type_simplified_entire_premium',
            'property_type_simplified_hotel_style'
        ]

        # One-hot encode categorical columns
        for col in categorical_columns:
            if col in df_final.columns:
                # Convert to string first to ensure proper encoding
                df_final[col] = df_final[col].astype(str)
                # Create dummy variables
                dummies = pd.get_dummies(df_final[col], prefix=col)
                # Add dummy columns to dataframe
                df_final = pd.concat([df_final, dummies], axis=1)
                # Drop original column
                df_final = df_final.drop(columns=[col])

        # Define numeric columns for different transformations
        log_transform_features = [
            'host_listings_count',
            'minimum_nights',
            'listing_age_days'
        ]

        scale_features = [
            'avg_review_score',
            'neighborhood_price_index',
            'availability_rate_30'
        ]

        # Store scalers as class attributes if not already present
        if not hasattr(self, 'standard_scaler'):
            self.standard_scaler = StandardScaler()
            self.minmax_scaler = MinMaxScaler()

        # Log transform highly skewed features
        for col in log_transform_features:
            if col in df_final.columns:
                df_final[col] = np.log1p(df_final[col])

        # Standard scale continuous features
        scale_cols = [col for col in scale_features if col in df_final.columns]
        if scale_cols:
            df_final[scale_cols] = self.standard_scaler.fit_transform(df_final[scale_cols])

        # Handle boolean columns
        boolean_columns = [
            'instant_bookable',
            'host_identity_verified',
            'is_experienced_host',
            'premium_location_amenities'
        ]

        for col in boolean_columns:
            if col in df_final.columns:
                df_final[col] = df_final[col].astype(int)

        # Handle any remaining infinities or NaN values
        df_final = df_final.replace([np.inf, -np.inf], np.nan)

        # Ensure all remaining columns are numeric and handle missing values
        for col in df_final.columns:
            if not pd.api.types.is_numeric_dtype(df_final[col]):
                try:
                    df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Dropping column {col} - could not convert to numeric. Error: {str(e)}")
                    df_final = df_final.drop(columns=[col])
                    continue

            # Fill missing values with median for numeric columns
            if df_final[col].isnull().any():
                df_final[col] = df_final[col].fillna(df_final[col].median())

        return df_final

    def process_data(self, filepath, is_test=False):
        """Main processing function"""
        # Load data
        df = self.load_data(filepath)

        if is_test:
            df['price'] = 0

        # Convert categorical columns to string type
        categorical_columns = df.select_dtypes(include=['category']).columns
        for col in categorical_columns:
            df[col] = df[col].astype(str)

        # Parse strings to lists
        df['amenities_list'] = df['amenities'].apply(self.parse_amenities)
        df['reviews_list'] = df['reviews'].apply(self.parse_reviews)

        # Process all features
        df = self.process_property_types(df)
        df = self.process_neighborhoods(df, is_test=is_test)
        df = self.process_categorical_features(df)
        df = self.process_amenities(df)
        df = self.process_binary_features(df)
        df = self.process_boolean_columns(df)

        # Create derived features
        df = self.create_time_features(df)
        df = self.process_review_scores(df)
        df = self.process_availability(df)
        df = self.create_host_features(df)
        df = self.create_location_features(df)
        df = self.create_interaction_features(df)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Encode categorical features
        df = self.encode_categorical_features(df)

        # Final preparation including feature selection
        df = self.prepare_final_features(df)

        if is_test:
            df = df.drop(columns=['price'])

        # Print final dataset information
        print("\nFinal Dataset Information:")
        print(f"Shape: {df.shape}")
        print("\nFeature types:")
        print(df.dtypes.value_counts())

        return df


def main():
    processor = AirbnbDataProcessor()
    input_file = "train.csv"
    output_file = "processed_airbnb_data_encoded.csv"

    try:
        print("Processing data...")
        processed_df = processor.process_data(input_file)

        print("\nProcessed and encoded dataset info:")
        print(f"Shape: {processed_df.shape}")
        print("\nFeature types:")
        print(processed_df.dtypes.value_counts())

        print("\nSample of encoded ordinal features:")
        ordinal_features = ['host_response_rate_cat', 'host_acceptance_rate_cat',
                          'host_listing_count_cat', 'response_time_category']
        print(processed_df[ordinal_features].head())

        print(f"\nSaving processed data to {output_file}")
        processed_df.to_csv(output_file, index=False)

    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
