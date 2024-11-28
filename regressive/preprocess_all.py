# preprocess_all.py
from preprocess import AirbnbDataProcessor

def main():
    processor = AirbnbDataProcessor()

    # Process training data first
    print("Processing training data...")
    train_df = processor.process_data('train.csv', is_test=False)
    train_df.to_csv('processed_train.csv', index=False)

    # Process test data using established mappings
    print("\nProcessing test data...")
    test_df = processor.process_data('test.csv', is_test=True)
    test_df.to_csv('processed_test.csv', index=False)

    # Verify columns match (except price and id)
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns) - {'id'}  # Remove id from comparison
    price_col = {'price'}

    if train_cols - price_col != test_cols:
        print("\nWarning: Feature mismatch between train and test sets!")
        print("Features in train but not in test:", train_cols - price_col - test_cols)
        print("Features in test but not in train:", test_cols - (train_cols - price_col))
    else:
        print("\nSuccess: Train and test sets have matching features!")
        print("Number of features for modeling:", len(train_cols - price_col))

if __name__ == "__main__":
    main()
