import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import shap

class FeatureImportanceAnalyzer:
    def __init__(self, data_path, target_column='price', random_state=42):
        self.data_path = data_path
        self.target_column = target_column
        self.random_state = random_state
        self.model = None
        self.X = None
        self.y = None
        self.feature_names = None

    def load_data(self):
        """Load the preprocessed data"""
        print("Loading data...")
        df = pd.read_csv(self.data_path)

        # Separate features and target
        self.X = df.drop(columns=[self.target_column])
        self.y = df[self.target_column]
        self.feature_names = self.X.columns.tolist()

        print(f"Loaded {len(self.feature_names)} features")

        # Print basic statistics
        print("\nFeature statistics:")
        print(self.X.describe().round(2).T)

        return self.X, self.y

    def train_model(self):
        """Train the Random Forest model"""
        print("\nTraining Random Forest model...")
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(self.X, self.y)

    def get_feature_importance(self):
        """Get feature importance from multiple methods"""
        importance_dict = {}

        # 1. Default feature importance
        importance_dict['default'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 2. Permutation importance
        print("\nCalculating permutation importance...")
        perm_importance = permutation_importance(
            self.model, self.X, self.y,
            n_repeats=10,
            random_state=self.random_state
        )

        importance_dict['permutation'] = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)

        return importance_dict

    def evaluate_model(self):
        """Evaluate model performance using cross-validation"""
        print("\nEvaluating model performance...")

        # Split data for holdout evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )

        # Cross-validation scores
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=5, scoring='r2', n_jobs=-1
        )

        # Predictions on test set
        y_pred = self.model.predict(X_test)

        metrics = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred)
        }

        return metrics

    def plot_feature_importance(self, importance_df, method='default', top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 8))

        # Get top N features
        plot_data = importance_df.head(top_n)

        # Create bar plot
        sns.barplot(
            data=plot_data,
            x='importance',
            y='feature',
            palette='viridis'
        )

        plt.title(f'Top {top_n} Most Important Features ({method} importance)')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        # Save plot
        plt.savefig(f'feature_importance_{method}.png')
        plt.close()

    def analyze_shap_values(self, sample_size=100):
        """Analyze SHAP values for feature importance"""
        print("\nCalculating SHAP values...")

        # Take a sample of the data if it's too large
        if len(self.X) > sample_size:
            X_sample = self.X.sample(n=sample_size, random_state=self.random_state)
        else:
            X_sample = self.X

        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)

        # Plot SHAP summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, X_sample,
            plot_type="bar",
            show=False
        )
        plt.tight_layout()
        plt.savefig('shap_importance.png')
        plt.close()

        # Calculate and return SHAP-based feature importance
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        return shap_importance

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        # Load and prepare data
        self.load_data()

        # Handle missing values before training
        self.X = self.X.fillna(self.X.mean())

        # Remove any infinite values
        self.X = self.X.replace([np.inf, -np.inf], np.nan)
        self.X = self.X.fillna(self.X.mean())

        # Train model
        self.train_model()

        # Get feature importance
        importance_dict = self.get_feature_importance()

        # Evaluate model
        metrics = self.evaluate_model()

        # Plot feature importance
        for method, importance_df in importance_dict.items():
            self.plot_feature_importance(importance_df, method=method)

        # Calculate SHAP values and importance
        shap_importance = self.analyze_shap_values()
        importance_dict['shap'] = shap_importance

        # Print results
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nTop 10 Most Important Features (Default):")
        print(importance_dict['default'].head(10))

        print("\nTop 10 Most Important Features (Permutation):")
        print(importance_dict['permutation'].head(10))

        print("\nTop 10 Most Important Features (SHAP):")
        print(importance_dict['shap'].head(10))

        # Save results to CSV
        importance_dict['default'].to_csv('feature_importance_default.csv', index=False)
        importance_dict['permutation'].to_csv('feature_importance_permutation.csv', index=False)
        importance_dict['shap'].to_csv('feature_importance_shap.csv', index=False)

def main():
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        data_path='processed_airbnb_data_encoded.csv',
        target_column='price',
        random_state=42
    )

    try:
        # Run analysis
        analyzer.run_analysis()
        print("\nAnalysis completed successfully!")
        print("Results have been saved to CSV files and plots.")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
