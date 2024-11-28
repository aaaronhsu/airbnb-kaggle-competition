import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import shap

def analyze_feature_importance(X, y):
    """Analyze and visualize feature importance using multiple methods"""
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # 1. Correlation with target
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)

    # 2. Random Forest importance
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_importance = pd.DataFrame({
        'feature': X.columns,
        'rf_importance': rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)

    # 3. XGBoost importance
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.feature_importances_
    }).sort_values('xgb_importance', ascending=False)

    # 4. SHAP values
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'shap_importance': np.abs(shap_values).mean(0)
    }).sort_values('shap_importance', ascending=False)

    # Combine all metrics
    results = (correlations
              .merge(rf_importance, on='feature')
              .merge(xgb_importance, on='feature')
              .merge(shap_importance, on='feature'))

    # Normalize scores
    for col in ['correlation', 'rf_importance', 'xgb_importance', 'shap_importance']:
        results[f'{col}_normalized'] = results[col] / results[col].max()

    # Calculate average importance
    importance_cols = [col for col in results.columns if col.endswith('_normalized')]
    results['avg_importance'] = results[importance_cols].mean(axis=1)

    return results, xgb_model, X_test

def plot_feature_importance(importance_df, top_n=30):
    """Create visualizations of feature importance"""
    # Set figure style
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 12

    # 1. Overall feature importance
    fig, ax = plt.subplots()
    data = importance_df.head(top_n)

    # Create bar plot
    sns.barplot(data=data, x='avg_importance', y='feature', ax=ax)
    ax.set_title(f'Top {top_n} Features by Average Importance', pad=20)
    ax.set_xlabel('Average Normalized Importance')
    ax.set_ylabel('Feature')

    plt.tight_layout()
    plt.savefig('feature_importance_overall.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Comparison of different importance metrics
    fig, ax = plt.subplots()
    importance_metrics = ['correlation_normalized', 'rf_importance_normalized',
                         'xgb_importance_normalized', 'shap_importance_normalized']

    # Melt the dataframe for easier plotting
    plot_data = importance_df.head(20).melt(
        id_vars=['feature'],
        value_vars=importance_metrics,
        var_name='Metric',
        value_name='Importance'
    )

    # Create grouped bar plot
    sns.barplot(data=plot_data, x='Importance', y='feature', hue='Metric', ax=ax)
    ax.set_title('Feature Importance by Different Metrics (Top 20)', pad=20)
    ax.set_xlabel('Normalized Importance')
    ax.set_ylabel('Feature')

    # Adjust legend
    ax.legend(title='Metric', bbox_to_anchor=(1.05, 1))

    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_shap_summary(xgb_model, X_test):
    """Create SHAP summary plot"""
    plt.figure(figsize=(12, 8))
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Load processed data
    print("Loading data...")
    df = pd.read_csv('processed_train.csv')

    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']

    # Add engineered features
    from feature_engineering import add_engineered_features
    X = add_engineered_features(X)

    # Remove any non-numeric columns
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
    X = X[numeric_cols]

    print("Analyzing feature importance...")
    importance_results, xgb_model, X_test = analyze_feature_importance(X, y)

    print("Creating visualizations...")
    plot_feature_importance(importance_results)
    plot_shap_summary(xgb_model, X_test)

    # Save detailed results
    importance_results.to_csv('feature_importance_detailed.csv', index=False)

    print("\nTop 20 most important features:")
    print(importance_results[['feature', 'avg_importance']].head(20).to_string())

    print("\nVisualization files created:")
    print("- feature_importance_overall.png")
    print("- feature_importance_comparison.png")
    print("- shap_summary.png")
    print("- feature_importance_detailed.csv")

if __name__ == "__main__":
    main()
