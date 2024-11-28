import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from models import get_model
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_importance():
    """Analyze feature importance using XGBoost and correlations"""
    # Load processed data
    print("Loading data...")
    df = pd.read_csv('processed_train.csv')
    X = df.drop('price', axis=1)
    y = df['price']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost model for feature importance
    print("\nTraining XGBoost model for feature importance analysis...")
    xgb_model = get_model('xgboost')
    xgb_model.fit(X_train, y_train)

    # Get feature importance from XGBoost
    xgb_importance = pd.DataFrame({
        'feature': X.columns,
        'xgb_importance': xgb_model.model.feature_importances_
    }).sort_values('xgb_importance', ascending=False)

    # Calculate correlation with target
    correlations = pd.DataFrame({
        'feature': X.columns,
        'correlation': [abs(X[col].corr(y)) for col in X.columns]
    }).sort_values('correlation', ascending=False)

    # Combine metrics
    importance_df = xgb_importance.merge(correlations, on='feature')

    # Normalize metrics
    for col in ['xgb_importance', 'correlation']:
        importance_df[f'{col}_normalized'] = importance_df[col] / importance_df[col].max()

    # Calculate average importance
    importance_df['avg_importance'] = importance_df[[
        'xgb_importance_normalized',
        'correlation_normalized'
    ]].mean(axis=1)

    # Sort by average importance
    importance_df = importance_df.sort_values('avg_importance', ascending=False)

    # Create visualizations
    create_importance_plots(importance_df)

    return importance_df

def create_importance_plots(importance_df):
    """Create feature importance visualizations"""
    # 1. Overall feature importance bar plot
    plt.figure(figsize=(12, 8))
    plt.barh(
        importance_df.head(15)['feature'],
        importance_df.head(15)['avg_importance'],
        color='skyblue'
    )
    plt.title('Top 15 Features by Average Importance')
    plt.xlabel('Average Normalized Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_overall.png')
    plt.close()

    # 2. Comparison of different importance metrics
    plt.figure(figsize=(12, 8))
    top_10_features = importance_df.head(10)['feature']

    x = np.arange(len(top_10_features))
    width = 0.35

    plt.bar(x - width/2,
           importance_df.head(10)['xgb_importance_normalized'],
           width,
           label='XGBoost',
           color='skyblue')
    plt.bar(x + width/2,
           importance_df.head(10)['correlation_normalized'],
           width,
           label='Correlation',
           color='lightcoral')

    plt.xlabel('Feature')
    plt.ylabel('Normalized Importance')
    plt.title('Feature Importance by Different Metrics (Top 10)')
    plt.xticks(x, top_10_features, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance_comparison.png')
    plt.close()

def analyze_feature_interactions(importance_df):
    """Analyze interactions between top features"""
    print("\nAnalyzing feature interactions...")
    df = pd.read_csv('processed_train.csv')
    top_features = importance_df.head(5)['feature'].tolist()

    correlations = df[top_features].corr()

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(correlations, cmap='coolwarm', aspect='auto')
    plt.colorbar()

    # Add correlation values
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            plt.text(j, i, f'{correlations.iloc[i, j]:.2f}',
                    ha='center', va='center')

    plt.xticks(range(len(correlations)), correlations.columns, rotation=45, ha='right')
    plt.yticks(range(len(correlations)), correlations.columns)
    plt.title('Correlations Between Top Features')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()

    return correlations

def main():
    # Analyze feature importance
    importance_df = analyze_feature_importance()

    # Print top features
    print("\nTop 15 features by average importance:")
    print(importance_df[['feature', 'avg_importance']].head(15).to_string(index=False))

    # Analyze feature interactions
    correlations = analyze_feature_interactions(importance_df)

    # Save detailed results
    importance_df.to_csv('feature_importance_detailed.csv', index=False)

    print("\nVisualization files created:")
    print("- feature_importance_overall.png")
    print("- feature_importance_comparison.png")
    print("- feature_correlations.png")
    print("- feature_importance_detailed.csv")

if __name__ == "__main__":
    main()
