import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def analyze_distributions(df):
    """Analyze feature distributions and scaling needs"""
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate statistics for each numeric column
    stats_df = pd.DataFrame({
        'mean': df[numeric_cols].mean(),
        'median': df[numeric_cols].median(),
        'std': df[numeric_cols].std(),
        'min': df[numeric_cols].min(),
        'max': df[numeric_cols].max(),
        'skew': df[numeric_cols].skew(),
        'kurtosis': df[numeric_cols].kurtosis(),
        'unique_values': df[numeric_cols].nunique(),
        'range_ratio': df[numeric_cols].max() / df[numeric_cols].min().replace(0, np.nan)
    }).round(2)
    
    # Calculate scale differences
    stats_df['scale_difference'] = np.log10(stats_df['max'].abs() / stats_df['std'].replace(0, np.nan))
    
    # Sort by scale difference to identify features that might need scaling
    print("\nFeatures that might need scaling (sorted by scale difference):")
    print(stats_df.sort_values('scale_difference', ascending=False))
    
    # Identify highly skewed features
    print("\nHighly skewed features (|skew| > 2):")
    print(stats_df[abs(stats_df['skew']) > 2].sort_values('skew', ascending=False))
    
    return stats_df

def plot_distributions(df, n_cols=3):
    """Plot distributions of numeric features"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    n_features = len(numeric_cols)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        sns.histplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig('feature_distributions.png')
    plt.close()

def analyze_correlations(df):
    """Analyze feature correlations"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlations')
    plt.tight_layout()
    plt.savefig('feature_correlations.png')
    plt.close()
    
    # Print highly correlated features
    print("\nHighly correlated feature pairs (|correlation| > 0.7):")
    high_corr = np.where(np.abs(corr_matrix) > 0.7)
    for i, j in zip(*high_corr):
        if i < j:  # Print each pair only once
            print(f"{numeric_cols[i]} - {numeric_cols[j]}: {corr_matrix.iloc[i, j]:.2f}")

def analyze_outliers(df):
    """Analyze outliers using IQR method"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    outlier_stats = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
        outlier_stats[col] = {
            'outlier_count': outlier_count,
            'outlier_percentage': (outlier_count / len(df)) * 100
        }
    
    outlier_df = pd.DataFrame(outlier_stats).T
    print("\nOutlier analysis:")
    print(outlier_df.sort_values('outlier_percentage', ascending=False))

def main():
    # Load the processed data
    print("Loading data...")
    df = pd.read_csv('processed_train.csv')
    
    # Analyze distributions and scaling needs
    stats_df = analyze_distributions(df)
    
    # Plot distributions
    plot_distributions(df)
    
    # Analyze correlations
    analyze_correlations(df)
    
    # Analyze outliers
    analyze_outliers(df)
    
    # Save detailed statistics
    stats_df.to_csv('feature_statistics.csv')
    
    print("\nAnalysis complete. Check feature_distributions.png and feature_correlations.png for visualizations.")

if __name__ == "__main__":
    main()
