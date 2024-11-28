import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import get_model
import seaborn as sns

def analyze_max_depth_impact():
    """Analyze the impact of max_depth on model performance"""
    print("Loading data...")
    train_df = pd.read_csv('processed_train.csv')
    
    # Separate features and target
    X = train_df.drop('price', axis=1)
    y = train_df['price']
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Test different max_depth values
    max_depths = range(3, 16, 1)  # Test depths from 3 to 15
    results = []
    
    print("\nTesting different max_depth values...")
    for depth in max_depths:
        # Create and train model with current max_depth
        model = get_model('xgboost', params={'max_depth': depth})
        model.fit(X_train, y_train)
        
        # Evaluate on both training and validation sets
        train_eval = model.evaluate(X_train, y_train)
        val_eval = model.evaluate(X_val, y_val)
        
        results.append({
            'max_depth': depth,
            'train_accuracy': train_eval['accuracy'],
            'val_accuracy': val_eval['accuracy'],
            'train_adjacent_accuracy': train_eval['adjacent_accuracy'],
            'val_adjacent_accuracy': val_eval['adjacent_accuracy']
        })
        
        print(f"max_depth={depth}: "
              f"Val Accuracy={val_eval['accuracy']:.4f}, "
              f"Val Adjacent Accuracy={val_eval['adjacent_accuracy']:.4f}")
    
    results_df = pd.DataFrame(results)
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy metrics
    plt.subplot(1, 2, 1)
    plt.plot(results_df['max_depth'], results_df['train_accuracy'], 
             marker='o', label='Training Accuracy')
    plt.plot(results_df['max_depth'], results_df['val_accuracy'], 
             marker='o', label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Impact of Max Depth on Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot adjacent accuracy metrics
    plt.subplot(1, 2, 2)
    plt.plot(results_df['max_depth'], results_df['train_adjacent_accuracy'], 
             marker='o', label='Training Adjacent Accuracy')
    plt.plot(results_df['max_depth'], results_df['val_adjacent_accuracy'], 
             marker='o', label='Validation Adjacent Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Adjacent Accuracy')
    plt.title('Impact of Max Depth on Adjacent Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('max_depth_analysis.png')
    plt.close()
    
    # Save numerical results
    results_df.to_csv('max_depth_analysis.csv', index=False)
    
    # Find optimal max_depth
    best_depth = results_df.loc[
        results_df['val_accuracy'].idxmax(), 'max_depth'
    ]
    
    print("\nAnalysis Results:")
    print(f"Optimal max_depth: {best_depth}")
    print(f"Best validation accuracy: {results_df['val_accuracy'].max():.4f}")
    print(f"Best validation adjacent accuracy: {results_df['val_adjacent_accuracy'].max():.4f}")
    print("\nResults saved to max_depth_analysis.csv")
    print("Plots saved to max_depth_analysis.png")

if __name__ == "__main__":
    analyze_max_depth_impact()
