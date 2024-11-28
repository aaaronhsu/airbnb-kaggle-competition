import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models import get_model
import xgboost as xgb
import graphviz

def visualize_xgboost_tree():
    """Create and save a visualization of an XGBoost tree"""
    print("Loading data...")
    train_df = pd.read_csv('processed_train.csv')
    
    # Separate features and target
    X = train_df.drop('price', axis=1)
    y = train_df['price']
    
    # Create and train model with specified max_depth
    print("\nTraining XGBoost model with max_depth=9...")
    model = get_model('xgboost', params={'max_depth': 9})
    model.fit(X, y)
    
    # Get feature names
    feature_names = list(X.columns)
    
    # Create tree visualization using dump_model
    print("\nCreating tree visualization...")
    dump = model.model.get_booster().get_dump(dump_format='dot')
    
    # Save the first tree
    with open('xgboost_tree.dot', 'w') as f:
        f.write(dump[0])
    
    # Convert to PDF using graphviz
    print("Converting to PDF...")
    try:
        graph = graphviz.Source(dump[0])
        graph.render('xgboost_tree', format='pdf', cleanup=True)
        print("\nTree visualization saved as 'xgboost_tree.pdf'")
    except Exception as e:
        print(f"Error rendering PDF: {str(e)}")
        print("Dot file saved as 'xgboost_tree.dot'")

if __name__ == "__main__":
    visualize_xgboost_tree()
