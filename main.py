import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from utils.dataset_generator import generate_synthetic_cybersecurity_data, save_dataset
from models.clustering import AnomalyDetectionModels, AttackDetectionModels
from utils.metrics import calculate_metrics, print_metrics, plot_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os

def preprocess_data(train_df, test_df):
    """Fit preprocessing on training rows only; transform held-out rows."""
    train = train_df.drop(columns=['timestamp', 'attack_type'], errors='ignore')
    test = test_df.drop(columns=['timestamp', 'attack_type'], errors='ignore')
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    categorical = [c for c in train.columns if c in
                   ['src_ip', 'dst_ip', 'src_mac', 'dst_mac', 'protocol', 'tcp_flags']
                   or not pd.api.types.is_numeric_dtype(train[c])]
    numeric = [c for c in train.columns if c not in categorical]
    transformer = ColumnTransformer([
        ('numeric', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical)
    ])
    return transformer.fit_transform(train), transformer.transform(test)


def analyze_attack_types(df):
    """
    Analyze the distribution of different attack types.
    """
    print("\nAttack Type Analysis:")
    print("-" * 50)
    
    # Count attack types
    attack_counts = df['attack_type'].value_counts()
    print("\nAttack Type Distribution:")
    print(attack_counts)
    
    # Plot attack type distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='attack_type')
    plt.title('Distribution of Attack Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/attack_distribution.png')
    plt.close()

def evaluate_models(X, y, X_test, y_test):
    """
    Evaluate multiple models and compare their performance.
    """
    # Initialize models
    anomaly_models = AnomalyDetectionModels()
    attack_models = AttackDetectionModels()
    
    # Dictionary to store results
    results = {}
    
    # Evaluate anomaly detection models
    print("\nEvaluating Anomaly Detection Models:")
    print("-" * 50)
    
    anomaly_methods = {
        'K-means': anomaly_models.detect_anomalies_kmeans,
        'DBSCAN': anomaly_models.detect_anomalies_dbscan,
        'Isolation Forest': anomaly_models.detect_anomalies_isolation_forest
    }
    
    for name, method in anomaly_methods.items():
        print(f"\nEvaluating {name}:")
        predictions = method(X_test)
        metrics = calculate_metrics(y_test, predictions)
        results[f'anomaly_{name.lower()}'] = metrics
        print_metrics(metrics)
        plot_metrics(metrics, f'plots/anomaly_{name.lower()}_metrics.png')
    
    # Evaluate attack detection models
    print("\nEvaluating Attack Detection Models:")
    print("-" * 50)
    
    attack_methods = {
        'KNN': attack_models.detect_attacks_knn,
        'DBSCAN': attack_models.detect_attacks_dbscan,
        'Isolation Forest': attack_models.detect_attacks_isolation_forest
    }
    
    for name, method in attack_methods.items():
        print(f"\nEvaluating {name}:")
        if name == 'KNN':
            # For KNN, we need to train on X and predict on X_test
            predictions = method(X, y, X_test)  # Pass X_test as the prediction data
            metrics = calculate_metrics(y_test, predictions)
        elif name == 'DBSCAN':
            # DBSCAN returns both predictions and metrics
            predictions, dbscan_metrics = method(X_test)
            metrics = calculate_metrics(y_test, predictions)
            # Merge DBSCAN-specific metrics with general metrics
            metrics.update(dbscan_metrics)
        else:
            predictions = method(X_test)
            metrics = calculate_metrics(y_test, predictions)
        
        results[f'attack_{name.lower()}'] = metrics
        print_metrics(metrics)
        plot_metrics(metrics, f'plots/attack_{name.lower()}_metrics.png')
    
    return results

def compare_models(results):
    """
    Compare the performance of different models with enhanced visualizations.
    """
    # Extract metrics for comparison
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, metrics in results.items():
        row = {'Model': model_name}
        for metric in metrics_to_compare:
            row[metric] = metrics[metric]
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Bar plot of basic metrics
    plt.subplot(2, 2, 1)
    x = np.arange(len(comparison_df))
    width = 0.2
    for i, metric in enumerate(metrics_to_compare):
        plt.bar(x + i*width, comparison_df[metric], width, label=metric)
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x + width*1.5, comparison_df['Model'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Radar chart for each model
    plt.subplot(2, 2, 2, projection='polar')
    angles = np.linspace(0, 2*np.pi, len(metrics_to_compare), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # close the plot
    
    for idx, row in comparison_df.iterrows():
        values = row[metrics_to_compare].values
        values = np.concatenate((values, [values[0]]))  # close the plot
        plt.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        plt.fill(angles, values, alpha=0.25)
    
    plt.xticks(angles[:-1], metrics_to_compare)
    plt.title('Model Performance Radar Chart')
    plt.legend(bbox_to_anchor=(1.3, 0.5))
    
    # 3. Heatmap of model performance
    plt.subplot(2, 2, 3)
    metrics_matrix = comparison_df[metrics_to_compare].values
    sns.heatmap(metrics_matrix, 
                annot=True, 
                fmt='.3f',
                xticklabels=metrics_to_compare,
                yticklabels=comparison_df['Model'],
                cmap='YlOrRd')
    plt.title('Model Performance Heatmap')
    
    # 4. Box plot of metrics distribution
    plt.subplot(2, 2, 4)
    metrics_data = []
    labels = []
    for metric in metrics_to_compare:
        metrics_data.append(comparison_df[metric].values)
        labels.extend([metric] * len(comparison_df))
    
    plt.boxplot(metrics_data, labels=metrics_to_compare)
    plt.title('Metrics Distribution')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print comparison table with additional statistics
    print("\nModel Performance Comparison:")
    print("-" * 80)
    print(comparison_df.to_string(index=False))
    
    # Print additional statistics
    print("\nAdditional Statistics:")
    print("-" * 80)
    for metric in metrics_to_compare:
        print(f"\n{metric.upper()} Statistics:")
        print(f"Mean: {comparison_df[metric].mean():.4f}")
        print(f"Std: {comparison_df[metric].std():.4f}")
        print(f"Best Model: {comparison_df.loc[comparison_df[metric].idxmax(), 'Model']} ({comparison_df[metric].max():.4f})")
        print(f"Worst Model: {comparison_df.loc[comparison_df[metric].idxmin(), 'Model']} ({comparison_df[metric].min():.4f})")

def main():
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate synthetic dataset
    print("Generating synthetic cybersecurity dataset...")
    df = generate_synthetic_cybersecurity_data(n_samples=10000, attack_ratio=0.2)
    
    # Save dataset
    save_dataset(df)
    
    # Analyze attack types
    analyze_attack_types(df)
    
    # Prepare data for modeling
    y = (df['attack_type'] != 'normal').astype(int).values
    
    # Split data
    train_df, test_df, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = preprocess_data(train_df, test_df)
    
    # Evaluate models
    results = evaluate_models(X_train, y_train, X_test, y_test)
    
    # Compare models
    compare_models(results)
    
    print("\nAnalysis complete! Check the 'plots' directory for visualizations.")

if __name__ == "__main__":
    main() 