# Multi-Model Cybersecurity Attack Detection System

This project implements a sophisticated multi-model system for detecting cybersecurity attacks using various machine learning algorithms and ensemble methods.

## Features

- Synthetic dataset generation with realistic network traffic patterns
- Multiple attack types (DoS, DDoS, PortScan, Probe, SQL Injection, XSS, Brute Force)
- Advanced feature engineering and dimensionality reduction
- Multiple detection methods:
  - K-Nearest Neighbors (KNN)
  - DBSCAN
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - Random Forest
  - Z-Score
  - Ensemble methods
- Comprehensive evaluation metrics and visualizations
- Detailed analysis of attack types and feature importance

## Project Structure

```
.
├── data/                   # Directory for storing datasets
├── models/                 # Model implementations
│   └── clustering.py      # Clustering and detection models
├── utils/                  # Utility functions
│   ├── dataset_generator.py  # Dataset generation
│   └── metrics.py         # Evaluation metrics
├── plots/                  # Directory for storing visualizations
├── main.py                # Main script
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir data plots
```

## Running the Project

1. Activate your virtual environment if not already activated:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Run the main script:
```bash
python main.py
```

The script will:
- Generate synthetic cybersecurity data
- Analyze attack types and features
- Train and evaluate multiple models
- Generate comprehensive visualizations
- Save results in the 'plots' directory

## Output

The project generates several visualizations in the 'plots' directory:
- Attack type distribution
- Feature correlation heatmap
- Feature distributions
- Model performance metrics
- ROC curves
- Precision-recall curves
- Confusion matrices
- Model comparison charts

## Model Details

### Anomaly Detection Models
- K-means clustering
- DBSCAN
- Isolation Forest
- Local Outlier Factor (LOF)
- Ensemble method

### Attack Detection Models
- K-Nearest Neighbors (KNN)
- DBSCAN
- Isolation Forest
- Random Forest
- Z-Score
- Ensemble method

## Evaluation Metrics

The system evaluates models using multiple metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Balanced Accuracy
- Matthews Correlation Coefficient
- ROC AUC Score
- Average Precision Score
- Detection Rate
- False Alarm Rate 