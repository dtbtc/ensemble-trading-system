# Ensemble Learning Cryptocurrency Trading Strategy System

## Project Overview

This project is a cryptocurrency trading strategy research system based on machine learning ensemble methods. It builds a reliable trading signal generation system by integrating multiple machine learning models with technical analysis indicators and provides a complete backtesting evaluation framework.

### Core Features

- Multi-model ensemble learning (Stacking/Blending)
- Automated feature engineering
- Complete backtesting system
- Strategy evaluation and optimization
- Risk management

## System Architecture
project/
├── config/ # Configuration files
│ └── config.yaml # Main configuration file
├── data/ # Data directory
│ ├── raw/ # Raw data
│ └── processed/ # Processed data
├── models/ # Saved models
├── notebooks/ # Jupyter notebooks
├── results/ # Output results
│ ├── figures/ # Chart outputs
│ └── reports/ # Report outputs
├── src/ # Source code
│ ├── backtest/ # Backtesting module
│ ├── data_processor/ # Data processing
│ ├── evaluation/ # Evaluation module
│ ├── models/ # Model definitions
│ ├── strategy/ # Strategy module
│ └── utils/ # Utility functions
└── tests/ # Unit tests

## Module Descriptions

### 1. Data Processing Module (data_processor)

- **Data Cleaning and Preprocessing**
  - Handle missing and anomalous values
  - Data format standardization
  - Time series alignment

- **Feature Engineering**
  - Technical indicator calculation
  - Time feature generation
  - Lag feature construction

- **Feature Selection**
  - Correlation analysis
  - Feature importance evaluation
  - Dimensionality reduction

### 2. Model Module (models)

- **Base Models**
  - LightGBM
  - XGBoost
  - RandomForest
  - Other machine learning models

- **Ensemble Methods**
  - Stacking ensemble
  - Blending ensemble
  - Dynamic weight adjustment

### 3. Backtesting System (backtest)

- **Signal Generation**
  - Model predictions
  - Signal filtering
  - Threshold optimization

- **Trade Simulation**
  - Order execution
  - Position management
  - Fee calculation

- **Performance Evaluation**
  - Return calculation
  - Risk metrics
  - Performance analysis

### 4. Evaluation Module (evaluation)

- **Model Evaluation**
  - Accuracy, Precision, Recall
  - ROC curve and AUC
  - Cross-validation

- **Strategy Evaluation**
  - Cumulative returns
  - Sharpe ratio
  - Maximum drawdown

## Usage Instructions

### 1. Environment Setup

```bash
# Create virtual environment
conda create -n trading_env python=3.8
conda activate trading_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Process raw data
python src/data_processor/processor.py
```

### 3. Model Training

```bash
# Train model
python main.py --mode train --model [model_name]
```

### 4. Backtesting Analysis

```bash
# Execute backtest
python main.py --mode backtest --model [model_name]
```

## Configuration Guide

Main configuration file: `config/config.yaml`

```yaml
# Data configuration
data:
  raw_data_path: "data.csv"
  test_size: 0.2
  val_size: 0.2

# Model configuration
models:
  LightGBM:
    n_estimators: 100
    learning_rate: 0.1
  XGBoost:
    n_estimators: 100
    max_depth: 5

# Backtest configuration
backtest:
  initial_capital: 10000
  commission: 0.001
```

## Important Notes

1. **Data Security**
   - Regular data backup
   - Data quality checks
   - Data integrity verification

2. **Model Training**
   - Avoid overfitting
   - Regular retraining
   - Monitor model performance

3. **Risk Control**
   - Set stop losses
   - Control position sizes
   - Diversify investments

## Changelog

### v0.1.0 (2025-01-06)
- Initial version release
- Implemented basic functional framework
- Support for multiple base models
- Added basic backtesting functionality




