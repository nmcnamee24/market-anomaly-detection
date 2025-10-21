# Market Anomaly Detection in top 5 S&P 500 Stocks

An unsupervised machine learning model to identify irregular trading days across major U.S. equities using the Isolation Forest algorithm.

## Overview

This project analyzes ten years of 5 major stocks in the S&P 500 price data to learn normal market behavior from stable periods and flags anomalies such as the 2020 COVID and other events that have caused disruption. The system uses cross-sectional market features to detect market-wide irregularities with high precision.

## Features

- **Multi-dimensional Analysis**: Analyzes daily returns, volatility, volume surprises, and sector dispersion
- **Robust Preprocessing**: Handles missing data, winsorization of outliers, and standardization
- **Isolation Forest Algorithm**: Uses 400 decision trees with 2% contamination rate for anomaly detection
- **Sector Analysis**: Incorporates sector mapping for Information Technology, Consumer Discretionary, and Communication Services
- **Model Persistence**: Saves trained models and scalers for production deployment

## Key Components

### Data Processing
- Daily return calculations with infinite value handling
- Cross-sectional feature extraction (fraction_up, volatility measures, percentiles)
- Volume surprise detection using 30-day rolling statistics
- Sector dispersion analysis

### Machine Learning Pipeline
- StandardScaler for feature normalization
- Isolation Forest with optimized hyperparameters
- Anomaly scoring and binary classification
- Model validation using known market events

### Output Files
- `iso_model.joblib`: Trained Isolation Forest model
- `scaler.joblib`: Fitted StandardScaler
- `market_anomaly_scores.csv`: Complete results with anomaly scores and flags

## Results

The model successfully identifies major market disruptions with interpretable anomaly scores. Performance is validated against known market events like the COVID-19 crash period.

## Technologies Used

- **Python**: Core programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and preprocessing
- **Jupyter Notebook**: Interactive development and visualization
- **joblib**: Model serialization

## Usage

1. Load and preprocess S&P 500 price data
2. Extract cross-sectional market features
3. Train Isolation Forest model on stable periods
4. Generate anomaly scores for all trading days
5. Analyze results and identify irregular market behavior

## Model Performance

The system demonstrates robust anomaly detection capabilities, successfully flagging major market events while maintaining low false positive rates during normal market conditions.
