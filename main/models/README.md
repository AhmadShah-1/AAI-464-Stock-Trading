# Stock Prediction Models

This directory contains the machine learning models used for stock price prediction and trading signal generation.

## 1. LightGBM Regression Model (`lightgbm_regression_model.py`)

**Current Status:** ✅ Active / Best Performing Model

### Objective
Predict the **5-day forward return** (continuous value) of a stock. Predictions are then converted into trading signals (BUY, SELL, HOLD) based on a threshold (e.g., ±2%).

### Implementation Details
- **Type:** Gradient Boosting Regressor (LightGBM).
- **Target:** 5-day future return (`(close_t+5 - close_t) / close_t`).
- **Data Source:** Alpaca API (Daily OHLCV).
- **Training Data:** 1 year of daily data for 14 financial sector stocks (e.g., JPM, BAC, GS, MS).

### Feature Engineering
The model utilizes **77+ features** across multiple categories:
1.  **Price Action:** Returns, Log Returns, SMA distances.
2.  **Momentum:** RSI, MACD, ROC, Stochastic Oscillator, Williams %R.
3.  **Volatility:** ATR, Bollinger Bands, Parkinson/Garman-Klass Volatility.
4.  **Volume:** Volume Change, OBV, VWAP distance.
5.  **Market Context:** Relative strength vs SPY, VXX (Volatility Index), XLF (Financial Sector).
6.  **News Sentiment:** Daily sentiment score, news volume, sentiment momentum (via Alpaca News API).

**Feature Selection:**
- Automatically selects the top ~35 features most correlated with the target.
- **Forced Inclusion:** News sentiment features are explicitly included to capture non-linear market sentiment effects.

### Hyperparameter Tuning
Optimized using **Optuna** (100 trials) to minimize RMSE.
- **Key Params:** `learning_rate=0.01`, `num_leaves=31`, `feature_fraction=0.8`, `bagging_freq=5`.

### Performance (Latest Run)
- **Directional Accuracy:** **77.00%** (Correctly predicts Up/Down direction).
- **R² Score:** **0.6409** (Explains 64% of variance).
- **MAE:** **2.21%** (Average error in return prediction).
- **Trading Action Accuracy:** **57.50%** (Correct BUY/SELL/HOLD classification).

---

## 2. Random Forest Model (`random_forest_model.py`)

**Current Status:** ⏸️ Baseline / Alternative

### Objective
Directly classify the next market move into **3 classes**:
- **0 (SELL):** Return < -2%
- **1 (HOLD):** -2% ≤ Return ≤ 2%
- **2 (BUY):** Return > 2%

### Implementation Details
- **Type:** Random Forest Classifier (Ensemble of Decision Trees).
- **Library:** `sklearn.ensemble.RandomForestClassifier`.
- **Class Balancing:** Uses `class_weight='balanced'` to handle the prevalence of "HOLD" periods.

### Feature Engineering
Uses a standard set of technical indicators similar to the LightGBM model, scaled using `StandardScaler` to normalize inputs (mean=0, variance=1).

### Tuning & Configuration
- **Estimators:** 100 trees (default).
- **Max Depth:** 10 (to prevent overfitting).
- **Criterion:** Gini Impurity.

### Performance
- Used primarily as a baseline to benchmark the LightGBM model.
- Generally offers lower directional accuracy compared to the boosted regression approach but provides robust probability estimates for class membership.
