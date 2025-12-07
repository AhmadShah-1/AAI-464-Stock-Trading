# Stock Prediction Models

This directory contains the machine learning models we developed for predicting stock prices and generating trading signals. We experimented with a few different approaches, primarily focusing on regression models to predict future returns, which we then converted into trading signals.

## 1. LightGBM Regression Model (`lightgbm_regression_model.py`)

This ended up being our primary and best-performing model. The main idea here was to predict the **5-day forward return** as a continuous value. Once we have that prediction, we classify it into BUY, SELL, or HOLD signals based on a simple threshold (we used ±2%).

### Features
We used a LightGBM regressor because it's generally fast and handles tabular data well. We fed it a lot of data—about a year's worth of daily OHLCV data for 14 different financial sector stocks (like JPM, GS, etc.).

### Features
We refined our feature engineering significantly to capture broader market dynamics and avoid "regime overfitting." We generated over 80 features including:

*   **Market Context (New):** We now explicitly model the market regime by pulling in **SPY** (Market), **VXX** (Volatility Index), and **XLF** (Financial Sector). Key features include *Beta to SPY*, *Relative Strength*, and *VXX Trend*. This helps the model distinguish between a stock-specific move and a broad market rally/crash.
*   **Cyclical Time:** We replaced raw calendar features (like Month/Quarter) with **Cyclical Sine/Cosine Encodings** for the day of the week. We removed long-term time features to prevent the model from memorizing specific past months.
*   **Technical Indicators:** RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillators, etc.
*   **Volume & Volatility:** Relative volume, VWAP distance, and volatility stability.
*   **News Sentiment:** Daily sentiment scores from the Alpaca News API.

We used a feature selection step to narrow this down, though we prioritized Market Context and Sentiment features.

### Performance
After tuning hyperparameters with Optuna (running about 100 trials), we got some pretty solid results:
*   **Directional Accuracy:** ~77% (It’s quite good at guessing if the stock will go up or down).
*   **R² Score:** 0.64 (Explains a good chunk of the variance).
*   **MAE:** 2.21%

## 2. CatBoost Regression Model (`catboost_regression_model.py`)

We also experimented with CatBoost to see if it could handle the noise in the data better than LightGBM. The setup was identical—predicting the 5-day forward return.

It actually performed slightly better in terms of raw error metrics (lower MAE and higher R² of 0.66), likely because of how it handles categorical nuances and overfitting. However, LightGBM still edged it out slightly when it came to just predicting the pure direction of the move.

## 3. Ensemble Model (`ensemble_model.py`)

Since we had two decent models, it made sense to combine them. We built a simple ensemble that takes a weighted average of the predictions from both LightGBM and CatBoost.

We gave CatBoost slightly more weight (60%) because of its stability, with LightGBM taking the remaining 40%.

### Results
This "Super Model" approach worked well. It managed to keep the high directional accuracy of the LightGBM model (77%) while pulling the error rates down closer to the CatBoost levels. It feels like the most robust option for actual trading since it balances the strengths of both.
