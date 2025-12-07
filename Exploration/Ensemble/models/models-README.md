# Stock Prediction Models

This directory contains the machine learning components designed to predict 5-day forward stock returns for our banking sector basket. We moved from single models to a weighted ensemble approach to balance variance and bias.

## 1. Feature Engineering Strategy
We overhauled the feature pipeline to address "regime overfitting." Instead of just technical indicators, we now explicitly model the market environment:
*   **Market Context:** We pull data for **SPY** (Market), **VXX** (Volatility), and **XLF** (Financial Sector). We calculate features like *Beta*, *Relative Strength*, and *Sector Correlation* to help the model ignore noise during broad market moves.
*   **Cyclical Time:** Calendar dates (Month, Day of Week) are encoded as Sine/Cosine waves to preserve continuity.
*   **News Sentiment:** We force the inclusion of sentiment scores (derived from Alpaca news) to capture external shocks.

## 2. Model Implementations

### LightGBM Regressor (`lightgbm_regression_model.py`)
Our primary gradient boosting model. It's fast and effective on tabular data.
-   **Configuration:** Optimized using `tune_hyperparameters.py`. We found that a lower learning rate (~0.033) and higher regularization (`reg_alpha`, `reg_lambda`) significantly improved generalization on unseen data.
-   **Training Tracking:** Now supports `plot_training_history()` to visualize RMSE curves and detect overfitting early.

### CatBoost Regressor (`catboost_regression_model.py`)
Used for its superior handling of noisy data and stability.
-   **Configuration:** We tuned this to be the "conservative" partner in the ensemble. It uses a very low learning rate (~0.009) and deep trees (Depth 10) with strict regularization.
-   **Role:** Provides a stable baseline prediction that dampens the volatility of LightGBM's outputs.

### Ensemble Model (`ensemble_model.py`)
Combines the two models using a weighted average strategy (**40% LightGBM / 60% CatBoost**).
-   **Logic:** $Prediction = (0.4 * LightGBM) + (0.6 * CatBoost)$
-   **Trading Logic:** Converts the continuous return prediction into signals:
    -   **BUY:** Predicted Return > +2.0%
    -   **SELL:** Predicted Return < -2.0%
    -   **HOLD:** Between -2% and +2%

## 3. Performance & Tuning
We use a dedicated script `tune_hyperparameters.py` to run Optuna optimization trials. This ensures our hyperparameters are data-driven rather than guessed.

**Recent Verification Results (Test Stock: Citigroup):**
-   **Directional Accuracy:** ~88.5% (Correctly predicts Up/Down movement)
-   **RMSE:** 0.0205
-   **R² Score:** 0.81

*Note: These metrics are from a specific walk-forward validation window and may vary as market regimes shift.*

## 4. Visualization
You can visually inspect the model's training process using the new plotting tools:
```python
# In your notebook or script:
model.train(X_train, y_train, X_test, y_test)
model.plot_training_history()  # Shows Train vs Val RMSE
model.plot_results(results)    # Shows Predictions vs Actuals
```
