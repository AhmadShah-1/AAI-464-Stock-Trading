
import optuna
import pandas as pd
import numpy as np
import sys
import os
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../main')))

from models.lightgbm_regression_model import LightGBMRegressionModel
from models.catboost_regression_model import CatBoostRegressionModel

def tune_lightgbm(X_train, y_train, X_val, y_val, trials=20):
    print("\n" + "="*50)
    print("TUNING LIGHTGBM")
    print("="*50)

    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'seed': 42,
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            param,
            train_data,
            num_boost_round=1000,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)

    print(f"\n✅ LightGBM Best RMSE: {study.best_value:.5f}")
    print("Best Params:", study.best_params)
    return study.best_params

def tune_catboost(X_train, y_train, X_val, y_val, trials=20):
    print("\n" + "="*50)
    print("TUNING CATBOOST")
    print("="*50)

    def objective(trial):
        param = {
            'loss_function': 'RMSE',
            'iterations': 1000,
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 50,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.6, 0.95),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
        }

        model = CatBoostRegressor(**param)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=trials)

    print(f"\n✅ CatBoost Best RMSE: {study.best_value:.5f}")
    print("Best Params:", study.best_params)
    return study.best_params

if __name__ == "__main__":
    TRAIN_SYMBOLS = ['BAC', 'JPM', 'WFC', 'GS', 'MS', 'USB', 'PNC', 'AXP', 'COF', 'SCHW', 'BLK', 'BK', 'STT', 'TFC']
    TEST_SYMBOLS = ['C']
    FORWARD_DAYS = 5
    
    print("Fetching and Preparing Data...")
    lgb_loader = LightGBMRegressionModel()
    train_df, test_df = lgb_loader.fetch_data(TRAIN_SYMBOLS, TEST_SYMBOLS)
    
    train_features = lgb_loader.prepare_data(train_df, FORWARD_DAYS).dropna()
    
    exclude_cols = ['target', 'forward_returns', 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap']
    all_feature_cols = [col for col in train_features.columns if col not in exclude_cols]
    
    correlations = train_features[all_feature_cols + ['target']].corr()['target'].drop('target')
    correlations_abs = correlations.abs().sort_values(ascending=False)
    TOP_N_FEATURES = 35
    top_features = correlations_abs.head(TOP_N_FEATURES).index.tolist()
    
    news_features = ['news_sentiment', 'news_volume', 'sentiment_momentum', 'sentiment_ma_5', 'high_news_volume', 'sentiment_impact']
    forced_news_features = [f for f in news_features if f in all_feature_cols]
    for f in forced_news_features:
        if f not in top_features:
            top_features.append(f)
            
    split_idx = int(len(train_features) * 0.8)
    
    X = train_features[top_features]
    y = train_features['target']
    
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    X_val = X.iloc[split_idx:]
    y_val = y.iloc[split_idx:]
    
    print(f"Tuning Data: Train={len(X_train)}, Val={len(X_val)}")
    
    best_lgb = tune_lightgbm(X_train, y_train, X_val, y_val, trials=30)
    best_cat = tune_catboost(X_train, y_train, X_val, y_val, trials=30)
    
    print("FINAL TUNED PARAMETERS")
    print()
    print("LightGBM Params:")
    print(best_lgb)
    print("\nCatBoost Params:")
    print(best_cat)
