# Project Proposal

**Project Title:** Stock Price Movement Classifier

**Team Members:** Karina Fayn, Harshil Patel, Ahmad Shah

## Proposal

This project aims to develop a machine learning system that classifies short-term stock price movements into three actionable categories: Buy, Sell, or Hold. We will begin with a literature review examining existing research on financial time series classification, deep learning approaches for stock prediction, traditional machine learning methods for trading signals, and feature engineering techniques for market data. We will utilize publicly available financial datasets from sources such as Alpaca Markets API or Yahoo Finance to obtain historical time-series data including features like open, high, low, close prices, and trading volume. Our approach will explore multiple pattern recognition techniques, including traditional classifiers (e.g., Random Forest, SVM), deep learning architectures (e.g., 1D CNNs, LSTMs), and potentially hybrid models to capture temporal patterns in fixed-size time windows (e.g., 7-day intervals). The system will be implemented in Python using libraries such as scikit-learn, TensorFlow/PyTorch, and pandas, with original code developed for data preprocessing, feature engineering, model training, and real-time simulation. We will evaluate performance using standard classification metrics—accuracy, precision, recall, F1-score, and confusion matrices—comparing different model architectures and window configurations. The final deliverable will include a demonstration of the trained classifier making real-time predictions on stock data, accompanied by visualizations and performance analysis that showcase the practical applicability of pattern recognition techniques to financial forecasting.
