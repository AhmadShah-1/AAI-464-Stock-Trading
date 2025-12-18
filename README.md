# AAI-464 Pattern Recognition & Classification: Stock Trading System

## Abstract
This project implements an automated stock trading system designed to predict 5-day forward returns for a universe of various stocks. The system uses a machine learning pipeline that integrates data acquisition from the Alpaca API, extensive feature engineering (technical indicators and sentiment analysis), and predictive modeling using both Random Forest and Weighted Ensemble methods (LightGBM and CatBoost). The objective is to identify profitable trading signals by recognizing patterns in historical price data and market sentiment.

## Methodology

### 1. Data Acquisition
The system utilizes the Alpaca API to fetch historical daily bar data (Open, High, Low, Close, Volume) for various stocks. To capture market sentiment, the data pipeline also retrieves and processes news headlines, generating sentiment scores that serve as exogenous variables for the predictive models.

### 2. Feature Engineering
For the implementation of banking stocks, a comprehensive set of over 77 technical and fundamental features is generated to model market dynamics. Key feature categories include:
*   **Momentum Indicators**: Relative Strength Index (RSI), Moving Average Convergence Divergence (MACD), Stochastic Oscillator, Williams %R.
*   **Volatility Metrics**: Bollinger Bands, Average True Range (ATR), Parkinson Volatility (High-Low range based).
*   **Market Context**: Relative strength comparisons against sector benchmarks (e.g., SPY, XLF).
*   **Sentiment Analysis**: Aggregated daily news sentiment scores and volume impact analysis.

### 3. Predictive Modeling
The project explores two primary modeling approaches:
*   **Random Forest Classifier**:   # TODO: Add details.
*   **Weighted Ensemble Model**: A sophisticated ensemble combining Gradient Boosting Decision Trees (LightGBM) and CatBoost. This approach aims to leverage the trend-following capabilities of LightGBM with the stability and categorical handling of CatBoost, optimized via a weighted average of their probabilistic outputs.

## Project Structure

```
AAI-464-Stock-Trading/
├── main/                   # Core execution pipeline
│   ├── main.ipynb          # Primary entry point for end-to-end execution
│   ├── config.py           # Configuration management and environment variables
│   ├── utils/              # Utility modules
│   │   ├── alpaca_client.py       # API interface for data fetching
│   │   ├── feature_engineering.py # Technical indicator generation
│   │   └── visualizer.py          # Performance plotting and analysis
│   └── models/             # Model definitions (BaseModel, RandomForest)
│
├── Exploration/            # Research and experimental sandbox
│   ├── Ensemble/           # Development of Weighted Ensemble model (LightGBM + CatBoost)
│   └── RF/                 # Random Forest implementation
│
└── requirements.txt        # Python dependency manifest
```

## Installation & Configuration

### Prerequisites
*   Python 3.8 or higher
*   An active Alpaca Markets account (Paper Trading) for API access.

### Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd AAI-464-Stock-Trading
    ```

2.  **Install Dependencies**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**
    Create a `.env` file in the project root directory to store your credentials. This file is critical for API access and configuration.
    ```ini
    # .env file
    ALPACA_API_KEY=your_api_key_here
    ALPACA_SECRET_KEY=your_secret_key_here
    ALPACA_BASE_URL=https://paper-api.alpaca.markets
    
    # Optional Configurations
    TRADING_ENABLED=False
    CONFIDENCE_THRESHOLD=0.6
    ```

## Usage Instructions

1. Random Forest: # TODO: Add instructions

2. Ensemble Model: 

In order to run the pipeline for this part correctly, several steps need to be followed.

**Step 1:** Ensure that the environment variables are set. These environment variables are crucial for the data engineering part.

**Step 2:** Ensure you’re in the correct directory:
```bash
cd Exploration/Ensemble
```

**Option 1 (Run the model file):**
Execute the python script:
```bash
python3 models/ensemble_model.py
```
Once the python file executes you should see print statements flowing in.

**Option 2 (Execute the notebook):**
Locate and run `ensemble_model.ipynb`.