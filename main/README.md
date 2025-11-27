# Stock Trading System - Main Module

## Quick Start

### 1. Create `.env` file in the project root (one level up from this directory)

```bash
# Navigate to project root
cd ..

# Create .env file with your credentials
# Copy the following content:
```

```
ALPACA_API_KEY=PKZNR4HYDRG3HEVC525MPNYLTB
ALPACA_SECRET_KEY=2YfMdrj27yRkWGWTot5fgwcas5ire4SknZ5ZZVU2b1Cb
ALPACA_BASE_URL=https://paper-api.alpaca.markets
TRADING_ENABLED=false
MAX_POSITION_SIZE=1000
CONFIDENCE_THRESHOLD=0.6
DEFAULT_SYMBOL=AAPL
HISTORICAL_DAYS=365
```


## Architecture

### Core Components

- **`main.py`**: Entry point - orchestrates the entire workflow
- **`config.py`**: Loads configuration from `.env` file

### Models Package (`models/`)

- **`base_model.py`**: Abstract base class defining the model interface
- **`random_forest_model.py`**: ML-based ensemble model using scikit-learn

### Utils Package (`utils/`)

- **`alpaca_client.py`**: Fetches stock data from Alpaca API
- **`feature_engineering`** Creates technical features from raw OHLCV data
- **`model_factory.py`**: Factory pattern for seamless model switching

## Features Generated

The system creates 15 technical features:

1. **Returns**: Daily price returns, log returns
2. **Volatility**: 5-day and 10-day rolling standard deviation
3. **Volume**: Volume changes and ratios
4. **Moving Averages**: 5, 10, and 20-day SMAs
5. **Price Ratios**: Price relative to each MA
6. **Crossovers**: MA crossover signals
7. **Momentum**: 5-day and 10-day price momentum
8. **Price Range**: High-low range analysis

## Output

The program provides:

- Current stock price
- Model recommendation (BUY/HOLD/SELL)
- Confidence score
- Model evaluation metrics
