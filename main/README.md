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

### 2. Install dependencies

```bash
# From project root
pip install -r requirements.txt
```

### 3. Run the program

```bash
# From the main directory
python main.py

# Or specify model directly:
python main.py rf          # Random Forest
python main.py ti          # Technical Indicator
```

## Architecture

### Core Components

- **`main.py`**: Entry point - orchestrates the entire workflow
- **`config.py`**: Loads configuration from `.env` file

### Models Package (`models/`)

- **`base_model.py`**: Abstract base class defining the model interface
- **`random_forest_model.py`**: ML-based ensemble model using scikit-learn
- **`technical_indicator_model.py`**: Rule-based strategy using technical indicators

### Utils Package (`utils/`)

- **`alpaca_client.py`**: Fetches stock data from Alpaca API
- **`feature_engineering.py`**: Creates technical features from raw OHLCV data
- **`model_factory.py`**: Factory pattern for seamless model switching

## Model Switching

The system uses the **Factory Pattern** to enable seamless model switching:

```python
from utils.model_factory import ModelFactory

# Create Random Forest model
model = ModelFactory.create_model('random_forest')

# Or create Technical Indicator model
model = ModelFactory.create_model('technical_indicator')

# Both have the same interface
model.train(X_train, y_train)
prediction = model.predict(X_test)
confidence = model.get_confidence(X_test)
```

## Model Comparison

| Feature | Random Forest | Technical Indicator |
|---------|--------------|---------------------|
| Type | Machine Learning | Rule-Based |
| Training | Learns from data | Analyzes patterns |
| Adaptability | High | Fixed rules |
| Interpretability | Medium (feature importance) | High (explicit signals) |
| Confidence | Probability-based | Signal strength + historical accuracy |

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
- Technical signal breakdown (for TI model)
- Class probabilities (for RF model)

## Design Patterns Used

1. **Factory Pattern** (`model_factory.py`): Creates model instances
2. **Template Method** (`base_model.py`): Defines standard model interface
3. **Strategy Pattern**: Swappable trading algorithms (RF vs TI)

This architecture makes it easy to add new models in the future - just inherit from `BaseModel` and register with `ModelFactory`.

