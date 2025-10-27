# AAI-464 Stock Trading System

An AI-powered stock trading system designed specifically for technology and AI-related stocks. This project utilizes Alpaca's trading API to execute automated trading strategies based on machine learning predictions and technical analysis.

## Features

- 🤖 **Multiple ML Models**: Support for Random Forest, XGBoost, Gradient Boosting, and Technical Indicators
- 📊 **Trend Detection**: Identifies bull and bear market runs with strength analysis
- 🎯 **Smart Trading**: Buy/sell/hold recommendations with confidence scores
- 📈 **Technical Analysis**: Comprehensive technical indicators (RSI, MACD, Bollinger Bands, etc.)
- 📑 **Reporting**: Generates detailed reports, statistics, and visualizations
- 🔄 **Model Flexibility**: Easy switching between different prediction models
- 🎨 **Visualization**: Beautiful charts and dashboards for analysis results
- 🔒 **Risk Management**: Built-in position sizing and confidence thresholds

## Tech Stack Focus

The system is pre-configured to analyze major technology and AI stocks including:
- AAPL (Apple)
- MSFT (Microsoft)
- GOOGL (Alphabet/Google)
- NVDA (NVIDIA)
- META (Meta/Facebook)
- AMZN (Amazon)
- TSLA (Tesla)
- AMD (Advanced Micro Devices)
- INTC (Intel)
- ORCL (Oracle)
- CRM (Salesforce)
- ADBE (Adobe)
- PLTR (Palantir)
- AI (C3.ai)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AhmadShah-1/AAI-464-Stock-Trading.git
cd AAI-464-Stock-Trading
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up configuration:
```bash
cp .env.example .env
# Edit .env with your Alpaca API credentials
```

## Configuration

Create a `.env` file with the following configuration:

```env
# Alpaca API Configuration
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
TRADING_ENABLED=false
MAX_POSITION_SIZE=1000
RISK_PER_TRADE=0.02

# Model Configuration
DEFAULT_MODEL=random_forest
PREDICTION_HORIZON=5

# Tech/AI Stock Watchlist
TECH_STOCKS=AAPL,MSFT,GOOGL,NVDA,META,AMZN,TSLA,AMD,INTC,ORCL,CRM,ADBE,PLTR,AI

# Reporting
REPORT_OUTPUT_DIR=./reports
```

## Usage

### Basic Analysis

Analyze tech stocks with the default model:
```bash
python main.py --mode analyze
```

### Use Different Models

Choose from available models:
```bash
# Random Forest (default)
python main.py --mode analyze --model random_forest

# XGBoost
python main.py --mode analyze --model xgboost

# Gradient Boosting
python main.py --mode analyze --model gradient_boosting

# Technical Indicators (rule-based)
python main.py --mode analyze --model technical_indicators
```

### Analyze Specific Stocks

```bash
python main.py --mode analyze --symbols AAPL MSFT NVDA
```

### Train a Model

```bash
python main.py --mode train --model random_forest
```

### Execute Trades (Paper Trading)

**Note**: Requires `TRADING_ENABLED=true` in `.env`

```bash
python main.py --mode analyze --execute
```

### Generate Reports Only

```bash
python main.py --mode analyze --report
```

## Project Structure

```
AAI-464-Stock-Trading/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── alpaca_client.py         # Alpaca API integration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py            # Base model interface
│   │   ├── ml_models.py             # ML model implementations
│   │   ├── technical_model.py       # Technical indicator model
│   │   └── model_factory.py         # Model factory for easy switching
│   ├── strategies/
│   │   ├── __init__.py
│   │   └── trading_strategy.py      # Trading strategy engine
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── features.py              # Feature engineering
│   │   └── trend_detection.py       # Trend detection utilities
│   ├── reports/
│   │   ├── __init__.py
│   │   └── report_generator.py      # Report generation
│   ├── __init__.py
│   └── config.py                    # Configuration management
├── tests/                           # Test files
├── reports/                         # Generated reports
├── main.py                          # Main application
├── requirements.txt                 # Python dependencies
├── .env.example                     # Example configuration
├── .gitignore
├── LICENSE
└── README.md
```

## Models

### Machine Learning Models

1. **Random Forest**: Ensemble method using decision trees
2. **XGBoost**: Gradient boosting with optimizations
3. **Gradient Boosting**: Sequential ensemble method

### Technical Indicator Model

Rule-based model using:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator

## Features Engineering

The system automatically generates comprehensive features:

- **Price Features**: Returns, volatility, price changes
- **Technical Indicators**: 30+ indicators via `ta` library
- **Lag Features**: Historical price and volume data
- **Target Variables**: Multi-class classification (buy/hold/sell)

## Trend Detection

Advanced trend detection capabilities:

- **Trend Classification**: Bullish, bearish, or neutral
- **Trend Strength**: Quantified strength metric (0-1)
- **Support/Resistance**: Key price levels
- **Breakout Detection**: Identifies price breakouts
- **Momentum Analysis**: Price momentum calculation

## Reports

The system generates multiple types of reports:

1. **Analysis Report** (JSON): Detailed analysis for each stock
2. **Portfolio Report** (JSON): Current portfolio status and metrics
3. **Visualizations** (PNG): Charts including:
   - Action distribution pie chart
   - Confidence distribution histogram
   - Trend distribution bar chart
   - Top buy recommendations

## API Reference

### AlpacaClient

```python
from src.api import AlpacaClient

client = AlpacaClient()
account = client.get_account_info()
data = client.get_historical_data('AAPL', days=365)
order = client.place_order('AAPL', qty=10, side='buy')
```

### Model Factory

```python
from src.models import ModelFactory

# Create a model
model = ModelFactory.create_model('random_forest')

# Get available models
models = ModelFactory.get_available_models()
```

### Trading Strategy

```python
from src.strategies import TradingStrategy
from src.api import AlpacaClient
from src.models import ModelFactory

client = AlpacaClient()
model = ModelFactory.create_model('random_forest')
strategy = TradingStrategy(model, client)

# Analyze a stock
analysis = strategy.analyze_stock('AAPL')

# Scan watchlist
results = strategy.scan_watchlist()
```

### Report Generator

```python
from src.reports import ReportGenerator

report_gen = ReportGenerator()
report_gen.generate_analysis_report(analyses)
report_gen.generate_visualization(analyses)
report_gen.print_summary(analyses)
```

## Safety Features

- **Paper Trading**: Default configuration uses Alpaca's paper trading
- **Confidence Thresholds**: Only execute high-confidence trades
- **Position Sizing**: Risk management with configurable limits
- **Trend Filters**: Avoid trades against strong trends
- **Trading Toggle**: Requires explicit enable in configuration

## Development

### Running Tests

```bash
pytest tests/
```

### Adding a New Model

1. Create a new class inheriting from `BaseModel`
2. Implement required methods: `train()`, `predict()`, `predict_proba()`
3. Register with `ModelFactory`:

```python
from src.models import ModelFactory, BaseModel

class MyCustomModel(BaseModel):
    # Implementation
    pass

ModelFactory.register_model('my_model', MyCustomModel)
```

## Logging

Logs are written to both console and `trading.log` file with timestamps and log levels.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Do not use it for actual trading without understanding the risks involved. The authors are not responsible for any financial losses incurred through the use of this software.

## Acknowledgments

- Alpaca API for trading capabilities
- scikit-learn, XGBoost for ML models
- TA-Lib for technical indicators
- All contributors and supporters

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Made with ❤️ for the AI and Tech Trading Community**