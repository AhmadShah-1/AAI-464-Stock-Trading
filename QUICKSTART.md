# Quick Start Guide

Get started with the AI-Powered Stock Trading System in minutes!

## Prerequisites

- Python 3.8 or higher
- Alpaca Trading Account (paper or live)
- Git

## Step 1: Clone the Repository

```bash
git clone https://github.com/AhmadShah-1/AAI-464-Stock-Trading.git
cd AAI-464-Stock-Trading
```

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Step 3: Configure API Credentials

1. Sign up for an Alpaca account at https://alpaca.markets
2. Get your API keys from the dashboard
3. Copy the example environment file:

```bash
cp .env.example .env
```

4. Edit `.env` and add your credentials:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## Step 4: Run Your First Analysis

### Analyze Tech Stocks

```bash
python main.py --mode analyze --model technical_indicators
```

This will:
- Analyze all tech stocks in the watchlist (AAPL, MSFT, NVDA, etc.)
- Generate buy/sell/hold recommendations
- Detect market trends (bull/bear)
- Create reports in the `reports/` directory

### Analyze Specific Stocks

```bash
python main.py --mode analyze --symbols AAPL MSFT NVDA
```

### Use Machine Learning Models

```bash
# Random Forest
python main.py --mode analyze --model random_forest

# XGBoost
python main.py --mode analyze --model xgboost

# Gradient Boosting
python main.py --mode analyze --model gradient_boosting
```

## Step 5: View Reports

After running an analysis, check the `reports/` directory for:

- **JSON Reports**: Detailed analysis data
- **PNG Charts**: Visual dashboards
- **Console Output**: Summary statistics

## Step 6: Run Examples

Try the example scripts:

```bash
python examples.py
```

This demonstrates:
- Basic stock analysis
- Multiple stock scanning
- Trend detection
- Model comparison

## Understanding the Output

### Action Recommendations

- **BUY**: Model predicts price increase
- **SELL**: Model predicts price decrease
- **HOLD**: No strong signal

### Confidence Scores

- **>70%**: High confidence signal
- **60-70%**: Moderate confidence
- **<60%**: Low confidence (filtered out by default)

### Trend Analysis

- **Bullish**: Upward trend detected
- **Bearish**: Downward trend detected
- **Neutral**: No clear trend

## Advanced Usage

### Execute Trades (Paper Trading)

1. Enable trading in `.env`:
```env
TRADING_ENABLED=true
```

2. Run with execute flag:
```bash
python main.py --mode analyze --execute
```

### Train a Custom Model

```bash
python main.py --mode train --model random_forest --symbols AAPL MSFT GOOGL NVDA
```

### Adjust Confidence Threshold

```bash
python main.py --mode analyze --min-confidence 0.7
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Troubleshooting

### "Configuration validation failed"

Make sure you've set `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in your `.env` file.

### "No module named 'src'"

Run from the project root directory or install the package:
```bash
pip install -e .
```

### "No data available for symbol"

- Check that your Alpaca account is active
- Verify the stock symbol is correct
- Ensure you have data access permissions

### Import errors

Install all dependencies:
```bash
pip install -r requirements.txt
```

## What's Next?

1. **Customize the Watchlist**: Edit `TECH_STOCKS` in `.env`
2. **Adjust Parameters**: Modify configuration values
3. **Add New Models**: Extend the model factory
4. **Implement Backtesting**: Test strategies on historical data
5. **Create Custom Strategies**: Build your own trading logic

## Safety Reminders

- Always start with **paper trading**
- Test thoroughly before using real money
- Understand the risks of automated trading
- Monitor your positions regularly
- Set appropriate stop-losses

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Review the [examples.py](examples.py) script
- Examine the test files for usage patterns
- Open an issue on GitHub for bugs or questions

Happy Trading! 📈🚀
