# Setup Instructions

## 1. Install Dependencies

First, install the required Python packages:

```bash
pip install -r requirements.txt
```

## 2. Configure API Credentials

Create a `.env` file in the project root directory with your Alpaca API credentials:

```bash
# Alpaca API Configuration (Paper Trading)
ALPACA_API_KEY=PKZNR4HYDRG3HEVC525MPNYLTB
ALPACA_SECRET_KEY=2YfMdrj27yRkWGWTot5fgwcas5ire4SknZ5ZZVU2b1Cb
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Trading Configuration
TRADING_ENABLED=false
MAX_POSITION_SIZE=1000
CONFIDENCE_THRESHOLD=0.6

# Stock Configuration
DEFAULT_SYMBOL=AAPL
HISTORICAL_DAYS=365
```

**IMPORTANT**: The `.env` file is already in `.gitignore` and will not be committed to version control. This keeps your API credentials secure.

## 3. Run the Program

Navigate to the `main` directory and run:

```bash
cd main
python main.py
```

### Command-Line Options

You can specify which model to use via command-line argument:

```bash
# Use Random Forest model
python main.py 1
# or
python main.py rf

# Use Technical Indicator model
python main.py 2
# or
python main.py ti
```

Without arguments, the program will prompt you to select a model interactively.

## 4. Expected Output

The program will:
1. Display configuration settings
2. Fetch historical AAPL data from Alpaca API
3. Generate technical features
4. Train the selected model
5. Evaluate model performance on test data
6. Make a prediction for the latest trading day
7. Display recommendation with confidence score

## Project Structure

```
AAI-464-Stock-Trading/
├── .env                    # API credentials (create this)
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── SETUP.md               # This file
└── main/
    ├── main.py            # Main entry point
    ├── config.py          # Configuration management
    ├── models/
    │   ├── __init__.py
    │   ├── base_model.py               # Abstract base class
    │   ├── random_forest_model.py      # Random Forest implementation
    │   └── technical_indicator_model.py # Technical Indicator implementation
    └── utils/
        ├── __init__.py
        ├── alpaca_client.py      # Alpaca API integration
        ├── feature_engineering.py # Feature creation
        └── model_factory.py       # Model switching logic
```

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the `main` directory:
```bash
cd main
python main.py
```

### API Errors

If you get Alpaca API errors:
- Verify your API credentials in `.env` are correct
- Check your internet connection
- Ensure you're using paper trading credentials (not live trading)

### Missing Dependencies

If you get module not found errors:
```bash
pip install -r requirements.txt
```

## Switching Between Models

The system uses a factory pattern for easy model switching:

1. **Interactive**: Run `python main.py` and select from menu
2. **Command-line**: Run `python main.py rf` or `python main.py ti`
3. **Programmatic**: In code, use `ModelFactory.create_model('random_forest')` or `ModelFactory.create_model('technical_indicator')`

Both models implement the same interface (`BaseModel`), so they can be swapped seamlessly without changing the main program logic.

