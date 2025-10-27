# Architecture Overview

This document describes the architecture and design decisions of the AAI-464 Stock Trading System.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                         │
│                    (main.py, examples.py)                   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│                  Trading Strategy Engine                     │
│              (src/strategies/trading_strategy.py)            │
│  - Analyzes stocks                                          │
│  - Generates signals                                        │
│  - Executes trades                                          │
└─────┬──────────────────┬─────────────────┬─────────────────┘
      │                  │                 │
┌─────▼─────┐  ┌────────▼────────┐  ┌─────▼──────────────┐
│   Models  │  │   Trend Detection│  │   Alpaca API       │
│  Factory  │  │                  │  │   Client           │
└─────┬─────┘  └────────┬────────┘  └─────┬──────────────┘
      │                  │                 │
┌─────▼──────────────────▼─────────────────▼──────────────────┐
│                    Core Components                           │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  ML Models   │  │   Features   │  │   Reports    │    │
│  │              │  │  Engineering │  │  Generator   │    │
│  │ - Random     │  │              │  │              │    │
│  │   Forest     │  │ - Technical  │  │ - JSON       │    │
│  │ - XGBoost    │  │   Indicators │  │ - Charts     │    │
│  │ - Gradient   │  │ - Price      │  │ - Summary    │    │
│  │   Boosting   │  │   Features   │  │              │    │
│  │ - Technical  │  │ - Lag        │  │              │    │
│  │   Rules      │  │   Features   │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Configuration (`src/config.py`)

**Purpose**: Centralized configuration management

**Key Features**:
- Environment variable loading
- Default values
- Configuration validation
- Type conversion

**Usage**:
```python
from src.config import Config
stocks = Config.TECH_STOCKS
```

### 2. API Integration (`src/api/`)

**Purpose**: Interface with Alpaca Trading API

**Components**:
- `alpaca_client.py`: API wrapper

**Key Methods**:
- `get_historical_data()`: Fetch stock data
- `place_order()`: Execute trades
- `get_positions()`: Query current positions
- `get_account_info()`: Account status

**Design Decisions**:
- Async-ready design for future scaling
- Error handling with logging
- Paper trading by default

### 3. Models (`src/models/`)

**Purpose**: Prediction and classification models

**Architecture Pattern**: Strategy Pattern + Factory Pattern

**Components**:
- `base_model.py`: Abstract base class
- `ml_models.py`: ML implementations
- `technical_model.py`: Rule-based model
- `model_factory.py`: Model creation

**Interface**:
```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y): pass
    
    @abstractmethod
    def predict(self, X): pass
    
    @abstractmethod
    def predict_proba(self, X): pass
```

**Why This Design**:
- Easy to add new models
- Consistent interface
- Testable components
- Flexible model switching

### 4. Feature Engineering (`src/utils/features.py`)

**Purpose**: Transform raw data into ML features

**Pipeline**:
```
Raw OHLCV Data
    ↓
Price Features (returns, volatility)
    ↓
Technical Indicators (RSI, MACD, etc.)
    ↓
Lag Features (historical values)
    ↓
Target Variables (buy/sell/hold)
```

**Key Functions**:
- `prepare_features()`: Complete pipeline
- `add_technical_indicators()`: 30+ indicators
- `add_price_features()`: Returns, volatility
- `create_target_variable()`: Classification labels

**Library Choice**:
- Uses `ta` library for indicators
- Well-tested, maintained
- Comprehensive coverage

### 5. Trend Detection (`src/utils/trend_detection.py`)

**Purpose**: Identify market trends and patterns

**Class**: `TrendDetector`

**Methods**:
- `detect_trend()`: Bull/bear/neutral
- `detect_trend_strength()`: 0-1 strength score
- `identify_support_resistance()`: Key levels
- `detect_breakout()`: Breakout patterns
- `get_trend_analysis()`: Complete analysis

**Algorithm**:
1. Calculate moving averages (20, 50 day)
2. Identify crossovers
3. Check price position
4. Calculate divergence strength
5. Detect support/resistance
6. Generate recommendation

### 6. Trading Strategy (`src/strategies/trading_strategy.py`)

**Purpose**: Main orchestration and decision engine

**Class**: `TradingStrategy`

**Workflow**:
```
1. Fetch historical data
2. Engineer features
3. Get model prediction
4. Analyze trend
5. Apply filters
6. Generate signal
7. Execute trade (optional)
```

**Safety Features**:
- Confidence thresholds
- Trend filters
- Position sizing
- Risk management

### 7. Reporting (`src/reports/report_generator.py`)

**Purpose**: Generate comprehensive reports

**Class**: `ReportGenerator`

**Output Types**:
1. **JSON Reports**: Detailed data
2. **Visualizations**: Charts and graphs
3. **Console Summary**: Quick overview
4. **Portfolio Reports**: Account status

**Charts**:
- Action distribution (pie)
- Confidence histogram
- Trend distribution (bar)
- Top recommendations (horizontal bar)

## Design Patterns

### 1. Factory Pattern (Models)

**Why**: Easy model creation and switching

```python
model = ModelFactory.create_model('random_forest')
```

### 2. Strategy Pattern (Trading)

**Why**: Flexible algorithm selection

```python
strategy = TradingStrategy(model, client)
```

### 3. Template Method (Base Model)

**Why**: Consistent model interface

```python
class BaseModel(ABC):
    def evaluate(self, X, y):
        predictions = self.predict(X)
        # Common evaluation logic
```

### 4. Dependency Injection

**Why**: Testability and flexibility

```python
def __init__(self, model: BaseModel, client: AlpacaClient):
    self.model = model
    self.client = client
```

## Data Flow

### Analysis Mode

```
User Input (symbols)
    ↓
Fetch Data (Alpaca API)
    ↓
Feature Engineering
    ↓
Model Prediction
    ↓
Trend Analysis
    ↓
Signal Generation
    ↓
Report Generation
    ↓
Output (JSON, Charts, Console)
```

### Trading Mode

```
Analysis Results
    ↓
Check Confidence
    ↓
Apply Filters
    ↓
Calculate Position Size
    ↓
Place Order (Alpaca API)
    ↓
Log Results
```

## Technology Choices

### Why Python?

- Rich ML ecosystem
- Financial libraries (pandas, ta)
- Rapid development
- Strong community

### Why Alpaca?

- Commission-free trading
- Modern REST API
- Paper trading support
- Good documentation

### Why scikit-learn/XGBoost?

- Industry standard
- Well-documented
- Production-ready
- Easy to use

### Why TA Library?

- Comprehensive indicators
- Maintained
- Pandas integration
- Accurate implementations

## Scalability Considerations

### Current Limitations

- Sequential stock processing
- In-memory data storage
- Single-threaded execution
- No persistent state

### Future Improvements

1. **Parallel Processing**
   ```python
   from concurrent.futures import ThreadPoolExecutor
   with ThreadPoolExecutor() as executor:
       results = executor.map(analyze_stock, symbols)
   ```

2. **Database Integration**
   - Store historical data
   - Cache features
   - Track trade history

3. **Async Operations**
   - Non-blocking API calls
   - Concurrent data fetching
   - Real-time updates

4. **Model Caching**
   - Save trained models
   - Quick loading
   - Version management

## Security Considerations

### API Keys

- Environment variables only
- Never commit to git
- Use .env files
- Separate keys per environment

### Trading Safety

- Paper trading default
- Explicit enable required
- Confidence thresholds
- Position limits

### Error Handling

- Graceful failures
- Comprehensive logging
- No silent errors
- Clear error messages

## Testing Strategy

### Unit Tests

- Model interfaces
- Feature engineering
- Trend detection
- Configuration

### Integration Tests

- API connectivity (mocked)
- End-to-end workflows
- Report generation

### Test Coverage Goals

- Core logic: >90%
- Utilities: >80%
- Overall: >75%

## Performance Considerations

### Bottlenecks

1. API rate limits
2. Feature calculation
3. Model training
4. Data fetching

### Optimizations

1. **Caching**
   - Historical data
   - Features
   - Model predictions

2. **Batch Processing**
   - Multiple symbols at once
   - Parallel API calls

3. **Efficient Libraries**
   - NumPy vectorization
   - Pandas optimizations

## Monitoring and Logging

### Log Levels

- **INFO**: Normal operations
- **WARNING**: Potential issues
- **ERROR**: Failures
- **DEBUG**: Detailed tracing

### Key Metrics

- API call count
- Model accuracy
- Trade success rate
- Execution time
- Error frequency

## Future Enhancements

### Short Term

- [ ] Model persistence
- [ ] More indicators
- [ ] Better visualization
- [ ] Email notifications

### Medium Term

- [ ] Web dashboard
- [ ] Real-time monitoring
- [ ] Backtesting framework
- [ ] Portfolio optimization

### Long Term

- [ ] Deep learning models
- [ ] Sentiment analysis
- [ ] Multi-asset support
- [ ] Cloud deployment

## Conclusion

The system is designed to be:

- **Modular**: Easy to extend
- **Maintainable**: Clear structure
- **Testable**: Good coverage
- **Scalable**: Room to grow
- **Safe**: Multiple safeguards

The architecture supports both learning and production use cases while maintaining code quality and safety standards.
