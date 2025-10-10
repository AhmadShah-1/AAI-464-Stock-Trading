# AGENTS.md

This file provides guidance to Coding Agents (claude.ai/code, gemini etc) when working with code in this repository.

## Project Overview

This is an AI-powered stock trading system for CpE646A Pattern Recognition and Classification (Fall 2025) at Stevens Institute of Technology. Team: Harshil, Karina, and Ahmad. Due: December 12, 2025.

The system applies multiple classification models (Random Forest, XGBoost, Gradient Boosting, Technical Indicators) to predict buy/hold/sell actions for tech stocks using Alpaca's trading API.

## Tech Stack

- **Language**: Python 3.x
- **ML Libraries**: scikit-learn, XGBoost
- **Data**: pandas, numpy, technical analysis libraries
- **Visualization**: matplotlib, seaborn
- **API**: Alpaca Trading API (paper trading by default)
- **Testing**: pytest

## Core Principles

### Configuration Management
- Use `.env` file for all secrets and environment-specific settings
- NEVER commit API keys, tokens, or `.env` file to version control
- Configuration should include: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `TRADING_ENABLED`, `MAX_POSITION_SIZE`

### Design Patterns to Follow
- **Factory Pattern**: For creating different ML models with a unified interface
- **Strategy Pattern**: For swappable trading algorithms
- **Template Method**: For consistent model interfaces (train, predict, evaluate, get_confidence)
- **Dependency Injection**: For testability and flexibility

## Target Stock Symbols

Focus on major technology and AI companies:
- **Tech Giants**: AAPL, MSFT, GOOGL, AMZN, META
- **AI Leaders**: NVDA, AI, PLTR
- **Chip Makers**: AMD, INTC
- **Software**: ADBE, CRM, ORCL
- **Other**: TSLA

## Machine Learning Approach

### Feature Engineering Guidelines
Generate comprehensive technical indicators for pattern recognition:
- **Price Features**: Returns, volatility, price changes, volume analysis
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Momentum
- **Lag Features**: Historical price and volume data (various time windows)
- **Target Variable**: Multi-class classification (sell=0, hold=1, buy=2)

### Model Types
The system should support multiple classification approaches:
- **Random Forest**: Ensemble learning with decision trees
- **XGBoost**: Gradient boosting with optimizations
- **Gradient Boosting**: Sequential ensemble method
- **Technical Indicator Model**: Rule-based strategy using traditional signals

## Safety Requirements

**CRITICAL**: This system interacts with real financial markets. Implement these safety layers:

1. **Paper Trading Default**: Use Alpaca's paper trading API unless explicitly enabled in config
2. **Confidence Thresholds**: Require minimum confidence (e.g., 60%) before executing trades
3. **Trend Filters**: Avoid trading against strong market trends
4. **Position Limits**: Enforce maximum position sizes
5. **Manual Override**: Allow reviewing trades before execution

**Always verify before executing trades**:
- `TRADING_ENABLED` is explicitly set to true
- Confidence score meets minimum threshold
- Current market trend aligns with recommended action
- Position size is within configured limits

## Testing Requirements

Target: 15+ unit tests with >34% code coverage for academic requirements.

**Priority test areas**:
1. Model creation and switching logic
2. Feature engineering calculations
3. Trend detection algorithms
4. Configuration loading and validation
5. Trading strategy logic

**Important**: Use mocks for external API calls to avoid actual transactions during tests.

## Academic Deliverables

**Course**: CpE646A Pattern Recognition and Classification, Fall 2025
**Due**: December 12, 2025
**Report**: 5 pages including abstract, methodology, results, and team contributions

### Code Requirements
- Clearly mark any external code with comments indicating source/origin
- Demonstrate original thinking beyond library function calls
- Include clear docstrings explaining pattern recognition approaches
- Maintain documentation for compilation/execution instructions

### Team Contributions
Track individual contributions in `docs/PROJECT_PLAN.md` as work progresses (Harshil, Karina, Ahmad).

## Key Design Philosophy

### Multi-Model Strategy
Use multiple classification approaches to compare pattern recognition effectiveness:
- Ensemble methods (Random Forest, Gradient Boosting)
- Optimized boosting (XGBoost)
- Rule-based systems (Technical Indicators)

### Trend-Aware Predictions
Detect overall market trends (bull/bear) and factor into recommendations. Example: downgrade "buy" signals during bear markets to "hold".

### Confidence-Based Decision Making
All predictions must include confidence scores. Low confidence defaults to "hold" rather than risky actions.

## Important Pitfalls to Avoid

### Security & Configuration
- Never commit `.env` file or hardcode API keys
- Ensure `.env` is in `.gitignore`

### Machine Learning Best Practices
- **Look-Ahead Bias**: Only use data available at prediction time (no future data leakage)
- **Overfitting**: Validate on out-of-sample data, not just training data
- **Data Quality**: Handle missing values, outliers, and market anomalies

### Trading Logic
- Don't make predictions in isolation; always consider current market trend
- Always test with paper trading before considering live execution
- Respect API rate limits (Alpaca free tier: ~200 requests/minute)
- Handle market hours gracefully (markets closed weekends/holidays)

## Development Best Practices

- Write tests alongside implementation for testability
- Document code with clear docstrings explaining ML/pattern recognition approach
- Track model performance metrics for final report
- Update team contributions in `docs/PROJECT_PLAN.md` regularly

## Reference Documentation

- **Project Vision**: `docs/PROJECT_PLAN.md` - detailed project requirements and goals
- **Alpaca API**: https://alpaca.markets/docs/ - trading API documentation
- **Course Deadline**: December 12, 2025 - 5-page report required
