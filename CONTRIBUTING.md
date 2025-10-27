# Contributing to AAI-464 Stock Trading System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what's best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior
4. Actual behavior
5. System information (Python version, OS, etc.)
6. Error messages or logs

### Suggesting Features

We welcome feature suggestions! Please:

1. Check if the feature has already been suggested
2. Clearly describe the feature and its use case
3. Explain why it would be valuable
4. Consider implementation complexity

### Contributing Code

#### Getting Started

1. Fork the repository
2. Clone your fork:
```bash
git clone https://github.com/your-username/AAI-464-Stock-Trading.git
cd AAI-464-Stock-Trading
```

3. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

4. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

#### Making Changes

1. **Write Clean Code**
   - Follow PEP 8 style guidelines
   - Use meaningful variable names
   - Add docstrings to functions and classes
   - Keep functions focused and small

2. **Add Tests**
   - Write tests for new features
   - Ensure existing tests still pass
   - Aim for high test coverage

3. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new code
   - Update QUICKSTART.md for user-facing changes

4. **Commit Messages**
   - Use clear, descriptive commit messages
   - Start with a verb (Add, Fix, Update, etc.)
   - Keep first line under 72 characters
   - Add details in the body if needed

Example:
```
Add XGBoost model support for predictions

- Implement XGBoostModel class
- Add model to factory
- Include tests for new model
- Update documentation
```

#### Testing

Run tests before submitting:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_trading_system.py -v
```

#### Submitting a Pull Request

1. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```

2. Open a pull request on GitHub

3. In the PR description:
   - Describe what changes you made
   - Reference any related issues
   - List any breaking changes
   - Include screenshots if applicable

4. Wait for review and address feedback

## Development Guidelines

### Project Structure

```
src/
├── api/          # External API integrations
├── models/       # ML models and prediction logic
├── strategies/   # Trading strategies
├── utils/        # Helper functions
└── reports/      # Report generation
```

### Adding a New Model

1. Create model class in `src/models/`:
```python
from src.models.base_model import BaseModel

class MyModel(BaseModel):
    def __init__(self):
        super().__init__('my_model')
    
    def train(self, X, y):
        # Implementation
        pass
    
    def predict(self, X):
        # Implementation
        pass
    
    def predict_proba(self, X):
        # Implementation
        pass
```

2. Register in model factory:
```python
from src.models.model_factory import ModelFactory

ModelFactory.register_model('my_model', MyModel)
```

3. Add tests in `tests/`:
```python
def test_my_model():
    model = ModelFactory.create_model('my_model')
    assert model.name == 'my_model'
```

### Adding a New Feature

1. Create feature in appropriate module
2. Add tests
3. Update documentation
4. Submit PR

### Code Style

We follow PEP 8 with some exceptions:

- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use type hints where beneficial

Example:
```python
def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns for a price series.
    
    Args:
        prices: Price series
        periods: Number of periods for returns
    
    Returns:
        Returns series
    """
    return prices.pct_change(periods)
```

### Documentation Style

Use Google-style docstrings:

```python
def my_function(param1: str, param2: int = 0) -> bool:
    """
    Brief description of function.
    
    Longer description if needed, explaining what the
    function does in more detail.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    
    Example:
        >>> my_function("test", 5)
        True
    """
    pass
```

## Areas for Contribution

### High Priority

- [ ] Backtesting framework
- [ ] Additional ML models (LSTM, Prophet, etc.)
- [ ] More technical indicators
- [ ] Portfolio optimization
- [ ] Risk management improvements

### Medium Priority

- [ ] Web UI dashboard
- [ ] Real-time monitoring
- [ ] Notification system (email, SMS)
- [ ] Database integration for historical trades
- [ ] Performance benchmarking

### Good First Issues

- [ ] Add more unit tests
- [ ] Improve documentation
- [ ] Add type hints
- [ ] Fix minor bugs
- [ ] Improve error messages

## Questions?

If you have questions about contributing:

1. Check existing documentation
2. Search closed issues
3. Open a new issue with the "question" label

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to the AAI-464 Stock Trading System! 🎉
