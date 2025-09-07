# Trade Bot - Production-Grade Crypto Trading System

A comprehensive, modular Python trading bot for cryptocurrency markets with robust risk management, multiple signal generation strategies, and extensive backtesting capabilities.

## 🚀 Features

### Core Architecture
- **Modular Design**: Clean separation of data, signals, execution, and risk management
- **Production-Ready**: Comprehensive error handling, logging, and monitoring
- **Type Safety**: Full type hints with mypy validation
- **Testing**: Extensive unit and integration tests with pytest

### Data Infrastructure
- **CCXT Integration**: Exchange-agnostic OHLCV data loading with rate limiting
- **Technical Indicators**: 20+ indicators including RSI, MACD, Bollinger Bands, ATR
- **No Lookahead Bias**: Strict validation to prevent future data leakage
- **Efficient Caching**: Parquet-based data caching with automatic cleanup

### Signal Generation
- **Momentum Breakout**: Trend-following with volatility compression filters
- **Mean Reversion**: RSI and z-score based contrarian signals
- **Machine Learning**: Gradient boosting classifier with probability calibration
- **Composite Signals**: Weighted combination of multiple strategies

### Risk Management
- **Position Sizing**: Volatility-targeted sizing with Kelly fraction optimization
- **Stop Losses**: ATR-based stops with trailing stop functionality
- **Daily Limits**: 2% daily loss limit with automatic kill-switch
- **Portfolio Limits**: Maximum positions and correlation controls

### Backtesting Engine
- **Event-Driven**: Realistic execution simulation with latency and slippage
- **Cost Modeling**: Commission and market impact simulation
- **Walk-Forward**: Rolling window optimization and validation
- **Comprehensive Metrics**: 15+ performance metrics including Sharpe, Sortino, Calmar ratios

### Paper Trading
- **Realistic Simulation**: FIFO fills, partial fills, and market impact
- **Live Data**: Real-time market data processing
- **Same Logic**: Identical order logic as backtesting for consistency

## 📁 Project Structure

```
src/
├── data/
│   ├── loaders.py          # CCXT-based OHLCV data loading
│   └── features.py         # Technical indicators and feature engineering
├── models/
│   ├── signals.py          # Signal generation strategies
│   └── labeling.py         # Triple-barrier labeling for ML
├── exec/
│   ├── position_sizing.py  # Volatility-targeted position sizing
│   └── risk.py            # Risk management and stop losses
├── backtest/
│   ├── engine.py          # Event-driven backtesting engine
│   └── metrics.py         # Performance analysis and metrics
├── live/
│   ├── broker_stub.py     # Paper trading simulation
│   └── websocket.py       # Market data stream simulation
└── utils/
    ├── config.py          # Pydantic configuration management
    ├── time.py            # Timezone-safe timestamp utilities
    └── io.py              # Data caching and persistence

scripts/
├── run_backtest.py        # CLI for running backtests
└── run_paper.py           # CLI for paper trading

tests/
├── unit/                  # Unit tests for individual components
└── integration/           # End-to-end system tests

configs/
└── settings.example.yaml  # Example configuration file
```

## 🛠 Installation

### Prerequisites
- Python 3.9+
- Poetry (recommended) or pip

### Using Poetry (Recommended)
```bash
git clone <repository-url>
cd Trade-bot
poetry install
poetry shell
```

### Using pip
```bash
git clone <repository-url>
cd Trade-bot
pip install -r requirements.txt  # You'll need to create this from pyproject.toml
```

## ⚙️ Configuration

Copy the example configuration and customize:
```bash
cp configs/settings.example.yaml configs/settings.yaml
# Edit configs/settings.yaml with your preferences
```

Key configuration sections:
- **Exchange**: API credentials and rate limits
- **Trading**: Symbols, timeframes, and risk parameters
- **Backtest**: Date ranges and cost models
- **Data**: Caching and storage settings

## 🚀 Quick Start

### Run a Backtest
```bash
python scripts/run_backtest.py \
    --symbols "BTC/USDT,ETH/USDT" \
    --start-date 2023-01-01 \
    --end-date 2023-12-31 \
    --initial-cash 100000 \
    --output-dir results/
```

### Start Paper Trading
```bash
python scripts/run_paper.py run \
    --symbols "BTC/USDT,ETH/USDT" \
    --initial-cash 100000 \
    --output-dir results/live/
```

### Test Market Data
```bash
python scripts/run_paper.py test-data \
    --symbols "BTC/USDT,ETH/USDT" \
    --duration 60
```

## 📊 Strategy Details

### Momentum Breakout Strategy
- Identifies breakouts above/below rolling highs/lows
- Filters with volatility compression (low vol periods)
- Confirms with trend direction (50/200 EMA slope)
- Targets 2-3% moves with 1% stop losses

### Mean Reversion Strategy
- Uses RSI(2) and RSI(14) for oversold/overbought conditions
- Z-score of returns for statistical mean reversion
- Filters for low volatility environments
- Quick exits on trend confirmation

### Machine Learning Strategy
- Gradient boosting classifier on lagged features
- Triple-barrier labeling for realistic targets
- Probability calibration for reliable confidence scores
- Only trades when P(up) > 60% and expected value > 0

## 🔧 Development

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/ tests/ scripts/

# Lint code
ruff src/ tests/ scripts/

# Type checking
mypy src/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

- **Return Metrics**: Total return, CAGR, volatility
- **Risk-Adjusted**: Sharpe ratio, Sortino ratio, Calmar ratio
- **Drawdown**: Maximum drawdown, recovery time
- **Risk Metrics**: VaR, CVaR, skewness, kurtosis
- **Trading**: Hit rate, profit factor, expectancy, turnover

## ⚠️ Risk Management

### Built-in Safeguards
- **Daily Loss Limit**: Automatic shutdown at 2% daily loss
- **Position Limits**: Maximum 10% of equity per position
- **Stop Losses**: ATR-based stops with trailing functionality
- **Kill Switch**: Emergency shutdown capability
- **No Future Data**: Strict validation prevents lookahead bias

### Risk Parameters
```yaml
trading:
  max_position_size: 0.1        # 10% max position
  max_daily_loss: 0.02          # 2% daily loss limit
  max_risk_per_trade: 0.005     # 0.5% risk per trade
  target_volatility: 0.15       # 15% target volatility
  kelly_cap: 0.25               # Cap Kelly fraction
```

## 🔒 Security & Safety

- **Paper Trading Only**: No real money trading implemented
- **No Private Keys**: System doesn't handle real exchange authentication
- **Sandbox Mode**: All exchange connections use sandbox/testnet
- **Configuration Validation**: Pydantic ensures config correctness
- **Comprehensive Logging**: Full audit trail of all decisions

## 📚 Documentation

### Key Components

1. **Data Loaders** (`src/data/loaders.py`): CCXT-based data fetching with error handling
2. **Features** (`src/data/features.py`): Technical indicators with no lookahead bias
3. **Signals** (`src/models/signals.py`): Multiple signal generation strategies
4. **Position Sizing** (`src/exec/position_sizing.py`): Kelly and volatility-based sizing
5. **Risk Management** (`src/exec/risk.py`): Comprehensive risk controls
6. **Backtesting** (`src/backtest/engine.py`): Event-driven simulation engine
7. **Paper Trading** (`src/live/broker_stub.py`): Realistic trading simulation

### Configuration Reference

See `configs/settings.example.yaml` for detailed configuration options including:
- Exchange settings and API configuration
- Trading parameters and risk limits
- Backtest settings and date ranges
- Feature engineering parameters
- Strategy-specific settings

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended for live trading with real money. Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🙏 Acknowledgments

- [CCXT](https://github.com/ccxt/ccxt) for exchange connectivity
- [pandas](https://pandas.pydata.org/) for data manipulation
- [scikit-learn](https://scikit-learn.org/) for machine learning
- [pytest](https://pytest.org/) for testing framework
- The quantitative finance community for research and insights