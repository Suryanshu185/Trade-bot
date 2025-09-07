"""Test configuration for the trading bot."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='5min', tz='UTC')
    
    # Generate realistic price data
    np.random.seed(42)
    n_periods = len(dates)
    
    # Random walk with drift
    returns = np.random.normal(0.0001, 0.02, n_periods)  # Small positive drift, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame(index=dates)
    data['close'] = prices
    data['open'] = data['close'].shift(1).fillna(prices[0])
    
    # Generate high/low with realistic spreads
    high_low_spread = np.random.uniform(0.001, 0.01, n_periods)  # 0.1% to 1% spread
    data['high'] = data['close'] * (1 + high_low_spread/2)
    data['low'] = data['close'] * (1 - high_low_spread/2)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
    data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
    
    # Generate volume
    data['volume'] = np.random.exponential(1000, n_periods)
    
    return data


@pytest.fixture
def sample_trades_data():
    """Create sample trades data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H', tz='UTC')
    
    trades = []
    for i, date in enumerate(dates[:100]):  # 100 trades
        trade = {
            'timestamp': date,
            'symbol': 'BTC/USDT',
            'side': 'buy' if i % 2 == 0 else 'sell',
            'quantity': np.random.uniform(0.1, 1.0),
            'price': 100 + np.random.normal(0, 5),  # Price around $100 with some variation
            'commission': 0.001,
            'slippage': 0.0001,
            'pnl': np.random.normal(0, 50)  # Random P&L
        }
        trades.append(trade)
    
    return pd.DataFrame(trades)


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary configuration file."""
    config_content = """
exchange:
  name: "binance"
  sandbox: true

trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  max_risk_per_trade: 0.01
  target_volatility: 0.15

backtest:
  initial_cash: 10000.0
  commission: 0.001
"""
    
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content)
    
    return config_file


@pytest.fixture
def mock_market_data():
    """Mock market data for testing."""
    return {
        'BTC/USDT': {
            'bid': 45000.0,
            'ask': 45050.0,
            'close': 45025.0,
            'volume': 1000.0,
            'timestamp': pd.Timestamp.now(tz='UTC')
        },
        'ETH/USDT': {
            'bid': 2000.0,
            'ask': 2005.0,
            'close': 2002.5,
            'volume': 5000.0,
            'timestamp': pd.Timestamp.now(tz='UTC')
        }
    }