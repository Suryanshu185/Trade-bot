"""Integration test for the complete trading system."""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path

from src.models.signals import create_default_signal_generators
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceAnalyzer
from src.live.broker_stub import PaperTradingEngine
from src.data.features import FeatureEngine
from src.utils.config import Config


class TestSystemIntegration:
    """Integration tests for the complete trading system."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create comprehensive historical data for testing."""
        # Create 3 months of 5-minute data
        start_date = pd.Timestamp('2023-01-01', tz='UTC')
        end_date = pd.Timestamp('2023-03-31', tz='UTC')
        timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
        
        data = {}
        symbols = ['BTC/USDT', 'ETH/USDT']
        
        for symbol in symbols:
            np.random.seed(42 if symbol == 'BTC/USDT' else 43)
            
            # Generate realistic price series
            n_periods = len(timestamps)
            base_price = 50000.0 if symbol == 'BTC/USDT' else 3000.0
            
            # Add trend and volatility
            trend = np.linspace(0, 0.2, n_periods)  # 20% trend over period
            volatility = 0.02  # 2% per period volatility
            
            returns = np.random.normal(trend / n_periods, volatility, n_periods)
            prices = base_price * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            df = pd.DataFrame(index=timestamps)
            df['close'] = prices
            df['open'] = df['close'].shift(1).fillna(prices[0])
            
            # Generate realistic high/low
            high_low_spread = np.random.uniform(0.001, 0.01, n_periods)
            df['high'] = df['close'] * (1 + high_low_spread/2)
            df['low'] = df['close'] * (1 - high_low_spread/2)
            
            # Ensure OHLC consistency
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            # Generate volume with some correlation to volatility
            vol_factor = np.abs(returns) + 0.5  # Higher volume on volatile days
            df['volume'] = np.random.exponential(1000) * vol_factor
            
            df['symbol'] = symbol
            data[symbol] = df
        
        return data
    
    def test_end_to_end_backtest(self, sample_historical_data):
        """Test complete backtest workflow."""
        # Setup
        symbols = list(sample_historical_data.keys())
        start_date = pd.Timestamp('2023-01-15', tz='UTC')  # Leave some data for features
        end_date = pd.Timestamp('2023-03-15', tz='UTC')
        
        # Initialize components
        feature_engine = FeatureEngine()
        signal_generator = create_default_signal_generators()
        
        # Process data and add features
        processed_data = {}
        for symbol, data in sample_historical_data.items():
            processed_data[symbol] = feature_engine.compute_all_features(data)
        
        # Train signal generators
        for symbol, data in processed_data.items():
            training_data = data[data.index < start_date]
            if len(training_data) > 100:
                signal_generator.fit(training_data)
                break
        
        # Run backtest
        engine = BacktestEngine(
            initial_cash=100000.0,
            commission_rate=0.001,
            slippage_bps=2.0,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add data and signals
        for symbol, data in processed_data.items():
            engine.add_data(symbol, data)
        
        engine.add_signal_generator('composite', signal_generator)
        
        # Execute backtest
        results = engine.run_backtest()
        
        # Validate results
        assert 'equity_curve' in results
        assert 'trades' in results
        assert not results['equity_curve'].empty
        
        # Check that equity curve makes sense
        equity_curve = results['equity_curve']
        assert equity_curve['equity'].iloc[0] == 100000.0  # Initial cash
        assert equity_curve['equity'].iloc[-1] > 0  # Final equity positive
        
        # Analyze performance
        analyzer = PerformanceAnalyzer()
        metrics = analyzer.analyze_backtest(equity_curve['equity'], results['trades'])
        
        # Basic sanity checks
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        
        # Max drawdown should be reasonable
        assert 0 <= metrics['max_drawdown'] <= 1
        
        print(f"Backtest completed: {metrics['total_return']:.2%} return, "
              f"{metrics['sharpe_ratio']:.2f} Sharpe, {metrics['max_drawdown']:.2%} max DD")
    
    @pytest.mark.asyncio
    async def test_paper_trading_integration(self, sample_historical_data):
        """Test paper trading system integration."""
        symbols = list(sample_historical_data.keys())
        
        # Create signal generators
        signal_generators = create_default_signal_generators()
        
        # Train on historical data
        for symbol, data in sample_historical_data.items():
            feature_engine = FeatureEngine()
            data_with_features = feature_engine.compute_all_features(data)
            signal_generators.fit(data_with_features)
            break  # Train on first symbol
        
        # Initialize paper trading engine
        engine = PaperTradingEngine(
            initial_cash=50000.0,
            symbols=symbols,
            signal_generators=signal_generators.generators
        )
        
        await engine.start()
        
        # Simulate market data updates
        test_duration = 60  # seconds
        update_interval = 5  # seconds
        
        try:
            for i in range(test_duration // update_interval):
                # Simulate market data
                for symbol in symbols:
                    # Get base price from historical data
                    base_price = sample_historical_data[symbol]['close'].iloc[-1]
                    
                    # Add some random movement
                    price_change = np.random.normal(0, 0.001)  # 0.1% volatility
                    current_price = base_price * (1 + price_change)
                    
                    market_data = {
                        'timestamp': pd.Timestamp.now(tz='UTC'),
                        'bid': current_price * 0.9995,
                        'ask': current_price * 1.0005,
                        'close': current_price,
                        'volume': np.random.exponential(1000),
                        'high_24h': current_price * 1.02,
                        'low_24h': current_price * 0.98
                    }
                    
                    engine.update_market_data(symbol, market_data)
                
                # Process trading cycle
                await engine.process_trading_cycle()
                
                # Wait before next update
                await asyncio.sleep(update_interval)
        
        finally:
            engine.stop()
        
        # Check final state
        account = engine.broker.get_account_summary()
        performance = engine.get_performance_summary()
        
        # Basic validations
        assert account['total_equity'] > 0
        assert account['cash'] > 0
        assert isinstance(performance, dict)
        
        print(f"Paper trading completed: "
              f"Final equity: ${account['total_equity']:,.2f}, "
              f"Trades: {account['total_trades']}")
    
    def test_signal_generation_pipeline(self, sample_historical_data):
        """Test the complete signal generation pipeline."""
        # Get sample data
        symbol = 'BTC/USDT'
        data = sample_historical_data[symbol]
        
        # Add technical features
        feature_engine = FeatureEngine()
        data_with_features = feature_engine.compute_all_features(data)
        
        # Create and train signal generators
        signal_generator = create_default_signal_generators()
        
        # Train on first half of data
        split_point = len(data_with_features) // 2
        training_data = data_with_features.iloc[:split_point]
        test_data = data_with_features.iloc[split_point:]
        
        signal_generator.fit(training_data)
        
        # Generate signals on test data
        signals = signal_generator.generate_signals(test_data)
        
        # Validate signals
        assert isinstance(signals, pd.DataFrame)
        assert 'composite' in signals.columns
        assert len(signals) <= len(test_data)
        
        # Check signal values are reasonable
        composite_signals = signals['composite'].dropna()
        assert all(abs(sig) <= 1.1 for sig in composite_signals), "Signals should be roughly between -1 and 1"
        
        # Check signal distribution
        signal_std = composite_signals.std()
        assert signal_std > 0, "Signals should have some variation"
        
        print(f"Generated {len(composite_signals)} signals with std={signal_std:.3f}")
    
    def test_risk_integration(self, sample_historical_data):
        """Test risk management integration with trading."""
        from src.exec.risk import RiskManager
        from src.exec.position_sizing import PositionSizer
        
        # Initialize components
        risk_manager = RiskManager(
            max_daily_loss=0.05,  # 5% daily loss limit
            max_positions=3
        )
        
        position_sizer = PositionSizer(
            max_risk_per_trade=0.02,  # 2% risk per trade
            target_volatility=0.15
        )
        
        # Simulate trading day
        portfolio_value = 100000.0
        risk_manager.reset_daily_tracking(portfolio_value)
        
        # Test position creation with size limits
        symbols = list(sample_historical_data.keys())
        
        for i, symbol in enumerate(symbols):
            current_price = sample_historical_data[symbol]['close'].iloc[-1]
            signal_strength = 0.5  # Moderate signal
            
            # Calculate position size
            size, details = position_sizer.calculate_size(
                signal_strength=signal_strength,
                price=current_price,
                volatility=0.20,  # 20% volatility
                portfolio_value=portfolio_value
            )
            
            # Add position through risk manager
            if size != 0:
                success = risk_manager.add_position(
                    symbol=symbol,
                    size=size,
                    entry_price=current_price,
                    stop_loss=position_sizer.calculate_stop_loss(
                        current_price, size, portfolio_value, atr=current_price * 0.02
                    )
                )
                
                if success:
                    print(f"Added position: {symbol} size={size:.4f} @ ${current_price:.2f}")
        
        # Test risk monitoring
        current_prices = {
            symbol: data['close'].iloc[-1] * 0.95  # 5% price drop
            for symbol, data in sample_historical_data.items()
        }
        
        # Update risk manager with new prices
        triggered_events = risk_manager.update_prices(current_prices)
        
        # Check risk summary
        risk_summary = risk_manager.get_position_risk_summary(current_prices)
        
        # Validate risk calculations
        assert 'total_exposure' in risk_summary
        assert 'total_unrealized_pnl' in risk_summary
        assert risk_summary['num_positions'] <= 3  # Respect position limit
        
        print(f"Risk summary: {risk_summary['num_positions']} positions, "
              f"${risk_summary['total_unrealized_pnl']:.2f} unrealized P&L")
    
    def test_configuration_integration(self, temp_config_file):
        """Test configuration system integration."""
        from src.utils.config import load_config
        
        # Load config from file
        config = load_config(temp_config_file)
        
        # Test that config affects component initialization
        assert config.trading.symbols == ["BTC/USDT", "ETH/USDT"]
        assert config.trading.max_risk_per_trade == 0.01
        assert config.backtest.initial_cash == 10000.0
        
        # Test creating components with config
        from src.exec.position_sizing import create_position_sizer
        from src.exec.risk import create_risk_manager
        
        sizer = create_position_sizer()
        risk_mgr = create_risk_manager()
        
        # Check that config values are applied
        assert sizer.max_risk_per_trade == config.trading.max_risk_per_trade
        assert sizer.target_volatility == config.trading.target_volatility
    
    def test_data_pipeline_integration(self, sample_historical_data):
        """Test data loading and processing pipeline."""
        from src.data.features import add_technical_features
        from src.utils.io import DataCache, save_dataframe, load_dataframe
        
        # Test data caching
        cache = DataCache()
        symbol = 'BTC/USDT'
        data = sample_historical_data[symbol]
        
        # Cache data
        cache_key = f"test_data_{symbol}"
        cache.set(cache_key, data)
        
        # Retrieve from cache
        cached_data = cache.get(cache_key)
        assert cached_data is not None
        assert len(cached_data) == len(data)
        
        # Test feature pipeline
        data_with_features = add_technical_features(data)
        
        # Validate features were added
        original_cols = set(data.columns)
        feature_cols = set(data_with_features.columns) - original_cols
        
        assert len(feature_cols) > 10, "Should have added multiple technical features"
        
        # Test that features have reasonable values
        for col in feature_cols:
            if 'rsi' in col.lower():
                rsi_values = data_with_features[col].dropna()
                if len(rsi_values) > 0:
                    assert all(0 <= val <= 100 for val in rsi_values), f"RSI values should be 0-100: {col}"
        
        print(f"Added {len(feature_cols)} features to {symbol} data")
    
    def test_performance_metrics_integration(self, sample_historical_data):
        """Test performance analysis integration."""
        from src.backtest.metrics import PerformanceAnalyzer
        
        # Create synthetic equity curve
        dates = sample_historical_data['BTC/USDT'].index[-100:]  # Last 100 periods
        
        # Generate realistic equity curve with some volatility and trend
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # Positive drift with volatility
        equity_values = 100000 * np.exp(np.cumsum(returns))
        
        equity_curve = pd.Series(equity_values, index=dates)
        
        # Analyze performance
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        metrics = analyzer.analyze_backtest(equity_curve)
        
        # Validate all expected metrics are present
        expected_metrics = [
            'total_return', 'cagr', 'volatility', 'sharpe_ratio',
            'sortino_ratio', 'max_drawdown', 'calmar_ratio'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert isinstance(metrics[metric], (int, float)), f"Metric {metric} should be numeric"
        
        # Generate performance report
        report = analyzer.generate_performance_report(equity_curve)
        assert isinstance(report, str)
        assert len(report) > 100  # Should be a substantial report
        
        print(f"Performance analysis: {metrics['total_return']:.2%} return, "
              f"{metrics['sharpe_ratio']:.2f} Sharpe ratio")