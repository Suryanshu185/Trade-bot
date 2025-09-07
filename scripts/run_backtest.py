#!/usr/bin/env python3
"""CLI script for running backtests."""

import asyncio
import click
from pathlib import Path
import pandas as pd
from loguru import logger

from src.data.loaders import load_crypto_data
from src.data.features import FeatureEngine
from src.models.signals import create_default_signal_generators
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceAnalyzer
from src.utils.config import load_config
from src.utils.io import ResultsWriter


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--strategy', '-s', default='default', help='Strategy name')
@click.option('--symbols', default='BTC/USDT,ETH/USDT', help='Comma-separated symbols')
@click.option('--start-date', type=click.DateTime(), help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=click.DateTime(), help='End date (YYYY-MM-DD)')
@click.option('--initial-cash', type=float, default=100000.0, help='Initial cash')
@click.option('--commission', type=float, default=0.001, help='Commission rate')
@click.option('--slippage', type=float, default=2.0, help='Slippage in basis points')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def main(
    config: str,
    strategy: str,
    symbols: str,
    start_date,
    end_date,
    initial_cash: float,
    commission: float,
    slippage: float,
    output_dir: str,
    verbose: bool
):
    """Run backtesting with specified parameters."""
    
    # Setup logging
    if verbose:
        logger.add("backtest.log", level="DEBUG")
    else:
        logger.add("backtest.log", level="INFO")
    
    logger.info("Starting backtest runner...")
    
    # Load configuration
    if config:
        config_obj = load_config(Path(config))
    else:
        config_obj = load_config()
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Set date range
    if start_date:
        start_date = pd.Timestamp(start_date).tz_localize('UTC')
    else:
        start_date = pd.Timestamp.now().tz_localize('UTC') - pd.Timedelta(days=365)
    
    if end_date:
        end_date = pd.Timestamp(end_date).tz_localize('UTC')
    else:
        end_date = pd.Timestamp.now().tz_localize('UTC')
    
    logger.info(f"Backtest configuration:")
    logger.info(f"  Strategy: {strategy}")
    logger.info(f"  Symbols: {symbol_list}")
    logger.info(f"  Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"  Initial cash: ${initial_cash:,.2f}")
    logger.info(f"  Commission: {commission:.4f}")
    logger.info(f"  Slippage: {slippage} bps")
    
    try:
        # Run backtest
        results = asyncio.run(run_backtest(
            strategy=strategy,
            symbols=symbol_list,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            commission=commission,
            slippage=slippage,
            output_dir=output_dir
        ))
        
        logger.info("Backtest completed successfully!")
        
        # Print summary
        if results and 'metrics' in results:
            print("\n" + "="*60)
            print("BACKTEST SUMMARY")
            print("="*60)
            
            metrics = results['metrics']
            print(f"Total Return:     {metrics.get('total_return', 0)*100:8.2f}%")
            print(f"CAGR:            {metrics.get('cagr', 0)*100:8.2f}%")
            print(f"Max Drawdown:    {metrics.get('max_drawdown', 0)*100:8.2f}%")
            print(f"Sharpe Ratio:    {metrics.get('sharpe_ratio', 0):8.2f}")
            print(f"Total Trades:    {metrics.get('total_trades', 0):8.0f}")
            print(f"Win Rate:        {metrics.get('hit_rate', 0)*100:8.2f}%")
            
            if output_dir:
                print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


async def run_backtest(
    strategy: str,
    symbols: list,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_cash: float,
    commission: float,
    slippage: float,
    output_dir: str = None
) -> dict:
    """Run the actual backtest."""
    
    # Load historical data
    logger.info("Loading historical data...")
    days_back = (end_date - start_date).days + 30  # Extra days for indicators
    
    try:
        data_dict = await load_crypto_data(
            symbols=symbols,
            timeframe="5m",
            days_back=days_back
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        # Create dummy data for testing
        logger.warning("Using dummy data for testing")
        data_dict = create_dummy_data(symbols, start_date, end_date)
    
    # Filter data to backtest period
    for symbol in symbols:
        if symbol in data_dict and not data_dict[symbol].empty:
            mask = (data_dict[symbol].index >= start_date) & (data_dict[symbol].index <= end_date)
            data_dict[symbol] = data_dict[symbol][mask]
    
    # Add features to data
    logger.info("Computing technical features...")
    feature_engine = FeatureEngine()
    
    for symbol in symbols:
        if symbol in data_dict and not data_dict[symbol].empty:
            data_dict[symbol] = feature_engine.compute_all_features(data_dict[symbol])
    
    # Create signal generators
    logger.info("Initializing signal generators...")
    signal_generator = create_default_signal_generators()
    
    # Train signal generators
    for symbol in symbols:
        if symbol in data_dict and not data_dict[symbol].empty:
            training_data = data_dict[symbol][data_dict[symbol].index < start_date]
            if len(training_data) > 100:
                signal_generator.fit(training_data)
                break
    
    # Initialize backtest engine
    logger.info("Setting up backtest engine...")
    engine = BacktestEngine(
        initial_cash=initial_cash,
        commission_rate=commission,
        slippage_bps=slippage,
        start_date=start_date,
        end_date=end_date
    )
    
    # Add data and signals
    for symbol in symbols:
        if symbol in data_dict and not data_dict[symbol].empty:
            engine.add_data(symbol, data_dict[symbol])
    
    engine.add_signal_generator(strategy, signal_generator)
    
    # Run backtest
    logger.info("Running backtest...")
    backtest_results = engine.run_backtest()
    
    # Analyze performance
    logger.info("Analyzing performance...")
    analyzer = PerformanceAnalyzer()
    
    equity_curve = backtest_results.get('equity_curve', pd.Series())
    trades_df = backtest_results.get('trades', pd.DataFrame())
    
    if not equity_curve.empty:
        metrics = analyzer.analyze_backtest(equity_curve['equity'], trades_df)
        
        # Generate performance report
        report = analyzer.generate_performance_report(equity_curve['equity'], trades_df)
        logger.info(f"Performance Report:\n{report}")
        
        # Save results
        if output_dir:
            writer = ResultsWriter(Path(output_dir))
            writer.save_backtest_results(
                strategy_name=strategy,
                equity_curve=equity_curve,
                trades=trades_df,
                metrics=metrics
            )
        
        return {
            'equity_curve': equity_curve,
            'trades': trades_df,
            'metrics': metrics,
            'report': report
        }
    
    else:
        logger.warning("No backtest results generated")
        return {}


def create_dummy_data(symbols: list, start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
    """Create dummy OHLCV data for testing."""
    logger.warning("Creating dummy data - for testing only!")
    
    data_dict = {}
    
    # Create 5-minute timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='5min', tz='UTC')
    
    for symbol in symbols:
        # Generate random walk price data
        np.random.seed(42)  # For reproducibility
        
        n_periods = len(timestamps)
        returns = np.random.normal(0, 0.02, n_periods)  # 2% volatility
        
        # Start from $100
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV data
        data = pd.DataFrame(index=timestamps)
        data['close'] = prices
        
        # Generate OHLC from close prices
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        
        # Generate high/low with some randomness
        noise = np.random.normal(0, 0.005, n_periods)  # 0.5% noise
        data['high'] = data['close'] * (1 + np.abs(noise))
        data['low'] = data['close'] * (1 - np.abs(noise))
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        # Generate volume
        data['volume'] = np.random.exponential(1000, n_periods)
        
        # Add symbol column
        data['symbol'] = symbol
        
        data_dict[symbol] = data
    
    return data_dict


if __name__ == "__main__":
    main()