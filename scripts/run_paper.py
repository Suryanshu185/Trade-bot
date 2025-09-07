#!/usr/bin/env python3
"""CLI script for running paper trading."""

import asyncio
import signal
import click
from pathlib import Path
import pandas as pd
from loguru import logger

from src.models.signals import create_default_signal_generators
from src.live.broker_stub import PaperTradingEngine
from src.live.websocket import create_market_data_feed
from src.data.loaders import load_crypto_data
from src.utils.config import load_config
from src.utils.io import ResultsWriter


class PaperTradingApp:
    """Paper trading application."""
    
    def __init__(
        self,
        symbols: list,
        initial_cash: float,
        config_path: str = None,
        output_dir: str = None
    ):
        self.symbols = symbols
        self.initial_cash = initial_cash
        self.config_path = config_path
        self.output_dir = output_dir
        
        # Components
        self.trading_engine = None
        self.market_feed = None
        self.signal_generators = None
        
        # State
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown_event.set()
    
    async def initialize(self):
        """Initialize paper trading components."""
        logger.info("Initializing paper trading engine...")
        
        # Load configuration
        if self.config_path:
            config = load_config(Path(self.config_path))
        else:
            config = load_config()
        
        # Create signal generators
        self.signal_generators = create_default_signal_generators()
        
        # Load some historical data for signal generation
        logger.info("Loading historical data for signals...")
        try:
            historical_data = await load_crypto_data(
                symbols=self.symbols,
                timeframe="5m",
                days_back=30
            )
        except Exception as e:
            logger.warning(f"Failed to load historical data: {e}")
            historical_data = {}
        
        # Train signal generators if we have data
        if historical_data:
            for symbol, data in historical_data.items():
                if not data.empty:
                    self.signal_generators.fit(data)
                    break  # Train on first available symbol
        
        # Initialize trading engine
        self.trading_engine = PaperTradingEngine(
            initial_cash=self.initial_cash,
            symbols=self.symbols,
            signal_generators=self.signal_generators.generators
        )
        
        # Initialize market data feed
        self.market_feed = await create_market_data_feed(
            symbols=self.symbols,
            historical_data=historical_data
        )
        
        # Connect market feed to trading engine
        self.market_feed.add_data_handler(self._handle_market_data)
        
        logger.info("Paper trading engine initialized successfully")
    
    def _handle_market_data(self, symbol: str, data: dict):
        """Handle incoming market data."""
        # Update trading engine with market data
        if self.trading_engine:
            self.trading_engine.update_market_data(symbol, data)
    
    async def run(self):
        """Run the paper trading engine."""
        try:
            await self.initialize()
            
            # Start trading engine
            await self.trading_engine.start()
            self.is_running = True
            
            logger.info("Paper trading started. Press Ctrl+C to stop.")
            
            # Main trading loop
            while self.is_running and not self.shutdown_event.is_set():
                # Process trading cycle
                await self.trading_engine.process_trading_cycle()
                
                # Wait before next cycle
                await asyncio.sleep(30)  # Process every 30 seconds
                
                # Check for shutdown
                if self.shutdown_event.is_set():
                    break
        
        except Exception as e:
            logger.error(f"Paper trading error: {e}")
            raise
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown paper trading engine."""
        logger.info("Shutting down paper trading engine...")
        
        self.is_running = False
        
        # Stop trading engine
        if self.trading_engine:
            self.trading_engine.stop()
        
        # Disconnect market feed
        if self.market_feed:
            self.market_feed.disconnect()
        
        # Save final results
        await self._save_results()
        
        logger.info("Paper trading shutdown complete")
    
    async def _save_results(self):
        """Save trading results."""
        if not self.trading_engine or not self.output_dir:
            return
        
        try:
            # Get performance summary
            performance = self.trading_engine.get_performance_summary()
            
            # Get account summary
            account = self.trading_engine.broker.get_account_summary()
            
            # Save daily report
            writer = ResultsWriter(Path(self.output_dir))
            
            daily_report = {
                'final_equity': account['total_equity'],
                'total_return': performance.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'max_drawdown': performance.get('max_drawdown', 0),
                'total_trades': performance.get('total_trades', 0),
                'win_rate': performance.get('win_rate', 0),
                'positions': account['positions'],
                'cash': account['cash']
            }
            
            writer.save_daily_report(daily_report)
            
            # Save trade history
            trades_df = pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'symbol': trade.symbol,
                    'side': trade.side,
                    'quantity': trade.quantity,
                    'price': trade.price,
                    'commission': trade.commission,
                    'slippage': trade.slippage
                }
                for trade in self.trading_engine.broker.trades
            ])
            
            if not trades_df.empty:
                writer.save_live_trades(trades_df)
            
            logger.info(f"Results saved to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--symbols', default='BTC/USDT,ETH/USDT', help='Comma-separated symbols')
@click.option('--initial-cash', type=float, default=100000.0, help='Initial cash amount')
@click.option('--output-dir', type=click.Path(), help='Output directory for results')
@click.option('--duration', type=int, help='Duration in hours (default: run indefinitely)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def main(
    config: str,
    symbols: str,
    initial_cash: float,
    output_dir: str,
    duration: int,
    verbose: bool
):
    """Run paper trading with specified parameters."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.add("paper_trading.log", level=log_level)
    
    logger.info("Starting paper trading runner...")
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Setup output directory
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Paper trading configuration:")
    logger.info(f"  Symbols: {symbol_list}")
    logger.info(f"  Initial cash: ${initial_cash:,.2f}")
    logger.info(f"  Duration: {duration or 'indefinite'} hours")
    logger.info(f"  Output dir: {output_dir or 'none'}")
    
    # Create and run paper trading app
    app = PaperTradingApp(
        symbols=symbol_list,
        initial_cash=initial_cash,
        config_path=config,
        output_dir=output_dir
    )
    
    try:
        if duration:
            # Run for specified duration
            asyncio.run(run_with_timeout(app, duration))
        else:
            # Run indefinitely
            asyncio.run(app.run())
            
    except KeyboardInterrupt:
        logger.info("Paper trading interrupted by user")
    except Exception as e:
        logger.error(f"Paper trading failed: {e}")
        raise


async def run_with_timeout(app: PaperTradingApp, duration_hours: int):
    """Run paper trading with timeout."""
    try:
        # Start the app
        app_task = asyncio.create_task(app.run())
        
        # Wait for either completion or timeout
        await asyncio.wait_for(app_task, timeout=duration_hours * 3600)
        
    except asyncio.TimeoutError:
        logger.info(f"Paper trading completed after {duration_hours} hours")
        app.shutdown_event.set()
        
        # Give some time for graceful shutdown
        try:
            await asyncio.wait_for(app_task, timeout=30)
        except asyncio.TimeoutError:
            logger.warning("Forced shutdown after timeout")


@click.command()
@click.option('--symbols', default='BTC/USDT,ETH/USDT', help='Comma-separated symbols')
@click.option('--duration', type=int, default=60, help='Test duration in seconds')
def test_market_data(symbols: str, duration: int):
    """Test market data feed."""
    from src.live.websocket import run_market_data_test
    
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    logger.info(f"Testing market data for {symbol_list}")
    asyncio.run(run_market_data_test(symbol_list, duration))


@click.command()
@click.option('--initial-cash', type=float, default=10000.0, help='Initial cash')
@click.option('--duration', type=int, default=300, help='Test duration in seconds')
def test_paper_broker(initial_cash: float, duration: int):
    """Test paper broker functionality."""
    from src.live.broker_stub import PaperBroker, OrderType
    
    async def run_broker_test():
        broker = PaperBroker(initial_cash)
        
        logger.info(f"Testing paper broker with ${initial_cash}")
        
        # Submit some test orders
        order1 = broker.submit_order("BTC/USDT", "buy", 0.1, OrderType.MARKET)
        order2 = broker.submit_order("ETH/USDT", "buy", 1.0, OrderType.LIMIT, 2000.0)
        
        logger.info(f"Submitted orders: {order1}, {order2}")
        
        # Simulate market data updates
        for i in range(duration):
            market_data = {
                "BTC/USDT": {
                    "bid": 45000 + i * 10,
                    "ask": 45050 + i * 10,
                    "close": 45025 + i * 10,
                    "volume": 1000
                },
                "ETH/USDT": {
                    "bid": 2000 + i * 2,
                    "ask": 2005 + i * 2,
                    "close": 2002.5 + i * 2,
                    "volume": 5000
                }
            }
            
            trades = broker.update_market_data(market_data)
            
            if trades:
                for trade in trades:
                    logger.info(f"Trade executed: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
            
            await asyncio.sleep(1)
        
        # Final account summary
        summary = broker.get_account_summary()
        logger.info(f"Final account summary: {summary}")
    
    asyncio.run(run_broker_test())


# Add test commands to CLI
@click.group()
def cli():
    """Paper trading CLI."""
    pass


cli.add_command(main, "run")
cli.add_command(test_market_data, "test-data")
cli.add_command(test_paper_broker, "test-broker")


if __name__ == "__main__":
    cli()