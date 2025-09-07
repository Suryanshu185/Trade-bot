"""Market data websocket simulator with clock skew handling."""

import asyncio
import json
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger

from ..data.loaders import OHLCVLoader
from ..utils.config import get_config
from ..utils.time import now_utc, to_utc


@dataclass
class MarketTick:
    """Market data tick."""
    symbol: str
    timestamp: pd.Timestamp
    bid: float
    ask: float
    last: float
    volume: float
    high_24h: float
    low_24h: float


class MarketDataSimulator:
    """Simulate realistic market data stream."""
    
    def __init__(
        self,
        symbols: List[str],
        base_data: Optional[Dict[str, pd.DataFrame]] = None,
        tick_frequency_ms: int = 1000,
        volatility_factor: float = 1.0,
        spread_bps: float = 5.0
    ):
        self.symbols = symbols
        self.base_data = base_data or {}
        self.tick_frequency_ms = tick_frequency_ms
        self.volatility_factor = volatility_factor
        self.spread_bps = spread_bps
        
        # Current state
        self.current_prices = {}
        self.last_timestamps = {}
        self.daily_stats = {}
        
        # Clock skew simulation
        self.clock_skew_ms = np.random.normal(0, 50)  # ±50ms skew
        
        # Initialize with base data if available
        self._initialize_from_base_data()
    
    def _initialize_from_base_data(self) -> None:
        """Initialize simulator state from base data."""
        for symbol, data in self.base_data.items():
            if not data.empty:
                latest = data.iloc[-1]
                self.current_prices[symbol] = latest['close']
                self.last_timestamps[symbol] = data.index[-1]
                
                # Calculate daily stats
                daily_high = data['high'].rolling(window=24*60, min_periods=1).max().iloc[-1]
                daily_low = data['low'].rolling(window=24*60, min_periods=1).min().iloc[-1]
                daily_volume = data['volume'].rolling(window=24*60, min_periods=1).sum().iloc[-1]
                
                self.daily_stats[symbol] = {
                    'high_24h': daily_high,
                    'low_24h': daily_low,
                    'volume_24h': daily_volume
                }
    
    def _generate_price_movement(self, symbol: str, current_price: float) -> float:
        """Generate realistic price movement."""
        # Base volatility (annualized)
        base_vol = 0.5  # 50% annualized volatility
        
        # Convert to per-tick volatility
        ticks_per_year = 365 * 24 * 60 * 60 * 1000 / self.tick_frequency_ms
        tick_vol = base_vol / np.sqrt(ticks_per_year)
        
        # Generate random return
        random_return = np.random.normal(0, tick_vol * self.volatility_factor)
        
        # Add some mean reversion
        if symbol in self.base_data and not self.base_data[symbol].empty:
            recent_data = self.base_data[symbol].tail(20)
            if not recent_data.empty:
                mean_price = recent_data['close'].mean()
                mean_reversion = -0.001 * (current_price - mean_price) / mean_price
                random_return += mean_reversion
        
        # Apply return to price
        new_price = current_price * (1 + random_return)
        
        # Ensure price stays positive and reasonable
        new_price = max(new_price, current_price * 0.95)  # Max 5% drop per tick
        new_price = min(new_price, current_price * 1.05)  # Max 5% gain per tick
        
        return new_price
    
    def _calculate_bid_ask(self, mid_price: float) -> tuple[float, float]:
        """Calculate bid/ask from mid price."""
        spread = mid_price * (self.spread_bps / 10000)
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        return bid, ask
    
    def generate_tick(self, symbol: str) -> MarketTick:
        """Generate a single market tick."""
        current_time = now_utc()
        
        # Apply clock skew
        adjusted_time = current_time + pd.Timedelta(milliseconds=self.clock_skew_ms)
        
        # Initialize price if needed
        if symbol not in self.current_prices:
            if symbol in self.base_data and not self.base_data[symbol].empty:
                self.current_prices[symbol] = self.base_data[symbol]['close'].iloc[-1]
            else:
                self.current_prices[symbol] = 100.0  # Default starting price
        
        # Generate new price
        current_price = self.current_prices[symbol]
        new_price = self._generate_price_movement(symbol, current_price)
        self.current_prices[symbol] = new_price
        
        # Calculate bid/ask
        bid, ask = self._calculate_bid_ask(new_price)
        
        # Generate volume (simplified)
        base_volume = 1000.0
        volume = np.random.exponential(base_volume)
        
        # Update daily stats
        if symbol not in self.daily_stats:
            self.daily_stats[symbol] = {
                'high_24h': new_price,
                'low_24h': new_price,
                'volume_24h': volume
            }
        else:
            stats = self.daily_stats[symbol]
            stats['high_24h'] = max(stats['high_24h'], new_price)
            stats['low_24h'] = min(stats['low_24h'], new_price)
            stats['volume_24h'] += volume
        
        return MarketTick(
            symbol=symbol,
            timestamp=adjusted_time,
            bid=bid,
            ask=ask,
            last=new_price,
            volume=volume,
            high_24h=self.daily_stats[symbol]['high_24h'],
            low_24h=self.daily_stats[symbol]['low_24h']
        )
    
    def reset_daily_stats(self) -> None:
        """Reset daily statistics."""
        for symbol in self.daily_stats:
            current_price = self.current_prices.get(symbol, 100.0)
            self.daily_stats[symbol] = {
                'high_24h': current_price,
                'low_24h': current_price,
                'volume_24h': 0.0
            }


class WebSocketMarketDataStream:
    """Simulated WebSocket market data stream."""
    
    def __init__(
        self,
        symbols: List[str],
        simulator: Optional[MarketDataSimulator] = None,
        tick_frequency_ms: int = 1000
    ):
        self.symbols = symbols
        self.simulator = simulator or MarketDataSimulator(symbols)
        self.tick_frequency_ms = tick_frequency_ms
        
        # Subscribers
        self.tick_handlers: List[Callable[[MarketTick], None]] = []
        self.aggregate_handlers: List[Callable[[Dict[str, MarketTick]], None]] = []
        
        # State
        self.is_streaming = False
        self.latest_ticks: Dict[str, MarketTick] = {}
        
        # Performance tracking
        self.message_count = 0
        self.last_stats_time = now_utc()
    
    def subscribe_to_ticks(self, handler: Callable[[MarketTick], None]) -> None:
        """Subscribe to individual tick updates."""
        self.tick_handlers.append(handler)
    
    def subscribe_to_aggregates(self, handler: Callable[[Dict[str, MarketTick]], None]) -> None:
        """Subscribe to aggregated tick updates."""
        self.aggregate_handlers.append(handler)
    
    async def start_stream(self) -> None:
        """Start the market data stream."""
        logger.info(f"Starting market data stream for {len(self.symbols)} symbols")
        self.is_streaming = True
        
        try:
            while self.is_streaming:
                # Generate ticks for all symbols
                current_ticks = {}
                
                for symbol in self.symbols:
                    tick = self.simulator.generate_tick(symbol)
                    current_ticks[symbol] = tick
                    self.latest_ticks[symbol] = tick
                    
                    # Notify individual tick handlers
                    for handler in self.tick_handlers:
                        try:
                            handler(tick)
                        except Exception as e:
                            logger.error(f"Error in tick handler: {e}")
                
                # Notify aggregate handlers
                for handler in self.aggregate_handlers:
                    try:
                        handler(current_ticks)
                    except Exception as e:
                        logger.error(f"Error in aggregate handler: {e}")
                
                self.message_count += len(self.symbols)
                
                # Log performance stats periodically
                if self.message_count % 1000 == 0:
                    self._log_performance_stats()
                
                # Wait for next tick
                await asyncio.sleep(self.tick_frequency_ms / 1000.0)
                
        except Exception as e:
            logger.error(f"Market data stream error: {e}")
        finally:
            self.is_streaming = False
    
    def stop_stream(self) -> None:
        """Stop the market data stream."""
        logger.info("Stopping market data stream")
        self.is_streaming = False
    
    def get_latest_tick(self, symbol: str) -> Optional[MarketTick]:
        """Get latest tick for symbol."""
        return self.latest_ticks.get(symbol)
    
    def get_latest_ticks(self) -> Dict[str, MarketTick]:
        """Get latest ticks for all symbols."""
        return self.latest_ticks.copy()
    
    def _log_performance_stats(self) -> None:
        """Log performance statistics."""
        current_time = now_utc()
        elapsed = (current_time - self.last_stats_time).total_seconds()
        
        if elapsed > 0:
            msg_per_sec = 1000 / elapsed
            logger.debug(f"Market data: {msg_per_sec:.1f} messages/sec, "
                        f"Total: {self.message_count}")
        
        self.last_stats_time = current_time


class MarketDataFeed:
    """High-level market data feed with reconnection and error handling."""
    
    def __init__(
        self,
        symbols: List[str],
        use_simulation: bool = True,
        reconnect_attempts: int = 3,
        heartbeat_interval: int = 30
    ):
        self.symbols = symbols
        self.use_simulation = use_simulation
        self.reconnect_attempts = reconnect_attempts
        self.heartbeat_interval = heartbeat_interval
        
        # Initialize components
        if use_simulation:
            self.simulator = MarketDataSimulator(symbols)
            self.stream = WebSocketMarketDataStream(symbols, self.simulator)
        
        # Connection state
        self.is_connected = False
        self.connection_attempts = 0
        self.last_heartbeat = now_utc()
        
        # Data handlers
        self.data_handlers: List[Callable[[str, Dict[str, float]], None]] = []
    
    def add_data_handler(self, handler: Callable[[str, Dict[str, float]], None]) -> None:
        """Add handler for market data updates."""
        self.data_handlers.append(handler)
    
    async def connect(self) -> bool:
        """Connect to market data feed."""
        logger.info("Connecting to market data feed...")
        
        if self.use_simulation:
            # Setup simulation handlers
            self.stream.subscribe_to_ticks(self._handle_tick)
            
            # Start simulation stream
            asyncio.create_task(self.stream.start_stream())
            asyncio.create_task(self._heartbeat_monitor())
            
            self.is_connected = True
            logger.info("Connected to simulated market data feed")
            return True
        
        else:
            # Would implement real exchange connection here
            logger.warning("Real market data feed not implemented")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from market data feed."""
        logger.info("Disconnecting from market data feed")
        
        if self.use_simulation and hasattr(self, 'stream'):
            self.stream.stop_stream()
        
        self.is_connected = False
    
    def _handle_tick(self, tick: MarketTick) -> None:
        """Handle incoming market tick."""
        # Convert tick to standard format
        data = {
            'timestamp': tick.timestamp,
            'bid': tick.bid,
            'ask': tick.ask,
            'close': tick.last,
            'volume': tick.volume,
            'high_24h': tick.high_24h,
            'low_24h': tick.low_24h
        }
        
        # Notify handlers
        for handler in self.data_handlers:
            try:
                handler(tick.symbol, data)
            except Exception as e:
                logger.error(f"Error in data handler for {tick.symbol}: {e}")
        
        self.last_heartbeat = now_utc()
    
    async def _heartbeat_monitor(self) -> None:
        """Monitor connection health via heartbeats."""
        while self.is_connected:
            await asyncio.sleep(self.heartbeat_interval)
            
            time_since_heartbeat = (now_utc() - self.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.heartbeat_interval * 2:
                logger.warning(f"No heartbeat for {time_since_heartbeat:.1f}s, connection may be stale")
                
                # Attempt reconnection
                if self.connection_attempts < self.reconnect_attempts:
                    logger.info("Attempting to reconnect...")
                    self.connection_attempts += 1
                    # Would implement reconnection logic here
                else:
                    logger.error("Max reconnection attempts reached, disconnecting")
                    self.disconnect()
                    break
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all symbols."""
        if self.use_simulation and hasattr(self, 'stream'):
            latest_ticks = self.stream.get_latest_ticks()
            return {symbol: tick.last for symbol, tick in latest_ticks.items()}
        return {}
    
    def get_market_summary(self) -> Dict[str, Dict[str, float]]:
        """Get market summary for all symbols."""
        if self.use_simulation and hasattr(self, 'stream'):
            latest_ticks = self.stream.get_latest_ticks()
            return {
                symbol: {
                    'price': tick.last,
                    'bid': tick.bid,
                    'ask': tick.ask,
                    'high_24h': tick.high_24h,
                    'low_24h': tick.low_24h,
                    'volume_24h': getattr(tick, 'volume_24h', 0)
                }
                for symbol, tick in latest_ticks.items()
            }
        return {}


async def create_market_data_feed(
    symbols: List[str],
    historical_data: Optional[Dict[str, pd.DataFrame]] = None
) -> MarketDataFeed:
    """Create and connect market data feed."""
    # Initialize simulator with historical data if provided
    simulator = None
    if historical_data:
        simulator = MarketDataSimulator(symbols, historical_data)
    
    feed = MarketDataFeed(symbols, use_simulation=True)
    if simulator:
        feed.simulator = simulator
        feed.stream = WebSocketMarketDataStream(symbols, simulator)
    
    # Connect to feed
    connected = await feed.connect()
    if not connected:
        raise RuntimeError("Failed to connect to market data feed")
    
    return feed


# Convenience function for testing
async def run_market_data_test(symbols: List[str], duration_seconds: int = 60) -> None:
    """Run market data feed test."""
    def print_tick(symbol: str, data: Dict[str, float]) -> None:
        price = data['close']
        timestamp = data['timestamp']
        print(f"{timestamp}: {symbol} = ${price:.2f}")
    
    feed = await create_market_data_feed(symbols)
    feed.add_data_handler(print_tick)
    
    logger.info(f"Running market data test for {duration_seconds} seconds...")
    await asyncio.sleep(duration_seconds)
    
    feed.disconnect()
    logger.info("Market data test completed")