"""CCXT-based OHLCV data loader with retries and rate limiting."""

import asyncio
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import ccxt
from loguru import logger

from ..utils.config import get_config
from ..utils.time import to_utc, now_utc, timeframe_to_timedelta, validate_no_future_data
from ..utils.io import DataCache


class RateLimiter:
    """Rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 1200):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0
    
    async def wait(self) -> None:
        """Wait if necessary to respect rate limits."""
        now = time.time()
        elapsed = now - self.last_call
        
        if elapsed < self.min_interval:
            wait_time = self.min_interval - elapsed
            await asyncio.sleep(wait_time)
        
        self.last_call = time.time()


class ExchangeClient:
    """Exchange client with error handling and rate limiting."""
    
    def __init__(self, exchange_name: str = "binance", sandbox: bool = True):
        self.exchange_name = exchange_name
        self.sandbox = sandbox
        
        config = get_config()
        self.rate_limiter = RateLimiter(config.exchange.rate_limit)
        
        # Initialize exchange
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """Initialize CCXT exchange instance."""
        config = get_config()
        
        exchange_class = getattr(ccxt, self.exchange_name)
        
        params = {
            'sandbox': self.sandbox,
            'timeout': config.exchange.timeout * 1000,  # CCXT uses milliseconds
        }
        
        if config.exchange.api_key and config.exchange.secret:
            params.update({
                'apiKey': config.exchange.api_key,
                'secret': config.exchange.secret,
            })
        
        self.exchange = exchange_class(params)
        
        logger.info(f"Initialized {self.exchange_name} exchange (sandbox: {self.sandbox})")
    
    async def fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        retries: int = 3
    ) -> List[List]:
        """Fetch OHLCV data with retries."""
        for attempt in range(retries):
            try:
                await self.rate_limiter.wait()
                
                # Fetch data
                ohlcv = await asyncio.to_thread(
                    self.exchange.fetch_ohlcv,
                    symbol, timeframe, since, limit
                )
                
                logger.debug(f"Fetched {len(ohlcv)} candles for {symbol} {timeframe}")
                return ohlcv
                
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch {symbol} {timeframe} after {retries} attempts: {e}")
                    raise
                
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Fetch attempt {attempt + 1} failed for {symbol}: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        return []
    
    async def fetch_ticker(self, symbol: str, retries: int = 3) -> Dict:
        """Fetch current ticker data."""
        for attempt in range(retries):
            try:
                await self.rate_limiter.wait()
                ticker = await asyncio.to_thread(self.exchange.fetch_ticker, symbol)
                return ticker
                
            except Exception as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to fetch ticker for {symbol}: {e}")
                    raise
                
                wait_time = 2 ** attempt
                logger.warning(f"Ticker fetch attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s")
                await asyncio.sleep(wait_time)
        
        return {}
    
    def close(self) -> None:
        """Close exchange connection."""
        if hasattr(self.exchange, 'close'):
            self.exchange.close()


class OHLCVLoader:
    """OHLCV data loader with caching and validation."""
    
    def __init__(self, exchange_client: Optional[ExchangeClient] = None):
        self.exchange_client = exchange_client or ExchangeClient()
        self.cache = DataCache()
        
    def _ohlcv_to_dataframe(self, ohlcv_data: List[List], symbol: str) -> pd.DataFrame:
        """Convert OHLCV list to DataFrame."""
        if not ohlcv_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df = df.set_index('timestamp')
        
        # Ensure proper data types
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]
        
        return df
    
    def _get_cache_key(self, symbol: str, timeframe: str, start_date: str, end_date: str) -> str:
        """Generate cache key for OHLCV data."""
        return f"ohlcv_{symbol}_{timeframe}_{start_date}_{end_date}"
    
    async def load_historical_data(
        self, 
        symbol: str,
        timeframe: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Load historical OHLCV data."""
        start_ts = to_utc(start_date)
        end_ts = to_utc(end_date)
        
        # Check cache first
        cache_key = self._get_cache_key(symbol, timeframe, start_ts.date().isoformat(), end_ts.date().isoformat())
        
        if use_cache:
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded {len(cached_data)} rows from cache for {symbol} {timeframe}")
                return cached_data
        
        # Fetch from exchange
        logger.info(f"Fetching {symbol} {timeframe} data from {start_ts.date()} to {end_ts.date()}")
        
        all_data = []
        current_start = int(start_ts.timestamp() * 1000)
        end_ms = int(end_ts.timestamp() * 1000)
        
        # Calculate chunk size based on timeframe
        timeframe_ms = int(timeframe_to_timedelta(timeframe).total_seconds() * 1000)
        max_candles = 1000  # Most exchanges limit to 1000 candles per request
        chunk_duration = max_candles * timeframe_ms
        
        while current_start < end_ms:
            chunk_end = min(current_start + chunk_duration, end_ms)
            
            try:
                ohlcv = await self.exchange_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_start,
                    limit=max_candles
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # Update current_start to the timestamp after the last candle
                last_timestamp = ohlcv[-1][0]
                current_start = last_timestamp + timeframe_ms
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to fetch chunk starting at {current_start}: {e}")
                break
        
        # Convert to DataFrame
        df = self._ohlcv_to_dataframe(all_data, symbol)
        
        if df.empty:
            logger.warning(f"No data fetched for {symbol} {timeframe}")
            return df
        
        # Filter to exact date range
        df = df[(df.index >= start_ts) & (df.index <= end_ts)]
        
        # Validate data
        self._validate_ohlcv_data(df, symbol, timeframe)
        
        # Cache the result
        if use_cache:
            self.cache.set(cache_key, df)
        
        logger.info(f"Loaded {len(df)} rows for {symbol} {timeframe}")
        return df
    
    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """Validate OHLCV data for consistency."""
        if df.empty:
            return
        
        # Check for future data
        validate_no_future_data(df)
        
        # Check for missing values
        null_counts = df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values in {symbol} data: {null_counts[null_counts > 0].to_dict()}")
        
        # Check OHLC consistency
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            invalid_count = invalid_ohlc.sum()
            logger.warning(f"Found {invalid_count} invalid OHLC rows in {symbol} data")
            # Drop invalid rows
            df = df[~invalid_ohlc]
        
        # Check for gaps in data
        expected_freq = timeframe_to_timedelta(timeframe)
        actual_gaps = df.index.to_series().diff()[1:]  # Skip first NaT
        large_gaps = actual_gaps > expected_freq * 2
        
        if large_gaps.any():
            gap_count = large_gaps.sum()
            logger.warning(f"Found {gap_count} large gaps in {symbol} {timeframe} data")
    
    async def get_latest_data(
        self, 
        symbol: str, 
        timeframe: str, 
        lookback_periods: int = 100
    ) -> pd.DataFrame:
        """Get latest OHLCV data for real-time use."""
        end_time = now_utc()
        period_delta = timeframe_to_timedelta(timeframe)
        start_time = end_time - (period_delta * lookback_periods)
        
        return await self.load_historical_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_time,
            end_date=end_time,
            use_cache=False  # Don't cache real-time data
        )
    
    async def load_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols concurrently."""
        tasks = [
            self.load_historical_data(symbol, timeframe, start_date, end_date, use_cache)
            for symbol in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {symbol}: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result
        
        return data_dict
    
    def close(self) -> None:
        """Close the loader and exchange client."""
        self.exchange_client.close()


# Convenience function for simple use cases
async def load_crypto_data(
    symbols: List[str],
    timeframe: str = "5m",
    days_back: int = 30,
    exchange: str = "binance"
) -> Dict[str, pd.DataFrame]:
    """Load crypto data for given symbols and timeframe."""
    end_date = now_utc()
    start_date = end_date - pd.Timedelta(days=days_back)
    
    client = ExchangeClient(exchange, sandbox=True)
    loader = OHLCVLoader(client)
    
    try:
        data = await loader.load_multiple_symbols(
            symbols=symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        return data
    finally:
        loader.close()