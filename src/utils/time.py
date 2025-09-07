"""Timezone-safe timestamp utilities."""

from datetime import datetime, timezone
from typing import Union, Optional
import pandas as pd
import pytz


def to_utc(dt: Union[datetime, pd.Timestamp, str]) -> pd.Timestamp:
    """Convert datetime to UTC timezone-aware timestamp."""
    if isinstance(dt, str):
        dt = pd.Timestamp(dt)
    elif isinstance(dt, datetime):
        dt = pd.Timestamp(dt)
    
    if dt.tz is None:
        # Assume UTC if no timezone
        dt = dt.tz_localize("UTC")
    else:
        # Convert to UTC
        dt = dt.tz_convert("UTC")
    
    return dt


def now_utc() -> pd.Timestamp:
    """Get current UTC timestamp."""
    return pd.Timestamp.now(tz="UTC")


def ensure_timezone(
    ts: Union[datetime, pd.Timestamp], 
    tz: Optional[Union[str, timezone]] = None
) -> pd.Timestamp:
    """Ensure timestamp has timezone, defaulting to UTC."""
    if isinstance(ts, datetime):
        ts = pd.Timestamp(ts)
    
    if ts.tz is None:
        if tz is None:
            tz = "UTC"
        ts = ts.tz_localize(tz)
    
    return ts


def align_to_timeframe(ts: pd.Timestamp, timeframe: str) -> pd.Timestamp:
    """Align timestamp to timeframe boundary."""
    if timeframe == "1m":
        return ts.floor("1min")
    elif timeframe == "5m":
        return ts.floor("5min")
    elif timeframe == "15m":
        return ts.floor("15min")
    elif timeframe == "1h":
        return ts.floor("1h")
    elif timeframe == "4h":
        return ts.floor("4h")
    elif timeframe == "1d":
        return ts.floor("1d")
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    """Convert timeframe string to pandas Timedelta."""
    if timeframe == "1m":
        return pd.Timedelta(minutes=1)
    elif timeframe == "5m":
        return pd.Timedelta(minutes=5)
    elif timeframe == "15m":
        return pd.Timedelta(minutes=15)
    elif timeframe == "1h":
        return pd.Timedelta(hours=1)
    elif timeframe == "4h":
        return pd.Timedelta(hours=4)
    elif timeframe == "1d":
        return pd.Timedelta(days=1)
    else:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def validate_no_future_data(df: pd.DataFrame, current_time: Optional[pd.Timestamp] = None) -> None:
    """Validate that dataframe contains no future timestamps."""
    if current_time is None:
        current_time = now_utc()
    
    current_time = to_utc(current_time)
    
    if df.index.max() > current_time:
        future_count = (df.index > current_time).sum()
        raise ValueError(
            f"Found {future_count} future timestamps. "
            f"Latest: {df.index.max()}, Current: {current_time}"
        )


def resample_ohlcv(
    df: pd.DataFrame, 
    from_timeframe: str, 
    to_timeframe: str,
    validate_alignment: bool = True
) -> pd.DataFrame:
    """Resample OHLCV data to different timeframe."""
    if validate_alignment:
        # Ensure the data is properly aligned to the source timeframe
        expected_freq = timeframe_to_timedelta(from_timeframe)
        actual_freq = df.index.to_series().diff().median()
        
        if abs(actual_freq - expected_freq) > pd.Timedelta(seconds=1):
            raise ValueError(
                f"Data frequency {actual_freq} doesn't match expected {expected_freq}"
            )
    
    # Convert timeframe to pandas frequency
    freq_map = {
        "1m": "1min",
        "5m": "5min", 
        "15m": "15min",
        "1h": "1h",
        "4h": "4h",
        "1d": "1d"
    }
    
    target_freq = freq_map.get(to_timeframe)
    if target_freq is None:
        raise ValueError(f"Unsupported target timeframe: {to_timeframe}")
    
    # Resample OHLCV data
    resampled = df.resample(target_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    return resampled


def get_trading_sessions(
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    session_tz: str = "UTC"
) -> pd.DatetimeIndex:
    """Get trading session timestamps for the given period."""
    start = to_utc(start_date)
    end = to_utc(end_date)
    
    # Crypto markets are 24/7, so every day is a trading day
    return pd.date_range(start=start.date(), end=end.date(), freq="D", tz=session_tz)


def is_market_hours(dt: pd.Timestamp, market: str = "crypto") -> bool:
    """Check if timestamp is during market hours."""
    if market == "crypto":
        # Crypto markets are 24/7
        return True
    else:
        raise ValueError(f"Unsupported market: {market}")


class TimeManager:
    """Centralized time management for backtesting and live trading."""
    
    def __init__(self, current_time: Optional[pd.Timestamp] = None):
        self._current_time = current_time
        self._is_live = current_time is None
    
    @property
    def current_time(self) -> pd.Timestamp:
        """Get current time (live or simulated)."""
        if self._is_live:
            return now_utc()
        return self._current_time
    
    def set_time(self, new_time: pd.Timestamp) -> None:
        """Set current time for backtesting."""
        if self._is_live:
            raise RuntimeError("Cannot set time in live mode")
        self._current_time = to_utc(new_time)
    
    def advance_time(self, delta: pd.Timedelta) -> None:
        """Advance time by delta for backtesting."""
        if self._is_live:
            raise RuntimeError("Cannot advance time in live mode")
        self._current_time += delta
    
    def is_live(self) -> bool:
        """Check if running in live mode."""
        return self._is_live