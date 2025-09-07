"""Technical indicators and feature engineering with no lookahead bias."""

from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger

from ..utils.time import validate_no_future_data


def validate_input(df: pd.DataFrame, required_cols: list) -> None:
    """Validate input DataFrame has required columns."""
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def safe_divide(numerator: pd.Series, denominator: pd.Series, fill_value: float = 0.0) -> pd.Series:
    """Safely divide two series, handling division by zero."""
    return numerator.div(denominator).fillna(fill_value).replace([np.inf, -np.inf], fill_value)


def returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate returns with proper handling of edge cases."""
    if len(prices) < periods + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    return prices.pct_change(periods=periods)


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Calculate log returns."""
    if len(prices) < periods + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    return np.log(prices / prices.shift(periods))


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range."""
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    return np.maximum(tr1, np.maximum(tr2, tr3))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    if len(high) < window:
        return pd.Series(index=high.index, dtype=float)
    
    tr = true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        return pd.Series(index=prices.index, dtype=float)
    
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    
    rs = safe_divide(avg_gain, avg_loss, fill_value=0)
    rsi_values = 100 - (100 / (1 + rs))
    
    return rsi_values


def ema(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    if len(prices) < window:
        return pd.Series(index=prices.index, dtype=float)
    
    return prices.ewm(span=window, adjust=False).mean()


def sma(prices: pd.Series, window: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    if len(prices) < window:
        return pd.Series(index=prices.index, dtype=float)
    
    return prices.rolling(window=window, min_periods=window).mean()


def macd(
    prices: pd.Series, 
    fast_window: int = 12, 
    slow_window: int = 26, 
    signal_window: int = 9
) -> pd.DataFrame:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    if len(prices) < slow_window:
        return pd.DataFrame(
            index=prices.index,
            columns=['macd', 'signal', 'histogram']
        )
    
    ema_fast = ema(prices, fast_window)
    ema_slow = ema(prices, slow_window)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })


def bollinger_bands(
    prices: pd.Series, 
    window: int = 20, 
    num_std: float = 2.0
) -> pd.DataFrame:
    """Calculate Bollinger Bands."""
    if len(prices) < window:
        return pd.DataFrame(
            index=prices.index,
            columns=['middle', 'upper', 'lower', 'pct_b']
        )
    
    middle = sma(prices, window)
    std = prices.rolling(window=window, min_periods=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    # Percent B: position within the bands
    pct_b = safe_divide(prices - lower, upper - lower, fill_value=0.5)
    
    return pd.DataFrame({
        'middle': middle,
        'upper': upper,
        'lower': lower,
        'pct_b': pct_b
    })


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Calculate rolling z-score."""
    if len(series) < window:
        return pd.Series(index=series.index, dtype=float)
    
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    
    return safe_divide(series - rolling_mean, rolling_std, fill_value=0)


def volatility(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """Calculate rolling volatility."""
    if len(returns) < window:
        return pd.Series(index=returns.index, dtype=float)
    
    vol = returns.rolling(window=window, min_periods=window).std()
    
    if annualize:
        # Assume daily frequency, adjust for other frequencies as needed
        vol = vol * np.sqrt(252)
    
    return vol


def momentum(prices: pd.Series, window: int) -> pd.Series:
    """Calculate price momentum."""
    if len(prices) < window:
        return pd.Series(index=prices.index, dtype=float)
    
    return prices / prices.shift(window) - 1


def volume_weighted_average_price(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    volume: pd.Series,
    window: int
) -> pd.Series:
    """Calculate Volume Weighted Average Price (VWAP)."""
    if len(high) < window:
        return pd.Series(index=high.index, dtype=float)
    
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    
    return vwap


def donchian_channel(
    high: pd.Series, 
    low: pd.Series, 
    window: int = 20
) -> pd.DataFrame:
    """Calculate Donchian Channel."""
    if len(high) < window:
        return pd.DataFrame(
            index=high.index,
            columns=['upper', 'lower', 'middle']
        )
    
    upper = high.rolling(window=window, min_periods=window).max()
    lower = low.rolling(window=window, min_periods=window).min()
    middle = (upper + lower) / 2
    
    return pd.DataFrame({
        'upper': upper,
        'lower': lower,
        'middle': middle
    })


def stochastic(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3
) -> pd.DataFrame:
    """Calculate Stochastic Oscillator."""
    if len(high) < k_window:
        return pd.DataFrame(
            index=high.index,
            columns=['%K', '%D']
        )
    
    lowest_low = low.rolling(window=k_window, min_periods=k_window).min()
    highest_high = high.rolling(window=k_window, min_periods=k_window).max()
    
    k_percent = safe_divide(close - lowest_low, highest_high - lowest_low, fill_value=0.5) * 100
    d_percent = k_percent.rolling(window=d_window, min_periods=d_window).mean()
    
    return pd.DataFrame({
        '%K': k_percent,
        '%D': d_percent
    })


class FeatureEngine:
    """Feature engineering pipeline with no lookahead bias."""
    
    def __init__(self, validate_no_lookahead: bool = True):
        self.validate_no_lookahead = validate_no_lookahead
    
    def compute_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute price-based features."""
        validate_input(df, ['open', 'high', 'low', 'close'])
        
        features = df.copy()
        
        # Returns
        features['returns_1'] = returns(df['close'], 1)
        features['returns_5'] = returns(df['close'], 5)
        features['returns_20'] = returns(df['close'], 20)
        
        # Log returns
        features['log_returns_1'] = log_returns(df['close'], 1)
        features['log_returns_5'] = log_returns(df['close'], 5)
        
        # Momentum
        features['momentum_10'] = momentum(df['close'], 10)
        features['momentum_20'] = momentum(df['close'], 20)
        features['momentum_50'] = momentum(df['close'], 50)
        
        # Moving averages
        features['sma_10'] = sma(df['close'], 10)
        features['sma_20'] = sma(df['close'], 20)
        features['sma_50'] = sma(df['close'], 50)
        features['ema_10'] = ema(df['close'], 10)
        features['ema_20'] = ema(df['close'], 20)
        
        # Price relative to moving averages
        features['price_vs_sma20'] = safe_divide(df['close'], features['sma_20'], 1.0) - 1
        features['price_vs_sma50'] = safe_divide(df['close'], features['sma_50'], 1.0) - 1
        
        # Moving average slopes (trend)
        features['sma20_slope'] = features['sma_20'].pct_change(5)
        features['sma50_slope'] = features['sma_50'].pct_change(10)
        
        return features
    
    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volatility-based features."""
        validate_input(df, ['high', 'low', 'close'])
        
        features = df.copy()
        
        # ATR
        features['atr_14'] = atr(df['high'], df['low'], df['close'], 14)
        features['atr_20'] = atr(df['high'], df['low'], df['close'], 20)
        
        # Volatility
        returns_series = returns(df['close'], 1)
        features['volatility_10'] = volatility(returns_series, 10)
        features['volatility_20'] = volatility(returns_series, 20)
        features['volatility_50'] = volatility(returns_series, 50)
        
        # Volatility ratio (current vs historical)
        features['vol_ratio_20_50'] = safe_divide(features['volatility_20'], features['volatility_50'], 1.0)
        
        # Range features
        features['high_low_pct'] = safe_divide(df['high'] - df['low'], df['close'], 0.0)
        features['close_vs_range'] = safe_divide(df['close'] - df['low'], df['high'] - df['low'], 0.5)
        
        return features
    
    def compute_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators."""
        validate_input(df, ['open', 'high', 'low', 'close'])
        
        features = df.copy()
        
        # RSI
        features['rsi_14'] = rsi(df['close'], 14)
        features['rsi_2'] = rsi(df['close'], 2)  # Short-term RSI for mean reversion
        
        # MACD
        macd_data = macd(df['close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        
        # Bollinger Bands
        bb_data = bollinger_bands(df['close'], 20)
        features['bb_middle'] = bb_data['middle']
        features['bb_upper'] = bb_data['upper']
        features['bb_lower'] = bb_data['lower']
        features['bb_pct_b'] = bb_data['pct_b']
        
        # Donchian Channel
        donchian_data = donchian_channel(df['high'], df['low'], 20)
        features['donchian_upper'] = donchian_data['upper']
        features['donchian_lower'] = donchian_data['lower']
        features['donchian_middle'] = donchian_data['middle']
        
        # Stochastic
        if len(df) >= 14:
            stoch_data = stochastic(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch_data['%K']
            features['stoch_d'] = stoch_data['%D']
        
        return features
    
    def compute_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume-based features."""
        validate_input(df, ['high', 'low', 'close', 'volume'])
        
        features = df.copy()
        
        # Volume moving averages
        features['volume_sma_10'] = sma(df['volume'], 10)
        features['volume_sma_20'] = sma(df['volume'], 20)
        
        # Volume ratio
        features['volume_ratio'] = safe_divide(df['volume'], features['volume_sma_20'], 1.0)
        
        # VWAP
        features['vwap_20'] = volume_weighted_average_price(
            df['high'], df['low'], df['close'], df['volume'], 20
        )
        features['price_vs_vwap'] = safe_divide(df['close'], features['vwap_20'], 1.0) - 1
        
        return features
    
    def compute_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling z-score features."""
        validate_input(df, ['close'])
        
        features = df.copy()
        
        # Price z-scores
        features['price_zscore_20'] = rolling_zscore(df['close'], 20)
        features['price_zscore_50'] = rolling_zscore(df['close'], 50)
        
        # Returns z-scores
        returns_series = returns(df['close'], 1)
        features['returns_zscore_20'] = rolling_zscore(returns_series, 20)
        features['returns_zscore_50'] = rolling_zscore(returns_series, 50)
        
        # Volume z-scores (if volume available)
        if 'volume' in df.columns:
            features['volume_zscore_20'] = rolling_zscore(df['volume'], 20)
        
        return features
    
    def compute_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features in one go."""
        logger.info(f"Computing features for {len(df)} rows")
        
        # Validate no future data
        if self.validate_no_lookahead:
            validate_no_future_data(df)
        
        # Start with original data
        features = df.copy()
        
        # Compute feature groups
        features = self.compute_price_features(features)
        features = self.compute_volatility_features(features)
        features = self.compute_technical_indicators(features)
        
        if 'volume' in df.columns:
            features = self.compute_volume_features(features)
        
        features = self.compute_zscore_features(features)
        
        # Drop any rows with all NaN features (usually at the beginning)
        feature_cols = [col for col in features.columns if col not in df.columns]
        features = features.dropna(subset=feature_cols, how='all')
        
        logger.info(f"Generated {len(feature_cols)} features, {len(features)} valid rows")
        
        return features


# Convenience function
def add_technical_features(df: pd.DataFrame, validate_no_lookahead: bool = True) -> pd.DataFrame:
    """Add technical features to OHLCV data."""
    engine = FeatureEngine(validate_no_lookahead=validate_no_lookahead)
    return engine.compute_all_features(df)