"""Trading signal generation with rule-based and ML strategies."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from loguru import logger

from ..data.features import FeatureEngine
from ..utils.config import get_config
from ..utils.time import validate_no_future_data


class BaseSignalGenerator(ABC):
    """Base class for signal generators."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from data."""
        pass
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the signal generator (default: no-op for rule-based)."""
        self.is_fitted = True
    
    def validate_data(self, data: pd.DataFrame, required_features: List[str]) -> None:
        """Validate input data has required features."""
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")


class MomentumBreakoutSignals(BaseSignalGenerator):
    """Momentum breakout strategy signals."""
    
    def __init__(
        self,
        lookback_high_low: int = 20,
        volatility_filter_window: int = 20,
        trend_filter_window: int = 50,
        min_compression_ratio: float = 0.8,
        min_trend_strength: float = 0.02
    ):
        super().__init__("momentum_breakout")
        self.lookback_high_low = lookback_high_low
        self.volatility_filter_window = volatility_filter_window
        self.trend_filter_window = trend_filter_window
        self.min_compression_ratio = min_compression_ratio
        self.min_trend_strength = min_trend_strength
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate momentum breakout signals."""
        required_features = [
            'high', 'low', 'close', 'atr_14', 'sma_50', 'volatility_20'
        ]
        self.validate_data(data, required_features)
        
        signals = pd.Series(0, index=data.index, name='momentum_signal')
        
        # Calculate rolling highs and lows
        rolling_high = data['high'].rolling(window=self.lookback_high_low).max()
        rolling_low = data['low'].rolling(window=self.lookback_high_low).min()
        
        # Volatility compression filter
        current_vol = data['volatility_20']
        avg_vol = current_vol.rolling(window=self.volatility_filter_window).mean()
        vol_compression = current_vol / avg_vol
        
        # Trend filter using moving average slope
        ma_slope = data['sma_50'].pct_change(10)  # 10-period slope
        
        # Breakout conditions
        upward_breakout = (
            (data['close'] > rolling_high.shift(1)) &  # Break above recent high
            (vol_compression < self.min_compression_ratio) &  # Low volatility period
            (ma_slope > self.min_trend_strength)  # Upward trend
        )
        
        downward_breakout = (
            (data['close'] < rolling_low.shift(1)) &  # Break below recent low
            (vol_compression < self.min_compression_ratio) &  # Low volatility period
            (ma_slope < -self.min_trend_strength)  # Downward trend
        )
        
        # Generate signals
        signals.loc[upward_breakout] = 1   # Long signal
        signals.loc[downward_breakout] = -1  # Short signal
        
        return signals


class MeanReversionSignals(BaseSignalGenerator):
    """Mean reversion strategy signals."""
    
    def __init__(
        self,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        rsi_short_oversold: float = 10,
        rsi_short_overbought: float = 90,
        zscore_threshold: float = 2.0,
        volatility_filter_percentile: float = 50
    ):
        super().__init__("mean_reversion")
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.rsi_short_oversold = rsi_short_oversold
        self.rsi_short_overbought = rsi_short_overbought
        self.zscore_threshold = zscore_threshold
        self.volatility_filter_percentile = volatility_filter_percentile
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate mean reversion signals."""
        required_features = [
            'close', 'rsi_14', 'rsi_2', 'returns_zscore_20', 'volatility_20'
        ]
        self.validate_data(data, required_features)
        
        signals = pd.Series(0, index=data.index, name='mean_reversion_signal')
        
        # Low volatility filter (mean reversion works better in low vol)
        vol_threshold = data['volatility_20'].rolling(window=252).quantile(
            self.volatility_filter_percentile / 100
        )
        low_vol_filter = data['volatility_20'] < vol_threshold
        
        # RSI-based signals
        rsi_long = (
            (data['rsi_14'] < self.rsi_oversold) |
            (data['rsi_2'] < self.rsi_short_oversold)
        )
        
        rsi_short = (
            (data['rsi_14'] > self.rsi_overbought) |
            (data['rsi_2'] > self.rsi_short_overbought)
        )
        
        # Z-score based signals
        zscore_long = data['returns_zscore_20'] < -self.zscore_threshold
        zscore_short = data['returns_zscore_20'] > self.zscore_threshold
        
        # Combined signals with volatility filter
        long_signal = (rsi_long | zscore_long) & low_vol_filter
        short_signal = (rsi_short | zscore_short) & low_vol_filter
        
        signals.loc[long_signal] = 1
        signals.loc[short_signal] = -1
        
        return signals


class MLSignalGenerator(BaseSignalGenerator):
    """Machine learning-based signal generator."""
    
    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        probability_threshold: float = 0.6,
        lookback_periods: int = 100,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        test_size: float = 0.2
    ):
        super().__init__("ml_signals")
        self.feature_columns = feature_columns
        self.probability_threshold = probability_threshold
        self.lookback_periods = lookback_periods
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.test_size = test_size
        
        self.model = None
        self.scaler = None
        self.feature_importance_ = None
    
    def _get_feature_columns(self, data: pd.DataFrame) -> List[str]:
        """Get feature columns for ML model."""
        if self.feature_columns is not None:
            return self.feature_columns
        
        # Default feature selection
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        # Remove any columns with too many NaN values
        valid_cols = []
        for col in feature_cols:
            if data[col].notna().sum() / len(data) > 0.8:  # At least 80% valid data
                valid_cols.append(col)
        
        return valid_cols
    
    def _create_labels(self, prices: pd.Series, forward_window: int = 5) -> pd.Series:
        """Create forward-looking labels for classification."""
        # Calculate forward returns
        forward_returns = prices.pct_change(forward_window).shift(-forward_window)
        
        # Create binary labels based on returns
        labels = pd.Series(0, index=prices.index)
        labels[forward_returns > 0.001] = 1  # Up if return > 0.1%
        labels[forward_returns < -0.001] = -1  # Down if return < -0.1%
        
        return labels
    
    def fit(self, data: pd.DataFrame) -> None:
        """Train the ML model on historical data."""
        logger.info(f"Training ML model on {len(data)} samples")
        
        # Get features
        feature_cols = self._get_feature_columns(data)
        if not feature_cols:
            raise ValueError("No valid feature columns found")
        
        # Prepare features and labels
        X = data[feature_cols].copy()
        y = self._create_labels(data['close'])
        
        # Remove rows with NaN values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < self.lookback_periods:
            raise ValueError(f"Insufficient data: {len(X)} < {self.lookback_periods}")
        
        # Remove neutral labels for training (focus on clear directional moves)
        directional_mask = y != 0
        X = X[directional_mask]
        y = y[directional_mask]
        
        # Convert to binary classification (1 for up, 0 for down)
        y_binary = (y > 0).astype(int)
        
        # Split data using time series split
        tscv = TimeSeriesSplit(n_splits=3, test_size=int(len(X) * self.test_size))
        
        best_score = 0
        best_model = None
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42
            )
            
            # Calibrate probabilities
            calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            val_score = calibrated_model.score(X_val_scaled, y_val)
            
            if val_score > best_score:
                best_score = val_score
                best_model = calibrated_model
                self.scaler = scaler
        
        self.model = best_model
        self.feature_columns = feature_cols
        self.feature_importance_ = dict(zip(
            feature_cols, 
            best_model.base_estimator.feature_importances_
        ))
        
        self.is_fitted = True
        logger.info(f"ML model trained. Best validation score: {best_score:.3f}")
        
        # Log feature importance
        importance_df = pd.Series(self.feature_importance_).sort_values(ascending=False)
        logger.info(f"Top 10 features:\n{importance_df.head(10)}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate ML-based signals."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        signals = pd.Series(0, index=data.index, name='ml_signal')
        
        # Prepare features
        X = data[self.feature_columns].copy()
        
        # Remove rows with NaN values
        valid_mask = X.notna().all(axis=1)
        
        if not valid_mask.any():
            logger.warning("No valid data for ML predictions")
            return signals
        
        X_valid = X[valid_mask]
        
        # Scale features
        X_scaled = self.scaler.transform(X_valid)
        
        # Get predictions and probabilities
        probabilities = self.model.predict_proba(X_scaled)
        
        # Extract probabilities for up movement (class 1)
        prob_up = probabilities[:, 1]
        prob_down = probabilities[:, 0]
        
        # Generate signals based on probability threshold
        long_signals = prob_up > self.probability_threshold
        short_signals = prob_down > self.probability_threshold
        
        # Apply signals to valid indices
        valid_indices = X.index[valid_mask]
        signals.loc[valid_indices[long_signals]] = 1
        signals.loc[valid_indices[short_signals]] = -1
        
        return signals
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from trained model."""
        return self.feature_importance_


class CompositeSignalGenerator:
    """Combine multiple signal generators with weights."""
    
    def __init__(
        self,
        generators: Dict[str, BaseSignalGenerator],
        weights: Optional[Dict[str, float]] = None
    ):
        self.generators = generators
        self.weights = weights or {name: 1.0 for name in generators.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit all generators that need training."""
        for name, generator in self.generators.items():
            logger.info(f"Fitting {name} generator")
            generator.fit(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate composite signals from all generators."""
        all_signals = {}
        
        for name, generator in self.generators.items():
            try:
                signals = generator.generate_signals(data)
                all_signals[name] = signals
            except Exception as e:
                logger.error(f"Failed to generate signals for {name}: {e}")
                all_signals[name] = pd.Series(0, index=data.index)
        
        # Create DataFrame with all signals
        signals_df = pd.DataFrame(all_signals)
        
        # Calculate weighted composite signal
        composite_signal = sum(
            signals_df[name] * weight 
            for name, weight in self.weights.items()
        )
        
        # Threshold composite signal
        signals_df['composite'] = 0
        signals_df.loc[composite_signal > 0.5, 'composite'] = 1
        signals_df.loc[composite_signal < -0.5, 'composite'] = -1
        
        return signals_df


def create_default_signal_generators() -> CompositeSignalGenerator:
    """Create default set of signal generators."""
    config = get_config()
    
    # Create individual generators
    momentum = MomentumBreakoutSignals(
        lookback_high_low=config.trading.momentum_lookback
    )
    
    mean_reversion = MeanReversionSignals()
    
    ml_generator = MLSignalGenerator(
        probability_threshold=config.trading.ml_probability_threshold,
        lookback_periods=config.trading.ml_lookback_periods
    )
    
    # Combine with equal weights
    generators = {
        'momentum': momentum,
        'mean_reversion': mean_reversion,
        'ml': ml_generator
    }
    
    weights = {
        'momentum': 0.4,
        'mean_reversion': 0.3,
        'ml': 0.3
    }
    
    return CompositeSignalGenerator(generators, weights)