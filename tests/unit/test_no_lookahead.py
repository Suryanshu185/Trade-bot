"""Test no lookahead bias in features and signals."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from src.data.features import FeatureEngine, add_technical_features
from src.models.signals import MomentumBreakoutSignals, MeanReversionSignals
from src.models.labeling import TripleBarrierLabeler
from src.utils.time import validate_no_future_data, now_utc


class TestNoLookaheadBias:
    """Test that no future information is used in features and signals."""
    
    def test_features_no_future_data(self, sample_ohlcv_data):
        """Test that features don't use future data."""
        # Create features
        feature_engine = FeatureEngine(validate_no_lookahead=True)
        
        # This should not raise an exception
        features = feature_engine.compute_all_features(sample_ohlcv_data)
        
        # Check that no features use future data
        # All features at time t should only use data up to time t
        for i in range(10, len(features)):  # Skip first few rows (may have NaN)
            current_time = features.index[i]
            
            # Get data up to current time
            historical_data = sample_ohlcv_data.loc[:current_time]
            
            # Recompute features with only historical data
            historical_features = feature_engine.compute_all_features(historical_data)
            
            if len(historical_features) > 0:
                # Features should match (allowing for small numerical differences)
                for col in features.columns:
                    if col in historical_features.columns and current_time in historical_features.index:
                        current_value = features.loc[current_time, col]
                        historical_value = historical_features.loc[current_time, col]
                        
                        if pd.notna(current_value) and pd.notna(historical_value):
                            assert abs(current_value - historical_value) < 1e-10, \
                                f"Feature {col} uses future data at {current_time}"
    
    def test_rsi_no_lookahead(self, sample_ohlcv_data):
        """Test RSI calculation doesn't use future data."""
        from src.data.features import rsi
        
        prices = sample_ohlcv_data['close']
        
        # Calculate RSI for full series
        full_rsi = rsi(prices, window=14)
        
        # Calculate RSI incrementally and compare
        for i in range(20, len(prices), 10):  # Test every 10th point
            partial_prices = prices.iloc[:i+1]
            partial_rsi = rsi(partial_prices, window=14)
            
            if len(partial_rsi) > 0:
                current_time = prices.index[i]
                if current_time in partial_rsi.index and current_time in full_rsi.index:
                    partial_value = partial_rsi.loc[current_time]
                    full_value = full_rsi.loc[current_time]
                    
                    if pd.notna(partial_value) and pd.notna(full_value):
                        assert abs(partial_value - full_value) < 1e-10, \
                            f"RSI uses future data at {current_time}"
    
    def test_moving_averages_no_lookahead(self, sample_ohlcv_data):
        """Test moving averages don't use future data."""
        from src.data.features import sma, ema
        
        prices = sample_ohlcv_data['close']
        
        # Test SMA
        full_sma = sma(prices, window=20)
        
        for i in range(25, len(prices), 15):
            partial_prices = prices.iloc[:i+1]
            partial_sma = sma(partial_prices, window=20)
            
            current_time = prices.index[i]
            if current_time in partial_sma.index and current_time in full_sma.index:
                partial_value = partial_sma.loc[current_time]
                full_value = full_sma.loc[current_time]
                
                if pd.notna(partial_value) and pd.notna(full_value):
                    assert abs(partial_value - full_value) < 1e-10, \
                        f"SMA uses future data at {current_time}"
        
        # Test EMA
        full_ema = ema(prices, window=20)
        
        for i in range(25, len(prices), 15):
            partial_prices = prices.iloc[:i+1]
            partial_ema = ema(partial_prices, window=20)
            
            current_time = prices.index[i]
            if current_time in partial_ema.index and current_time in full_ema.index:
                partial_value = partial_ema.loc[current_time]
                full_value = full_ema.loc[current_time]
                
                if pd.notna(partial_value) and pd.notna(full_value):
                    assert abs(partial_value - full_value) < 1e-10, \
                        f"EMA uses future data at {current_time}"
    
    def test_signals_no_lookahead(self, sample_ohlcv_data):
        """Test that signals don't use future data."""
        # Add features first
        data_with_features = add_technical_features(sample_ohlcv_data)
        
        # Test momentum signals
        momentum_generator = MomentumBreakoutSignals()
        
        # Generate signals for full dataset
        full_signals = momentum_generator.generate_signals(data_with_features)
        
        # Test incrementally
        for i in range(50, len(data_with_features), 20):
            partial_data = data_with_features.iloc[:i+1]
            partial_signals = momentum_generator.generate_signals(partial_data)
            
            current_time = data_with_features.index[i]
            if current_time in partial_signals.index and current_time in full_signals.index:
                partial_value = partial_signals.loc[current_time]
                full_value = full_signals.loc[current_time]
                
                assert partial_value == full_value, \
                    f"Momentum signal uses future data at {current_time}"
    
    def test_labels_no_future_leakage(self, sample_ohlcv_data):
        """Test that labeling doesn't leak future information inappropriately."""
        prices = sample_ohlcv_data['close']
        
        # Create labels
        labeler = TripleBarrierLabeler(max_holding_days=2)
        labels, events = labeler.create_labels(prices)
        
        # Check that labels are properly aligned with events
        for event_time in events.index:
            if event_time in labels.index:
                # Label should only use information available at event time
                # The actual label value depends on future prices, which is correct
                # But the decision to create the event should not use future info
                
                # Check that event parameters don't use future information
                event_data = events.loc[event_time]
                
                # t1 (barrier time) should be in the future
                assert event_data['t1'] > event_time, \
                    f"Barrier time {event_data['t1']} not in future of event time {event_time}"
                
                # Target should be based on information available at event time
                historical_prices = prices.loc[:event_time]
                assert len(historical_prices) > 0, "No historical data for event"
    
    def test_time_validation_catches_future_data(self):
        """Test that validation catches future data."""
        current_time = now_utc()
        
        # Create DataFrame with future data
        future_time = current_time + pd.Timedelta(hours=1)
        df_with_future = pd.DataFrame(
            index=[current_time, future_time],
            data={'value': [1, 2]}
        )
        
        # Should raise exception
        with pytest.raises(ValueError, match="future timestamps"):
            validate_no_future_data(df_with_future, current_time)
        
        # DataFrame without future data should pass
        df_no_future = pd.DataFrame(
            index=[current_time - pd.Timedelta(hours=1), current_time],
            data={'value': [1, 2]}
        )
        
        # Should not raise exception
        validate_no_future_data(df_no_future, current_time)
    
    @patch('src.utils.time.now_utc')
    def test_features_respect_current_time(self, mock_now, sample_ohlcv_data):
        """Test that features respect the current time when validating."""
        # Set mock current time to middle of data
        mid_point = sample_ohlcv_data.index[len(sample_ohlcv_data) // 2]
        mock_now.return_value = mid_point
        
        # Create feature engine with validation
        feature_engine = FeatureEngine(validate_no_lookahead=True)
        
        # Using all data should raise exception due to future data
        with pytest.raises(ValueError, match="future timestamps"):
            feature_engine.compute_all_features(sample_ohlcv_data)
        
        # Using only data up to current time should work
        historical_data = sample_ohlcv_data.loc[:mid_point]
        features = feature_engine.compute_all_features(historical_data)
        
        assert len(features) > 0
        assert features.index.max() <= mid_point