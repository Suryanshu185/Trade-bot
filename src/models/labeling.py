"""Triple-barrier labeling with volatility-based horizons."""

from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from loguru import logger

from ..utils.time import validate_no_future_data


def get_daily_vol(prices: pd.Series, lookback: int = 100) -> pd.Series:
    """Calculate daily volatility for barrier sizing."""
    returns = prices.pct_change().dropna()
    
    # Use exponentially weighted standard deviation
    vol = returns.ewm(span=lookback).std()
    
    # Annualize assuming 252 trading days
    daily_vol = vol * np.sqrt(252)
    
    return daily_vol


def apply_triple_barrier(
    prices: pd.Series,
    events: pd.DataFrame,
    pt_sl: Optional[pd.Series] = None,
    molecule: Optional[list] = None
) -> pd.DataFrame:
    """
    Apply triple-barrier method to get labels.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series (close prices)
    events : pd.DataFrame
        DataFrame with columns:
        - t1: timestamp for vertical barrier (max holding period)
        - trgt: target (profit taking threshold)
        - side: side of the bet (1 for long, -1 for short, 0 for both)
    pt_sl : pd.Series, optional
        Profit taking and stop loss thresholds as multiple of volatility
    molecule : list, optional
        Subset of events to process
        
    Returns:
    --------
    pd.DataFrame with columns:
    - t1: timestamp when barrier was hit
    - ret: return achieved at barrier
    - bin: binary label (-1, 0, 1)
    """
    if molecule is None:
        molecule = events.index
    
    # Validate events DataFrame
    required_cols = ['t1', 'trgt', 'side']
    missing_cols = [col for col in required_cols if col not in events.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in events: {missing_cols}")
    
    events_ = events.loc[molecule]
    out = pd.DataFrame(index=events_.index)
    
    for event_idx, event in events_.iterrows():
        # Get price path from event start to barrier end
        start_time = event_idx
        end_time = event['t1']
        
        # Get price subset
        price_path = prices.loc[start_time:end_time]
        
        if len(price_path) < 2:
            # Not enough data
            out.loc[event_idx, ['t1', 'ret', 'bin']] = [end_time, 0, 0]
            continue
        
        # Calculate returns path
        p0 = price_path.iloc[0]
        returns_path = price_path / p0 - 1
        
        # Apply side (position direction)
        if event['side'] != 0:
            returns_path *= event['side']
        
        # Get barriers
        target = event['trgt']
        
        # Profit taking barrier (positive return)
        pt_barrier = target
        
        # Stop loss barrier (negative return)
        if pt_sl is not None and event_idx in pt_sl.index:
            sl_barrier = -pt_sl.loc[event_idx]
        else:
            sl_barrier = -target  # Symmetric stop loss
        
        # Find first barrier hit
        hit_pt = returns_path >= pt_barrier
        hit_sl = returns_path <= sl_barrier
        
        hit_any = hit_pt | hit_sl
        
        if hit_any.any():
            # Barrier was hit
            hit_time = hit_any.idxmax()  # First True value
            hit_return = returns_path.loc[hit_time]
            
            # Determine label
            if hit_pt.loc[hit_time]:
                label = 1  # Profit target hit
            elif hit_sl.loc[hit_time]:
                label = -1  # Stop loss hit
            else:
                label = 0  # Shouldn't happen, but safe fallback
        else:
            # No barrier hit, use end return
            hit_time = end_time
            hit_return = returns_path.iloc[-1]
            
            # Label based on final return
            if hit_return >= pt_barrier:
                label = 1
            elif hit_return <= sl_barrier:
                label = -1
            else:
                label = 0  # Neutral
        
        out.loc[event_idx, ['t1', 'ret', 'bin']] = [hit_time, hit_return, label]
    
    return out


def get_events(
    prices: pd.Series,
    volatility: Optional[pd.Series] = None,
    max_holding_days: int = 5,
    min_return_threshold: float = 0.001,
    side_prediction: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Get events for triple-barrier labeling.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    volatility : pd.Series, optional
        Volatility estimates for dynamic barriers
    max_holding_days : int
        Maximum holding period in days
    min_return_threshold : float
        Minimum return threshold for event generation
    side_prediction : pd.Series, optional
        Predicted side for each timestamp (1, -1, or 0)
        
    Returns:
    --------
    pd.DataFrame with events
    """
    events = pd.DataFrame(index=prices.index)
    
    # Calculate volatility if not provided
    if volatility is None:
        volatility = get_daily_vol(prices)
    
    # Align volatility with prices
    volatility = volatility.reindex(prices.index, method='ffill')
    
    # Set vertical barrier (max holding time)
    # For intraday data, convert days to appropriate frequency
    if isinstance(prices.index, pd.DatetimeIndex):
        freq = pd.infer_freq(prices.index)
        if freq:
            if 'min' in freq.lower():
                # Minutes data
                periods_per_day = 24 * 60 // int(freq.replace('min', ''))
                max_periods = max_holding_days * periods_per_day
            elif 'h' in freq.lower():
                # Hourly data
                max_periods = max_holding_days * 24
            else:
                # Daily or other
                max_periods = max_holding_days
        else:
            max_periods = max_holding_days
    else:
        max_periods = max_holding_days
    
    # Set t1 (vertical barrier)
    events['t1'] = prices.index.to_series().shift(-max_periods)
    
    # Handle end-of-series cases
    events['t1'] = events['t1'].fillna(prices.index[-1])
    
    # Set target threshold based on volatility
    events['trgt'] = volatility * 2.0  # 2 sigma moves
    
    # Ensure minimum threshold
    events['trgt'] = np.maximum(events['trgt'], min_return_threshold)
    
    # Set side (position direction)
    if side_prediction is not None:
        side_aligned = side_prediction.reindex(prices.index, method='ffill')
        events['side'] = side_aligned.fillna(0)
    else:
        events['side'] = 0  # Both sides
    
    # Remove events too close to end
    valid_events = events['t1'] > events.index
    events = events[valid_events]
    
    # Remove rows with NaN values
    events = events.dropna()
    
    return events


def get_bins(
    prices: pd.Series,
    events: pd.DataFrame,
    side_prediction: Optional[pd.Series] = None
) -> pd.Series:
    """
    Generate binary labels using triple-barrier method.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
    events : pd.DataFrame
        Events DataFrame from get_events()
    side_prediction : pd.Series, optional
        Side predictions for meta-labeling
        
    Returns:
    --------
    pd.Series with binary labels
    """
    # Apply triple barrier
    labels_df = apply_triple_barrier(prices, events)
    
    # Extract binary labels
    labels = labels_df['bin']
    
    # Meta-labeling: filter by side prediction
    if side_prediction is not None:
        side_aligned = side_prediction.reindex(labels.index, method='ffill')
        
        # Only keep labels that agree with side prediction
        agreement_mask = (
            ((labels == 1) & (side_aligned >= 0)) |
            ((labels == -1) & (side_aligned <= 0)) |
            (labels == 0)
        )
        
        labels = labels.where(agreement_mask, 0)
    
    return labels


def get_barrier_times(
    prices: pd.Series,
    events: pd.DataFrame
) -> pd.Series:
    """Get the times when barriers were touched."""
    labels_df = apply_triple_barrier(prices, events)
    return labels_df['t1']


def purge_overlapping_events(events: pd.DataFrame, barrier_times: pd.Series) -> pd.DataFrame:
    """
    Remove overlapping events to avoid data leakage.
    
    Parameters:
    -----------
    events : pd.DataFrame
        Events DataFrame
    barrier_times : pd.Series
        Times when barriers were hit
        
    Returns:
    --------
    pd.DataFrame with non-overlapping events
    """
    events_clean = events.copy()
    
    # Sort by index
    events_clean = events_clean.sort_index()
    
    # Remove events that start before previous event's barrier
    prev_barrier_time = None
    indices_to_drop = []
    
    for idx in events_clean.index:
        if prev_barrier_time is not None and idx <= prev_barrier_time:
            indices_to_drop.append(idx)
        else:
            # Update previous barrier time
            if idx in barrier_times.index:
                prev_barrier_time = barrier_times.loc[idx]
            else:
                # Fallback to t1 if barrier time not available
                prev_barrier_time = events_clean.loc[idx, 't1']
    
    # Drop overlapping events
    events_clean = events_clean.drop(indices_to_drop)
    
    logger.info(f"Removed {len(indices_to_drop)} overlapping events")
    
    return events_clean


class TripleBarrierLabeler:
    """Triple-barrier labeling with configurable parameters."""
    
    def __init__(
        self,
        max_holding_days: int = 5,
        volatility_lookback: int = 100,
        profit_taking_multiplier: float = 2.0,
        stop_loss_multiplier: Optional[float] = None,
        min_return_threshold: float = 0.001,
        purge_overlapping: bool = True
    ):
        self.max_holding_days = max_holding_days
        self.volatility_lookback = volatility_lookback
        self.profit_taking_multiplier = profit_taking_multiplier
        self.stop_loss_multiplier = stop_loss_multiplier or profit_taking_multiplier
        self.min_return_threshold = min_return_threshold
        self.purge_overlapping = purge_overlapping
    
    def create_labels(
        self,
        prices: pd.Series,
        side_prediction: Optional[pd.Series] = None,
        volatility: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Create triple-barrier labels.
        
        Returns:
        --------
        labels : pd.Series
            Binary labels
        events : pd.DataFrame
            Events used for labeling
        """
        # Validate no future data
        validate_no_future_data(pd.DataFrame(index=prices.index))
        
        # Calculate volatility if not provided
        if volatility is None:
            volatility = get_daily_vol(prices, self.volatility_lookback)
        
        # Get events
        events = get_events(
            prices=prices,
            volatility=volatility,
            max_holding_days=self.max_holding_days,
            min_return_threshold=self.min_return_threshold,
            side_prediction=side_prediction
        )
        
        if events.empty:
            logger.warning("No valid events generated")
            return pd.Series(dtype=int, name='labels'), events
        
        # Adjust target based on multiplier
        events['trgt'] = events['trgt'] * self.profit_taking_multiplier
        
        # Generate labels
        if self.purge_overlapping:
            # First pass to get barrier times
            temp_labels = apply_triple_barrier(prices, events)
            barrier_times = temp_labels['t1']
            
            # Purge overlapping events
            events_clean = purge_overlapping_events(events, barrier_times)
            
            # Final labeling with clean events
            labels = get_bins(prices, events_clean, side_prediction)
        else:
            labels = get_bins(prices, events, side_prediction)
        
        logger.info(f"Generated {len(labels)} labels from {len(events)} events")
        
        # Log label distribution
        label_counts = labels.value_counts().sort_index()
        logger.info(f"Label distribution: {label_counts.to_dict()}")
        
        return labels, events


# Convenience function
def create_triple_barrier_labels(
    prices: pd.Series,
    side_prediction: Optional[pd.Series] = None,
    max_holding_days: int = 5,
    volatility_multiplier: float = 2.0
) -> pd.Series:
    """Create triple-barrier labels with default parameters."""
    labeler = TripleBarrierLabeler(
        max_holding_days=max_holding_days,
        profit_taking_multiplier=volatility_multiplier
    )
    
    labels, _ = labeler.create_labels(prices, side_prediction)
    return labels