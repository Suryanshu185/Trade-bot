"""Volatility-targeted position sizing with Kelly fraction optimization."""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from ..utils.config import get_config


def kelly_fraction(
    win_rate: float, 
    avg_win: float, 
    avg_loss: float,
    cap: float = 0.25
) -> float:
    """
    Calculate Kelly fraction for position sizing.
    
    Parameters:
    -----------
    win_rate : float
        Probability of winning trade (0-1)
    avg_win : float
        Average winning trade return
    avg_loss : float
        Average losing trade return (positive value)
    cap : float
        Maximum Kelly fraction (risk control)
        
    Returns:
    --------
    float : Kelly fraction capped at maximum
    """
    if avg_loss <= 0 or avg_win <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    
    # Kelly formula: f = (bp - q) / b
    # where b = avg_win/avg_loss, p = win_rate, q = 1 - win_rate
    b = avg_win / avg_loss
    kelly_f = (b * win_rate - (1 - win_rate)) / b
    
    # Cap the Kelly fraction and ensure non-negative
    kelly_f = max(0, min(kelly_f, cap))
    
    return kelly_f


def volatility_target_size(
    target_vol: float,
    price_vol: float,
    price: float,
    portfolio_value: float,
    risk_per_trade: float = 0.005
) -> float:
    """
    Calculate position size based on volatility targeting.
    
    Parameters:
    -----------
    target_vol : float
        Target portfolio volatility (annualized)
    price_vol : float  
        Price volatility (annualized)
    price : float
        Current price
    portfolio_value : float
        Total portfolio value
    risk_per_trade : float
        Maximum risk per trade as fraction of portfolio
        
    Returns:
    --------
    float : Position size (number of shares/contracts)
    """
    if price_vol <= 0 or price <= 0 or portfolio_value <= 0:
        return 0.0
    
    # Calculate position value needed for target volatility
    position_value = (target_vol / price_vol) * portfolio_value
    
    # Calculate position size
    position_size = position_value / price
    
    # Apply risk per trade limit
    max_position_value = risk_per_trade * portfolio_value
    max_position_size = max_position_value / price
    
    # Take minimum to respect risk limits
    final_size = min(position_size, max_position_size)
    
    return final_size


def atr_position_size(
    atr: float,
    price: float,
    portfolio_value: float,
    risk_per_trade: float = 0.005,
    atr_multiplier: float = 2.0
) -> float:
    """
    Calculate position size based on ATR (Average True Range).
    
    Parameters:
    -----------
    atr : float
        Average True Range
    price : float
        Current price
    portfolio_value : float
        Total portfolio value
    risk_per_trade : float
        Risk per trade as fraction of portfolio
    atr_multiplier : float
        ATR multiplier for stop distance
        
    Returns:
    --------
    float : Position size
    """
    if atr <= 0 or price <= 0 or portfolio_value <= 0:
        return 0.0
    
    # Calculate stop distance
    stop_distance = atr * atr_multiplier
    
    # Calculate risk amount in currency
    risk_amount = portfolio_value * risk_per_trade
    
    # Calculate position size
    position_size = risk_amount / stop_distance
    
    return position_size


class PositionSizer:
    """Advanced position sizing with multiple strategies."""
    
    def __init__(
        self,
        target_volatility: float = 0.15,
        max_risk_per_trade: float = 0.005,
        kelly_cap: float = 0.25,
        min_position_size: float = 0.01,
        max_position_size: float = 0.5,
        sizing_method: str = "volatility_target"
    ):
        self.target_volatility = target_volatility
        self.max_risk_per_trade = max_risk_per_trade
        self.kelly_cap = kelly_cap
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.sizing_method = sizing_method
        
        # Track performance for Kelly calculation
        self.trade_history = []
    
    def add_trade_result(self, return_pct: float) -> None:
        """Add trade result for Kelly fraction calculation."""
        self.trade_history.append(return_pct)
        
        # Keep only recent trades for adaptation
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
    
    def get_kelly_fraction(self) -> float:
        """Calculate current Kelly fraction from trade history."""
        if len(self.trade_history) < 10:
            return 0.1  # Conservative default
        
        returns = np.array(self.trade_history)
        
        # Calculate win rate and average win/loss
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.1
        
        win_rate = len(wins) / len(returns)
        avg_win = np.mean(wins)
        avg_loss = np.mean(np.abs(losses))
        
        return kelly_fraction(win_rate, avg_win, avg_loss, self.kelly_cap)
    
    def calculate_size(
        self,
        signal_strength: float,
        price: float,
        volatility: float,
        portfolio_value: float,
        atr: Optional[float] = None,
        kelly_override: Optional[float] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate position size using selected method.
        
        Parameters:
        -----------
        signal_strength : float
            Signal strength (-1 to 1)
        price : float
            Current price
        volatility : float
            Price volatility (annualized)
        portfolio_value : float
            Total portfolio value
        atr : float, optional
            Average True Range
        kelly_override : float, optional
            Override Kelly fraction
            
        Returns:
        --------
        tuple : (position_size, sizing_details)
        """
        if abs(signal_strength) < 0.01:  # No meaningful signal
            return 0.0, {"reason": "weak_signal"}
        
        sizing_details = {
            "method": self.sizing_method,
            "signal_strength": signal_strength,
            "volatility": volatility,
            "kelly_fraction": kelly_override or self.get_kelly_fraction()
        }
        
        # Base position size calculation
        if self.sizing_method == "volatility_target":
            base_size = volatility_target_size(
                target_vol=self.target_volatility,
                price_vol=volatility,
                price=price,
                portfolio_value=portfolio_value,
                risk_per_trade=self.max_risk_per_trade
            )
        
        elif self.sizing_method == "atr" and atr is not None:
            base_size = atr_position_size(
                atr=atr,
                price=price,
                portfolio_value=portfolio_value,
                risk_per_trade=self.max_risk_per_trade
            )
        
        elif self.sizing_method == "kelly":
            kelly_f = kelly_override or self.get_kelly_fraction()
            base_size = (kelly_f * portfolio_value) / price
        
        else:
            # Fixed fraction fallback
            base_size = (self.max_risk_per_trade * portfolio_value) / price
        
        # Apply signal strength scaling
        scaled_size = base_size * abs(signal_strength)
        
        # Apply position limits
        max_size_value = self.max_position_size * portfolio_value
        max_size = max_size_value / price
        
        min_size_value = self.min_position_size * portfolio_value
        min_size = min_size_value / price
        
        # Final size with limits
        final_size = np.clip(scaled_size, min_size, max_size)
        
        # Apply direction
        final_size = final_size * np.sign(signal_strength)
        
        sizing_details.update({
            "base_size": base_size,
            "scaled_size": scaled_size,
            "final_size": final_size,
            "position_value": abs(final_size) * price,
            "position_pct": abs(final_size) * price / portfolio_value
        })
        
        return final_size, sizing_details
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_size: float,
        portfolio_value: float,
        atr: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> float:
        """Calculate stop loss price."""
        if position_size == 0:
            return entry_price
        
        direction = np.sign(position_size)
        
        # Calculate stop distance based on available information
        if atr is not None:
            stop_distance = atr * 2.0  # 2x ATR stop
        elif volatility is not None:
            # Use volatility-based stop (2 standard deviations)
            stop_distance = entry_price * volatility * 2.0 / np.sqrt(252)
        else:
            # Fallback to fixed percentage
            stop_distance = entry_price * 0.02  # 2% stop
        
        # Calculate stop price
        if direction > 0:  # Long position
            stop_price = entry_price - stop_distance
        else:  # Short position
            stop_price = entry_price + stop_distance
        
        # Ensure stop doesn't risk more than max per trade
        position_value = abs(position_size) * entry_price
        max_loss = portfolio_value * self.max_risk_per_trade
        max_stop_distance = max_loss / abs(position_size)
        
        if direction > 0:
            min_stop_price = entry_price - max_stop_distance
            stop_price = max(stop_price, min_stop_price)
        else:
            max_stop_price = entry_price + max_stop_distance
            stop_price = min(stop_price, max_stop_price)
        
        return stop_price
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_price: float,
        position_size: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """Calculate take profit price based on risk/reward ratio."""
        if position_size == 0:
            return entry_price
        
        direction = np.sign(position_size)
        stop_distance = abs(entry_price - stop_price)
        
        if direction > 0:  # Long position
            take_profit = entry_price + (stop_distance * risk_reward_ratio)
        else:  # Short position
            take_profit = entry_price - (stop_distance * risk_reward_ratio)
        
        return take_profit


class PortfolioSizer:
    """Portfolio-level position sizing with correlation and concentration limits."""
    
    def __init__(
        self,
        max_positions: int = 10,
        max_sector_weight: float = 0.3,
        max_correlation: float = 0.7,
        rebalance_threshold: float = 0.05
    ):
        self.max_positions = max_positions
        self.max_sector_weight = max_sector_weight
        self.max_correlation = max_correlation
        self.rebalance_threshold = rebalance_threshold
        
        self.current_positions = {}
        self.position_sizer = PositionSizer()
    
    def add_position(self, symbol: str, size: float, price: float, sector: str = "unknown") -> None:
        """Add or update position."""
        self.current_positions[symbol] = {
            "size": size,
            "price": price,
            "value": abs(size) * price,
            "sector": sector
        }
    
    def remove_position(self, symbol: str) -> None:
        """Remove position."""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value."""
        return sum(pos["value"] for pos in self.current_positions.values())
    
    def check_concentration_limits(self, symbol: str, new_size: float, price: float) -> bool:
        """Check if new position violates concentration limits."""
        portfolio_value = self.get_portfolio_value()
        new_value = abs(new_size) * price
        
        # Check position count limit
        if symbol not in self.current_positions and len(self.current_positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        # Check single position weight
        position_weight = new_value / portfolio_value if portfolio_value > 0 else 0
        if position_weight > self.max_sector_weight:
            logger.warning(f"Position weight {position_weight:.2%} exceeds limit {self.max_sector_weight:.2%}")
            return False
        
        return True
    
    def calculate_portfolio_size(
        self,
        signals: Dict[str, float],
        prices: Dict[str, float],
        volatilities: Dict[str, float],
        portfolio_value: float,
        **kwargs
    ) -> Dict[str, Tuple[float, Dict]]:
        """Calculate position sizes for entire portfolio."""
        results = {}
        
        # Sort signals by strength
        sorted_signals = sorted(signals.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for symbol, signal in sorted_signals:
            if symbol not in prices or symbol not in volatilities:
                continue
            
            # Calculate individual position size
            size, details = self.position_sizer.calculate_size(
                signal_strength=signal,
                price=prices[symbol],
                volatility=volatilities[symbol],
                portfolio_value=portfolio_value,
                **kwargs
            )
            
            # Check concentration limits
            if size != 0 and self.check_concentration_limits(symbol, size, prices[symbol]):
                results[symbol] = (size, details)
            else:
                results[symbol] = (0, {"reason": "concentration_limit"})
        
        return results


def create_position_sizer(config_override: Optional[Dict] = None) -> PositionSizer:
    """Create position sizer with configuration."""
    config = get_config()
    
    params = {
        "target_volatility": config.trading.target_volatility,
        "max_risk_per_trade": config.trading.max_risk_per_trade,
        "kelly_cap": config.trading.kelly_cap,
        "max_position_size": config.trading.max_position_size
    }
    
    if config_override:
        params.update(config_override)
    
    return PositionSizer(**params)