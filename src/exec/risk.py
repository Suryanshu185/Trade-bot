"""Risk management with stop losses, position limits, and kill-switch."""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from ..utils.config import get_config
from ..utils.time import now_utc


class RiskEvent(Enum):
    """Risk event types."""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    MAX_RISK_EXCEEDED = "max_risk_exceeded"
    CORRELATION_LIMIT = "correlation_limit"
    KILL_SWITCH = "kill_switch"


class Position:
    """Individual position with risk parameters."""
    
    def __init__(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        entry_time: pd.Timestamp,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None
    ):
        self.symbol = symbol
        self.size = size  # Positive for long, negative for short
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.trailing_stop_distance = trailing_stop_distance
        
        # Track high/low for trailing stop
        self.highest_price = entry_price if size > 0 else None
        self.lowest_price = entry_price if size < 0 else None
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.size > 0
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.size < 0
    
    def update_price(self, current_price: float) -> Optional[RiskEvent]:
        """Update position with current price and check for risk events."""
        if self.size == 0:
            return None
        
        # Update trailing highs/lows
        if self.is_long and (self.highest_price is None or current_price > self.highest_price):
            self.highest_price = current_price
        elif self.is_short and (self.lowest_price is None or current_price < self.lowest_price):
            self.lowest_price = current_price
        
        # Check stop loss
        if self.stop_loss is not None:
            if (self.is_long and current_price <= self.stop_loss) or \
               (self.is_short and current_price >= self.stop_loss):
                return RiskEvent.STOP_LOSS
        
        # Check take profit
        if self.take_profit is not None:
            if (self.is_long and current_price >= self.take_profit) or \
               (self.is_short and current_price <= self.take_profit):
                return RiskEvent.TAKE_PROFIT
        
        # Check trailing stop
        if self.trailing_stop_distance is not None:
            trailing_stop = self._calculate_trailing_stop()
            if trailing_stop is not None:
                if (self.is_long and current_price <= trailing_stop) or \
                   (self.is_short and current_price >= trailing_stop):
                    return RiskEvent.TRAILING_STOP
        
        return None
    
    def _calculate_trailing_stop(self) -> Optional[float]:
        """Calculate current trailing stop price."""
        if self.trailing_stop_distance is None:
            return None
        
        if self.is_long and self.highest_price is not None:
            return self.highest_price - self.trailing_stop_distance
        elif self.is_short and self.lowest_price is not None:
            return self.lowest_price + self.trailing_stop_distance
        
        return None
    
    def get_current_pnl(self, current_price: float) -> float:
        """Calculate current P&L."""
        if self.size == 0:
            return 0.0
        
        if self.is_long:
            return self.size * (current_price - self.entry_price)
        else:
            return -self.size * (current_price - self.entry_price)
    
    def get_current_return(self, current_price: float) -> float:
        """Calculate current return percentage."""
        pnl = self.get_current_pnl(current_price)
        position_value = abs(self.size) * self.entry_price
        
        if position_value == 0:
            return 0.0
        
        return pnl / position_value


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(
        self,
        max_daily_loss: float = 0.02,
        max_total_risk: float = 0.1,
        max_positions: int = 10,
        max_correlation: float = 0.7,
        kill_switch_enabled: bool = True
    ):
        self.max_daily_loss = max_daily_loss
        self.max_total_risk = max_total_risk
        self.max_positions = max_positions
        self.max_correlation = max_correlation
        self.kill_switch_enabled = kill_switch_enabled
        
        # Track positions and risk
        self.positions: Dict[str, Position] = {}
        self.daily_pnl = 0.0
        self.daily_start_equity = 0.0
        self.kill_switch_active = False
        self.last_reset_date = None
        
        # Risk event history
        self.risk_events: List[Dict] = []
    
    def reset_daily_tracking(self, current_equity: float) -> None:
        """Reset daily P&L tracking."""
        current_date = now_utc().date()
        
        if self.last_reset_date != current_date:
            self.daily_pnl = 0.0
            self.daily_start_equity = current_equity
            self.last_reset_date = current_date
            self.kill_switch_active = False
            
            logger.info(f"Reset daily tracking. Starting equity: ${current_equity:,.2f}")
    
    def add_position(
        self,
        symbol: str,
        size: float,
        entry_price: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        trailing_stop_distance: Optional[float] = None
    ) -> bool:
        """Add new position with risk checks."""
        # Check if trading is disabled
        if self.kill_switch_active:
            logger.warning("Kill switch active - no new positions allowed")
            return False
        
        # Check position count limit
        if len(self.positions) >= self.max_positions:
            logger.warning(f"Maximum positions ({self.max_positions}) reached")
            return False
        
        # Create position
        position = Position(
            symbol=symbol,
            size=size,
            entry_price=entry_price,
            entry_time=now_utc(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_distance=trailing_stop_distance
        )
        
        self.positions[symbol] = position
        
        logger.info(f"Added position: {symbol} size={size} entry=${entry_price}")
        return True
    
    def remove_position(self, symbol: str, exit_price: float, reason: str = "manual") -> Optional[float]:
        """Remove position and calculate P&L."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        pnl = position.get_current_pnl(exit_price)
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Log the exit
        logger.info(f"Closed position: {symbol} P&L=${pnl:.2f} reason={reason}")
        
        # Remove from positions
        del self.positions[symbol]
        
        return pnl
    
    def update_prices(self, prices: Dict[str, float]) -> List[Tuple[str, RiskEvent]]:
        """Update all positions with current prices and check for risk events."""
        triggered_events = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                risk_event = position.update_price(current_price)
                
                if risk_event:
                    triggered_events.append((symbol, risk_event))
                    
                    # Log the event
                    self._log_risk_event(symbol, risk_event, current_price)
        
        return triggered_events
    
    def check_daily_loss_limit(self, current_prices: Dict[str, float]) -> bool:
        """Check if daily loss limit is exceeded."""
        if self.daily_start_equity == 0:
            return False
        
        # Calculate current daily P&L including open positions
        total_pnl = self.daily_pnl
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_pnl = position.get_current_pnl(current_prices[symbol])
                total_pnl += position_pnl
        
        # Check daily loss
        daily_loss_pct = -total_pnl / self.daily_start_equity
        
        if daily_loss_pct >= self.max_daily_loss:
            logger.critical(f"Daily loss limit exceeded: {daily_loss_pct:.2%} >= {self.max_daily_loss:.2%}")
            
            if self.kill_switch_enabled:
                self.activate_kill_switch()
            
            return True
        
        return False
    
    def check_total_risk(self, current_prices: Dict[str, float], portfolio_value: float) -> bool:
        """Check if total portfolio risk is exceeded."""
        if portfolio_value <= 0:
            return False
        
        total_risk = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value = abs(position.size) * current_prices[symbol]
                position_risk = position_value / portfolio_value
                total_risk += position_risk
        
        if total_risk > self.max_total_risk:
            logger.warning(f"Total risk exceeded: {total_risk:.2%} > {self.max_total_risk:.2%}")
            return True
        
        return False
    
    def activate_kill_switch(self) -> None:
        """Activate kill switch to stop all trading."""
        self.kill_switch_active = True
        logger.critical("KILL SWITCH ACTIVATED - All trading disabled")
        
        # Record event
        self._log_risk_event("SYSTEM", RiskEvent.KILL_SWITCH, 0.0)
    
    def deactivate_kill_switch(self) -> None:
        """Manually deactivate kill switch."""
        self.kill_switch_active = False
        logger.info("Kill switch deactivated")
    
    def get_position_risk_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get comprehensive risk summary."""
        summary = {
            "num_positions": len(self.positions),
            "kill_switch_active": self.kill_switch_active,
            "daily_pnl": self.daily_pnl,
            "positions": {}
        }
        
        total_exposure = 0.0
        total_unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = abs(position.size) * current_price
                unrealized_pnl = position.get_current_pnl(current_price)
                
                total_exposure += position_value
                total_unrealized_pnl += unrealized_pnl
                
                summary["positions"][symbol] = {
                    "size": position.size,
                    "entry_price": position.entry_price,
                    "current_price": current_price,
                    "value": position_value,
                    "unrealized_pnl": unrealized_pnl,
                    "return_pct": position.get_current_return(current_price),
                    "stop_loss": position.stop_loss,
                    "take_profit": position.take_profit,
                    "trailing_stop": position._calculate_trailing_stop()
                }
        
        summary.update({
            "total_exposure": total_exposure,
            "total_unrealized_pnl": total_unrealized_pnl,
            "total_pnl": self.daily_pnl + total_unrealized_pnl
        })
        
        return summary
    
    def _log_risk_event(self, symbol: str, event: RiskEvent, price: float) -> None:
        """Log risk event."""
        event_data = {
            "timestamp": now_utc(),
            "symbol": symbol,
            "event": event.value,
            "price": price
        }
        
        self.risk_events.append(event_data)
        
        # Keep only recent events
        if len(self.risk_events) > 1000:
            self.risk_events = self.risk_events[-500:]
    
    def get_risk_report(self) -> Dict:
        """Generate comprehensive risk report."""
        recent_events = [e for e in self.risk_events 
                        if (now_utc() - e["timestamp"]).days <= 7]
        
        event_counts = {}
        for event in recent_events:
            event_type = event["event"]
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        return {
            "current_date": now_utc().date(),
            "kill_switch_active": self.kill_switch_active,
            "daily_pnl": self.daily_pnl,
            "daily_start_equity": self.daily_start_equity,
            "num_positions": len(self.positions),
            "max_positions": self.max_positions,
            "recent_events": event_counts,
            "config": {
                "max_daily_loss": self.max_daily_loss,
                "max_total_risk": self.max_total_risk,
                "max_positions": self.max_positions
            }
        }


class StopLossManager:
    """Specialized stop loss management."""
    
    def __init__(self):
        self.trailing_stops: Dict[str, float] = {}
    
    def set_trailing_stop(self, symbol: str, distance: float) -> None:
        """Set trailing stop distance for symbol."""
        self.trailing_stops[symbol] = distance
    
    def update_trailing_stops(
        self, 
        positions: Dict[str, Position], 
        current_prices: Dict[str, float]
    ) -> Dict[str, float]:
        """Update trailing stop levels."""
        updated_stops = {}
        
        for symbol, position in positions.items():
            if symbol in current_prices and symbol in self.trailing_stops:
                current_price = current_prices[symbol]
                distance = self.trailing_stops[symbol]
                
                if position.is_long:
                    new_stop = current_price - distance
                    if position.stop_loss is None or new_stop > position.stop_loss:
                        position.stop_loss = new_stop
                        updated_stops[symbol] = new_stop
                
                elif position.is_short:
                    new_stop = current_price + distance
                    if position.stop_loss is None or new_stop < position.stop_loss:
                        position.stop_loss = new_stop
                        updated_stops[symbol] = new_stop
        
        return updated_stops


def create_risk_manager(config_override: Optional[Dict] = None) -> RiskManager:
    """Create risk manager with configuration."""
    config = get_config()
    
    params = {
        "max_daily_loss": config.trading.max_daily_loss,
        "max_total_risk": config.trading.max_position_size,
        "kill_switch_enabled": True
    }
    
    if config_override:
        params.update(config_override)
    
    return RiskManager(**params)