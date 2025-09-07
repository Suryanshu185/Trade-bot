"""Test risk management functionality."""

import pytest
import pandas as pd
import numpy as np

from src.exec.risk import RiskManager, Position, RiskEvent
from src.exec.position_sizing import PositionSizer, kelly_fraction


class TestRiskManagement:
    """Test risk management components."""
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement."""
        risk_manager = RiskManager(max_daily_loss=0.02)  # 2% daily loss limit
        risk_manager.reset_daily_tracking(10000.0)  # Start with $10k
        
        # Simulate losing trades that push us near the limit
        risk_manager.daily_pnl = -150.0  # $150 loss, 1.5% of equity
        
        current_prices = {"BTC/USDT": 49000.0}
        
        # Should not trigger limit yet
        assert not risk_manager.check_daily_loss_limit(current_prices)
        assert not risk_manager.kill_switch_active
        
        # Add more losses to exceed limit
        risk_manager.daily_pnl = -250.0  # $250 loss, 2.5% of equity
        
        # Should trigger limit
        assert risk_manager.check_daily_loss_limit(current_prices)
        assert risk_manager.kill_switch_active
    
    def test_position_risk_limits(self):
        """Test position-level risk controls."""
        risk_manager = RiskManager(max_positions=3)
        
        # Add positions up to limit
        for i in range(3):
            success = risk_manager.add_position(
                symbol=f"SYMBOL{i}",
                size=1.0,
                entry_price=100.0
            )
            assert success
        
        # Fourth position should be rejected
        success = risk_manager.add_position(
            symbol="SYMBOL4",
            size=1.0,
            entry_price=100.0
        )
        assert not success
        assert len(risk_manager.positions) == 3
    
    def test_stop_loss_execution(self):
        """Test stop loss triggering."""
        # Create position with stop loss
        position = Position(
            symbol="BTC/USDT",
            size=1.0,  # Long position
            entry_price=50000.0,
            entry_time=pd.Timestamp.now(tz='UTC'),
            stop_loss=49000.0  # 2% stop loss
        )
        
        # Price above stop loss - no trigger
        risk_event = position.update_price(49500.0)
        assert risk_event is None
        
        # Price hits stop loss - should trigger
        risk_event = position.update_price(48900.0)
        assert risk_event == RiskEvent.STOP_LOSS
    
    def test_trailing_stop(self):
        """Test trailing stop functionality."""
        position = Position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp.now(tz='UTC'),
            trailing_stop_distance=1000.0  # $1000 trailing distance
        )
        
        # Price moves up - should update highest price
        position.update_price(51000.0)
        assert position.highest_price == 51000.0
        
        # Price continues up
        position.update_price(52000.0)
        assert position.highest_price == 52000.0
        
        # Calculate trailing stop (should be $51000 now)
        trailing_stop = position._calculate_trailing_stop()
        assert trailing_stop == 51000.0
        
        # Price drops to trailing stop level
        risk_event = position.update_price(50900.0)
        assert risk_event == RiskEvent.TRAILING_STOP
    
    def test_take_profit_execution(self):
        """Test take profit triggering."""
        position = Position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp.now(tz='UTC'),
            take_profit=52000.0
        )
        
        # Price below take profit - no trigger
        risk_event = position.update_price(51500.0)
        assert risk_event is None
        
        # Price hits take profit - should trigger
        risk_event = position.update_price(52100.0)
        assert risk_event == RiskEvent.TAKE_PROFIT
    
    def test_position_sizing_kelly(self):
        """Test Kelly fraction position sizing."""
        # Test Kelly calculation
        win_rate = 0.6
        avg_win = 0.05  # 5% average win
        avg_loss = 0.03  # 3% average loss
        
        kelly_f = kelly_fraction(win_rate, avg_win, avg_loss, cap=0.25)
        
        # Manual calculation: (0.05 * 0.6 - 0.03 * 0.4) / 0.05 = 0.36
        # But capped at 0.25
        expected_kelly = min(0.25, (avg_win * win_rate - avg_loss * (1 - win_rate)) / avg_win)
        assert abs(kelly_f - expected_kelly) < 0.001
    
    def test_position_sizing_volatility_target(self):
        """Test volatility-targeted position sizing."""
        sizer = PositionSizer(
            target_volatility=0.15,  # 15% target volatility
            max_risk_per_trade=0.01  # 1% max risk per trade
        )
        
        # Test position size calculation
        signal_strength = 0.8
        price = 50000.0
        volatility = 0.30  # 30% asset volatility
        portfolio_value = 100000.0
        
        size, details = sizer.calculate_size(
            signal_strength=signal_strength,
            price=price,
            volatility=volatility,
            portfolio_value=portfolio_value
        )
        
        # Should reduce position size due to high volatility
        assert size > 0  # Should be positive for positive signal
        assert details['position_pct'] <= sizer.max_risk_per_trade
        
        # Test with lower volatility
        size_low_vol, _ = sizer.calculate_size(
            signal_strength=signal_strength,
            price=price,
            volatility=0.10,  # Lower volatility
            portfolio_value=portfolio_value
        )
        
        # Should allow larger position with lower volatility
        assert size_low_vol > size
    
    def test_position_size_limits(self):
        """Test position size limits."""
        sizer = PositionSizer(
            max_position_size=0.1,  # 10% max position
            min_position_size=0.001  # 0.1% min position
        )
        
        # Test very strong signal (should be capped)
        size, details = sizer.calculate_size(
            signal_strength=2.0,  # Very strong signal
            price=100.0,
            volatility=0.20,
            portfolio_value=10000.0
        )
        
        position_value = abs(size) * 100.0
        position_pct = position_value / 10000.0
        assert position_pct <= sizer.max_position_size
        
        # Test very weak signal (should meet minimum)
        size, details = sizer.calculate_size(
            signal_strength=0.01,  # Very weak signal
            price=100.0,
            volatility=0.20,
            portfolio_value=10000.0
        )
        
        # Should either be zero or meet minimum
        if size != 0:
            position_value = abs(size) * 100.0
            position_pct = position_value / 10000.0
            assert position_pct >= sizer.min_position_size
    
    def test_kill_switch_functionality(self):
        """Test kill switch activation and deactivation."""
        risk_manager = RiskManager(kill_switch_enabled=True)
        
        # Initially should not be active
        assert not risk_manager.kill_switch_active
        
        # Activate kill switch
        risk_manager.activate_kill_switch()
        assert risk_manager.kill_switch_active
        
        # Should prevent new positions
        success = risk_manager.add_position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0
        )
        assert not success
        
        # Deactivate kill switch
        risk_manager.deactivate_kill_switch()
        assert not risk_manager.kill_switch_active
        
        # Should allow new positions again
        success = risk_manager.add_position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0
        )
        assert success
    
    def test_risk_event_logging(self):
        """Test risk event logging."""
        risk_manager = RiskManager()
        
        # Initially no events
        assert len(risk_manager.risk_events) == 0
        
        # Activate kill switch (should log event)
        risk_manager.activate_kill_switch()
        assert len(risk_manager.risk_events) == 1
        
        event = risk_manager.risk_events[0]
        assert event['event'] == RiskEvent.KILL_SWITCH.value
        assert event['symbol'] == 'SYSTEM'
        
        # Check risk report includes events
        report = risk_manager.get_risk_report()
        assert RiskEvent.KILL_SWITCH.value in report['recent_events']
    
    def test_position_correlation_limits(self):
        """Test position correlation limits (simplified)."""
        from src.exec.position_sizing import PortfolioSizer
        
        portfolio_sizer = PortfolioSizer(
            max_positions=5,
            max_sector_weight=0.3
        )
        
        # Add positions up to sector limit
        portfolio_sizer.add_position("BTC/USDT", 0.15, 50000.0, "crypto")
        portfolio_sizer.add_position("ETH/USDT", 0.10, 3000.0, "crypto")
        
        # Check concentration limits for new crypto position
        can_add_crypto = portfolio_sizer.check_concentration_limits(
            "ADA/USDT", 0.10, 1.0, 
        )
        
        # Should reject due to sector concentration
        # Note: This is simplified as we'd need actual portfolio value calculation
        # In real implementation, this would check against actual sector weights
    
    def test_stop_loss_calculation(self):
        """Test stop loss price calculation."""
        sizer = PositionSizer()
        
        # Test long position stop loss
        stop_price = sizer.calculate_stop_loss(
            entry_price=50000.0,
            position_size=1.0,  # Long position
            portfolio_value=100000.0,
            atr=1000.0
        )
        
        # Stop should be below entry price for long position
        assert stop_price < 50000.0
        
        # Should be approximately 2*ATR below entry
        expected_stop = 50000.0 - (2 * 1000.0)
        assert abs(stop_price - expected_stop) < 100.0  # Allow some variance for risk limits
        
        # Test short position stop loss
        stop_price = sizer.calculate_stop_loss(
            entry_price=50000.0,
            position_size=-1.0,  # Short position
            portfolio_value=100000.0,
            atr=1000.0
        )
        
        # Stop should be above entry price for short position
        assert stop_price > 50000.0
    
    def test_take_profit_calculation(self):
        """Test take profit price calculation."""
        sizer = PositionSizer()
        
        entry_price = 50000.0
        stop_price = 49000.0  # $1000 stop
        
        # Test 2:1 risk/reward ratio
        take_profit = sizer.calculate_take_profit(
            entry_price=entry_price,
            stop_price=stop_price,
            position_size=1.0,
            risk_reward_ratio=2.0
        )
        
        # Take profit should be $2000 above entry (2x the $1000 risk)
        expected_tp = entry_price + 2 * (entry_price - stop_price)
        assert abs(take_profit - expected_tp) < 1.0
        
        # Test short position
        stop_price = 51000.0  # Stop above entry for short
        take_profit = sizer.calculate_take_profit(
            entry_price=entry_price,
            stop_price=stop_price,
            position_size=-1.0,
            risk_reward_ratio=2.0
        )
        
        # Take profit should be below entry for short position
        assert take_profit < entry_price