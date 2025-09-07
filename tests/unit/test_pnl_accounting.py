"""Test P&L calculation and accounting accuracy."""

import pytest
import pandas as pd
import numpy as np

from src.backtest.engine import BacktestEngine, ExecutionEngine, Order, OrderType, OrderStatus
from src.backtest.metrics import calculate_returns, calculate_cagr, calculate_sharpe_ratio
from src.live.broker_stub import PaperBroker
from src.exec.risk import RiskManager, Position


class TestPnLAccounting:
    """Test P&L calculation accuracy and accounting consistency."""
    
    def test_simple_trade_pnl(self):
        """Test basic trade P&L calculation."""
        broker = PaperBroker(initial_cash=10000.0, commission_rate=0.001)
        
        # Buy 1 BTC at $50,000
        order_id = broker.submit_order("BTC/USDT", "buy", 1.0, OrderType.MARKET)
        
        # Simulate market data to fill order
        market_data = {
            "BTC/USDT": {
                "bid": 49950.0,
                "ask": 50050.0,
                "close": 50000.0,
                "volume": 1000
            }
        }
        
        trades = broker.update_market_data(market_data)
        assert len(trades) == 1
        
        trade = trades[0]
        position_value = trade.quantity * trade.price
        expected_cash = 10000.0 - position_value - trade.commission
        
        # Check cash accounting
        assert abs(broker.cash - expected_cash) < 0.01
        
        # Check position
        assert broker.positions["BTC/USDT"] == trade.quantity
        
        # Sell the position
        sell_order_id = broker.submit_order("BTC/USDT", "sell", 1.0, OrderType.MARKET)
        
        # Simulate price increase
        market_data["BTC/USDT"]["close"] = 51000.0
        market_data["BTC/USDT"]["bid"] = 50950.0
        market_data["BTC/USDT"]["ask"] = 51050.0
        
        sell_trades = broker.update_market_data(market_data)
        assert len(sell_trades) == 1
        
        sell_trade = sell_trades[0]
        
        # Calculate expected P&L
        buy_cost = trade.quantity * trade.price + trade.commission
        sell_proceeds = sell_trade.quantity * sell_trade.price - sell_trade.commission
        expected_pnl = sell_proceeds - buy_cost
        
        # Check final cash (should be initial + P&L)
        expected_final_cash = 10000.0 + expected_pnl
        assert abs(broker.cash - expected_final_cash) < 0.01
        
        # Position should be closed
        assert broker.positions.get("BTC/USDT", 0) == 0
    
    def test_multiple_trades_pnl_consistency(self):
        """Test P&L consistency across multiple trades."""
        broker = PaperBroker(initial_cash=10000.0, commission_rate=0.001)
        initial_cash = broker.cash
        
        trades_executed = []
        
        # Execute multiple trades
        for i in range(5):
            # Buy
            buy_order = broker.submit_order("BTC/USDT", "buy", 0.1, OrderType.MARKET)
            
            market_data = {
                "BTC/USDT": {
                    "bid": 50000 + i * 100,
                    "ask": 50100 + i * 100,
                    "close": 50050 + i * 100,
                    "volume": 1000
                }
            }
            
            buy_trades = broker.update_market_data(market_data)
            trades_executed.extend(buy_trades)
            
            # Sell
            sell_order = broker.submit_order("BTC/USDT", "sell", 0.1, OrderType.MARKET)
            
            market_data["BTC/USDT"]["close"] = 50100 + i * 100
            market_data["BTC/USDT"]["bid"] = 50050 + i * 100
            
            sell_trades = broker.update_market_data(market_data)
            trades_executed.extend(sell_trades)
        
        # Calculate total P&L from trades
        total_pnl = 0
        buy_cost = 0
        sell_proceeds = 0
        
        for trade in trades_executed:
            trade_value = trade.quantity * trade.price
            if trade.side == "buy":
                buy_cost += trade_value + trade.commission
            else:
                sell_proceeds += trade_value - trade.commission
        
        total_pnl = sell_proceeds - buy_cost
        
        # Check cash consistency
        expected_cash = initial_cash + total_pnl
        assert abs(broker.cash - expected_cash) < 0.01
        
        # All positions should be closed
        assert sum(broker.positions.values()) == 0
    
    def test_commission_calculation(self):
        """Test commission calculation accuracy."""
        broker = PaperBroker(initial_cash=10000.0, commission_rate=0.001)
        
        # Place order
        order_id = broker.submit_order("BTC/USDT", "buy", 0.5, OrderType.MARKET)
        
        market_data = {
            "BTC/USDT": {
                "bid": 49950.0,
                "ask": 50050.0,
                "close": 50000.0,
                "volume": 1000
            }
        }
        
        trades = broker.update_market_data(market_data)
        trade = trades[0]
        
        # Check commission calculation
        expected_commission = trade.quantity * trade.price * broker.commission_rate
        assert abs(trade.commission - expected_commission) < 0.001
        
        # Check that commission is deducted from cash
        expected_cash = 10000.0 - (trade.quantity * trade.price) - trade.commission
        assert abs(broker.cash - expected_cash) < 0.01
    
    def test_backtest_pnl_consistency(self, sample_ohlcv_data):
        """Test P&L consistency in backtesting engine."""
        # Create simple backtest
        engine = BacktestEngine(
            initial_cash=10000.0,
            commission_rate=0.001,
            slippage_bps=1.0
        )
        
        # Add data
        engine.add_data("BTC/USDT", sample_ohlcv_data)
        
        # Track equity curve
        initial_equity = engine.cash
        
        # Manually execute a few trades to test accounting
        engine.execution_engine.submit_order(
            "BTC/USDT", OrderType.MARKET, "buy", 0.1,
            timestamp=sample_ohlcv_data.index[10]
        )
        
        # Process the order
        current_prices = {"BTC/USDT": sample_ohlcv_data.iloc[10]['close']}
        trades = engine.execution_engine.process_orders(
            current_prices, sample_ohlcv_data.index[10]
        )
        
        # Apply trade
        for trade in trades:
            engine._apply_trade(trade)
        
        # Check position value + cash equals initial equity (minus costs)
        position_value = engine.positions.get("BTC/USDT", 0) * current_prices["BTC/USDT"]
        total_equity = engine.cash + position_value
        
        # Account for commission and slippage
        total_costs = sum(trade.commission + trade.slippage for trade in trades)
        expected_equity = initial_equity - total_costs
        
        assert abs(total_equity - expected_equity) < 0.01
    
    def test_position_class_pnl(self):
        """Test Position class P&L calculation."""
        # Long position
        long_position = Position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp.now(tz='UTC')
        )
        
        # Test current P&L
        current_price = 51000.0
        pnl = long_position.get_current_pnl(current_price)
        expected_pnl = 1.0 * (51000.0 - 50000.0)
        assert abs(pnl - expected_pnl) < 0.01
        
        # Test return percentage
        return_pct = long_position.get_current_return(current_price)
        expected_return = (51000.0 - 50000.0) / 50000.0
        assert abs(return_pct - expected_return) < 0.001
        
        # Short position
        short_position = Position(
            symbol="BTC/USDT",
            size=-1.0,
            entry_price=50000.0,
            entry_time=pd.Timestamp.now(tz='UTC')
        )
        
        # Test short P&L
        pnl = short_position.get_current_pnl(current_price)
        expected_pnl = -(-1.0) * (51000.0 - 50000.0)  # Negative size, so profit on price decrease
        assert abs(pnl - expected_pnl) < 0.01
    
    def test_equity_curve_calculation(self, sample_trades_data):
        """Test equity curve calculation from trades."""
        # Create equity curve from trades
        initial_equity = 10000.0
        running_equity = initial_equity
        equity_points = [initial_equity]
        
        for _, trade in sample_trades_data.iterrows():
            # Apply trade P&L
            running_equity += trade['pnl']
            equity_points.append(running_equity)
        
        equity_series = pd.Series(equity_points)
        
        # Calculate returns
        returns = calculate_returns(equity_series)
        
        # Check that returns are calculated correctly
        for i in range(1, len(equity_series)):
            expected_return = (equity_series.iloc[i] - equity_series.iloc[i-1]) / equity_series.iloc[i-1]
            actual_return = returns.iloc[i]
            
            if not pd.isna(actual_return):
                assert abs(actual_return - expected_return) < 1e-10
        
        # Test CAGR calculation
        if len(equity_series) > 1:
            cagr = calculate_cagr(equity_series, periods_per_year=252)
            
            # Manual CAGR calculation
            total_return = equity_series.iloc[-1] / equity_series.iloc[0] - 1
            years = len(equity_series) / 252
            manual_cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            if years > 0:
                assert abs(cagr - manual_cagr) < 1e-10
    
    def test_risk_manager_pnl_tracking(self):
        """Test risk manager P&L tracking."""
        risk_manager = RiskManager()
        risk_manager.reset_daily_tracking(10000.0)
        
        # Add position
        risk_manager.add_position(
            symbol="BTC/USDT",
            size=1.0,
            entry_price=50000.0
        )
        
        # Test P&L calculation
        current_prices = {"BTC/USDT": 51000.0}
        
        # Get risk summary
        summary = risk_manager.get_position_risk_summary(current_prices)
        
        # Check unrealized P&L
        expected_unrealized = 1.0 * (51000.0 - 50000.0)
        assert abs(summary['total_unrealized_pnl'] - expected_unrealized) < 0.01
        
        # Close position and check realized P&L
        realized_pnl = risk_manager.remove_position("BTC/USDT", 51000.0)
        assert abs(realized_pnl - expected_unrealized) < 0.01
        
        # Check daily P&L update
        assert abs(risk_manager.daily_pnl - expected_unrealized) < 0.01
    
    def test_slippage_impact_on_pnl(self):
        """Test that slippage correctly impacts P&L."""
        execution_engine = ExecutionEngine(
            commission_rate=0.001,
            slippage_bps=10.0  # 10 bps slippage
        )
        
        # Submit market buy order
        order_id = execution_engine.submit_order(
            "BTC/USDT", OrderType.MARKET, "buy", 1.0
        )
        
        # Process with market data
        current_prices = {"BTC/USDT": 50000.0}
        current_time = pd.Timestamp.now(tz='UTC')
        
        trades = execution_engine.process_orders(current_prices, current_time)
        assert len(trades) == 1
        
        trade = trades[0]
        
        # Check that slippage increased the buy price
        expected_slippage = 50000.0 * (10.0 / 10000)  # 10 bps
        assert trade.slippage > 0
        assert trade.price > 50000.0  # Buy price should be higher due to slippage
        
        # Check that final price includes slippage
        expected_final_price = 50000.0 + expected_slippage
        assert abs(trade.price - expected_final_price) < 1.0  # Allow small variance