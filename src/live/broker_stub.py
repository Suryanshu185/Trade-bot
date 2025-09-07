"""Paper trading simulator with FIFO fills, latency, and fees."""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from ..backtest.engine import Order, Trade, OrderType, OrderStatus
from ..exec.risk import RiskManager, Position
from ..exec.position_sizing import PositionSizer
from ..utils.config import get_config
from ..utils.time import now_utc


@dataclass
class FillSimulation:
    """Simulated fill with realistic characteristics."""
    timestamp: pd.Timestamp
    price: float
    quantity: float
    latency_ms: float
    market_impact_bps: float


class PaperBroker:
    """Paper trading broker simulator."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        min_latency_ms: float = 10.0,
        max_latency_ms: float = 100.0,
        market_impact_model: str = "sqrt"
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.min_latency_ms = min_latency_ms
        self.max_latency_ms = max_latency_ms
        self.market_impact_model = market_impact_model
        
        # Account state
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.pending_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"paper_{self._order_counter:06d}"
    
    def submit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None
    ) -> str:
        """Submit order to paper broker."""
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=abs(quantity),
            price=price,
            timestamp=now_utc()
        )
        
        self.pending_orders[order_id] = order
        logger.info(f"Submitted {side} order: {symbol} {quantity} @ {price or 'MARKET'}")
        
        return order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.status = OrderStatus.CANCELLED
            del self.pending_orders[order_id]
            logger.info(f"Cancelled order {order_id}")
            return True
        return False
    
    def update_market_data(self, market_data: Dict[str, Dict[str, float]]) -> List[Trade]:
        """Update with latest market data and process orders."""
        executed_trades = []
        
        for order_id, order in list(self.pending_orders.items()):
            if order.symbol in market_data:
                symbol_data = market_data[order.symbol]
                trade = self._try_fill_order(order, symbol_data)
                
                if trade:
                    executed_trades.append(trade)
                    self.trades.append(trade)
                    self._apply_trade(trade)
                    
                    # Remove from pending if fully filled
                    if order.status == OrderStatus.FILLED:
                        del self.pending_orders[order_id]
                        self.filled_orders.append(order)
        
        # Update unrealized P&L
        self._update_unrealized_pnl(market_data)
        
        return executed_trades
    
    def _try_fill_order(self, order: Order, market_data: Dict[str, float]) -> Optional[Trade]:
        """Try to fill an order with market data."""
        bid = market_data.get('bid', market_data.get('close'))
        ask = market_data.get('ask', market_data.get('close'))
        last = market_data.get('close', market_data.get('last'))
        volume = market_data.get('volume', 1000000)  # Default high volume
        
        if not all([bid, ask, last]):
            return None
        
        # Determine if order should fill
        should_fill = False
        fill_price = None
        
        if order.order_type == OrderType.MARKET:
            should_fill = True
            fill_price = ask if order.side == 'buy' else bid
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == 'buy' and order.price >= ask:
                should_fill = True
                fill_price = min(order.price, ask)
            elif order.side == 'sell' and order.price <= bid:
                should_fill = True
                fill_price = max(order.price, bid)
        
        if not should_fill:
            return None
        
        # Simulate realistic fill characteristics
        fill_simulation = self._simulate_fill(order, fill_price, volume)
        
        # Create trade
        trade = Trade(
            id=f"trade_{len(self.trades) + 1:06d}",
            symbol=order.symbol,
            timestamp=fill_simulation.timestamp,
            side=order.side,
            quantity=fill_simulation.quantity,
            price=fill_simulation.price,
            commission=fill_simulation.quantity * fill_simulation.price * self.commission_rate,
            slippage=fill_simulation.market_impact_bps / 10000 * fill_simulation.price
        )
        
        # Update order status
        order.filled_quantity += fill_simulation.quantity
        order.avg_fill_price = fill_simulation.price
        order.commission += trade.commission
        order.slippage += trade.slippage
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        return trade
    
    def _simulate_fill(self, order: Order, base_price: float, volume: float) -> FillSimulation:
        """Simulate realistic fill with latency and market impact."""
        # Simulate latency
        latency = np.random.uniform(self.min_latency_ms, self.max_latency_ms)
        fill_time = order.timestamp + pd.Timedelta(milliseconds=latency)
        
        # Calculate market impact
        order_value = order.quantity * base_price
        volume_ratio = order_value / (volume * base_price) if volume > 0 else 0.01
        
        if self.market_impact_model == "sqrt":
            impact_bps = 1.0 * np.sqrt(volume_ratio) * 10000  # Square root model
        elif self.market_impact_model == "linear":
            impact_bps = 2.0 * volume_ratio * 10000  # Linear model
        else:
            impact_bps = 1.0  # Fixed impact
        
        # Cap market impact
        impact_bps = min(impact_bps, 50.0)  # Max 50 bps impact
        
        # Apply impact to price
        if order.side == 'buy':
            final_price = base_price * (1 + impact_bps / 10000)
        else:
            final_price = base_price * (1 - impact_bps / 10000)
        
        # Simulate partial fills for large orders
        fill_quantity = order.quantity
        if order_value > 10000:  # Orders > $10k might get partial fills
            if np.random.random() < 0.1:  # 10% chance of partial fill
                fill_quantity = order.quantity * np.random.uniform(0.7, 0.95)
        
        return FillSimulation(
            timestamp=fill_time,
            price=final_price,
            quantity=fill_quantity,
            latency_ms=latency,
            market_impact_bps=impact_bps
        )
    
    def _apply_trade(self, trade: Trade) -> None:
        """Apply trade to account."""
        # Update cash
        trade_value = trade.quantity * trade.price
        total_cost = trade_value + trade.commission
        
        if trade.side == 'buy':
            self.cash -= total_cost
            position_change = trade.quantity
        else:
            self.cash += trade_value - trade.commission
            position_change = -trade.quantity
        
        # Update position
        current_position = self.positions.get(trade.symbol, 0.0)
        new_position = current_position + position_change
        
        if abs(new_position) < 1e-8:
            self.positions.pop(trade.symbol, None)
        else:
            self.positions[trade.symbol] = new_position
        
        # Track realized P&L for closed positions
        if current_position * position_change < 0:  # Position reducing trade
            # Calculate P&L on closed portion
            # Simplified - would need FIFO/LIFO logic for complete accuracy
            pass
    
    def _update_unrealized_pnl(self, market_data: Dict[str, Dict[str, float]]) -> None:
        """Update unrealized P&L."""
        total_unrealized = 0.0
        
        for symbol, quantity in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol].get('close', 0.0)
                position_value = quantity * current_price
                # Would need average cost basis for accurate unrealized P&L
                # This is simplified
                total_unrealized += position_value
        
        self.unrealized_pnl = total_unrealized
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get current account summary."""
        total_position_value = sum(
            abs(qty) * 100  # Simplified - would use current market prices
            for qty in self.positions.values()
        )
        
        total_equity = self.cash + total_position_value
        
        return {
            'timestamp': now_utc(),
            'cash': self.cash,
            'positions': self.positions.copy(),
            'total_position_value': total_position_value,
            'total_equity': total_equity,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'daily_pnl': self.daily_pnl,
            'num_pending_orders': len(self.pending_orders),
            'num_filled_orders': len(self.filled_orders),
            'total_trades': len(self.trades)
        }
    
    def get_positions(self) -> Dict[str, float]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_pending_orders(self) -> List[Order]:
        """Get pending orders."""
        return list(self.pending_orders.values())
    
    def get_trade_history(self, days: int = 7) -> List[Trade]:
        """Get recent trade history."""
        cutoff_date = now_utc() - pd.Timedelta(days=days)
        return [trade for trade in self.trades if trade.timestamp >= cutoff_date]


class PaperTradingEngine:
    """Complete paper trading engine."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        symbols: Optional[List[str]] = None,
        signal_generators: Optional[Dict] = None
    ):
        config = get_config()
        
        self.symbols = symbols or config.trading.symbols
        self.broker = PaperBroker(initial_cash)
        self.risk_manager = RiskManager()
        self.position_sizer = PositionSizer()
        
        # Signal generation
        self.signal_generators = signal_generators or {}
        
        # Market data
        self.latest_data: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.equity_history: List[Dict] = []
        self.daily_reports: List[Dict] = []
        
        # Control flags
        self.is_running = False
        self.last_signal_time = {}
    
    async def start(self) -> None:
        """Start paper trading engine."""
        logger.info("Starting paper trading engine...")
        self.is_running = True
        
        # Initialize risk manager with current equity
        account = self.broker.get_account_summary()
        self.risk_manager.reset_daily_tracking(account['total_equity'])
        
        logger.info(f"Paper trading started with ${account['total_equity']:,.2f}")
    
    def stop(self) -> None:
        """Stop paper trading engine."""
        logger.info("Stopping paper trading engine...")
        self.is_running = False
        
        # Cancel all pending orders
        pending_orders = self.broker.get_pending_orders()
        for order in pending_orders:
            self.broker.cancel_order(order.id)
        
        # Generate final report
        self._generate_daily_report()
    
    def update_market_data(self, symbol: str, data: Dict[str, float]) -> None:
        """Update market data for a symbol."""
        self.latest_data[symbol] = data.copy()
        self.latest_data[symbol]['timestamp'] = now_utc()
    
    async def process_trading_cycle(self) -> None:
        """Process one trading cycle."""
        if not self.is_running:
            return
        
        current_time = now_utc()
        
        # Process pending orders with latest market data
        executed_trades = self.broker.update_market_data(self.latest_data)
        
        # Update risk manager with executed trades
        for trade in executed_trades:
            # Update risk manager positions (simplified)
            symbol = trade.symbol
            if symbol not in self.risk_manager.positions:
                # Would create Position object with proper parameters
                pass
        
        # Generate signals if we have sufficient market data
        signals = self._generate_signals(current_time)
        
        # Check risk limits
        account = self.broker.get_account_summary()
        current_prices = {symbol: data.get('close', 0) for symbol, data in self.latest_data.items()}
        
        if not self.risk_manager.check_daily_loss_limit(current_prices):
            # Generate new orders based on signals
            await self._generate_orders(signals, current_prices, account)
        
        # Update equity tracking
        self._update_equity_tracking(account)
        
        # Log status periodically
        if len(self.equity_history) % 100 == 0:
            logger.info(f"Paper trading status: Equity=${account['total_equity']:,.2f}, "
                       f"Positions={len(account['positions'])}, "
                       f"Pending Orders={account['num_pending_orders']}")
    
    def _generate_signals(self, current_time: pd.Timestamp) -> Dict[str, float]:
        """Generate trading signals."""
        signals = {}
        
        for symbol in self.symbols:
            if symbol not in self.latest_data:
                continue
            
            # Skip if we generated signals too recently
            if symbol in self.last_signal_time:
                time_since_last = current_time - self.last_signal_time[symbol]
                if time_since_last < pd.Timedelta(minutes=5):  # Minimum 5 min between signals
                    continue
            
            # Generate signals using available generators
            symbol_signals = []
            for name, generator in self.signal_generators.items():
                try:
                    # Would need historical data for proper signal generation
                    # This is simplified
                    signal = 0.0  # Placeholder
                    symbol_signals.append(signal)
                except Exception as e:
                    logger.warning(f"Failed to generate {name} signal for {symbol}: {e}")
            
            if symbol_signals:
                combined_signal = np.mean(symbol_signals)
                if abs(combined_signal) > 0.1:  # Minimum signal strength
                    signals[symbol] = combined_signal
                    self.last_signal_time[symbol] = current_time
        
        return signals
    
    async def _generate_orders(
        self,
        signals: Dict[str, float],
        current_prices: Dict[str, float],
        account: Dict[str, Any]
    ) -> None:
        """Generate orders based on signals."""
        portfolio_value = account['total_equity']
        
        for symbol, signal in signals.items():
            if symbol not in current_prices:
                continue
            
            current_position = account['positions'].get(symbol, 0.0)
            current_price = current_prices[symbol]
            
            # Calculate target position size
            target_size, sizing_details = self.position_sizer.calculate_size(
                signal_strength=signal,
                price=current_price,
                volatility=0.2,  # Would calculate from historical data
                portfolio_value=portfolio_value
            )
            
            # Calculate order size
            order_size = target_size - current_position
            
            if abs(order_size) < 0.01:  # Minimum order size
                continue
            
            # Submit order
            side = 'buy' if order_size > 0 else 'sell'
            order_id = self.broker.submit_order(
                symbol=symbol,
                side=side,
                quantity=abs(order_size),
                order_type=OrderType.MARKET
            )
            
            logger.info(f"Generated order: {symbol} {side} {abs(order_size):.2f} @ market")
    
    def _update_equity_tracking(self, account: Dict[str, Any]) -> None:
        """Update equity curve tracking."""
        equity_point = {
            'timestamp': account['timestamp'],
            'equity': account['total_equity'],
            'cash': account['cash'],
            'unrealized_pnl': account['unrealized_pnl'],
            'num_positions': len(account['positions'])
        }
        
        self.equity_history.append(equity_point)
        
        # Keep only recent history to manage memory
        if len(self.equity_history) > 10000:
            self.equity_history = self.equity_history[-5000:]
    
    def _generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily trading report."""
        account = self.broker.get_account_summary()
        trades_today = self.broker.get_trade_history(days=1)
        
        # Calculate daily P&L
        daily_pnl = 0.0
        if self.equity_history and len(self.equity_history) > 1:
            today_start = self.equity_history[0]['equity']
            current_equity = account['total_equity']
            daily_pnl = current_equity - today_start
        
        report = {
            'date': now_utc().date(),
            'starting_equity': self.equity_history[0]['equity'] if self.equity_history else account['total_equity'],
            'ending_equity': account['total_equity'],
            'daily_pnl': daily_pnl,
            'daily_return': daily_pnl / self.equity_history[0]['equity'] if self.equity_history else 0.0,
            'trades_count': len(trades_today),
            'total_commission': sum(trade.commission for trade in trades_today),
            'positions': account['positions'],
            'cash': account['cash'],
            'risk_events': self.risk_manager.get_risk_report()
        }
        
        self.daily_reports.append(report)
        
        logger.info(f"Daily Report: P&L=${daily_pnl:,.2f} ({report['daily_return']:.2%}), "
                   f"Trades={len(trades_today)}, Equity=${account['total_equity']:,.2f}")
        
        return report
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.equity_history:
            return {}
        
        equity_series = pd.Series([point['equity'] for point in self.equity_history])
        returns = equity_series.pct_change().dropna()
        
        return {
            'total_return': (equity_series.iloc[-1] / equity_series.iloc[0] - 1) if len(equity_series) > 1 else 0.0,
            'volatility': returns.std() * np.sqrt(252 * 24 * 60 / 5),  # Assuming 5-min data
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252 * 24 * 60 / 5) if returns.std() > 0 else 0.0,
            'max_drawdown': (equity_series.expanding().max() - equity_series).max() / equity_series.expanding().max().max(),
            'total_trades': len(self.broker.trades),
            'win_rate': len([t for t in self.broker.trades if t.price > 0]) / len(self.broker.trades) if self.broker.trades else 0.0
        }


async def run_paper_trading(
    symbols: List[str],
    signal_generators: Dict,
    initial_cash: float = 100000.0,
    duration_hours: int = 24
) -> PaperTradingEngine:
    """Run paper trading for specified duration."""
    engine = PaperTradingEngine(initial_cash, symbols, signal_generators)
    
    await engine.start()
    
    try:
        # Run for specified duration
        end_time = now_utc() + pd.Timedelta(hours=duration_hours)
        
        while now_utc() < end_time and engine.is_running:
            await engine.process_trading_cycle()
            await asyncio.sleep(30)  # Process every 30 seconds
        
    except KeyboardInterrupt:
        logger.info("Paper trading interrupted by user")
    
    finally:
        engine.stop()
    
    return engine