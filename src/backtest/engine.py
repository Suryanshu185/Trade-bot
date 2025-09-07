"""Event-driven backtesting engine with realistic execution simulation."""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger

from ..data.features import FeatureEngine
from ..models.signals import BaseSignalGenerator
from ..exec.position_sizing import PositionSizer
from ..exec.risk import RiskManager, Position, RiskEvent
from ..utils.config import get_config
from ..utils.time import TimeManager, now_utc


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Trading order."""
    id: str
    symbol: str
    order_type: OrderType
    side: str  # 'buy' or 'sell'
    quantity: float
    price: Optional[float]  # None for market orders
    timestamp: pd.Timestamp
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Trade:
    """Executed trade."""
    id: str
    symbol: str
    timestamp: pd.Timestamp
    side: str
    quantity: float
    price: float
    commission: float
    slippage: float
    signal_strength: float = 0.0
    strategy: str = ""


@dataclass
class PortfolioState:
    """Portfolio state at a point in time."""
    timestamp: pd.Timestamp
    cash: float
    positions: Dict[str, float]  # symbol -> quantity
    equity: float
    returns: float
    drawdown: float


class ExecutionEngine:
    """Handles order execution with realistic simulation."""
    
    def __init__(
        self,
        commission_rate: float = 0.001,
        slippage_bps: float = 2.0,
        market_impact_factor: float = 0.1,
        partial_fill_prob: float = 0.05
    ):
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.market_impact_factor = market_impact_factor
        self.partial_fill_prob = partial_fill_prob
        
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self._order_counter = 0
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        self._order_counter += 1
        return f"order_{self._order_counter:06d}"
    
    def submit_order(
        self,
        symbol: str,
        order_type: OrderType,
        side: str,
        quantity: float,
        price: Optional[float] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> str:
        """Submit trading order."""
        order_id = self._generate_order_id()
        
        order = Order(
            id=order_id,
            symbol=symbol,
            order_type=order_type,
            side=side,
            quantity=abs(quantity),
            price=price,
            timestamp=timestamp or now_utc()
        )
        
        self.orders[order_id] = order
        return order_id
    
    def process_orders(
        self,
        current_prices: Dict[str, float],
        current_time: pd.Timestamp,
        volumes: Optional[Dict[str, float]] = None
    ) -> List[Trade]:
        """Process pending orders and return executed trades."""
        executed_trades = []
        
        for order_id, order in list(self.orders.items()):
            if order.status != OrderStatus.PENDING:
                continue
            
            if order.symbol not in current_prices:
                continue
            
            current_price = current_prices[order.symbol]
            executed_trade = self._try_execute_order(order, current_price, current_time, volumes)
            
            if executed_trade:
                executed_trades.append(executed_trade)
                self.trades.append(executed_trade)
        
        return executed_trades
    
    def _try_execute_order(
        self,
        order: Order,
        current_price: float,
        current_time: pd.Timestamp,
        volumes: Optional[Dict[str, float]] = None
    ) -> Optional[Trade]:
        """Try to execute a single order."""
        # Check if order should be executed
        should_execute = False
        
        if order.order_type == OrderType.MARKET:
            should_execute = True
        elif order.order_type == OrderType.LIMIT:
            if order.side == 'buy' and current_price <= order.price:
                should_execute = True
            elif order.side == 'sell' and current_price >= order.price:
                should_execute = True
        
        if not should_execute:
            return None
        
        # Calculate execution details
        execution_price = self._calculate_execution_price(order, current_price)
        
        # Check for partial fills
        fill_quantity = order.quantity
        if np.random.random() < self.partial_fill_prob:
            fill_quantity = order.quantity * np.random.uniform(0.5, 0.9)
        
        # Calculate commission and slippage
        commission = fill_quantity * execution_price * self.commission_rate
        slippage_amount = execution_price * (self.slippage_bps / 10000)
        
        # Adjust price for slippage
        if order.side == 'buy':
            final_price = execution_price + slippage_amount
        else:
            final_price = execution_price - slippage_amount
        
        # Update order status
        order.filled_quantity += fill_quantity
        order.avg_fill_price = final_price
        order.commission += commission
        order.slippage += slippage_amount
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Create trade
        trade = Trade(
            id=f"trade_{len(self.trades) + 1:06d}",
            symbol=order.symbol,
            timestamp=current_time,
            side=order.side,
            quantity=fill_quantity,
            price=final_price,
            commission=commission,
            slippage=slippage_amount
        )
        
        return trade
    
    def _calculate_execution_price(self, order: Order, current_price: float) -> float:
        """Calculate realistic execution price."""
        if order.order_type == OrderType.MARKET:
            return current_price
        elif order.order_type == OrderType.LIMIT:
            return order.price
        else:
            return current_price
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id in self.orders and self.orders[order_id].status == OrderStatus.PENDING:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 2.0,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ):
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.start_date = start_date
        self.end_date = end_date
        
        # Initialize components
        self.execution_engine = ExecutionEngine(commission_rate, slippage_bps)
        self.position_sizer = PositionSizer()
        self.risk_manager = RiskManager()
        self.time_manager = TimeManager()
        
        # Portfolio state
        self.cash = initial_cash
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.equity_curve: List[PortfolioState] = []
        
        # Performance tracking
        self.peak_equity = initial_cash
        self.max_drawdown = 0.0
        
        # Data and signals
        self.data: Dict[str, pd.DataFrame] = {}
        self.signal_generators: Dict[str, BaseSignalGenerator] = {}
        
    def add_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Add price data for a symbol."""
        # Ensure data is sorted by timestamp
        data = data.sort_index()
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")
        
        self.data[symbol] = data
        logger.info(f"Added data for {symbol}: {len(data)} rows from {data.index[0]} to {data.index[-1]}")
    
    def add_signal_generator(self, name: str, generator: BaseSignalGenerator) -> None:
        """Add signal generator."""
        self.signal_generators[name] = generator
    
    def run_backtest(self) -> Dict[str, Any]:
        """Run the complete backtest."""
        logger.info("Starting backtest...")
        
        if not self.data:
            raise ValueError("No data provided for backtesting")
        
        # Determine date range
        all_dates = []
        for symbol_data in self.data.values():
            all_dates.extend(symbol_data.index)
        
        all_dates = pd.Index(all_dates).unique().sort_values()
        start_date = self.start_date or all_dates[0]
        end_date = self.end_date or all_dates[-1]
        
        # Filter data to date range
        backtest_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
        
        logger.info(f"Backtesting from {start_date} to {end_date} ({len(backtest_dates)} periods)")
        
        # Train signal generators on initial data
        self._train_signal_generators(start_date)
        
        # Initialize time manager for backtesting
        self.time_manager = TimeManager(start_date)
        
        # Run simulation
        for i, current_time in enumerate(backtest_dates):
            self.time_manager.set_time(current_time)
            self._process_timestamp(current_time)
            
            # Log progress
            if i % 1000 == 0:
                progress = (i / len(backtest_dates)) * 100
                logger.info(f"Backtest progress: {progress:.1f}%")
        
        # Final portfolio update
        self._update_portfolio_state(backtest_dates[-1])
        
        logger.info("Backtest completed")
        
        # Generate results
        return self._generate_backtest_results()
    
    def _train_signal_generators(self, start_date: pd.Timestamp) -> None:
        """Train signal generators on initial data."""
        logger.info("Training signal generators...")
        
        for name, generator in self.signal_generators.items():
            # Get training data (use first symbol for now)
            if self.data:
                first_symbol = list(self.data.keys())[0]
                training_data = self.data[first_symbol][self.data[first_symbol].index < start_date]
                
                if len(training_data) > 100:  # Ensure sufficient training data
                    # Add features if needed
                    feature_engine = FeatureEngine()
                    training_data_with_features = feature_engine.compute_all_features(training_data)
                    
                    generator.fit(training_data_with_features)
                    logger.info(f"Trained {name} generator on {len(training_data)} samples")
    
    def _process_timestamp(self, current_time: pd.Timestamp) -> None:
        """Process a single timestamp in the backtest."""
        # Get current market data
        current_prices = {}
        current_data = {}
        
        for symbol, data in self.data.items():
            if current_time in data.index:
                row = data.loc[current_time]
                current_prices[symbol] = row['close']
                current_data[symbol] = row
        
        if not current_prices:
            return
        
        # Process pending orders
        executed_trades = self.execution_engine.process_orders(current_prices, current_time)
        
        # Update positions from executed trades
        for trade in executed_trades:
            self._apply_trade(trade)
        
        # Update risk manager
        self.risk_manager.update_prices(current_prices)
        
        # Check for risk events
        risk_events = self.risk_manager.update_prices(current_prices)
        for symbol, event in risk_events:
            self._handle_risk_event(symbol, event, current_prices[symbol], current_time)
        
        # Generate new signals
        signals = self._generate_signals(current_time, current_data)
        
        # Generate new orders based on signals
        if not self.risk_manager.kill_switch_active:
            self._generate_orders(signals, current_prices, current_time)
        
        # Update portfolio state
        self._update_portfolio_state(current_time)
    
    def _generate_signals(self, current_time: pd.Timestamp, current_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Generate trading signals for current timestamp."""
        signals = {}
        
        for symbol in current_data.keys():
            # Get historical data up to current time
            symbol_data = self.data[symbol][self.data[symbol].index <= current_time]
            
            if len(symbol_data) < 50:  # Need sufficient history
                continue
            
            # Add features
            feature_engine = FeatureEngine()
            try:
                data_with_features = feature_engine.compute_all_features(symbol_data)
                
                # Get signals from all generators
                symbol_signals = {}
                for name, generator in self.signal_generators.items():
                    try:
                        signal_series = generator.generate_signals(data_with_features)
                        if not signal_series.empty and current_time in signal_series.index:
                            symbol_signals[name] = signal_series.loc[current_time]
                    except Exception as e:
                        logger.warning(f"Failed to generate {name} signal for {symbol}: {e}")
                
                # Combine signals (simple average for now)
                if symbol_signals:
                    combined_signal = np.mean(list(symbol_signals.values()))
                    signals[symbol] = combined_signal
                    
            except Exception as e:
                logger.warning(f"Failed to process features for {symbol}: {e}")
        
        return signals
    
    def _generate_orders(self, signals: Dict[str, float], current_prices: Dict[str, float], current_time: pd.Timestamp) -> None:
        """Generate orders based on signals."""
        portfolio_value = self._calculate_portfolio_value(current_prices)
        
        for symbol, signal in signals.items():
            if abs(signal) < 0.1:  # Minimum signal threshold
                continue
            
            current_position = self.positions.get(symbol, 0.0)
            current_price = current_prices[symbol]
            
            # Calculate target position size
            # Simple implementation - can be enhanced with position sizer
            target_size = signal * 0.1 * portfolio_value / current_price  # 10% max position
            
            # Calculate trade size
            trade_size = target_size - current_position
            
            if abs(trade_size) < 0.01:  # Minimum trade size
                continue
            
            # Submit order
            side = 'buy' if trade_size > 0 else 'sell'
            self.execution_engine.submit_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=side,
                quantity=abs(trade_size),
                timestamp=current_time
            )
    
    def _apply_trade(self, trade: Trade) -> None:
        """Apply executed trade to portfolio."""
        # Update cash
        trade_value = trade.quantity * trade.price
        if trade.side == 'buy':
            self.cash -= trade_value + trade.commission
            position_change = trade.quantity
        else:
            self.cash += trade_value - trade.commission
            position_change = -trade.quantity
        
        # Update position
        current_position = self.positions.get(trade.symbol, 0.0)
        new_position = current_position + position_change
        
        if abs(new_position) < 1e-8:  # Close to zero
            self.positions.pop(trade.symbol, None)
        else:
            self.positions[trade.symbol] = new_position
    
    def _handle_risk_event(self, symbol: str, event: RiskEvent, price: float, timestamp: pd.Timestamp) -> None:
        """Handle risk events by closing positions."""
        if symbol in self.positions:
            position_size = self.positions[symbol]
            side = 'sell' if position_size > 0 else 'buy'
            
            # Submit market order to close position
            self.execution_engine.submit_order(
                symbol=symbol,
                order_type=OrderType.MARKET,
                side=side,
                quantity=abs(position_size),
                timestamp=timestamp
            )
    
    def _calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if symbol in current_prices:
                position_value += quantity * current_prices[symbol]
        
        return self.cash + position_value
    
    def _update_portfolio_state(self, timestamp: pd.Timestamp) -> None:
        """Update portfolio state tracking."""
        # Get current prices for portfolio valuation
        current_prices = {}
        for symbol in self.positions.keys():
            if symbol in self.data and timestamp in self.data[symbol].index:
                current_prices[symbol] = self.data[symbol].loc[timestamp, 'close']
        
        # Calculate current equity
        current_equity = self._calculate_portfolio_value(current_prices)
        
        # Calculate returns
        if self.equity_curve:
            prev_equity = self.equity_curve[-1].equity
            returns = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
        else:
            returns = (current_equity - self.initial_cash) / self.initial_cash
        
        # Calculate drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            drawdown = 0.0
        else:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Create portfolio state
        state = PortfolioState(
            timestamp=timestamp,
            cash=self.cash,
            positions=self.positions.copy(),
            equity=current_equity,
            returns=returns,
            drawdown=drawdown
        )
        
        self.equity_curve.append(state)
    
    def _generate_backtest_results(self) -> Dict[str, Any]:
        """Generate comprehensive backtest results."""
        if not self.equity_curve:
            return {"error": "No equity curve data"}
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame([
            {
                'timestamp': state.timestamp,
                'equity': state.equity,
                'returns': state.returns,
                'drawdown': state.drawdown,
                'cash': state.cash
            }
            for state in self.equity_curve
        ]).set_index('timestamp')
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame([
            {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'price': trade.price,
                'value': trade.quantity * trade.price,
                'commission': trade.commission,
                'slippage': trade.slippage
            }
            for trade in self.execution_engine.trades
        ])
        
        return {
            'equity_curve': equity_df,
            'trades': trades_df,
            'initial_cash': self.initial_cash,
            'final_equity': equity_df['equity'].iloc[-1] if not equity_df.empty else self.initial_cash,
            'max_drawdown': self.max_drawdown,
            'total_trades': len(self.execution_engine.trades),
            'total_commission': sum(trade.commission for trade in self.execution_engine.trades),
            'total_slippage': sum(trade.slippage for trade in self.execution_engine.trades)
        }