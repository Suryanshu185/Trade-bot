"""Performance metrics calculation for backtesting and live trading."""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from loguru import logger


def calculate_returns(equity_curve: pd.Series) -> pd.Series:
    """Calculate returns from equity curve."""
    return equity_curve.pct_change().fillna(0.0)


def annualize_return(returns: Union[float, pd.Series], periods_per_year: int = 252) -> Union[float, pd.Series]:
    """Annualize returns."""
    if isinstance(returns, pd.Series):
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    else:
        return returns * periods_per_year


def calculate_cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate Compound Annual Growth Rate."""
    if len(equity_curve) < 2:
        return 0.0
    
    start_value = equity_curve.iloc[0]
    end_value = equity_curve.iloc[-1]
    years = len(equity_curve) / periods_per_year
    
    if start_value <= 0 or years <= 0:
        return 0.0
    
    cagr = (end_value / start_value) ** (1 / years) - 1
    return cagr


def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Calculate annualized volatility."""
    if len(returns) < 2:
        return 0.0
    
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    if excess_returns.std() == 0:
        return 0.0
    
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """Calculate Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return np.inf if excess_returns.mean() > 0 else 0.0
    
    downside_deviation = downside_returns.std()
    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation


def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Union[float, pd.Timestamp]]:
    """Calculate maximum drawdown and related metrics."""
    if len(equity_curve) < 2:
        return {'max_drawdown': 0.0, 'max_dd_duration': pd.Timedelta(0), 'recovery_time': pd.Timedelta(0)}
    
    # Calculate running maximum (peak)
    running_max = equity_curve.expanding().max()
    
    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max
    
    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find the peak before maximum drawdown
    peak_before_max_dd = running_max.loc[:max_dd_date].idxmax()
    
    # Find recovery date (when equity reaches new high after max drawdown)
    recovery_date = None
    peak_value = running_max.loc[max_dd_date]
    
    post_dd_curve = equity_curve.loc[max_dd_date:]
    recovery_mask = post_dd_curve >= peak_value
    
    if recovery_mask.any():
        recovery_date = post_dd_curve[recovery_mask].index[0]
        recovery_time = recovery_date - peak_before_max_dd
    else:
        recovery_time = equity_curve.index[-1] - peak_before_max_dd
    
    # Calculate drawdown duration
    dd_duration = max_dd_date - peak_before_max_dd
    
    return {
        'max_drawdown': abs(max_dd),
        'max_dd_date': max_dd_date,
        'peak_before_dd': peak_before_max_dd,
        'max_dd_duration': dd_duration,
        'recovery_date': recovery_date,
        'recovery_time': recovery_time
    }


def calculate_calmar_ratio(cagr: float, max_drawdown: float) -> float:
    """Calculate Calmar ratio (CAGR / Max Drawdown)."""
    if max_drawdown == 0:
        return np.inf if cagr > 0 else 0.0
    
    return cagr / max_drawdown


def calculate_hit_rate(trades_df: pd.DataFrame) -> float:
    """Calculate hit rate (percentage of winning trades)."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return 0.0
    
    winning_trades = (trades_df['pnl'] > 0).sum()
    total_trades = len(trades_df)
    
    return winning_trades / total_trades if total_trades > 0 else 0.0


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """Calculate profit factor (gross profit / gross loss)."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return 0.0
    
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_expectancy(trades_df: pd.DataFrame) -> float:
    """Calculate expectancy per trade."""
    if trades_df.empty or 'pnl' not in trades_df.columns:
        return 0.0
    
    hit_rate = calculate_hit_rate(trades_df)
    
    winning_trades = trades_df[trades_df['pnl'] > 0]['pnl']
    losing_trades = trades_df[trades_df['pnl'] < 0]['pnl']
    
    avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
    avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0
    
    expectancy = (hit_rate * avg_win) + ((1 - hit_rate) * avg_loss)
    return expectancy


def calculate_turnover(trades_df: pd.DataFrame, avg_equity: float) -> float:
    """Calculate portfolio turnover rate."""
    if trades_df.empty or avg_equity <= 0:
        return 0.0
    
    total_traded_value = trades_df['value'].sum() if 'value' in trades_df.columns else 0.0
    
    # Annualize turnover (assuming daily data)
    days = (trades_df.index.max() - trades_df.index.min()).days if len(trades_df) > 1 else 1
    annual_traded_value = total_traded_value * (365 / days)
    
    return annual_traded_value / avg_equity


def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Value at Risk."""
    if len(returns) < 2:
        return 0.0
    
    return returns.quantile(confidence_level)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    if len(returns) < 2:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    return returns[returns <= var].mean()


def calculate_skewness(returns: pd.Series) -> float:
    """Calculate skewness of returns."""
    if len(returns) < 3:
        return 0.0
    
    return returns.skew()


def calculate_kurtosis(returns: pd.Series) -> float:
    """Calculate kurtosis of returns."""
    if len(returns) < 4:
        return 0.0
    
    return returns.kurtosis()


def calculate_information_ratio(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate Information Ratio against benchmark."""
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    tracking_error = excess_returns.std()
    
    if tracking_error == 0:
        return 0.0
    
    return excess_returns.mean() / tracking_error


def calculate_beta(returns: pd.Series, market_returns: pd.Series) -> float:
    """Calculate beta against market."""
    if len(returns) != len(market_returns) or len(returns) < 2:
        return 0.0
    
    covariance = np.cov(returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    
    if market_variance == 0:
        return 0.0
    
    return covariance / market_variance


def calculate_alpha(returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """Calculate alpha (excess return over CAPM)."""
    if len(returns) != len(market_returns) or len(returns) < 2:
        return 0.0
    
    beta = calculate_beta(returns, market_returns)
    
    portfolio_return = returns.mean() * 252  # Annualize
    market_return = market_returns.mean() * 252  # Annualize
    
    expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
    alpha = portfolio_return - expected_return
    
    return alpha


class PerformanceAnalyzer:
    """Comprehensive performance analysis."""
    
    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
    
    def analyze_backtest(
        self,
        equity_curve: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Perform comprehensive backtest analysis."""
        if len(equity_curve) < 2:
            logger.warning("Insufficient data for performance analysis")
            return {}
        
        # Calculate returns
        returns = calculate_returns(equity_curve)
        
        # Basic performance metrics
        metrics = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if equity_curve.iloc[0] > 0 else 0.0,
            'cagr': calculate_cagr(equity_curve, self.periods_per_year),
            'volatility': calculate_volatility(returns, self.periods_per_year),
            'sharpe_ratio': calculate_sharpe_ratio(returns, self.risk_free_rate, self.periods_per_year),
            'sortino_ratio': calculate_sortino_ratio(returns, self.risk_free_rate, self.periods_per_year),
        }
        
        # Drawdown metrics
        dd_metrics = calculate_max_drawdown(equity_curve)
        metrics.update({
            'max_drawdown': dd_metrics['max_drawdown'],
            'calmar_ratio': calculate_calmar_ratio(metrics['cagr'], dd_metrics['max_drawdown']),
            'max_dd_duration_days': dd_metrics['max_dd_duration'].days,
            'recovery_time_days': dd_metrics['recovery_time'].days if dd_metrics['recovery_time'] else np.nan
        })
        
        # Risk metrics
        metrics.update({
            'var_5pct': calculate_var(returns, 0.05),
            'cvar_5pct': calculate_cvar(returns, 0.05),
            'skewness': calculate_skewness(returns),
            'kurtosis': calculate_kurtosis(returns),
        })
        
        # Trade-based metrics
        if trades_df is not None and not trades_df.empty:
            # Add P&L column if not present
            if 'pnl' not in trades_df.columns and 'value' in trades_df.columns:
                trades_df['pnl'] = trades_df['value']  # Simplified
            
            metrics.update({
                'total_trades': len(trades_df),
                'hit_rate': calculate_hit_rate(trades_df),
                'profit_factor': calculate_profit_factor(trades_df),
                'expectancy': calculate_expectancy(trades_df),
                'avg_equity': equity_curve.mean(),
            })
            
            # Add turnover if possible
            metrics['turnover'] = calculate_turnover(trades_df, metrics['avg_equity'])
        
        # Benchmark comparison
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            metrics.update({
                'information_ratio': calculate_information_ratio(returns, benchmark_returns),
                'beta': calculate_beta(returns, benchmark_returns),
                'alpha': calculate_alpha(returns, benchmark_returns, self.risk_free_rate)
            })
        
        return metrics
    
    def generate_performance_report(
        self,
        equity_curve: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None
    ) -> str:
        """Generate a formatted performance report."""
        metrics = self.analyze_backtest(equity_curve, trades_df, benchmark_returns)
        
        if not metrics:
            return "Insufficient data for performance analysis"
        
        report = []
        report.append("=" * 60)
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Return metrics
        report.append("\nRETURN METRICS:")
        report.append(f"Total Return:      {metrics.get('total_return', 0) * 100:8.2f}%")
        report.append(f"CAGR:             {metrics.get('cagr', 0) * 100:8.2f}%")
        report.append(f"Volatility:       {metrics.get('volatility', 0) * 100:8.2f}%")
        
        # Risk-adjusted metrics
        report.append("\nRISK-ADJUSTED METRICS:")
        report.append(f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):8.2f}")
        report.append(f"Sortino Ratio:    {metrics.get('sortino_ratio', 0):8.2f}")
        report.append(f"Calmar Ratio:     {metrics.get('calmar_ratio', 0):8.2f}")
        
        # Drawdown metrics
        report.append("\nDRAWDOWN METRICS:")
        report.append(f"Max Drawdown:     {metrics.get('max_drawdown', 0) * 100:8.2f}%")
        report.append(f"Max DD Duration:  {metrics.get('max_dd_duration_days', 0):8.0f} days")
        
        # Risk metrics
        report.append("\nRISK METRICS:")
        report.append(f"VaR (5%):         {metrics.get('var_5pct', 0) * 100:8.2f}%")
        report.append(f"CVaR (5%):        {metrics.get('cvar_5pct', 0) * 100:8.2f}%")
        report.append(f"Skewness:         {metrics.get('skewness', 0):8.2f}")
        report.append(f"Kurtosis:         {metrics.get('kurtosis', 0):8.2f}")
        
        # Trading metrics
        if 'total_trades' in metrics:
            report.append("\nTRADING METRICS:")
            report.append(f"Total Trades:     {metrics.get('total_trades', 0):8.0f}")
            report.append(f"Hit Rate:         {metrics.get('hit_rate', 0) * 100:8.2f}%")
            report.append(f"Profit Factor:    {metrics.get('profit_factor', 0):8.2f}")
            report.append(f"Expectancy:       {metrics.get('expectancy', 0):8.2f}")
            report.append(f"Turnover:         {metrics.get('turnover', 0):8.2f}x")
        
        # Benchmark metrics
        if 'beta' in metrics:
            report.append("\nBENCHMARK COMPARISON:")
            report.append(f"Beta:             {metrics.get('beta', 0):8.2f}")
            report.append(f"Alpha:            {metrics.get('alpha', 0) * 100:8.2f}%")
            report.append(f"Information Ratio: {metrics.get('information_ratio', 0):8.2f}")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def calculate_rolling_metrics(
        self,
        returns: pd.Series,
        window: int = 252
    ) -> pd.DataFrame:
        """Calculate rolling performance metrics."""
        if len(returns) < window:
            logger.warning(f"Insufficient data for rolling metrics (need {window}, have {len(returns)})")
            return pd.DataFrame()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling Sharpe ratio
        rolling_metrics['sharpe_ratio'] = returns.rolling(window).apply(
            lambda x: calculate_sharpe_ratio(x, self.risk_free_rate, self.periods_per_year)
        )
        
        # Rolling volatility
        rolling_metrics['volatility'] = returns.rolling(window).std() * np.sqrt(self.periods_per_year)
        
        # Rolling max drawdown
        equity_curve = (1 + returns).cumprod()
        rolling_metrics['max_drawdown'] = equity_curve.rolling(window).apply(
            lambda x: calculate_max_drawdown(pd.Series(x))['max_drawdown']
        )
        
        return rolling_metrics.dropna()


def create_performance_analyzer(risk_free_rate: float = 0.02) -> PerformanceAnalyzer:
    """Create performance analyzer with default parameters."""
    return PerformanceAnalyzer(risk_free_rate=risk_free_rate)