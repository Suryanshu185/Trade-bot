"""I/O utilities for data caching and persistence."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd
from loguru import logger

from .config import get_config


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(
    df: pd.DataFrame, 
    file_path: Union[str, Path], 
    format: str = "parquet"
) -> None:
    """Save DataFrame to file with specified format."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    if format == "parquet":
        df.to_parquet(file_path, compression="snappy")
    elif format == "csv":
        df.to_csv(file_path, index=True)
    elif format == "feather":
        df.reset_index().to_feather(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.debug(f"Saved DataFrame to {file_path} ({len(df)} rows)")


def load_dataframe(
    file_path: Union[str, Path], 
    format: str = "parquet"
) -> Optional[pd.DataFrame]:
    """Load DataFrame from file, return None if not exists."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        if format == "parquet":
            df = pd.read_parquet(file_path)
        elif format == "csv":
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif format == "feather":
            df = pd.read_feather(file_path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.debug(f"Loaded DataFrame from {file_path} ({len(df)} rows)")
        return df
    
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    """Save dictionary to JSON file."""
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    logger.debug(f"Saved JSON to {file_path}")


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Load dictionary from JSON file, return None if not exists."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.debug(f"Loaded JSON from {file_path}")
        return data
    
    except Exception as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


class DataCache:
    """Centralized data caching with automatic cleanup."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        config = get_config()
        self.cache_dir = cache_dir or config.data.cache_dir
        self.max_age_days = config.data.max_cache_age_days
        ensure_dir(self.cache_dir)
    
    def _get_cache_path(self, key: str, format: str = "parquet") -> Path:
        """Generate cache file path from key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        extension = {"parquet": ".parquet", "csv": ".csv", "json": ".json"}[format]
        return self.cache_dir / f"{safe_key}{extension}"
    
    def get(self, key: str, format: str = "parquet") -> Optional[Union[pd.DataFrame, Dict]]:
        """Get cached data by key."""
        cache_path = self._get_cache_path(key, format)
        
        if not cache_path.exists():
            return None
        
        # Check age
        age_days = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit='s')).days
        if age_days > self.max_age_days:
            logger.debug(f"Cache expired for {key} (age: {age_days} days)")
            cache_path.unlink()
            return None
        
        if format == "json":
            return load_json(cache_path)
        else:
            return load_dataframe(cache_path, format)
    
    def set(
        self, 
        key: str, 
        data: Union[pd.DataFrame, Dict], 
        format: str = "parquet"
    ) -> None:
        """Cache data by key."""
        cache_path = self._get_cache_path(key, format)
        
        if format == "json":
            save_json(data, cache_path)
        else:
            save_dataframe(data, cache_path, format)
    
    def invalidate(self, key: str, format: str = "parquet") -> None:
        """Remove cached data by key."""
        cache_path = self._get_cache_path(key, format)
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Invalidated cache for {key}")
    
    def cleanup(self) -> None:
        """Remove expired cache files."""
        if not self.cache_dir.exists():
            return
        
        removed_count = 0
        for cache_file in self.cache_dir.glob("*"):
            age_days = (pd.Timestamp.now() - pd.Timestamp(cache_file.stat().st_mtime, unit='s')).days
            if age_days > self.max_age_days:
                cache_file.unlink()
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned up {removed_count} expired cache files")


class ResultsWriter:
    """Writer for backtest and trading results."""
    
    def __init__(self, results_dir: Optional[Path] = None):
        config = get_config()
        self.results_dir = results_dir or config.data.results_dir
        ensure_dir(self.results_dir)
    
    def save_backtest_results(
        self, 
        strategy_name: str,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        metrics: Dict[str, Any],
        run_id: Optional[str] = None
    ) -> Path:
        """Save complete backtest results."""
        if run_id is None:
            run_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = self.results_dir / strategy_name / run_id
        ensure_dir(run_dir)
        
        # Save equity curve
        save_dataframe(equity_curve, run_dir / "equity_curve.parquet")
        
        # Save trades
        save_dataframe(trades, run_dir / "trades.parquet")
        
        # Save metrics
        save_json(metrics, run_dir / "metrics.json")
        
        logger.info(f"Saved backtest results to {run_dir}")
        return run_dir
    
    def save_live_trades(self, trades: pd.DataFrame, date: Optional[str] = None) -> Path:
        """Save live trading results."""
        if date is None:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        file_path = self.results_dir / "live" / f"trades_{date}.parquet"
        save_dataframe(trades, file_path)
        
        logger.info(f"Saved live trades to {file_path}")
        return file_path
    
    def save_daily_report(self, report: Dict[str, Any], date: Optional[str] = None) -> Path:
        """Save daily trading report."""
        if date is None:
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        
        file_path = self.results_dir / "reports" / f"daily_report_{date}.json"
        save_json(report, file_path)
        
        logger.info(f"Saved daily report to {file_path}")
        return file_path