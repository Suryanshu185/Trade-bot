"""Configuration management using Pydantic settings."""

from typing import Dict, List, Optional
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    
    name: str = "binance"
    sandbox: bool = True
    api_key: Optional[str] = None
    secret: Optional[str] = None
    rate_limit: int = 1200  # requests per minute
    timeout: int = 30  # seconds


class TradingConfig(BaseModel):
    """Trading configuration."""
    
    symbols: List[str] = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
    timeframes: List[str] = ["1m", "5m", "1h"]
    primary_timeframe: str = "5m"
    
    # Risk parameters
    max_position_size: float = Field(default=0.1, ge=0.01, le=1.0)  # fraction of equity
    max_daily_loss: float = Field(default=0.02, ge=0.001, le=0.1)  # 2% daily loss limit
    max_risk_per_trade: float = Field(default=0.005, ge=0.001, le=0.05)  # 0.5% per trade
    target_volatility: float = Field(default=0.15, ge=0.05, le=0.5)  # 15% annualized
    kelly_cap: float = Field(default=0.25, ge=0.1, le=0.5)  # Kelly fraction cap
    
    # Strategy parameters
    momentum_lookback: int = Field(default=20, ge=5, le=100)
    mean_reversion_lookback: int = Field(default=14, ge=5, le=50)
    volatility_lookback: int = Field(default=20, ge=10, le=100)
    
    # ML parameters
    ml_probability_threshold: float = Field(default=0.6, ge=0.5, le=0.95)
    ml_lookback_periods: int = Field(default=100, ge=50, le=500)


class BacktestConfig(BaseModel):
    """Backtesting configuration."""
    
    start_date: str = "2021-01-01"
    end_date: str = "2024-01-01"
    initial_cash: float = Field(default=100000.0, ge=1000.0)
    
    # Cost model
    commission: float = Field(default=0.001, ge=0.0, le=0.01)  # 0.1% commission
    slippage_bps: float = Field(default=2.0, ge=0.0, le=10.0)  # 2 bps slippage
    
    # Walk-forward optimization
    train_months: int = Field(default=6, ge=3, le=24)
    test_months: int = Field(default=1, ge=1, le=6)


class DataConfig(BaseModel):
    """Data configuration."""
    
    cache_dir: Path = Path("data/cache")
    data_dir: Path = Path("data/raw")
    results_dir: Path = Path("data/results")
    
    # Data refresh settings
    refresh_interval_hours: int = Field(default=1, ge=1, le=24)
    max_cache_age_days: int = Field(default=7, ge=1, le=30)


class Config(BaseSettings):
    """Main configuration class."""
    
    model_config = SettingsConfigDict(
        env_prefix="TRADE_BOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Environment
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configs
    exchange: ExchangeConfig = ExchangeConfig()
    trading: TradingConfig = TradingConfig()
    backtest: BacktestConfig = BacktestConfig()
    data: DataConfig = DataConfig()
    
    def __init__(self, config_file: Optional[Path] = None, **kwargs) -> None:
        """Initialize config, optionally loading from YAML file."""
        if config_file and config_file.exists():
            import yaml
            with open(config_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            super().__init__(**{**yaml_data, **kwargs})
        else:
            super().__init__(**kwargs)
    
    def save_to_yaml(self, file_path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, indent=2)


# Global config instance
config = Config()


def load_config(config_file: Optional[Path] = None) -> Config:
    """Load configuration from file or environment."""
    global config
    config = Config(config_file=config_file)
    return config


def get_config() -> Config:
    """Get the current configuration."""
    return config