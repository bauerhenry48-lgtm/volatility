"""Symbol configuration management."""

from typing import Dict
from synth.miner.define import symbol_list


def get_symbol_config(symbol: str) -> Dict:
    """Get configuration dictionary for a given symbol."""
    symbol_data = next((item for item in symbol_list if item['asset'] == symbol), None)
    if symbol_data is None:
        raise ValueError(f"Symbol {symbol} not found in symbol_list")
    return symbol_data


def get_outlier_threshold(symbol: str) -> float:
    """Get outlier threshold for the symbol."""
    config = get_symbol_config(symbol)
    return config.get('outlier_threshold', 0.01)


def get_volatility_scaling_factor(symbol: str) -> float:
    """Get volatility scaling factor for the symbol."""
    config = get_symbol_config(symbol)
    return config.get('volatility_scaling_factor', 5)


def get_week_for_width_calculation(symbol: str) -> int:
    """Get number of weeks for width calculation."""
    config = get_symbol_config(symbol)
    return config.get('week_for_width_calculation', 12)


def get_min_max_multiplier(symbol: str) -> float:
    """Get min-max multiplier for the symbol."""
    config = get_symbol_config(symbol)
    return config.get('min_max_multiplier', 2.5)

