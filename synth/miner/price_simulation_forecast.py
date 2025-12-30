"""Price path simulation for forecasting."""

from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
from synth.miner.symbol_config import get_volatility_scaling_factor
from synth.miner.lookup_tables import get_sigma_for_time, get_volatility_for_time


def simulate_single_price_path_for_forecast(
    current_price: float,
    time_increment: int,
    time_length: int,
    start_time: datetime,
    sigma_lookup: Dict[tuple, float],
    volatility_lookup: Dict[tuple, float],
    symbol: str,
    historical_sigma_df: Optional[np.ndarray] = None,
    drift: float = 0.0
) -> np.ndarray:
    """Simulate a single price path for forecasting."""
    num_steps = int(time_length / time_increment)
    price_change_pcts = np.zeros(num_steps)
    weekday_returns = np.ones(num_steps)
    
    current_datetime = start_time
    weekday = current_datetime.strftime('%A')
    
    # Load lookback sigma values from historical sigma data
    # Use the third last and second last sigma values from the CSV file
    lookback_sigma = np.zeros(2)
    
    if historical_sigma_df is not None and len(historical_sigma_df) >= 3:
        # Get third last sigma (index -3)
        lookback_sigma[1] = float(historical_sigma_df.iloc[-3]['sigma'])
        # Get second last sigma (index -2)
        lookback_sigma[0] = float(historical_sigma_df.iloc[-2]['sigma'])
    
    current_sigma = get_sigma_for_time(sigma_lookup, weekday, current_datetime)
    
    for step in range(num_steps):
        # Update sigma for current time
        current_sigma = get_sigma_for_time(sigma_lookup, weekday, current_datetime)

        if step == 0:
            current_sigma = (current_sigma * 1 + (lookback_sigma[0] * 2 + lookback_sigma[1]) / 3) / 2
        if step == 1:
            current_sigma = (current_sigma * 2 + lookback_sigma[0]) / 3

        # Get volatility for current 5-minute step
        volatility = get_volatility_for_time(volatility_lookup, weekday, current_datetime)

        # If total steps < 6, scale volatility by num_steps/6, otherwise use full volatility
        steps_per_30min = 6  # 30 minutes / 5 minutes per step
        volatility_scaling_factor = get_volatility_scaling_factor(symbol)
        if num_steps < steps_per_30min:
            weekday_returns[step] = 1 + volatility * (num_steps / steps_per_30min) / volatility_scaling_factor
        else:
            weekday_returns[step] = 1 + volatility / volatility_scaling_factor

        # Generate random price change using sigma
        price_change_pcts[step] = np.random.normal(0, current_sigma)

        # Move to next time step
        current_datetime += timedelta(seconds=time_increment)
        weekday = current_datetime.strftime('%A')
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + price_change_pcts + drift)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    
    # Apply weekday returns
    weekday_returns = np.insert(weekday_returns, 0, 1.0)
    
    # Calculate final price path
    price_path = current_price * cumulative_returns * weekday_returns
    
    return price_path


def simulate_crypto_price_paths_for_forecast(
    current_price: float,
    time_increment: int,
    time_length: int,
    start_time: datetime,
    num_simulations: int,
    sigma_lookup: Dict[tuple, float],
    volatility_lookup: Dict[tuple, float],
    width: float,
    symbol: str,
    historical_sigma_df: Optional[np.ndarray] = None,
    drift: float = 0.0
) -> np.ndarray:
    """Generate valid price paths within price bounds."""
    valid_paths = []
    min_price = (1 - width) * current_price
    max_price = (1 + width) * current_price
    extended_min_price = (1 - width * 1.3) * current_price
    extended_max_price = (1 + width * 1.3) * current_price
    
    while len(valid_paths) < num_simulations:
        price_path = simulate_single_price_path_for_forecast(
            current_price, time_increment, time_length, start_time,
            sigma_lookup, volatility_lookup, symbol, historical_sigma_df, drift
        )
        # Check if final price is within bounds and any price is within extended bounds
        final_price_valid = min_price <= price_path[-1] <= max_price
        any_price_valid = np.any((extended_min_price <= price_path) & (price_path <= extended_max_price))
        if final_price_valid and any_price_valid:
            valid_paths.append(price_path)
    
    return np.array(valid_paths)

