"""Price path simulation for forecasting."""

from datetime import datetime, timedelta
from typing import Dict, Optional
import numpy as np
from synth.miner.symbol_config import get_volatility_scaling_factor
from synth.miner.lookup_tables import get_sigma_for_time, get_volatility_for_time


# def simulate_single_price_path_for_forecast(
#     current_price: float,
#     time_increment: int,
#     time_length: int,
#     start_time: datetime,
#     sigma_lookup: Dict[tuple, float],
#     volatility_lookup: Dict[tuple, float],
#     symbol: str,
#     historical_sigma_df: Optional[np.ndarray] = None,
#     drift: float = 0.0
# ) -> np.ndarray:
#     """Simulate a single price path for forecasting."""
#     num_steps = int(time_length / time_increment)
#     price_change_pcts = np.zeros(num_steps)
#     weekday_returns = np.ones(num_steps)
    
#     current_datetime = start_time
#     weekday = current_datetime.strftime('%A')
    
#     # Load lookback sigma values from historical sigma data
#     # Use the third last and second last sigma values from the CSV file
#     lookback_sigma = np.zeros(2)
    
#     if historical_sigma_df is not None and len(historical_sigma_df) >= 3:
#         # Get third last sigma (index -3)
#         lookback_sigma[1] = float(historical_sigma_df.iloc[-3]['sigma'])
#         # Get second last sigma (index -2)
#         lookback_sigma[0] = float(historical_sigma_df.iloc[-2]['sigma'])
    
#     current_sigma = get_sigma_for_time(sigma_lookup, weekday, current_datetime)
    
#     for step in range(num_steps):
#         # Update sigma for current time
#         current_sigma = get_sigma_for_time(sigma_lookup, weekday, current_datetime)

#         if step == 0:
#             current_sigma = (current_sigma * 1 + (lookback_sigma[0] * 2 + lookback_sigma[1]) / 3) / 2
#         if step == 1:
#             current_sigma = (current_sigma * 2 + lookback_sigma[0]) / 3

#         # Get volatility for current 5-minute step
#         # volatility = get_volatility_for_time(volatility_lookup, weekday, current_datetime)

#         # If total steps < 6, scale volatility by num_steps/6, otherwise use full volatility
#         # steps_per_30min = 6  # 30 minutes / 5 minutes per step
#         # volatility_scaling_factor = get_volatility_scaling_factor(symbol)
#         # if num_steps < steps_per_30min:
#         #     weekday_returns[step] = 1 + volatility * (num_steps / steps_per_30min) / volatility_scaling_factor
#         # else:
#         #     weekday_returns[step] = 1 + volatility / volatility_scaling_factor

#         # Generate random price change using sigma
#         price_change_pcts[step] = np.random.normal(0, current_sigma)

#         # Move to next time step
#         current_datetime += timedelta(seconds=time_increment)
#         weekday = current_datetime.strftime('%A')
    
#     # Calculate cumulative returns
#     cumulative_returns = np.cumprod(1 + price_change_pcts + drift)
#     cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    
#     # # Apply weekday returns
#     # weekday_returns = np.insert(weekday_returns, 0, 1.0)
    
#     # Calculate final price path
#     price_path = current_price * cumulative_returns #* weekday_returns
    
#     return price_path


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
    """Generate price paths with final prices symmetrically normally distributed around current_price."""
    num_steps = int(time_length / time_increment)
    
    # Estimate the standard deviation for final prices based on the width
    # For a normal distribution, ~95% of values fall within 2 std dev
    # So width * current_price ≈ 2 * std_dev_final (conservative estimate)
    std_dev_final = (width * current_price) / 2.0
    
    # Generate target final prices from a symmetric normal distribution
    target_final_prices = np.random.normal(current_price, std_dev_final, size=num_simulations + 10)
    
    # Clip to reasonable bounds (within 2 * width to ensure reasonable paths)
    max_deviation = width * 2 * current_price
    target_final_prices = np.clip(target_final_prices, 
                                   current_price - max_deviation, 
                                   current_price + max_deviation)
    # remove the 5 largest and 5 smallest values
    target_final_prices = np.sort(target_final_prices)[5:-5]
    print(len(target_final_prices))
    
    valid_paths = []
    
    for target_final_price in target_final_prices:
        # Generate a path that ends close to the target final price
        price_path = simulate_single_price_path_with_target(
            current_price, target_final_price, time_increment, time_length, start_time,
            sigma_lookup, volatility_lookup, symbol, historical_sigma_df, drift, num_steps
        )
        valid_paths.append(price_path)
    
    return np.array(valid_paths)


def simulate_single_price_path_with_target(
    current_price: float,
    target_final_price: float,
    time_increment: int,
    time_length: int,
    start_time: datetime,
    sigma_lookup: Dict[tuple, float],
    volatility_lookup: Dict[tuple, float],
    symbol: str,
    historical_sigma_df: Optional[np.ndarray] = None,
    drift: float = 0.0,
    num_steps: Optional[int] = None
) -> np.ndarray:
    """Simulate a price path that ends close to the target final price using a guided random walk."""
    if num_steps is None:
        num_steps = int(time_length / time_increment)
    
    if num_steps == 0:
        return np.array([current_price])
    
    price_change_pcts = np.zeros(num_steps)
    
    current_datetime = start_time
    weekday = current_datetime.strftime('%A')
    
    # Load lookback sigma values from historical sigma data
    lookback_sigma = np.zeros(2)
    if historical_sigma_df is not None and len(historical_sigma_df) >= 3:
        lookback_sigma[1] = float(historical_sigma_df.iloc[-3]['sigma'])
        lookback_sigma[0] = float(historical_sigma_df.iloc[-2]['sigma'])
    
    # Calculate the required total return to reach target
    target_total_return = target_final_price / current_price
    
    # Calculate the average drift per step needed to reach target (geometric mean)
    if target_total_return > 0:
        required_drift_per_step = (target_total_return ** (1.0 / num_steps)) - 1.0
    else:
        required_drift_per_step = 0.0
    
    # Generate the path with a combination of random walk and slight bias toward target
    for step in range(num_steps):
        # Update sigma for current time
        current_datetime = start_time + timedelta(seconds=step * time_increment)
        current_sigma = get_sigma_for_time(sigma_lookup, weekday, current_datetime)
        
        if step == 0:
            current_sigma = (current_sigma * 1 + (lookback_sigma[0] * 2 + lookback_sigma[1]) / 3) / 2
        if step == 1:
            current_sigma = (current_sigma * 2 + lookback_sigma[0]) / 3
        
        # Generate random price change using sigma (stochastic component)
        random_change = np.random.normal(0, current_sigma)
        
        # Add a small bias toward the target that diminishes as we progress
        # This ensures paths naturally converge toward target while maintaining randomness
        remaining_steps = num_steps - step
        bias_factor = max(0.0, 0.2 * (remaining_steps / num_steps))  # Decreases from 0.2 to 0
        
        # Combine random walk with slight bias toward target
        price_change_pcts[step] = (1 - bias_factor) * random_change + bias_factor * required_drift_per_step
        
        # Move to next time step
        current_datetime += timedelta(seconds=time_increment)
        weekday = current_datetime.strftime('%A')
    
    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + price_change_pcts + drift)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    
    # Calculate final price path
    price_path = current_price * cumulative_returns
    
    return price_path

