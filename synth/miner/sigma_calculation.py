"""Sigma calculation and optimization functionality."""

from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from synth.miner.config import TIME_INCREMENT_5MIN, TIME_INCREMENT_30MIN
from synth.miner.symbol_config import get_symbol_config
from synth.miner.utils import round_to_30_minutes, convert_to_datetime
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.miner.xau_market_closure import (
    insert_xau_closure_period_data,
    is_xau_weekend_closure
)


def _interpolate_missing_5min_intervals(data: pd.DataFrame) -> pd.DataFrame:
    """Fill in missing 5-minute intervals using linear interpolation.
    
    Creates a complete datetime index with 5-minute intervals from the first
    to last timestamp in the data, and fills missing values using linear interpolation.
    
    Args:
        data: DataFrame with datetime index and 'price' column
        
    Returns:
        DataFrame with complete 5-minute intervals and interpolated prices
    """
    if len(data) == 0:
        return data
    
    # Get the first and last timestamps
    first_time = data.index[0]
    last_time = data.index[-1]
    
    # Create a complete datetime index with 5-minute intervals
    complete_index = pd.date_range(
        start=first_time,
        end=last_time,
        freq='5min'
    )
    
    # Reindex the data to include all 5-minute intervals
    data_reindexed = data.reindex(complete_index)
    
    # Interpolate missing values linearly
    data_reindexed['price'] = data_reindexed['price'].interpolate(method='linear')
    
    return data_reindexed


def calculate_optimal_sigma(
    symbol: str,
    data: pd.DataFrame,
    min_time: Optional[datetime] = None
) -> pd.DataFrame:
    """Calculate optimal sigma values for a symbol."""
    symbol_data = get_symbol_config(symbol)
    
    # For XAU, preprocess data to interpolate prices during closure periods

    if symbol == 'XAU':
        print(f"[SIGMA] Inserting missing data for XAU...")
        data = insert_xau_closure_period_data(data)
    
    # Fill in missing 5-minute intervals using linear interpolation
    print(f"[SIGMA] Interpolating missing 5-minute intervals for {symbol}...")
    data = _interpolate_missing_5min_intervals(data)
    length = len(data['price'])
    
    # Get the last datetime from the data and round down to nearest 30 minutes
    last_time = convert_to_datetime(data.index[-1])
    start_time = round_to_30_minutes(last_time)
    
    # Convert min_time to datetime if provided
    min_datetime = None
    if min_time is not None:
        min_datetime = round_to_30_minutes(convert_to_datetime(min_time))
    
    optimal_sigmas = []
    datetime_indexes = []
    
    # Calculate total number of iterations for progress bar
    total_iterations = 0
    temp_count = 0
    for i in range(length, -1, -6):
        dt_index = start_time - timedelta(minutes=30 * temp_count)
        if min_datetime is not None and dt_index < min_datetime:
            break
        total_iterations += 1
        temp_count += 1
    
    # Iterate backwards through the data in chunks of 6
    iteration_count = 0
    print(f"[SIGMA] Calculating sigmas for {symbol}...")
    
    # Create progress bar
    with tqdm(total=total_iterations, desc=f"[SIGMA] {symbol}", unit="iter", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for i in range(length, -1, -6):
            offset = length - i
            dt_index = start_time - timedelta(minutes=30 * iteration_count)
            
            # Only calculate if >= min_time (if specified)
            if min_datetime is not None and dt_index < min_datetime:
                break
            
            # For XAU, handle different closure periods
            if symbol == 'XAU':
                if is_xau_weekend_closure(dt_index):
                    best_sigma = 0.0
                    optimal_sigmas.append(best_sigma)
                    datetime_indexes.append(dt_index)
                    iteration_count += 1
                    pbar.update(1)
                    continue
            
            # Find optimal sigma
            best_sigma, _ = _find_optimal_sigma(data, offset, symbol_data)
            
            optimal_sigmas.append(best_sigma)
            datetime_indexes.append(dt_index)
            iteration_count += 1
            pbar.update(1)
    
    # Create DataFrame with datetime index and sigma values
    if optimal_sigmas:
        sigma_df = pd.DataFrame({
            'sigma': optimal_sigmas
        }, index=pd.DatetimeIndex(datetime_indexes, name='time'))
        return sigma_df
    else:
        return pd.DataFrame(columns=['sigma'])


def _find_optimal_sigma(data: pd.DataFrame, offset: int, symbol_data: dict) -> tuple:
    """Find the optimal sigma value for a given offset."""
    best_sigma = 0
    best_crps = 1000000
    
    for s in range(0, symbol_data['search_count']):
        sigma = round(
            symbol_data['origin_sigma'] + s * symbol_data['sigma_interval'],
            symbol_data['round_num']
        )
        crps = calculate_optimal_crps(data, sigma, offset)
        
        if crps > 0 and crps < best_crps:
            best_sigma = sigma
            best_crps = crps
    
    return best_sigma, best_crps


def calculate_optimal_crps(data: pd.DataFrame, sigma: float, offset: int) -> float:
    """Calculate optimal CRPS for a given sigma and offset."""
    before_index = len(data['price']) - offset - 7
    if before_index < 0:
        return 0

    close_price = data['price'].iloc[before_index]
    real_path = data['price'][before_index: before_index + 7]
    
    # Generate 6 steps to match real_path (7 prices = 6 intervals)
    price_paths = simulate_price_paths_for_sigma(
        close_price, TIME_INCREMENT_5MIN, TIME_INCREMENT_30MIN, 500, sigma
    )
    numeric_value, _ = calculate_crps_for_miner(price_paths, np.array(real_path), TIME_INCREMENT_5MIN)
    return numeric_value


def simulate_single_price_path_for_sigma(
    current_price: float,
    time_increment: int,
    time_length: int,
    sigma: float
) -> np.ndarray:
    """Simulate a single price path using random normal distribution."""
    num_steps = int(time_length / time_increment)
    price_change_pcts = np.random.normal(0, sigma, size=num_steps)
    cumulative_returns = np.cumprod(1 + price_change_pcts)
    cumulative_returns = np.insert(cumulative_returns, 0, 1.0)
    return current_price * cumulative_returns


def simulate_price_paths_for_sigma(
    current_price: float,
    time_increment: int,
    time_length: int,
    num_simulations: int,
    sigma: float
) -> np.ndarray:
    """Generate multiple price paths for a given sigma."""
    price_paths = [
        simulate_single_price_path_for_sigma(current_price, time_increment, time_length, sigma)
        for _ in range(num_simulations)
    ]
    return np.array(price_paths)

