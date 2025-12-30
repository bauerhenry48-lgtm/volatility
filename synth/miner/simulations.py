from synth.miner.price_simulation import get_asset_price
from synth.utils.helpers import convert_prices_to_time_format
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from synth.miner.config import TIME_INCREMENT_5MIN, DEFAULT_FORECAST_LENGTH, PRESENT_NUM_SIMULATIONS
from synth.miner.symbol_config import get_min_max_multiplier
from synth.miner.lookup_tables import load_sigma_and_volatility_lookup
from synth.miner.price_simulation_forecast import simulate_crypto_price_paths_for_forecast
from synth.validator.crps_calculation import calculate_crps_for_miner
from synth.miner.xau_market_closure import is_xau_market_closure, adjust_time_for_xau_closure
from synth.miner.symbol_config import get_week_for_width_calculation
from synth.miner.data_loader import load_or_fetch_data

def generate_simulations(
    asset="BTC",
    start_time: str = "",
    time_increment=300,
    time_length=86400,
    num_simulations=1,
    sigma=0.01,
):
    """
    Generate simulated price paths.

    Parameters:
        asset (str): The asset to simulate. Default is 'BTC'.
        start_time (str): The start time of the simulation. Defaults to current time.
        time_increment (int): Time increment in seconds.
        time_length (int): Total time length in seconds.
        num_simulations (int): Number of simulation runs.
        sigma (float): Standard deviation of the simulated price path.

    Returns:
        numpy.ndarray: Simulated price paths.
    """
    if start_time == "":
        raise ValueError("Start time must be provided.")

    start_time_dt = datetime.fromisoformat(start_time)

    # Wait until current time almost arrives at start time
    # def_generation_time = 30
    # current_time_dt = datetime.now(timezone.utc)
    # delay = (start_time_dt - current_time_dt).total_seconds()
    # delay = max(delay - def_generation_time, 0)
    # time.sleep(delay)

    # Retry loading data up to 100 times until it succeeds
    data = None
    max_retries = 100
    for attempt in range(max_retries):
        try:
            data = load_or_fetch_data(asset, fetch_new_data=False)
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed to load data: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                print(f"[ERROR] Failed to load data after {max_retries} attempts: {e}")
                raise  # Re-raise the exception if all retries failed
    
    if data is None:
        raise RuntimeError(f"Failed to load data for asset {asset} after {max_retries} attempts")

    current_price = get_asset_price(asset)
    if current_price is None:
        current_price = data['price'].iloc[-1]

    simulations = generate_optimized_paths(asset, data, start_time_dt, current_price, time_increment, time_length, num_simulations)

    predictions = convert_prices_to_time_format(
        simulations.tolist(), start_time, time_increment
    )

    return predictions

def generate_optimized_paths(
    asset: str,
    data: pd.DataFrame,
    start_time: datetime,
    start_price: float,
    time_increment,
    time_length,
    num_simulations,
):
    
    # Load sigma and volatility lookup tables based on remainder
    sigma_lookup, volatility_lookup = load_sigma_and_volatility_lookup(asset)
    
    # Load historical sigma data for lookback sigma values based on remainder
    historical_sigma_df = _load_historical_sigma(asset)

    # calculate end time
    end_time = start_time + timedelta(seconds=time_length)
    
    # Calculate width based on historical 1-hour moving averages
    width = calculate_forecast_width(data, start_time, end_time, asset)
    
    # Simulate price paths
    drift = 0.0
    simulations = simulate_crypto_price_paths_for_forecast(
        start_price, time_increment, time_length, start_time,
        num_simulations, sigma_lookup, volatility_lookup,
        width, asset, historical_sigma_df, drift
    )

    return simulations

def plot_price_prediction_paths(
    data: pd.DataFrame,
    symbol: str,
    timepoints: Optional[List[datetime]] = None,
    num_simulations: int = 100
):
    time_increment = TIME_INCREMENT_5MIN
    forecast_length = DEFAULT_FORECAST_LENGTH
    data_is_aware = data.index.tz is not None
    
    # Determine which timepoints to use
    forecast_timepoints = _prepare_timepoints(timepoints, data_is_aware)
    fetch_start_time = datetime.now(timezone.utc)
    one_minute_data = load_or_fetch_data(symbol, fetch_new_data=False)
    if len(one_minute_data) == 0:
        print(f"[PLOT] Warning: No 1-minute data available for timepoint {fetch_start_time}. Skipping.")
        return
    # Process each timepoint
    for timepoint_idx, current_start in enumerate(forecast_timepoints):

        # Get the starting price
        # start_price = _get_start_price(data, current_start)
        # Fetch 6 days of 1-minute data for this timepoint
        
        # Get the starting price from 1-minute data (closest price at or before current_start)
        available_data = one_minute_data[one_minute_data.index <= current_start]
        if len(available_data) == 0:
            print(f"[PLOT] Warning: No 1-minute data available before {current_start}. Skipping.")
            continue
        
        start_price = available_data['price'].iloc[-1]
        
        current_end = current_start + timedelta(seconds=forecast_length)
        
        # Get the real price path from 1-minute data for comparison
        real_path_data = one_minute_data[
            (one_minute_data.index >= current_start) & 
            (one_minute_data.index <= current_end)
        ]
        print(f"Current start: {current_start}, Current end: {current_end}")
        
        if len(real_path_data) == 0:
            print(f"[PLOT] Warning: No real price data available for timepoint {current_start}. Skipping.")
            continue
        
        # Simulate price paths (still using 5-minute intervals)
        price_paths = generate_optimized_paths(
            symbol, data, current_start, start_price,
            time_increment, forecast_length, num_simulations
        )
        
        # Calculate CRPS using 1-minute actual data
        # Note: We need to align 1-minute actual data with 5-minute simulations
        real_prices_padded, num_expected_steps = _prepare_real_prices_for_one_minute(
            real_path_data, current_start, time_increment, price_paths.shape[1]
        )
        
        numeric_value, _ = calculate_crps_for_miner(
            price_paths, real_prices_padded, time_increment
        )
        
        # Create and save plot
        min_max_multiplier = get_min_max_multiplier(symbol)
        _create_plot(
            price_paths, real_path_data, current_start, time_increment,
            num_expected_steps, numeric_value, min_max_multiplier,
            symbol, timepoint_idx
        )
        
        print(f"[PLOT] CRPS for timepoint {timepoint_idx + 1} ({current_end.strftime('%Y-%m-%d %H:%M:%S')}): {numeric_value:.6f}")
    
    print(f"[PLOT] Generated {len(forecast_timepoints)} plots for {symbol}")

def calculate_forecast_width(
    data: pd.DataFrame,
    current_start: datetime,
    current_end: datetime,
    symbol: str
) -> float:
    # Calculate forecast width based on historical 1-hour moving averages.
    # For XAU, adjust current_end if it falls during market closure
    if symbol == 'XAU':
        if is_xau_market_closure(current_end):
            current_end = adjust_time_for_xau_closure(current_end, symbol)
            current_end = current_end + timedelta(hours=1)
        if is_xau_market_closure(current_start):
            weekday = current_start.weekday()
            if weekday <= 4:
                current_start = current_start.replace(hour=22, minute=0, second=0, microsecond=0)
            elif weekday == 5:
                current_start = (current_start - timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
            elif weekday == 6:
                current_start = (current_start - timedelta(days=2)).replace(hour=22, minute=0, second=0, microsecond=0)

    weeks_for_width_calculation = get_week_for_width_calculation(symbol)
    width_values = []
    
    for i in range(1, weeks_for_width_calculation + 1):
        # Calculate time points i weeks ago
        start_time_i = current_start - timedelta(weeks=i)
        end_time_i = current_end - timedelta(weeks=i)
        
        # Get 1-hour window of data before each time point for moving average
        ma_window_start = start_time_i - timedelta(hours=1)
        ma_window_end_start = start_time_i
        ma_window_end = end_time_i - timedelta(hours=1)
        ma_window_end_end = end_time_i
        
        # Get price data for 1-hour moving average at start_time_i
        ma_start_data = data[
            (data.index >= ma_window_start) & 
            (data.index <= ma_window_end_start)
        ]
        
        # Get price data for 1-hour moving average at end_time_i
        ma_end_data = data[
            (data.index >= ma_window_end) & 
            (data.index <= ma_window_end_end)
        ]
        
        # Calculate 1-hour moving averages
        if len(ma_start_data) > 0 and len(ma_end_data) > 0:
            ma_cs_i = ma_start_data['price'].mean()
            ma_ce_i = ma_end_data['price'].mean()
            
            # Calculate absolute difference ratio
            if ma_ce_i > 0:
                ratio = abs(ma_cs_i - ma_ce_i) / ma_ce_i
                width_values.append(ratio)
    
    # Get the third largest value among all calculated ratios
    min_max_multiplier = get_min_max_multiplier(symbol)
    if len(width_values) >= 3:
        width_values_sorted = sorted(width_values, reverse=True)
        width = width_values_sorted[3] * min_max_multiplier
    else:
        # Fallback if we don't have enough values
        width = 0.025
    return width


def _load_historical_sigma(symbol: str) -> Optional[pd.DataFrame]:
    """Load historical sigma data from CSV.
    
    Args:
        symbol: Symbol name (e.g., 'BTC', 'XAU')
    
    Returns:
        DataFrame with historical sigma data or None if file not found
    """
    
    filename = f'{symbol.lower()}_sigma.csv'
    
    # Retry loading data up to 100 times until it succeeds (for transient errors)
    historical_sigma_df = None
    max_retries = 100
    for attempt in range(max_retries):
        try:
            historical_sigma_df = pd.read_csv(filename, index_col=0, parse_dates=True)
            print(f"Loaded {len(historical_sigma_df)} sigma values from {filename}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed to load {filename}: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                print(f"++++++ [ERROR] Failed to load {filename} after {max_retries} attempts: {e}")
                raise  # Re-raise the exception if all retries failed
    
    if historical_sigma_df is None:
        raise RuntimeError(f"++++++ Failed to load {filename} after {max_retries} attempts")

    return historical_sigma_df

def _prepare_timepoints(
    timepoints: Optional[List[datetime]],
    data_is_aware: bool,
) -> List[datetime]:
    """Prepare timepoints for forecasting."""
    if timepoints is not None:
        if len(timepoints) < 1:
            raise ValueError(f"++++++ Timepoints must be at least 1, but got {len(timepoints)}")
        
        forecast_timepoints = []
        for tp in timepoints:
            tp = tp.replace(tzinfo=timezone.utc) - timedelta(days=1)
            # tp = tp.replace(tzinfo=None) - timedelta(days=1)
            forecast_timepoints.append(tp)
        return forecast_timepoints

def _prepare_real_prices_for_one_minute(
    real_path_data: pd.DataFrame,
    current_start: datetime,
    time_increment: int,
    num_expected_steps: int
) -> tuple:
    """
    Prepare real prices from 1-minute data for CRPS calculation.
    This function aligns 1-minute actual data with 5-minute simulation intervals
    by taking the price at each 5-minute mark from the 1-minute data.
    """
    real_path_data_clean = real_path_data['price'].dropna()
    
    if len(real_path_data_clean) == 0:
        return None, num_expected_steps
    
    # Pad real_prices to match simulation length with NaN for missing time points
    real_prices_padded = np.full(num_expected_steps, np.nan)
    
    # For each simulation step (5-minute interval), find the closest 1-minute data point
    for step_index in range(num_expected_steps):
        target_time = current_start + timedelta(seconds=step_index * time_increment)
        
        # Find the closest 1-minute data point to this target time
        # Get data points within a small window around the target time
        window_start = target_time - timedelta(seconds=time_increment // 2)
        window_end = target_time + timedelta(seconds=time_increment // 2)
        
        window_data = real_path_data_clean[
            (real_path_data_clean.index >= window_start) & 
            (real_path_data_clean.index <= window_end)
        ]
        
        if len(window_data) > 0:
            # Use the price closest to the target time
            time_diffs = pd.Series(
                abs((window_data.index - target_time).total_seconds()),
                index=window_data.index
            )
            closest_idx = time_diffs.idxmin()
            real_prices_padded[step_index] = real_path_data_clean.loc[closest_idx]
        else:
            # If no data in window, try to find the closest overall
            time_diffs = pd.Series(
                abs((real_path_data_clean.index - target_time).total_seconds()),
                index=real_path_data_clean.index
            )
            if len(time_diffs) > 0:
                closest_idx = time_diffs.idxmin()
                # Only use if within reasonable range (e.g., within 2.5 minutes)
                if time_diffs.loc[closest_idx] <= time_increment / 2:
                    real_prices_padded[step_index] = real_path_data_clean.loc[closest_idx]
    
    return real_prices_padded, num_expected_steps


def _create_plot(
    price_paths: np.ndarray,
    real_path_data: pd.DataFrame,
    current_start: datetime,
    time_increment: int,
    num_expected_steps: int,
    numeric_value: float,
    min_max_multiplier: float,
    symbol: str,
    timepoint_idx: int
):
    """Create and save a plot for a timepoint."""
    # Create time axis for plotting
    time_steps = np.arange(num_expected_steps) * time_increment / 3600  # Convert to hours
    time_points = [current_start + timedelta(seconds=i * time_increment) for i in range(num_expected_steps)]
    
    # Get real price path
    real_path_data_clean = real_path_data['price'].dropna()
    real_prices = real_path_data_clean.values
    real_timestamps = real_path_data_clean.index
    real_time_steps = [(ts - current_start).total_seconds() / 3600 for ts in real_timestamps]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    
    # Plot simulated paths
    colors = plt.cm.tab20(np.linspace(0, 1, PRESENT_NUM_SIMULATIONS))
    for i in range(PRESENT_NUM_SIMULATIONS):
        plt.plot(time_steps, price_paths[i], color=colors[i], alpha=0.5, linewidth=0.5, markersize=0.5, marker='o')
    
    # Plot real price path
    plt.plot(real_time_steps, real_prices, 'r-', linewidth=1.5, label='Real Price Path', zorder=10, marker='o', markersize=1.5)
    
    # Add labels and title
    plt.xlabel('Time (hours from start)', fontsize=13)
    plt.ylabel('Price', fontsize=13)
    plt.title(f'{symbol} Price Prediction Paths - Timepoint {timepoint_idx + 1}\n{current_start.strftime("%Y-%m-%d %H:%M:%S")}', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    # Add CRPS and min_max_multiplier as text box
    textstr = f'CRPS: {numeric_value:.6f}\nmin_max_multiplier: {min_max_multiplier}'
    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5))
    
    # Format x-axis
    ax = plt.gca()
    num_labels = min(12, len(time_points))
    step = max(1, len(time_points) // num_labels)
    x_ticks = time_steps[::step]
    x_labels = [time_points[i].strftime("%H:%M") for i in range(0, len(time_points), step)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'{symbol.lower()}_price_paths_timepoint_{timepoint_idx + 1}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')

