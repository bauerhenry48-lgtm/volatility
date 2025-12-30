"""Main data processing functionality."""

from datetime import datetime
from typing import Tuple, Optional
import pandas as pd
import os
import shutil
import time
from synth.miner.sigma_calculation import calculate_optimal_sigma

def load_or_create_sigma_data(output_filename: str) -> Tuple[Optional[pd.DataFrame], Optional[datetime]]:
    """Load existing sigma CSV or prepare for new calculation."""
    existing_sigma_df = None
    start_from_time = None
    
    try:
        existing_sigma_df = pd.read_csv(output_filename, index_col=0, parse_dates=True)
        if len(existing_sigma_df) > 0:
            # Get the most recent time (first row since it's sorted with most recent first)
            start_from_time = existing_sigma_df.index[0]
            # Remove the first row (most recent time)
            existing_sigma_df = existing_sigma_df.iloc[1:].copy()
            print(f"[INFO] Found {len(existing_sigma_df)} sigmas. Will recalculate from {start_from_time}.")
    except FileNotFoundError:
        print(f"++++++ No existing sigma CSV found. Creating new file.")
        start_from_time = None
    
    return existing_sigma_df, start_from_time


def process_sigma_calculation(symbol: str, price_data: pd.DataFrame, output_filename: str) -> pd.DataFrame:
    """Calculate and save sigma data."""
      
    existing_sigma_df, start_from_time = load_or_create_sigma_data(output_filename)
    
    # Calculate new sigmas starting from the deleted time using the loaded data
    min_time_for_calc = start_from_time
    print(f"[CALCULATE] Calculating new sigmas starting from {min_time_for_calc if min_time_for_calc else 'historical data'}...")
    new_sigma_df = calculate_optimal_sigma(symbol, price_data, min_time=min_time_for_calc)
    
    # For sigma data, order matters (most recent first), so we need to rewrite
    # But we'll use a safe approach with temporary files   
    # Combine existing and new sigmas
    # Note: new_sigma_df is always a DataFrame (never None) from calculate_optimal_sigma
    if existing_sigma_df is not None and not existing_sigma_df.empty:
        # Both exist: combine new sigmas (most recent first) + existing sigmas (oldest first)
        if not new_sigma_df.empty:
            combined_df = pd.concat([new_sigma_df, existing_sigma_df])
        else:
            # Only existing sigmas have data
            combined_df = existing_sigma_df
    else:
        # No existing sigmas, use new sigmas (even if empty)
        combined_df = new_sigma_df
    
    # Remove oldest rows equal to the number of new rows added to maintain file size
    num_new_rows = len(new_sigma_df)-1 if not new_sigma_df.empty else 0
    if num_new_rows > 0 and len(combined_df) > num_new_rows:
        # Remove oldest rows from the end (since structure is: new_sigma_df (recent) + existing_sigma_df (older))
        combined_df = combined_df.iloc[:-num_new_rows].copy()
        print(f"[SAVE] Removed {num_new_rows} oldest sigma data points to maintain file size")
    
    # Write to temporary file first, then atomically replace original (safer)
    temp_file = output_filename + '.tmp'
    combined_df.to_csv(temp_file)
    
    # Atomically replace the original file
    # On Windows, file handles may not be immediately released, so we use retry logic
    max_retries = 5
    retry_delay = 0.1  # 100ms delay between retries
    
    for attempt in range(max_retries):
        try:
            if os.name == 'nt':  # Windows
                # On Windows, use shutil.move which handles file handles better
                if os.path.exists(output_filename):
                    os.remove(output_filename)
                # Small delay to ensure file handle is released
                if attempt > 0:
                    time.sleep(retry_delay)
                shutil.move(temp_file, output_filename)
            else:  # Unix-like
                # On Unix, rename is atomic
                os.rename(temp_file, output_filename)
            break  # Success, exit retry loop
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                # Wait a bit longer before retrying
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                # Last attempt failed
                print(f"[ERROR] Failed to replace sigma file after {max_retries} attempts: {e}")
                # Clean up temp file if rename failed
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                raise
        except Exception as e:
            print(f"[ERROR] Failed to replace sigma file: {e}")
            # Clean up temp file if rename failed
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            raise
    
    return combined_df


def process_symbol(symbol: str, fetch_new_data: bool = True) -> pd.DataFrame:
    """
    Process a single symbol: load data, prepare it, calculate sigma, and calculate EWM.
    Splits data into 5 groups based on timestamp.minute % 5 and processes each group separately.
    
    Returns:
        Combined sigma DataFrame (from remainder 0 group)
    """
    from synth.miner.data_loader import load_or_fetch_data
    from synth.miner.ewm_calculation import calculate_volatility_ewm_by_weekday, calculate_sigma_ewm_by_weekday
    
    # Load price data
    data = load_or_fetch_data(symbol, fetch_new_data=fetch_new_data)
    print(f"[LOAD] Loaded {len(data)} price data points")

    # Create output filenames with remainder suffix
    sigma_filename = f'{symbol.lower()}_sigma.csv'
    volatility_filename = f'{symbol.lower()}_weekday_volatility.csv'
    sigma_ewm_filename = f'{symbol.lower()}_weekday_sigma.csv'
    # Process sigma calculation
    # filter data to only include 5-minute intervals
    data = data[data.index.minute % 5 == 0]
    combined_df = process_sigma_calculation(symbol, data, sigma_filename)
    
    # Calculate EWM by weekday
    calculate_volatility_ewm_by_weekday(symbol, data, output_file=volatility_filename)
    calculate_sigma_ewm_by_weekday(symbol, combined_df, output_file=sigma_ewm_filename)

    return combined_df

