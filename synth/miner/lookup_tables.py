"""Lookup table management for sigma and volatility values."""

from typing import Dict, Tuple, Optional
from datetime import datetime
import pandas as pd
import time

def load_sigma_and_volatility_lookup(symbol: str) -> Tuple[Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    """Load sigma and volatility lookup tables from CSV files.
    
    Args:
        symbol: Symbol name (e.g., 'BTC', 'XAU')
        remainder: Optional remainder (0-4) to load remainder-specific files. If None, uses default files.
    
    Returns:
        Tuple of (sigma_lookup, volatility_lookup) dictionaries
    """


    
    # Load sigma data (30-minute intervals) with retry logic
    sigma_filename = f'{symbol.lower()}_weekday_sigma.csv'
    sigma_df = None
    max_retries = 100
    for attempt in range(max_retries):
        try:
            sigma_df = pd.read_csv(sigma_filename)
            if len(sigma_df) == 0:
                raise pd.errors.EmptyDataError(f"File {sigma_filename} is empty")
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed to load {sigma_filename}: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                print(f"++++++ Failed to load {sigma_filename} after {max_retries} attempts: {e}")
                raise  # Re-raise the exception if all retries failed
    
    if sigma_df is None:
        raise RuntimeError(f"++++++ Failed to load {sigma_filename} after {max_retries} attempts")
    
    sigma_lookup = {(row['weekday'], row['time']): row['ewm12_sigma'] for _, row in sigma_df.iterrows()}
    
    # Load volatility data (5-minute intervals) with retry logic
    volatility_filename = f'{symbol.lower()}_weekday_volatility.csv'
    volatility_df = None
    for attempt in range(max_retries):
        try:
            volatility_df = pd.read_csv(volatility_filename)
            if len(volatility_df) == 0:
                raise pd.errors.EmptyDataError(f"File {volatility_filename} is empty")
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed to load {volatility_filename}: {e}. Retrying...")
                time.sleep(1)  # Wait 1 second before retrying
            else:
                print(f"++++++ Failed to load {volatility_filename} after {max_retries} attempts: {e}")
                raise  # Re-raise the exception if all retries failed
    
    if volatility_df is None:
        raise RuntimeError(f"++++++ Failed to load {volatility_filename} after {max_retries} attempts")
    
    volatility_lookup = {(row['weekday'], row['time']): row['ewm12_volatility'] for _, row in volatility_df.iterrows()}
    
    return sigma_lookup, volatility_lookup


def get_sigma_for_time(sigma_lookup: Dict[Tuple[str, str], float], weekday: str, time: datetime) -> float:
    """Get sigma value for a specific weekday and time."""
    minute = 30 if time.minute >= 30 else 0
    time_str = f"{time.hour:02d}:{minute:02d}"
    return sigma_lookup.get((weekday, time_str), 0.0)


def get_volatility_for_time(volatility_lookup: Dict[Tuple[str, str], float], weekday: str, time: datetime) -> float:
    """Get volatility value for a specific weekday and time."""
    minute = (time.minute // 5) * 5
    time_str = f"{time.hour:02d}:{minute:02d}"
    return volatility_lookup.get((weekday, time_str), 0.0)


