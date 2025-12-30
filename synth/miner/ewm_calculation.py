"""Exponential Weighted Moving Average (EWM) calculation functionality."""

from typing import Optional
import pandas as pd
from synth.miner.config import EWM_SPAN, MA_WINDOW, EWM_INDEX, WEEKDAY_NAMES


def calculate_volatility_ewm_by_weekday(symbol: str, data: pd.DataFrame, output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Calculate volatility EWM by weekday."""
    if output_file is None:
        output_file = f'{symbol.lower()}_weekday_volatility.csv'
    
    data = data.copy()
    data['MA14'] = data['price'].rolling(window=MA_WINDOW, min_periods=1).mean()
    data['volatility'] = (data['price'] - data['MA14']) / data['MA14']
    
    return _calculate_ewm_by_weekday(
        data, time_block_size=5, blocks_per_day=288,
        value_column='volatility', output_column='ewm12_volatility',
        output_file=output_file
    )


def calculate_sigma_ewm_by_weekday(symbol: str, sigma_df: pd.DataFrame, output_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Calculate sigma EWM by weekday."""
    if output_file is None:
        output_file = f'{symbol.lower()}_weekday_sigma.csv'
    
    return _calculate_ewm_by_weekday(
        sigma_df, time_block_size=30, blocks_per_day=48,
        value_column='sigma', output_column='ewm12_sigma',
        output_file=output_file
    )


def _calculate_ewm_by_weekday(
    data: pd.DataFrame,
    time_block_size: int,
    blocks_per_day: int,
    value_column: str,
    output_column: str,
    output_file: str
) -> Optional[pd.DataFrame]:
    """Calculate EWM by weekday and time position."""
    print(f"[EWM] Processing {value_column} data...")
    
    if data is None or len(data) == 0:
        print(f"++++++ [EWM] Error: No {value_column} data provided")
        return None
    
    data = data.copy()
    data['weekday'] = data.index.day_name()
    data['time_position'] = (data.index.hour * 60 + data.index.minute) // time_block_size
    
    # Sort by time to ensure chronological order for EWM
    data = data.sort_index()
    
    # Create structure: {day: {position: [(timestamp, value)]}}
    structure = _create_weekday_structure(blocks_per_day)
    
    # Populate the structure
    for idx, row in data.iterrows():
        day = row['weekday']
        pos = int(row['time_position'])
        if day in structure and pos < blocks_per_day and not pd.isna(row[value_column]):
            structure[day][pos].append((idx, row[value_column]))
    
    results = _calculate_ewm_values(structure, time_block_size, value_column, output_column)
    
    # Create DataFrame and save
    if results:
        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values(['weekday', 'time'])
        result_df.to_csv(output_file, index=False)
        return result_df
    else:
        print(f"++++++ [EWM] No data found for EWM calculation")
        return None


def _create_weekday_structure(blocks_per_day: int) -> dict:
    """Create empty structure for organizing data by weekday and time position."""
    structure = {}
    for day in WEEKDAY_NAMES:
        structure[day] = {}
        for pos in range(blocks_per_day):
            structure[day][pos] = []
    return structure


def _calculate_ewm_values(structure: dict, time_block_size: int, value_column: str, output_column: str) -> list:
    """Calculate EWM values for each weekday and time position."""
    results = []
    
    for day in WEEKDAY_NAMES:
        if day not in structure:
            continue
        
        for pos in range(len(structure[day])):
            value_tuples = structure[day][pos]
            if len(value_tuples) > 0:
                # Extract values in chronological order
                values = [v for _, v in value_tuples]
                timestamps = [t for t, _ in value_tuples]
                
                # Convert to pandas Series for EWM calculation
                values_series = pd.Series(values, index=timestamps)
                
                # Remove the min and max value in the values_series
                if values_series.size >= 3:
                    values_series = values_series.drop(values_series.idxmin())
                    values_series = values_series.drop(values_series.idxmax())
                else:
                    # If there are less than 3 values, set the final EWM12 to 0
                    continue
                
                # Calculate EWM12
                ewm12 = values_series.ewm(span=EWM_SPAN, adjust=False).mean()
                if len(ewm12) < abs(EWM_INDEX):
                    print(f"++++++ [EWM] Not enough data for {value_column} EWM calculation")
                    print(f"++++++ [EWM] {len(ewm12)} data points available")
                    final_ewm12 = 0.0
                else:
                    final_ewm12 = ewm12.iloc[EWM_INDEX]
                
                # Calculate time string in HH:MM format
                hour = pos * time_block_size // 60
                minute = (pos * time_block_size) % 60
                time_str = f"{hour:02d}:{minute:02d}"
                
                # Store result
                results.append({
                    'weekday': day,
                    'time': time_str,
                    output_column: final_ewm12,
                    'sample_count': len(value_tuples)
                })
    
    return results

