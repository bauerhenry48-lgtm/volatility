"""XAU market closure handling and data transformation."""

from datetime import datetime, timedelta, timezone
import pandas as pd
from synth.miner.utils import convert_to_datetime


def is_xau_market_closure(dt: datetime) -> bool:
    """
    Check if a datetime falls during XAU market closure period.
    Closure: 10:00 PM - 11:00 PM (Mon-Thu), 10:00 PM Fri - 11:00 PM Sun
    """
    dt = convert_to_datetime(dt)
    
    hour = dt.hour
    minute = dt.minute
    weekday = dt.weekday()  # 0=Monday, 4=Friday, 6=Sunday
    
    if weekday < 4:  # Monday-Thursday
        return hour == 22 and 1 <= minute <= 58
    elif weekday == 4:  # Friday
        return hour >= 22 and minute >= 1
    elif weekday == 5:  # Saturday
        return True
    elif weekday == 6:  # Sunday
        return hour < 22 or (hour == 22 and minute <= 58)
    
    return False


def is_xau_weekend_closure(dt: datetime) -> bool:
    """Check if datetime falls during XAU weekend closure (Friday 10:00 PM to Sunday 10:00 PM)."""
    dt = convert_to_datetime(dt)
    weekday = dt.weekday()
    hour = dt.hour
    
    if weekday == 4:  # Friday
        return hour >= 22  # Friday 10:00 PM onwards
    elif weekday == 5:  # Saturday
        return True
    elif weekday == 6:  # Sunday
        return hour < 22  # Up to and including Sunday 10:00 PM
    
    return False


def adjust_time_for_xau_closure(dt: datetime, symbol: str) -> datetime:
    """
    Adjust a datetime to avoid market closure periods for XAU.
    Moves the time to the last valid 5-minute interval before closure.
    """
    if symbol != 'XAU':
        return dt
    
    dt = convert_to_datetime(dt)
    
    if not is_xau_market_closure(dt):
        return dt
    
    weekday = dt.weekday()
    
    if weekday < 5:  # Monday-Friday
        result = dt.replace(hour=22, minute=0, second=0, microsecond=0)
    elif weekday >= 5:  # Saturday-Sunday
        days_back = weekday - 4  # 1 for Saturday, 2 for Sunday
        result = (dt - timedelta(days=days_back)).replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        result = dt
    
    return result


def transform_xau_data_for_market_closure(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform XAU data to align with new market closure times.
    
    Before November 2025: Market closed 9:00 PM - 10:00 PM (Mon-Thu), 9:00 PM Fri - 10:00 PM Sun
    After November 2025: Market closed 10:00 PM - 11:00 PM (Mon-Thu), 10:00 PM Fri - 11:00 PM Sun
    
    Transformation:
    For data before November 2025:
       - Stretch 8:00 PM - 9:00 PM data to 8:00 PM - 10:00 PM
       - Compress 10:00 PM - 12:00 AM data to 11:00 PM - 12:00 AM
    """
    if price_data is None or len(price_data) == 0:
        return price_data
    
    data = price_data.copy()
    data = data.sort_index()
    
    # November 2025 cutoff date
    nov_2025_cutoff = datetime(2025, 11, 1, tzinfo=timezone.utc)
    
    # Separate data before and after November 2025
    before_nov = data[data.index < nov_2025_cutoff].copy()
    after_nov = data[data.index >= nov_2025_cutoff].copy()
    print(f"[TRANSFORM] Before November 2025: {len(before_nov)} price data points")
    transformed_rows = []
    
    # Process data before November 2025
    if len(before_nov) > 0:
        for idx, row in before_nov.iterrows():
            dt = convert_to_datetime(idx)
            hour = dt.hour
            minute = dt.minute
            price_current = row['price']
            
            # Transform 8:00 PM - 9:00 PM data to 8:00 PM - 10:00 PM
            if hour == 20:
                double_minutes = minute * 2
                double_hour = 20 + (double_minutes // 60)
                double_minutes = double_minutes % 60
                
                double_dt = dt.replace(hour=double_hour, minute=double_minutes, second=0, microsecond=0)
                transformed_rows.append({'time': double_dt, 'price': price_current})
                
                after_1min_dt = dt + timedelta(minutes=1)
                intermediate_time = minute * 2 + 1
                intermediate_hour = 20 + (intermediate_time // 60)
                intermediate_minutes = intermediate_time % 60
                intermediate_dt = dt.replace(hour=intermediate_hour, minute=intermediate_minutes, second=0, microsecond=0)
                intermediate_price = (price_current + before_nov.loc[after_1min_dt, 'price']) / 2
                transformed_rows.append({'time': intermediate_dt, 'price': intermediate_price})

            # Handle 9:00 PM - 10:00 PM (market closure period)
            elif hour == 21:
                if minute == 0:
                    new_dt = dt.replace(hour=22, minute=0, second=0, microsecond=0)
                    transformed_rows.append({'time': new_dt, 'price': price_current})
            
            # Transform 10:00 PM - 12:00 AM data to 11:00 PM - 12:00 AM
            elif hour == 22:
                if minute % 2 == 0:
                    compressed_minute = int(minute / 2)
                    new_dt = dt.replace(hour=23, minute=compressed_minute, second=0, microsecond=0)
                    transformed_rows.append({'time': new_dt, 'price': price_current})
          
            elif hour == 23:
                if minute % 2 == 0:
                    compressed_minute = 30 + int(minute / 2)
                    new_dt = dt.replace(hour=23, minute=compressed_minute, second=0, microsecond=0)
                    transformed_rows.append({'time': new_dt, 'price': price_current})
            else:
                # All other times remain unchanged
                transformed_rows.append({'time': dt, 'price': row['price']})
    
    print(f"[TRANSFORM] Transformed {len(transformed_rows)} price data points from before November 2025")
    
    # Process data after November 2025 (no transformation needed)
    if len(after_nov) > 0:
        for idx, row in after_nov.iterrows():
            transformed_rows.append({'time': idx, 'price': row['price']})
    
    # Create DataFrame from transformed rows
    if not transformed_rows:
        return price_data
    
    transformed_df = pd.DataFrame(transformed_rows)
    transformed_df.set_index('time', inplace=True)
    transformed_df = transformed_df.sort_index()
    
    # Remove duplicates, keeping the first occurrence
    before_dedup = len(transformed_df)
    transformed_df = transformed_df[~transformed_df.index.duplicated(keep='first')]
    after_dedup = len(transformed_df)
    
    if before_dedup != after_dedup:
        print(f"[TRANSFORM] Removed {before_dedup - after_dedup} duplicate entries")
    
    print(f"[TRANSFORM] Final transformed data: {len(transformed_df)} price data points")
    
    return transformed_df

def insert_xau_closure_period_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Insert missing data for XAU closure periods at 5-minute intervals.
    - Mon-Thu 10 PM to 11 PM: Linear interpolation between 10 PM and 11 PM
    - Sun 10 PM to 11 PM: Use Friday 10 PM price, then interpolate to 11 PM
    - Friday 10 PM to Sunday 10 PM: No data inserted (market closed, sigma = 0)
    """
    if len(data) == 0:
        return data
    
    data = data.copy()
    data = data.sort_index()
    
    unique_dates = sorted(set(data.index.date))
    new_rows = []
    
    for date in unique_dates:
        date_data = data[data.index.date == date]
        if len(date_data) == 0:
            continue
        
        first_dt = convert_to_datetime(date_data.index[0])
        weekday = first_dt.weekday()
        
        # Monday-Thursday: Insert data for 10 PM to 11 PM
        if weekday < 4:
            new_rows.extend(_insert_weekday_closure_data(data, first_dt))
        
        # Sunday: Insert data for 10 PM to 11 PM using Friday 10 PM price
        elif weekday == 6:
            new_rows.extend(_insert_sunday_closure_data(data, first_dt, date))
    
    # Add new rows to the DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df.set_index('time', inplace=True)
        data = pd.concat([data, new_df])
        data = data.sort_index()
        data = data[~data.index.duplicated(keep='first')]
    
    return data


def _insert_weekday_closure_data(data: pd.DataFrame, first_dt: datetime) -> list:
    """Insert closure period data for weekdays."""
    new_rows = []
    day_start = first_dt.replace(hour=22, minute=0, second=0, microsecond=0)
    day_end = first_dt.replace(hour=22, minute=59, second=0, microsecond=0)
    
    # Get price at 10 PM
    start_price = None
    if day_start in data.index:
        start_price = data.loc[day_start, 'price']
    else:
        before_10pm = data[data.index <= day_start]
        if len(before_10pm) > 0:
            start_price = before_10pm['price'].iloc[-1]
    
    # Get price at 11 PM
    end_price = None
    if day_end in data.index:
        end_price = data.loc[day_end, 'price']
    else:
        after_11pm = data[data.index >= day_end]
        if len(after_11pm) > 0:
            end_price = after_11pm['price'].iloc[0]
    
    # Insert interpolated prices at 5-minute intervals
    if start_price is not None and end_price is not None:
        current_time = day_start + timedelta(minutes=5)
        while current_time < day_end:
            if current_time not in data.index:
                total_minutes = 60
                minutes_from_10pm = current_time.minute
                weight = minutes_from_10pm / total_minutes if total_minutes > 0 else 0
                interpolated_price = start_price * (1 - weight) + end_price * weight
                new_rows.append({'time': current_time, 'price': interpolated_price})
            current_time += timedelta(minutes=5)
    
    return new_rows


def _insert_sunday_closure_data(data: pd.DataFrame, first_dt: datetime, date) -> list:
    """Insert closure period data for Sunday using Friday prices."""
    new_rows = []
    
    # Find Friday 10 PM price (2 days before Sunday)
    friday_date = date - timedelta(days=2)
    friday_10pm = first_dt.replace(
        year=friday_date.year, month=friday_date.month, day=friday_date.day,
        hour=22, minute=0, second=0, microsecond=0
    )
    
    friday_price = None
    if friday_10pm in data.index:
        friday_price = data.loc[friday_10pm, 'price']
    else:
        friday_data = data[data.index.date == friday_date]
        if len(friday_data) > 0:
            friday_22h = friday_data[friday_data.index.hour == 22]
            if len(friday_22h) > 0:
                friday_price = friday_22h['price'].iloc[0]
            else:
                friday_price = friday_data['price'].iloc[-1]
    
    # Get Sunday 11 PM price
    sunday_10pm = first_dt.replace(hour=22, minute=0, second=0, microsecond=0)
    sunday_11pm = first_dt.replace(hour=23, minute=0, second=0, microsecond=0)
    
    sunday_price = None
    if sunday_11pm in data.index:
        sunday_price = data.loc[sunday_11pm, 'price']
    else:
        sunday_after_11pm = data[(data.index.date == date) & (data.index.hour >= 23)]
        if len(sunday_after_11pm) > 0:
            sunday_price = sunday_after_11pm['price'].iloc[0]
    
    # Insert Sunday 10 PM price (using Friday 10 PM price)
    if friday_price is not None:
        if sunday_10pm not in data.index:
            new_rows.append({'time': sunday_10pm, 'price': friday_price})
        
        # Insert interpolated prices from 10:05 PM to 10:55 PM
        if sunday_price is not None:
            current_time = sunday_10pm + timedelta(minutes=5)
            while current_time < sunday_11pm:
                if current_time not in data.index:
                    total_minutes = 60
                    minutes_from_10pm = current_time.minute
                    weight = minutes_from_10pm / total_minutes if total_minutes > 0 else 0
                    interpolated_price = friday_price * (1 - weight) + sunday_price * weight
                    new_rows.append({'time': current_time, 'price': interpolated_price})
                current_time += timedelta(minutes=5)
    
    return new_rows

