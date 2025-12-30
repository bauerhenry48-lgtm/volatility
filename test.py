"""
Cryptocurrency and asset price forecasting - Main execution script.

This script processes multiple symbols (BTC, ETH, SOL, XAU) by:
1. Loading or fetching price data
2. Preparing data (transforming XAU, calculating volatilities, detecting outliers)
3. Calculating optimal sigma values
4. Calculating EWM (Exponential Weighted Moving Average) by weekday
5. Optionally plotting price prediction paths
"""

from datetime import datetime, timezone, timedelta
from synth.miner.data_loader import load_or_fetch_data
from synth.miner.data_processing import process_symbol
from synth.miner.simulations import plot_price_prediction_paths, generate_simulations
import time

def main():
    """Main execution function."""
    # List of symbols to process
    # symbols = ['BTC']
    # symbols = ['XAU', 'SOL', 'ETH', 'BTC']
    # Process each symbol

    # while True:
    #     for symbol in symbols:
    #         print(f"\n{'='*60}")
    #         print(f"Processing {symbol}")
    #         print(f"{'='*60}")
    #         process_symbol(symbol, fetch_new_data=True)
    #     time.sleep(300)
 
    symbol = 'BTC'
    data = load_or_fetch_data(symbol, fetch_new_data=True)

    process_symbol(symbol, fetch_new_data=True)
    timepoints = [
        datetime(2025, 11, 18, 0, 37, 0),
        datetime(2025, 11, 17, 23, 1, 0),
        datetime(2025, 11, 17, 18, 29, 0),
        datetime(2025, 11, 17, 17, 0, 0),
        datetime(2025, 11, 17, 15, 26, 0),
        datetime(2025, 11, 17, 13, 57, 0),
        datetime(2025, 11, 17, 10, 57, 0),
        datetime(2025, 11, 17, 9, 28, 0),
        datetime(2025, 11, 17, 7, 59, 0),
        datetime(2025, 11, 17, 6, 30, 0),
        datetime(2025, 11, 17, 4, 59, 0),
        datetime(2025, 11, 17, 3, 29, 0),
    ] # BTC

    # Plot price prediction paths
    plot_price_prediction_paths(data, symbol, timepoints=timepoints, num_simulations=100)

    # test for live version.
    # current_utc = datetime.now(timezone.utc).replace(microsecond=0)
    # future_utc = current_utc + timedelta(minutes=0)
    # iso_string = future_utc.isoformat()
    # generate_simulations('BTC', iso_string)

if __name__ == "__main__":
    main()
