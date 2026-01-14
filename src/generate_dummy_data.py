import pandas as pd
import numpy as np
import os

def generate_synthetic_market_data(days=30):
    """
    Generates dummy Day-Ahead prices and load data for Germany (DE_LU).
    Mimics typical daily volatility found in ENTSO-E data.
    """
    # Create time range
    start = pd.Timestamp('2025-12-01', tz='Europe/Berlin')
    end = start + pd.Timedelta(days=days, hours=-1)
    time_index = pd.date_range(start=start, end=end, freq='h')
    
    # 1. Simulate Day-Ahead Prices (€/MWh)
    # Base price + Daily seasonality + Random volatility
    base_price = 80
    hour_effect = -15 * np.sin(2 * np.pi * (time_index.hour - 4) / 24) # Peak at 4 PM
    noise = np.random.normal(0, 10, len(time_index))
    prices = base_price + hour_effect + noise
    
    # 2. Simulate Actual Load (MW)
    base_load = 50000
    load_seasonality = 10000 * np.sin(2 * np.pi * (time_index.hour - 8) / 24)
    load_noise = np.random.normal(0, 2000, len(time_index))
    load = base_load + load_seasonality + load_noise
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': time_index,
        'price_eur_mwh': prices,
        'load_mw': load
    }).set_index('timestamp')
    
    # Ensure directory exists and save
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/de_lu_market_data.csv')
    print(f"Phase I Complete: Generated {len(df)} rows of synthetic data in data/raw/.")

if __name__ == "__main__":
    generate_synthetic_market_data(days=60) # Generate 2 months for forecasting training