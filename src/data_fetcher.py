import os
import pandas as pd
from entsoe import EntsoePandasClient
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

def fetch_entsoe_prices(country_code='DE_LU', start_date='2024-01-01', end_date='2024-01-02'):
    """
    Fetches historical Day-Ahead prices from ENTSO-E using key from .env.
    """
    api_key = os.getenv('ENTSOE_API_KEY')
    
    if not api_key:
        raise ValueError("API Key not found. Ensure ENTSOE_API_KEY is set in your .env file.")
        
    client = EntsoePandasClient(api_key=api_key)
    
    start = pd.Timestamp(start_date, tz='UTC')
    end = pd.Timestamp(end_date, tz='UTC')
    
    try:
        ts = client.query_day_ahead_prices(country_code, start=start, end=end)
        df = ts.to_frame(name='price')
        
        # Save raw data to avoid redundant API hits
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv('data/raw/entsoe_prices.csv')
        return df
    except Exception as e:
        print(f"Error fetching data from ENTSO-E: {e}")
        return None