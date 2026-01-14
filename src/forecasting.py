import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import os


def generate_quantile_forecasts(input_file, output_file):
    
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    df['hour'] = df.index.hour
    X = df[['hour']].values
    y = df['price_eur_mwh'].values
    
    quantiles = [0.25, 0.50, 0.75]
    forecasts = pd.DataFrame(index=df.index)
    forecasts['actual'] = y
    
    for q in quantiles:
        model = GradientBoostingRegressor(
            loss='quantile', alpha=q, n_estimators=100, max_depth=3
        )
        model.fit(X,y)
        forecasts[f'q_{q}'] = model.predict(X)
        
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    forecasts.to_csv(output_file)
    print("generated quantile forecasts and saved to", output_file)
    return forecasts

def plot_risk_bands(df):
    plt.figure(figsize=(12, 6))
    plt.fill_between(df.index[:168], df['q_0.25'][:168], df['q_0.75'][:168], 
                     alpha=0.3, color='orange', label='Risk Band (IQR)')
    plt.plot(df.index[:168], df['q_0.5'][:168], color='red', label='Median Forecast')
    plt.scatter(df.index[:168], df['actual'][:168], color='black', s=10, label='Actual Data', alpha=0.5)
    plt.title("Electricity Price Risk Bands (First Week)")
    plt.ylabel("EUR/MWh")
    plt.legend()
    plt.savefig('results/price_risk_bands.png')
    plt.show()
    
if __name__ == "__main__":
    results = generate_quantile_forecasts('data/raw/de_lu_market_data.csv', 
                                          'data/processed/quantile_forecasts.csv')
    plot_risk_bands(results)