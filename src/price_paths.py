import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

def generate_scenarios(input_file, n_samples=1000, n_clusters=10):
    
    df = pd.read_csv(input_file, index_col=0, parse_dates=True)
    
    mu = df["q_0.5"].values
    sigma = (df['q_0.75'] - df['q_0.25']).values / 1.349
    
    n_timesteps = len(mu)
    samples = np.zeros((n_samples, n_timesteps))
    
    print(f"Generating {n_samples} paths using Latin HyperCube Sampling")
    for t in range(n_timesteps):
        # create uniform intervals 
        probs = np.linspace(0,1, n_samples+1)
        # sample randomly within each interval
        u = np.random.uniform(probs[:-1], probs[1:])
        # transform to normal distribution
        samples[:, t] = norm.ppf(u, loc=mu[t], scale=max(sigma[t], 0.1))
    
    print(f"Reducing to {n_clusters} representative scenarios...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(samples)
    representative_scenarios = kmeans.cluster_centers_
    
    scenario_df = pd.DataFrame(representative_scenarios).transpose()
    scenario_df.columns = [f'S{i}' for i in range(n_clusters)]
    scenario_df['timestamp'] = df.index
    scenario_df.set_index('timestamp', inplace=True)
    
    os.makedirs('data/processed', exist_ok=True)
    scenario_df.to_csv('data/processed/representative_scenarios.csv')
    print("Phase III Complete: Scenarios saved.")
    
    return scenario_df, samples

def generate_market_scenarios(base_price_df, num_scenarios=10, volatility=0.2):
    """
    Takes real ENTSO-E prices and creates stochastic paths for CVaR optimization.
    """
    # ENTSO-E data usually comes in 1-hour or 15-min intervals
    # Ensure we have a clean numpy array of the base prices
    base_prices = base_price_df['price'].values
    T = len(base_prices)
    
    scenarios = {}
    time_labels = [f"t{i}" for i in range(T)]
    
    for s in range(num_scenarios):
        # Create a 'Random Walk' or Gaussian noise around the real price
        # This simulates potential market fluctuations
        noise = np.random.normal(0, volatility * np.mean(base_prices), T)
        scenarios[f"S{s}"] = base_prices + noise
        
    df_scenarios = pd.DataFrame(scenarios, index=time_labels)
    
    # Save for the optimization agents to read
    os.makedirs('data/processed', exist_ok=True)
    df_scenarios.to_csv('data/processed/representative_scenarios.csv')
    
    print(f"Generated {num_scenarios} scenarios based on real ENTSO-E data.")
    return df_scenarios

def plot_scenarios(representative_df, all_samples):
    plt.figure(figsize=(12, 6))
    
    # 1. Get the actual timestamps for the x-axis
    time_axis = representative_df.index[:48]
    
    # 2. Pass the time_axis explicitly to the first plot
    # This ensures the gray lines and colored lines align on the same dates
    plt.plot(time_axis, all_samples[:100, :48].T, color='gray', alpha=0.1)
    
    # 3. Plot the representative scenarios (this uses the index automatically)
    plt.plot(representative_df.iloc[:48], linewidth=2)
    
    plt.title("Scenario Reduction: 1,000 Paths Reduced to 10 Representative Scenarios")
    plt.xlabel("Time")
    plt.ylabel("Price (€/MWh)")
    plt.xticks(rotation=45) # Rotate dates for better readability
    plt.tight_layout()
    plt.savefig('results/scenarios_reduced.png')
    plt.show()

if __name__ == "__main__":
    rep_df, raw_samples = generate_scenarios('data/processed/quantile_forecasts.csv')
    plot_scenarios(rep_df, raw_samples)
    
    