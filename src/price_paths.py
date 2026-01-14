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
    
    