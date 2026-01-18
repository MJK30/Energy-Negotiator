import os
import pandas as pd
import numpy as np
import time
from dotenv import load_dotenv
from data_fetcher import fetch_entsoe_prices
from price_paths import generate_market_scenarios
from optimization import run_RiskAwareOptimization
from aggregator import solve_Aggregator_QP

# Load environment variables from .env file
load_dotenv()

def run_VPP_Coordination():
    """
    Runs the Virtual Power Plant (VPP) coordination algorithm using an ADMM-like consensus method.
    This coordinates multiple prosumers (energy producers/consumers) to achieve system-wide balance
    while respecting individual constraints and risk preferences.
    """
    start_time = time.time()
    
    # Load historical market price data and generate scenario forecasts
    raw_data_path = 'data/raw/entsoe_prices.csv'
    raw_prices = pd.read_csv(raw_data_path, index_col=0)
    scen_df = generate_market_scenarios(raw_prices, num_scenarios=10)
    T = len(scen_df)  # Number of time periods
    
    # Algorithm parameters for stability and convergence
    num_prosumers = 3  # Number of prosumers in the neighborhood
    alpha_step = 0.2   # Relaxation parameter for consensus updates (smaller values ensure stability)
    tolerance = 1e-3   # Convergence tolerance for mismatch norm
    max_iterations = 100  # Maximum iterations to prevent infinite loops

    # Define prosumer personas with their characteristics:
    # [lambda_risk, rho_stiffness, capacity (kWh), power_limit (kW), alpha_confidence]
    # lambda_risk: Risk aversion parameter (0 = risk-neutral, higher = more risk-averse)
    # rho_stiffness: Penalty for deviating from reference schedule (higher = stiffer)
    # capacity: Battery storage capacity
    # power_limit: Maximum power output/input
    # alpha_confidence: Confidence level for scenario optimization
    persona_profiles = {
        0: [0.0, 0.8, 2.0, 1.0, 0.90],  # Baker: Flexible, low risk aversion, small battery
        1: [0.3, 1.0, 5.0, 2.5, 0.95],  # Family: Balanced risk and flexibility
        2: [0.7, 1.5, 20.0, 10.0, 0.99] # School: High risk aversion, stiff schedule, large battery
    }
    
    # Extract parameters for the coordination algorithm
    rho_list = [p[1] for p in persona_profiles.values()]  # Stiffness penalties
    lambda_dual = np.zeros(T)  # Dual variables (shadow prices) for power balance constraints
    x_consensus = [np.zeros(T) for _ in range(num_prosumers)]  # Consensus reference schedules
    history = {'mismatch': []}  # Track convergence history
    p_limits = [profile[3] for profile in persona_profiles.values()]  # Power limits

    print(f"Starting Stabilized Coordination (Alpha={alpha_step})...")
    
    for k in range(max_iterations):
        agent_schedules = []
        for p in range(num_prosumers):
            profile = persona_profiles[p]
            # Solve risk-aware optimization for this prosumer
            # This minimizes expected cost under uncertainty, considering:
            # - Risk aversion (CVaR or similar)
            # - Battery constraints (capacity, power limits)
            # - Deviation penalties from consensus reference
            # - Dual variables from aggregator (price signals)
            net_p = run_RiskAwareOptimization(
                'data/processed/representative_scenarios.csv',
                lambda_risk=profile[0],  # Risk parameter
                rho_val=profile[1],      # Stiffness penalty
                max_cap=profile[2],      # Battery capacity
                max_p=profile[3],        # Power limit
                alpha_conf=profile[4],   # Confidence level
                lambda_val=lambda_dual,  # Dual variables (prices)
                x_ref_val=x_consensus[p] # Consensus reference
            )
            agent_schedules.append(net_p)
            
        # Step 2: Aggregator solves coordination problem
        # This is a quadratic program that minimizes total system mismatch
        # while respecting individual power limits and applying stiffness penalties
        delta_df, new_lambda_series = solve_Aggregator_QP(num_prosumers, agent_schedules, 
                                                      rho_list, p_limits)
        
        # Step 3: Check convergence by computing system mismatch norm
        # Mismatch is the total power imbalance across all prosumers
        total_internal_p = pd.concat(agent_schedules, axis=1).sum(axis=1)
        mismatch_norm = np.linalg.norm(total_internal_p)
        
        history['mismatch'].append(mismatch_norm)
        
        print(f"\n{'='*20} Iteration {k} {'='*20}")
        print(f"SYSTEM MISMATCH (Norm): {mismatch_norm:.6f}")
        
        # Update consensus references for each prosumer
        for p in range(num_prosumers):
            d_x = delta_df[delta_df['i'] == f'p{p}'].set_index('t')['level']
            # Relaxed consensus update: weighted average of current schedule and corrected schedule
            x_consensus[p] = (1 - alpha_step) * agent_schedules[p] + alpha_step * (agent_schedules[p] + d_x)
            if mismatch_norm < tolerance:
                print(f"\n✅ Convergence achieved in {k} iterations.")
                break
            
        # Update dual variables (ADMM dual ascent step)
        # This adjusts shadow prices to drive the system toward balance
        # Subtracting moves against the gradient of the mismatch
        lambda_dual -= alpha_step * new_lambda_series.values
        # Clip prices to realistic bounds to prevent numerical instability
        lambda_dual = np.clip(lambda_dual, -500, 500)

        # Standard ADMM primal update: apply corrections to consensus references
        for p in range(num_prosumers):
            d_x_agent = delta_df[delta_df['i'] == f'p{p}'].set_index('t')['level']
            x_consensus[p] = agent_schedules[p] + d_x_agent.values

    # Save final results
    os.makedirs('results', exist_ok=True)
    final_results = pd.concat(agent_schedules, axis=1)
    final_results.columns = ['Baker (Solar)', 'Family (Tesla)', 'School (Battery)']
    final_results.to_csv('results/final_neighborhood_dispatch.csv')
    pd.DataFrame(history).to_csv('results/convergence_history.csv')

if __name__ == "__main__":
    run_VPP_Coordination()