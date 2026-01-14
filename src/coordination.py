from optimization import run_RiskAwareOptimization
from aggregator import solve_Aggregator_QP
import numpy as np
import pandas as pd

def run_VPP_Coordination():
    # Load metadata
    scen_path = 'data/processed/representative_scenarios.csv'
    df_temp = pd.read_csv(scen_path, index_col=0)
    T = len(df_temp) # 1440
    
    num_prosumers = 3
    rho = 100
    alpha_step = 0.1
    
    # Correctly initialized vectors
    lambda_dual = np.zeros(T)
    x_consensus = [np.zeros(T) for _ in range(num_prosumers)]
    
    print(f"Starting ALADIN Coordination (T={T})...")
    for k in range(25):
        agent_schedules = []
        for p in range(num_prosumers):
            net_p_series = run_RiskAwareOptimization(
                scen_path, lambda_risk=0.1, 
                lambda_val=lambda_dual, x_ref_val=x_consensus[p], rho_val=rho
            )
            agent_schedules.append(net_p_series)
            
        # 2. Aggregator Step 
        delta_df, new_lambda_series = solve_Aggregator_QP(num_prosumers, agent_schedules, rho)
        
        # 3. Update Step with Safety Check
        total_p = pd.concat(agent_schedules, axis=1).sum(axis=1)
        mismatch_norm = np.linalg.norm(total_p)
        
        for p in range(num_prosumers):
            # Safe column check to avoid KeyError
            if not delta_df.empty and 'i' in delta_df.columns:
                d_x = delta_df[delta_df['i'] == f'p{p}'].set_index('t')['level']
                x_consensus[p] = agent_schedules[p] + alpha_step * d_x
            else:
                print(f"Agent {p} adjustment skipped due to empty Master result.")

        # Update price signal lambda
        lambda_dual = lambda_dual + alpha_step * (new_lambda_series.values - lambda_dual)
        print(f"Lambda_Dual Value: {lambda_dual}")
        
        print(f"Iteration {k} | System Mismatch Norm: {mismatch_norm:.4f}")

if __name__ == "__main__":
    run_VPP_Coordination()