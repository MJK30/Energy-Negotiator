import pandas as pd
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum
import os
import sys

def run_RiskAwareOptimization(scenario_file, lambda_risk=0.0, alpha_conf=0.95):
    # 1. Load Data
    df_scen = pd.read_csv(scenario_file, index_col=0)
    time_labels = [f"t{i}" for i in range(len(df_scen))]
    df_scen.index = time_labels
    
    m = Container()
    
    # 2. Sets
    t = Set(m, name='t', records=time_labels)
    s = Set(m, name="s", records=df_scen.columns.tolist())
    
    # 3. Parameters
    price_df = df_scen.reset_index().melt(id_vars='index')
    price_df.columns = ['t', 's', 'value']
    Price = Parameter(m, name="Price", domain=[s,t], records=price_df[['s', 't', 'value']])
    prob = Parameter(m, name="prob", domain=s, records=pd.Series(1.0/len(s.records), index=s.records))
    
    # BESS Constants
    eff_ch, eff_dis = 0.85, 0.90
    P_limit, SoC_max, SoC_min = 1.8, 1.8, 0.2
    initial_soc = 1.0
    
    # 4. Variables (Continuous for LP)
    p_ch = Variable(m, name="p_ch", domain=[t], type="positive")
    p_dis = Variable(m, name="p_dis", domain=[t], type="positive")
    soc = Variable(m, name="soc", domain=[t], type="positive")
    
    p_ch.up[t] = P_limit
    p_dis.up[t] = P_limit
    soc.up[t] = SoC_max
    soc.lo[t] = SoC_min
    
    profit = Variable(m, name="profit", domain=s)
    zeta = Variable(m, name="zeta") 
    aux = Variable(m, name="aux", domain=s, type="positive")
    
    # 5. Equations
    soc_dyn = Equation(m, name="soc_dyn", domain=[t])
    prof_calc = Equation(m, name="prof_calc", domain=s)
    cvar_limit = Equation(m, name="cvar_limit", domain=s)
    term_con = Equation(m, name="term_con")
    
    lambda_dual = Parameter(m, name="lambda_dual", domain=[t])
    x_ref = Parameter(m, name="x_ref", domain=[t])
    rho = 0.1 # Penalty weight
    
    net_p = Variable(m, name="net_p", domain=[t])
    net_p_def = Equation(m, name="net_p_def", domain=[t])
    net_p_def[t] = net_p[t] == (p_dis[t] - p_ch[t])
    
    # Dynamics & Constraints
    soc_dyn[t].where[t.pos == 1] = soc[t] == initial_soc + (eff_ch * p_ch[t] - p_dis[t] / eff_dis)
    soc_dyn[t].where[t.pos > 1] = soc[t] == soc[t.lag(1)] + (eff_ch * p_ch[t] - p_dis[t] / eff_dis)
    
    term_con[...] = soc[time_labels[-1]] >= initial_soc
    
    prof_calc[s] = profit[s] == Sum(t, Price[s, t] * (p_dis[t] - p_ch[t]))
    cvar_limit[s] = aux[s] >= (zeta - profit[s])
    
    # 6. Objective (Expected Profit - Risk)
    expected_profit = Sum(s, prob[s] * profit[s])
    cvar_term = zeta + (1 / (1 - alpha_conf)) * Sum(s, prob[s] * aux[s])
    cvar_profit = zeta - (1 / (1 - alpha_conf)) * Sum(s, prob[s] * aux[s])
    obj = (1 - lambda_risk) * expected_profit + lambda_risk * cvar_profit
    
    aladin_obj = obj - Sum(t, lambda_dual[t] * net_p[t]) - (rho/2) * Sum(t, (net_p[t] - x_ref[t])**2)
    
    # 7. Solve as LP
    bess_model = Model(m, name="ALADIN_Agent", equations=m.getEquations(), 
                    problem="qcp", sense="max", objective=aladin_obj)
    
    print(f"Solving BESS Risk-Aware Model (LP) [lambda={lambda_risk}, alpha={alpha_conf}]...")
    bess_model.solve(output=sys.stdout)
    
    if profit.records is None:
        return m, None, None, None, None
        
    return m, soc.records, p_ch.records, p_dis.records, profit.records

if __name__ == "__main__":
    df_filepath = 'data/processed/representative_scenarios.csv'
    
    # Test with a moderate lambda and high alpha
    m_cont, soc_res, pch_res, pdis_res, prof_res = run_RiskAwareOptimization(
        df_filepath, lambda_risk=0.0, alpha_conf=0.95
    )
    
    if prof_res is not None:
        print(f"Optimization Successful!")
        print(f"Mean Expected Profit: {prof_res['level'].mean():.2f} EUR")
        os.makedirs('data/results', exist_ok=True)
        soc_res.to_csv('results/optimized_soc.csv', index=False)















# import pandas as pd
# import numpy as np
# from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum
# import os

# def run_RiskAwareOptimization(scenario_file, lambda_risk=0.0, alpha_conf=0.5):
#     # 1. Load Data & Force Label Alignment
#     df_scen = pd.read_csv(scenario_file, index_col=0)
#     # Ensure index is exactly 't0', 't1', ... to match GAMSPy Set
#     time_labels = [f"t{i}" for i in range(len(df_scen))]
#     df_scen.index = time_labels
    
#     m = Container()
    
#     # 2. Sets
#     t = Set(m, name='t', records=time_labels)
#     s = Set(m, name="s", records=df_scen.columns.tolist())
    
#     # 3. Parameters (Fixed Melt logic for matching)
#     price_df = df_scen.reset_index().melt(id_vars='index')
#     price_df.columns = ['t', 's', 'value']
#     # Reorder for domain [s, t]
#     Price = Parameter(m, name="Price", domain=[s,t], records=price_df[['s', 't', 'value']])
    
#     prob = Parameter(m, name="prob", domain=s, records=pd.Series(1.0/len(s.records), index=s.records))
    
#     # BESS Constants
#     eff_ch = 0.85
#     eff_dis = 0.90
#     P_limit = 1.8
#     SoC_max = 1.8
#     SoC_min = 0.2
#     initial_soc = 1.0
    
#     # 4. Variables
#     # Dispatch is Scenario-Independent (Monti et al. Standard)
#     p_ch = Variable(m, name="p_ch", domain=[t], type="positive")
#     p_dis = Variable(m, name="p_dis", domain=[t], type="positive")
#     y_ch = Variable(m, name="y_ch", domain=[t], type="binary")
#     y_dis = Variable(m, name="y_dis", domain=[t], type="binary")
#     soc = Variable(m, name="soc", domain=[t], type="positive")
    
#     # Risk Variables
#     profit = Variable(m, name="profit", domain=s)
#     zeta = Variable(m, name="zeta") 
#     aux = Variable(m, name="aux", domain=s, type="positive")
    
#     # 5. Equations
#     soc_dyn = Equation(m, name="soc_dyn", domain=[t])
#     exclusivity = Equation(m, name="exclusivity", domain=[t])
#     p_ch_lim = Equation(m, name="p_ch_lim", domain=[t]) # FIXED: Use equations, not .up
#     p_dis_lim = Equation(m, name="p_dis_lim", domain=[t]) # FIXED: Use equations, not .up
#     prof_calc = Equation(m, name="prof_calc", domain=s)
#     cvar_limit = Equation(m, name="cvar_limit", domain=s)
#     term_con = Equation(m, name="term_con")
    
#     # SoC Dynamics
#     soc_dyn[t].where[t.pos == 1] = soc[t] == initial_soc + (eff_ch * p_ch[t] - p_dis[t] / eff_dis)
#     soc_dyn[t].where[t.pos > 1] = soc[t] == soc[t.lag(1)] + (eff_ch * p_ch[t] - p_dis[t] / eff_dis)
    
#     # Power Limits (Linked to Binaries)
#     p_ch_lim[t] = p_ch[t] <= P_limit * y_ch[t]
#     p_dis_lim[t] = p_dis[t] <= P_limit * y_dis[t]
#     exclusivity[t] = y_ch[t] + y_dis[t] <= 1
    
#     # Physical SoC Bounds
#     soc.up[t] = SoC_max
#     soc.lo[t] = SoC_min
    
#     # Terminal Constraint (Battery must end where it started)
#     term_con[...] = soc[time_labels[-1]] >= initial_soc
    
#     # Financials
#     prof_calc[s] = profit[s] == Sum(t, Price[s, t] * (p_dis[t] - p_ch[t]))
#     cvar_limit[s] = aux[s] >= (zeta - profit[s])
    
#     # 6. Objective
#     expected_profit = Sum(s, prob[s] * profit[s])
#     cvar_term = zeta + (1 / (1 - alpha_conf)) * Sum(s, prob[s] * aux[s])
#     obj = expected_profit - lambda_risk * cvar_term
    
#     # 7. Solve
#     bess_model = Model(m, name="BESS_Risk", equations=m.getEquations(), 
#                        problem="mip", sense="max", objective=obj)
    
#     print("Solving BESS Risk-Aware Model...")
#     bess_model.solve()
#     print(bess_model.status)
    
#     # Return results (with safety check)
#     if profit.records is None:
#         return m, None, None, None, None
        
#     return m, soc.records, p_ch.records, p_dis.records, profit.records

# if __name__ == "__main__":
#     df_filepath = 'data/processed/representative_scenarios.csv'
    
#     # Lambda = 0.05 (Risk-Aware but Profit-Seeking)
#     m_cont, soc_res, pch_res, pdis_res, prof_res = run_RiskAwareOptimization(df_filepath, lambda_risk=0.05)
    
#     if prof_res is not None:
#         print(f"Optimization Successful!")
#         print(f"Mean Expected Profit: {prof_res['level'].mean():.2f} EUR")
        
#         # Save to results/ folder
#         os.makedirs('results', exist_ok=True)
#         soc_res.to_csv('results/optimized_soc.csv', index=False)
#         prof_res.to_csv('results/scenario_profits.csv', index=False)
#         print("Results saved to results/optimized_soc.csv")
#     else:
#         print("CRITICAL: Optimization failed. Status: The model logic still contains a conflict.")