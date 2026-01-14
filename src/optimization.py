import pandas as pd
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum

def run_RiskAwareOptimization(scenario_file, lambda_risk=0.1, alpha_conf=0.95, 
                               lambda_val=None, x_ref_val=None, rho_val=0.5):
    # 1. Load data and force label alignment
    df_scen = pd.read_csv(scenario_file, index_col=0)
    time_labels = [f"t{i}" for i in range(len(df_scen))]
    df_scen.index = time_labels
    
    m = Container()
    t = Set(m, name='t', records=time_labels)
    s = Set(m, name="s", records=df_scen.columns.tolist())
    
    # Prices and Probabilities
    # Ensure melt id matches our forced index 'index'
    price_df = df_scen.reset_index().melt(id_vars='index')
    price_df.columns = ['t', 's', 'value']
    Price = Parameter(m, name="Price", domain=[s,t], records=price_df[['s', 't', 'value']])
    prob = Parameter(m, name="prob", domain=s, records=pd.Series(1.0/len(s.records), index=s.records))
    
    # 2. ALADIN Inputs
    # Force lambda and x_ref to match the actual length (1440)
    l_rec = pd.Series(lambda_val, index=time_labels) if lambda_val is not None else 0
    x_rec = pd.Series(x_ref_val, index=time_labels) if x_ref_val is not None else 0
    lambda_dual = Parameter(m, name="lambda_dual", domain=[t], records=l_rec)
    x_ref = Parameter(m, name="x_ref", domain=[t], records=x_rec)
    rho = Parameter(m, name="rho", records=rho_val)

    # 3. Variables & Bounds
    p_ch = Variable(m, name="p_ch", domain=[t], type="positive")
    p_dis = Variable(m, name="p_dis", domain=[t], type="positive")
    soc = Variable(m, name="soc", domain=[t], type="positive")
    net_p = Variable(m, name="net_p", domain=[t])
    
    p_ch.up[t] = 1.8
    p_dis.up[t] = 1.8
    soc.up[t] = 1.8
    soc.lo[t] = 0.2
    
    profit = Variable(m, name="profit", domain=s)
    zeta = Variable(m, name="zeta") 
    aux = Variable(m, name="aux", domain=s, type="positive")
    
    # 4. Equations
    soc_dyn = Equation(m, domain=[t])
    net_p_def = Equation(m, domain=[t])
    prof_calc = Equation(m, domain=s)
    cvar_limit = Equation(m, domain=s)

    net_p_def[t] = net_p[t] == (p_dis[t] - p_ch[t])
    
    soc_dyn[t].where[t.pos == 1] = soc[t] == 1.0 + (0.85 * p_ch[t] - p_dis[t] / 0.90)
    soc_dyn[t].where[t.pos > 1] = soc[t] == soc[t.lag(1)] + (0.85 * p_ch[t] - p_dis[t] / 0.90)

    prof_calc[s] = profit[s] == Sum(t, Price[s, t] * net_p[t])
    cvar_limit[s] = aux[s] >= (zeta - profit[s])
    
    # 5. ALADIN Objective (Maximize Profit - Penalty)
    expected_profit = Sum(s, prob[s] * profit[s])
    cvar_profit = zeta - (1 / (1 - alpha_conf)) * Sum(s, prob[s] * aux[s])
    base_obj = (1 - lambda_risk) * expected_profit + lambda_risk * cvar_profit
    
    aladin_penalty = Sum(t, lambda_dual[t] * net_p[t] + (rho/2) * (net_p[t] - x_ref[t])**2)
    
    bess_model = Model(m, name="Agent", equations=m.getEquations(), 
                        problem="qcp", sense="max", objective=base_obj - aladin_penalty)
    bess_model.solve()
    
    if net_p.records is None:
        return pd.Series(0.0, index=time_labels)
        
    return net_p.records.set_index('t')['level']