import pandas as pd
import numpy as np
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum

def run_RiskAwareOptimization(scenario_file, lambda_risk, rho_val, max_cap, max_p,
                              alpha_conf=0.95, lambda_val=None, x_ref_val=None):
    """
    Solves the risk-aware optimization problem for a battery energy storage system (BESS).
    This maximizes expected profit under uncertainty while considering risk (via CVaR),
    battery constraints, and coordination penalties from the ADMM algorithm.
    Returns the optimal net power schedule.
    """
    # Load scenario data (price forecasts) and align time labels
    df_scen = pd.read_csv(scenario_file, index_col=0)
    time_labels = [f"t{i}" for i in range(len(df_scen))]
    df_scen.index = time_labels

    # Scale prices to prevent large economic terms from dominating coordination penalties
    price_scale = df_scen.values.max() if df_scen.values.max() > 0 else 1.0

    m = Container()  # GAMSPy model container
    t = Set(m, name='t', records=time_labels)  # Time periods
    s = Set(m, name="s", records=df_scen.columns.tolist())  # Scenarios

    # Load price data and assume equal probability for each scenario
    price_df = df_scen.reset_index().melt(id_vars='index')
    price_df.columns = ['t', 's', 'value']
    Price = Parameter(m, name="Price", domain=[s,t], records=price_df[['s', 't', 'value']])
    prob = Parameter(m, name="prob", domain=s, records=pd.Series(1.0/len(s.records), index=s.records))

    # Load ADMM coordination inputs: dual variables (prices) and reference schedules
    l_rec = pd.Series(lambda_val, index=time_labels) if lambda_val is not None else 0
    x_rec = pd.Series(x_ref_val, index=time_labels) if x_ref_val is not None else 0
    lambda_dual = Parameter(m, name="lambda_dual", domain=[t], records=l_rec)
    x_ref = Parameter(m, name="x_ref", domain=[t], records=x_rec)
    rho = Parameter(m, name="rho", records=rho_val)  # Stiffness penalty

    # Define decision variables for battery operation
    p_ch = Variable(m, name="p_ch", domain=[t], type="positive")    # Charging power
    p_dis = Variable(m, name="p_dis", domain=[t], type="positive")  # Discharging power
    soc = Variable(m, name="soc", domain=[t], type="positive")      # State of charge
    net_p = Variable(m, name="net_p", domain=[t])                   # Net power (positive = discharge, negative = charge)

    # Set bounds on variables
    p_ch.up[t] = max_p      # Max charging power
    p_dis.up[t] = max_p     # Max discharging power
    soc.up[t] = max_cap     # Max SOC
    soc.lo[t] = 0.05 * max_cap  # Min SOC (5% buffer)

    # Variables for risk-aware profit calculation (CVaR)
    profit = Variable(m, name="profit", domain=s)  # Profit per scenario
    zeta = Variable(m, name="zeta")                # CVaR threshold
    aux = Variable(m, name="aux", domain=s, type="positive")  # Auxiliary for CVaR

    # Define model equations
    soc_dyn = Equation(m, name="soc_dyn", domain=[t])          # SOC dynamics
    net_p_def = Equation(m, name="net_p_def", domain=[t])      # Net power definition
    prof_calc = Equation(m, name="prof_calc", domain=s)        # Profit calculation
    cvar_limit = Equation(m, name="cvar_limit", domain=s)      # CVaR constraint
    final_soc_con = Equation(m, name="final_soc_con")          # Final SOC constraint

    # Net power is discharge minus charge
    net_p_def[t] = net_p[t] == (p_dis[t] - p_ch[t])

    # SOC dynamics with charging/discharging efficiencies (90% each)
    # Initial SOC starts at 50% of capacity
    soc_dyn[t].where[t.pos == 1] = soc[t] == (max_cap * 0.5) + (0.90 * p_ch[t] - p_dis[t] / 0.90)
    # Subsequent periods: SOC[t] = SOC[t-1] + efficient charge - efficient discharge
    soc_dyn[t].where[t.pos > 1] = soc[t] == soc[t.lag(1)] + (0.90 * p_ch[t] - p_dis[t] / 0.90)

    # Duplicate initial SOC equation (for syntax reasons)
    soc_dyn[t].where[t.pos == 1] = soc[t] == (max_cap * 0.5) + (0.90 * p_ch[t] - p_dis[t] / 0.90)
    # Ensure final SOC is at least 50% of capacity
    final_soc_con[...] = soc[time_labels[-1]] >= (max_cap * 0.5)

    # Profit per scenario: sum of price * net_power over time
    prof_calc[s] = profit[s] == Sum(t, Price[s, t] * net_p[t])
    # CVaR auxiliary constraint: aux >= zeta - profit (for losses)
    cvar_limit[s] = aux[s] >= (zeta - profit[s])

    # Redundant parameter definitions (for compatibility)
    lam = Parameter(m, name="lam", domain=[t], records=np.array(lambda_val))
    ref = Parameter(m, name="ref", domain=[t], records=np.array(x_ref_val))

    # Define objective components
    lam_p = Parameter(m, name="lam_p", domain=[t], records=pd.Series(lambda_val, index=time_labels))
    ref_p = Parameter(m, name="ref_p", domain=[t], records=pd.Series(x_ref_val, index=time_labels))

    # Expected profit across scenarios
    expected_profit = Sum(s, prob[s] * profit[s])
    # CVaR: expected value of losses beyond the (1-alpha) quantile
    # Mathematically: CVaR = zeta - (1/(1-alpha)) * E[ max(zeta - profit, 0) ]
    cvar_profit = zeta - (1 / (1 - alpha_conf)) * Sum(s, prob[s] * aux[s])

    # Economic objective: weighted average of expected profit and CVaR
    economic_obj = (1 - lambda_risk) * expected_profit + lambda_risk * cvar_profit

    # Coordination term from ADMM: linear penalty + quadratic penalty on deviation from reference
    # This enforces consensus: penalizes deviation from reference schedule and responds to dual prices
    coordination_term = Sum(t, lam_p[t] * net_p[t] +
                           (0.5 * rho_val) * (net_p[t] - ref_p[t])**2)

    # Full objective: maximize economic benefit minus coordination penalty
    bess_model = Model(
        m, name="Agent", equations=m.getEquations(),
        problem="qcp", sense="max", objective=economic_obj - coordination_term
    )

    bess_model.solve()

    # Return the optimized net power schedule
    if net_p.records is not None:
        return net_p.records.set_index('t')['level']
    else:
        # Fallback for solve failure
        return pd.Series(0.0, index=time_labels)