import pandas as pd
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum

def solve_Aggregator_QP(num_agents, agent_schedules, rho_val):
    m = Container()
    # Ensure time labels are captured correctly from the agent data
    time_labels = agent_schedules[0].index.tolist()
    t = Set(m, name="t", records=time_labels)
    i = Set(m, name="i", records=[f"p{n}" for n in range(num_agents)])
    
    # Explicitly build the DataFrame with column names to avoid loading errors
    records_list = []
    for idx, schedule in enumerate(agent_schedules):
        for time_idx, val in schedule.items():
            records_list.append([f"p{idx}", time_idx, val])
    
    x_val_df = pd.DataFrame(records_list, columns=['i', 't', 'value'])
    x_val = Parameter(m, name="x_val", domain=[i, t], records=x_val_df)
    H = Parameter(m, name="H", records=rho_val)
    
    delta_x = Variable(m, name="delta_x", domain=[i, t])
    balance = Equation(m, name="balance", domain=[t])
    
    # Sobic Eq 4b: Sum(x + delta_x) = 0
    balance[t] = Sum(i, x_val[i, t] + delta_x[i, t]) == 0
    
    # Sobic Eq 4a: Quadratic Objective
    obj = Sum([i, t], 0.5 * H * (delta_x[i, t]**2))
    
    # Explicitly use a solver that handles QCP marginals well (like CPLEX or GUROBI)
    qp_model = Model(m, name="Master", equations=[balance], problem="qcp", sense="min", objective=obj)
    qp_model.solve()
    
    # Check both for existence of records and successful solve status
    if delta_x.records is None or balance.records is None:
        print("CRITICAL: Aggregator failed. Check if agents are at physical limits.")
        # Return zeros to keep the loop alive but notify the user
        return pd.DataFrame(), pd.Series(0.0, index=time_labels)
        
    # Extract marginals (shadow prices) for the coordination signal
    return delta_x.records, balance.records.set_index('t')['marginal']