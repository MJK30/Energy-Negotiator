import pandas as pd
from gamspy import Container, Set, Parameter, Variable, Equation, Model, Sum

def solve_Aggregator_QP(num_agents, agent_schedules, rho_list, p_limits):
    """
    Solves the aggregator's quadratic programming problem in the ADMM consensus algorithm.
    This finds optimal adjustments (corrections) to each agent's schedule to achieve system-wide
    power balance, while penalizing deviations based on agent stiffness (rho values).
    Returns the adjustments and shadow prices (marginals) for the balance constraints.
    """
    m = Container()  # GAMSPy model container
    time_labels = agent_schedules[0].index.tolist()
    t = Set(m, name="t", records=time_labels)  # Time periods set

    # Define agent set including prosumers and a balancing grid agent
    agent_indices = [f"p{n}" for n in range(num_agents)] + ["p_grid"]
    i = Set(m, name="i", records=agent_indices)

    # Power limits for each agent (grid has high capacity)
    p_max_vals = pd.Series(p_limits + [1000.0], index=agent_indices)
    p_max = Parameter(m, name="p_max", domain=[i], records=p_max_vals)

    # Penalty weights (rho) for deviations; grid has high penalty to act as last resort
    rho_vals = pd.Series(rho_list + [20.0], index=agent_indices)
    H = Parameter(m, name="H", domain=[i], records=rho_vals)

    # Load current agent schedules into parameter
    records = []
    for idx, sched in enumerate(agent_schedules):
        for time_idx, val in sched.items():
            records.append([f"p{idx}", time_idx, val])
    # Grid starts with zero schedule (neutral)
    for time_idx in time_labels:
        records.append(["p_grid", time_idx, 0.0])

    x_val = Parameter(m, name="x_val", domain=[i, t], records=pd.DataFrame(records))

    # Decision variables: adjustments to schedules
    delta_x = Variable(m, name="delta_x", domain=[i, t])

    # Balance constraint: total power (schedules + adjustments) must sum to zero across all agents
    balance = Equation(m, name="balance", domain=[t])
    balance[t] = Sum(i, x_val[i, t] + delta_x[i, t]) == 0
    
    # Bounding the adjustments
    delta_x.up[i, t] = p_max[i] - x_val[i, t]
    delta_x.lo[i, t] = -p_max[i] - x_val[i, t]
    
    # Objective: Minimize the weighted squared deviation (ADMM form)
    obj = Sum([i, t], 0.5 * H[i] * (delta_x[i, t]**2))

    # Solve the quadratic constrained programming problem
    qp_model = Model(m, name="Aggregator", equations=[balance],
                     problem="qcp", sense="min", objective=obj)
    qp_model.solve()

    # Handle solve failure
    if delta_x.records is None:
        return pd.DataFrame(), pd.Series(0.0, index=time_labels)

    # Return adjustments and shadow prices (dual variables for balance constraints)
    return delta_x.records, balance.records.set_index('t')['marginal']


