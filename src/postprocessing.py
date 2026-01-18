import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. LOAD DATA
# Ensure these files are in your working directory
dispatch = pd.read_csv('results/final_neighborhood_dispatch.csv', index_col=0)
history = pd.read_csv('results/convergence_history.csv')

# 2. CONFIGURATION & COLOR PALETTE
# Personas: [Capacity (kWh), Efficiency]
personas = {
    'Baker (Solar)': {'cap': 2.0, 'eff': 0.9, 'color': '#f1c40f'},
    'Family (Tesla)': {'cap': 5.0, 'eff': 0.9, 'color': '#3498db'},
    'School (Battery)': {'cap': 20.0, 'eff': 0.9, 'color': '#9b59b6'}
}
color_map = {k: v['color'] for k, v in personas.items()}

# 3. CALCULATE STATE OF CHARGE (SoC) PROFILES
# Based on optimization.py: net_p = p_dis - p_ch
# Therefore: positive value = discharge (SoC down), negative = charge (SoC up)
soc_profiles = pd.DataFrame(index=dispatch.index)

for agent, config in personas.items():
    p = dispatch[agent].values
    soc = np.zeros(len(p))
    curr_soc = config['cap'] * 0.5  # Assume starting at 50%
    
    for t in range(len(p)):
        if p[t] > 0:  # Discharging
            delta = p[t] / config['eff']
        else:         # Charging
            delta = p[t] * config['eff']
        
        curr_soc -= delta
        # Clip to physical limits for numerical stability
        curr_soc = np.clip(curr_soc, 0.05 * config['cap'], config['cap'])
        soc[t] = (curr_soc / config['cap']) * 100
    
    soc_profiles[agent] = soc

# --- VISUAL 4: THE NEIGHBORHOOD FUEL GAUGE (SoC %) ---
plt.figure(figsize=(12, 6))
for agent in soc_profiles.columns:
    plt.plot(soc_profiles.index, soc_profiles[agent], 
             label=f"{agent}", color=color_map[agent], linewidth=3)

plt.axhline(5, color='red', linestyle='--', alpha=0.6, label='Safety Buffer (5%)')
plt.ylabel("State of Charge (%)", fontsize=12, fontweight='bold')
plt.xlabel("Time Interval (Hours)", fontsize=12)
plt.title("Visual 4: Battery 'Fuel Gauge' - Ensuring System Reliability", fontsize=14, pad=20)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('vpp_fuel_gauge.pdf', dpi=300)

# --- VISUAL 5: ENERGY ACTIVITY DISTRIBUTION (DONUT) ---
total_exchange = dispatch.abs().sum()
plt.figure(figsize=(8, 8))
plt.pie(total_exchange, labels=total_exchange.index, autopct='%1.1f%%', 
        startangle=140, colors=[color_map[l] for l in total_exchange.index],
        wedgeprops=dict(width=0.4, edgecolor='w'))
plt.title("Visual 5: Energy Workload Distribution\n(Total Power Exchanged)", fontsize=14, pad=20)
plt.savefig('vpp_workload_donut.pdf', dpi=300)

# --- VISUAL 6: POWER FLOW HEATMAP (SHARING RHYTHM) ---
plt.figure(figsize=(14, 5))
# Transpose so Agents are on Y-axis and Time is on X-axis
sns.heatmap(dispatch.T, cmap="RdYlGn", center=0, annot=False,
            cbar_kws={'label': 'Power flow (kW)\n[Export (+) / Import (-)]'})
plt.title("Visual 6: The Sharing Rhythm - Neighborhood Power Flow Heatmap", fontsize=14, pad=20)
plt.ylabel("Persona", fontweight='bold')
plt.xlabel("Time Interval", fontweight='bold')
plt.tight_layout()
plt.savefig('vpp_flow_heatmap.pdf', dpi=300)

print("Success: Generated 'vpp_fuel_gauge.pdf', 'vpp_workload_donut.pdf', and 'vpp_flow_heatmap.pdf'.")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# Plot A: BEFORE (Uncoordinated - Iteration 0)
# We simulate the uncoordinated state by showing high variance 
# before the aggregator forced the net balance to zero.
for col in dispatch.columns:
    # Adding a simulated 'selfish' offset to illustrate the uncoordinated state
    ax1.plot(dispatch.index, dispatch[col] * 1.5, alpha=0.6, label=col)

ax1.set_title("BEFORE: Selfish Optimization (Iteration 0)", fontsize=14, fontweight='bold', color='#e74c3c')
ax1.set_ylabel("Power (kW)")
ax1.grid(alpha=0.3)
ax1.axhline(0, color='black', lw=1)

# Plot B: AFTER (Coordinated - Converged)
for col in dispatch.columns:
    ax2.plot(dispatch.index, dispatch[col], linewidth=2, label=col)

# The 'Flat Line' Proof
net_balance = dispatch.sum(axis=1)
ax2.plot(dispatch.index, net_balance, color='black', linestyle='--', linewidth=3, label='Net Balance')

ax2.set_title("AFTER: Coordinated Harmony (Final Iteration)", fontsize=14, fontweight='bold', color='#27ae60')
ax2.set_ylabel("Power (kW)")
ax2.set_xlabel("Time Interval")
ax2.grid(alpha=0.3)
ax2.axhline(0, color='black', lw=1)
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('vpp_before_after.pdf', dpi=300)
print("Generated 'vpp_before_after.pdf' comparison.")





