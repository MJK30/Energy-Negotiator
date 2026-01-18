# The Energy Negotiator: Orchestrating Private Assets for Neighborhood Grid Harmony

**"The Energy Negotiator"** is a decentralized, risk-aware optimization framework built with **GAMSPY**. It solves the "Duck Curve" challenge at the neighborhood level by coordinating independent energy agents—without compromising their privacy—to ensure local transformer stability and grid harmony.

---

## 1. Project Vision: Solving the Duck Curve
Modern power grids face a fundamental mismatch between midday solar generation peaks and evening demand cycles. At the neighborhood level, when individual households optimize their batteries selfishly, they create synchronized power peaks that strain local infrastructure.

This project implements a **Virtual Power Plant (VPP)** logic that acts as a "Market Negotiator," using shadow price signals to harmonize the behavior of diverse residents into a single, balanced community resource.

---

## 2. The Neighborhood Personas
The system coordinates three distinct "Characters," each with unique economic motives and technical constraints:

| Persona | Asset | Capacity | Power Max | Risk Weight ($\beta$) | Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baker (Solar)** | Solar + BESS | 2.0 kWh | 1.0 kW | 0.1 | The midday energy provider |
| **Family (Tesla)** | Powerwall | 5.0 kWh | 2.5 kW | 0.7 | The risk-averse consumer |
| **School (Battery)** | Industrial BESS | 20.0 kWh | 10.0 kW | 0.4 | The neighborhood anchor |

![The Distributed Architecture](DISTRIBUTED.png)

---

## 3. Core Methodology

### **The Negotiator (ADMM Aggregator)**
Unlike centralized control systems, the Negotiator uses the **Alternating Direction Method of Multipliers (ADMM)**.
- **Privacy First**: The central aggregator never sees private battery levels or household schedules.
- **Iterative Consensus**: It issues "Shadow Prices" ($\lambda$) based on the neighborhood's current imbalance. Agents adjust their behavior until a perfect system balance is reached.

### **Risk-Aware Agent Optimization (CVaR)**
Each agent manages market uncertainty through **Conditional Value at Risk (CVaR)**.
- **Objective**: $\max \quad (1-\beta) \mathbb{E}[\text{Profit}] + \beta \text{CVaR}_{\alpha}$
- **$\beta$ (The Tuning Knob)**: Allows residents to choose their comfort level between aggressive profit-seeking (Baker) and conservative safety (Family).

---

## 4. Technical Architecture
The repository follows a modular optimization pipeline:

* `src/optimization.py`: Local agent models (Battery physics + CVaR + ADMM proximal terms).
* `src/aggregator.py`: The master problem that calculates the neighborhood's coordination signal.
* `src/coordination.py`: The main orchestrator that manages the iterative "Negotiation" loop.
* `src/postprocessing.py`: Generates  visuals for results analysis.

---

## 5. Performance & Results
The framework achieves a state of mathematical harmony that ensures zero impact on the external grid:
* **Precision**: Reached a system mismatch norm of **$2.17 \times 10^{-6}$**.
* **Stability**: Converged from a chaotic 44 kW imbalance to a perfectly flat line within 100 iterations.
* **Synergy**: Effectively "shaves" the evening peak by using the Baker’s solar surplus to charge the School's anchor battery.



---

## 6. How to Run
1.  **Initialize Parameters**: Ensure the persona profiles in `coordination.py` match your neighborhood scenario.
2.  **Run Negotiation**: 
    ```bash
    python src/coordination.py
    ```
3.  **Generate Visuals**:
    ```bash
    python src/postprocessing.py
    ```
    * `vpp_before_after.png`: Visual proof of peak reduction.
    * `vpp_flow_heatmap.png`: The "Handshake" between solar supply and battery demand.
    * `vpp_fuel_gauge.png`: Real-time monitoring of battery State of Charge (SoC).

---
