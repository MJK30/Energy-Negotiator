# Multi-Agent-Battery-Orchestration-Platform-with-Probabilistic-Risk-Hedging.

This project implements a decentralized, risk-aware optimization framework for a **Virtual Power Plant (VPP)** consisting of Battery Energy Storage Systems (BESS). By leveraging the **ALADIN (Augmented Lagrangian based Alternating Direction Inexact Newton)** algorithm, the system coordinates multiple independent prosumers to achieve grid balance while explicitly managing market price uncertainty through **Conditional Value at Risk (CVaR)**.

---

## 1. Project Overview
In modern energy markets, battery operators face significant financial risks due to price volatility. This project provides a technical solution that:
1.  **Forecasts** market uncertainty using probabilistic quantile regression.
2.  **Optimizes** local battery dispatch to maximize profit while "hedging" against tail risks.
3.  **Coordinates** multiple batteries using a privacy-preserving distributed algorithm (ALADIN) to ensure collective grid stability without sharing sensitive local data.

---

## 2. General Project Structure
The repository is organized into a modular pipeline, separating data science tasks from optimization and control:

| Directory/File | Purpose |
| :--- | :--- |
| `data/raw/` | Original ENTSO-E or synthetic market data. |
| `data/processed/` | Quantile forecasts and clustered scenarios. |
| `results/` | Visualizations (Risk bands, schedules, convergence). |
| `src/data_ingress.py` | Phase I: Data fetching and synthetic generation. |
| `src/forecasting.py` | Phase II: Probabilistic quantile regression. |
| `src/scenarios.py` | Phase III: Sampling and K-Means reduction. |
| `src/optimization.py` | Phase IV: Local Risk-Aware BESS (ALADIN Agent). |
| `src/aggregator.py` | Phase V: Coupling Quadratic Program (ALADIN Master). |
| `src/coordination.py` | Phase V: Main Orchestrator / ALADIN Loop. |

---

## 3. Core Algorithms

### **Conditional Value at Risk (CVaR)**
To manage market uncertainty, the local agent uses CVaR to penalize the expected loss in the worst $\alpha\%$ (e.g., 5%) of scenarios.
* **Loss Definition**: $Loss_s = Profit_{baseline} - Profit_s$.
* **Optimization Goal**: $\max (E[Profit] - \lambda \cdot CVaR_{\alpha})$.



### **ALADIN (Distributed Optimization)**
ALADIN is used to solve the coordination problem by decomposing the global objective into local subproblems.
* **Local Step**: Agents solve their own risk-aware profit maximization, including a dual price signal and a proximal penalty to ensure convergence.
* **Consensus Step**: The aggregator solves a **Coupling Quadratic Program (QP)** to balance the total power mismatch across the community.



---

## 4. Technical Specifications
The models are parameterized based on technical specifications common in energy research:
* **Power Limit**: 1.8 MW Ramp Limit.
* **Capacity**: 2.0 MWh (with 10% - 90% SoC operational window).
* **Efficiency**: 85% Charging / 90% Discharging.
* **Time Step**: 1.0 Hour (Standard Day-Ahead resolution).

---

## 5. How to Run
1.  **Generate Scenarios**: Run `src/scenarios.py` to create the representative price paths.
2.  **Start Coordination**: Execute `python src/coordination.py`.
3.  **Analyze Results**: Convergence logs will appear in the console, and optimized schedules will be saved to `results/`.
