# README

## Purpose  
This repository contains simulations for a G-function–based model of VEGF and GLUT1 expression in cancer. The model treats VEGF-mediated resource recruitment and GLUT1-mediated resource uptake as co-evolving quantitative traits shaped by local cell–cell interactions within tumor neighborhoods.  

A key feature of the model is the **degree of resource sharing (q)**, which determines whether interactions resemble a **tragedy of the commons** or a **public goods game**, and influences both evolutionary dynamics and therapeutic response.

---

## Folder Layout  

```
repo/
├── Adaptive-Landscapes.ipynb
├── Dynamic-Neighborhood-Simulations.ipynb
├── Dynamic-Neighborhood-Team-Sweeps-Sensitivity.ipynb
├── Dynamic-Neighborhood-ESS-Sweep-Sensitivity.ipynb
├── Fixed-Neighborhood-Simulations.ipynb
├── Fixed-Neighborhood-Team-Sweeps-Sensitivity.ipynb
├── Fixed-Neighborhood-ESS-Sweeps-Sensitivity.ipynb
├── Therapy-Simulations.ipynb
├── Viability-Values.ipynb
│
├── outputs/
│   ├── dynamic_team_opt/
│   ├── evolving_neighborhoods/
│   ├── fixed_neighborhoods/
│   └── team_opt_dynamics/
│
├── utils/
│   ├── __init__.py
│   ├── evolving_neighborhood.py
│   ├── fixed_neighborhood.py
│   ├── models_therapy.py
│   ├── model_params.py
│   └── README.md
```

---

## Set-Up and How to Run  

Install required packages:
```
pip install numpy scipy matplotlib pandas
```

Launch Jupyter:
```
jupyter notebook
```

---

## Workflow and Notebook Guide  

### A. Core Simulations (No Therapy)

#### 1. Fixed Neighborhood Dynamics  
**Open:** `Fixed-Neighborhood-Simulations.ipynb`  
- Simulates VEGF and GLUT1 dynamics in fixed-size neighborhoods  
- Compares ESS and team optimum  
- Generates time dynamics and equilibrium behavior  

---

#### 2. Dynamic Neighborhoods  
**Open:** `Dynamic-Neighborhood-Simulations.ipynb`  
- Neighborhood size evolves dynamically  
- Produces coupled dynamics of population, VEGF, and GLUT1  
- Used to generate trajectories for adaptive landscapes  

---

#### 3. Adaptive Landscapes  
**Open:** `Adaptive-Landscapes.ipynb`  
- Computes and visualizes fitness landscapes  
- Uses trajectory outputs from simulations  
- Allows comparison of ESS vs team optimum structure  

---

### B. Parameter Sweeps and Sensitivity  

#### 4. Fixed Neighborhood Sweeps  
- `Fixed-Neighborhood-Team-Sweeps-Sensitivity.ipynb`  
- `Fixed-Neighborhood-ESS-Sweeps-Sensitivity.ipynb`  

Run all cells to:
- Compute equilibrium strategies across parameter ranges  
- Generate sweep plots (e.g., vs R, q, costs, etc.)  
- Compare ESS and team optimum  

---

#### 5. Dynamic Neighborhood Sweeps  
- `Dynamic-Neighborhood-Team-Sweeps-Sensitivity.ipynb`  
- `Dynamic-Neighborhood-ESS-Sweep-Sensitivity.ipynb`  

Run all cells to:
- Perform parameter sweeps when neighborhood size evolves  
- Extract equilibrium values after convergence  
- Generate summary plots  

---

### C. Therapy Simulations  

#### 6. Therapy Dynamics  
**Open:** `Therapy-Simulations.ipynb`  

- Implements therapy as reductions in:
  - VEGF effectiveness (`a`) → anti-angiogenic therapy  
  - GLUT1 effectiveness (`k`) → glucose uptake inhibition  
- Supports different therapy schedules and strengths  
- Outputs time-series dynamics  

---

#### 7. Viability and Outcome Metrics  
**Open:** `Viability-Values.ipynb`  

- Computes viability metrics and summary statistics  
- Used to compare therapy strategies  
- Can be used downstream for plotting or analysis  

---

## Outputs  

Simulation outputs are stored in:

```
outputs/
├── dynamic_team_opt/
├── evolving_neighborhoods/
├── fixed_neighborhoods/
└── team_opt_dynamics/
```

These folders contain:
- Time-series data  
- Equilibrium values  
- Parameter sweep summaries  

---

## Utilities  

All model logic is centralized in `utils/`:

- `model_params.py` → default parameters  
- `fixed_neighborhood.py` → fixed neighborhood dynamics  
- `evolving_neighborhood.py` → dynamic neighborhood model  
- `models_therapy.py` → therapy extensions  

This structure avoids duplication and keeps notebooks focused on simulation + visualization.

---

## Notes  

- Simulations should be run long enough to ensure convergence  
- Parameter sweeps can be computationally intensive  
- Performance can be improved using parallelization if needed  
- Outputs are reused across notebooks to avoid recomputation  

---
