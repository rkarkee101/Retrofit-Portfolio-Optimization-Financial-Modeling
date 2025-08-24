# Retrofit Portfolio Optimization & Financial Modeling (with Uncertainty)

A Python tool to **prioritize building energy-efficiency retrofits** under a **budget constraint** using multiple financial metrics and optimization methods. It supports **uncertainty in annual savings** (low/base/high ranges + optional Monte Carlo), and generates publication-quality plots, including **NPV-vs-budget** curves with **uncertainty bands**.

> Built for school/building retrofit planning, but easily adapts to any project-selection / capital budgeting problem.

---

## ✨ Key Features

- **Multi-metric evaluation** per measure (with low/base/high):
  - **NPV** (Net Present Value)
  - **IRR** (Internal Rate of Return)
  - **Payback** (simple and discounted)
  - **PI** (Profitability Index = PV/Cost)
- **Optimization methods**:
  - Exact **0/1 knapsack (DP)** to maximize total NPV
  - **Greedy ROI** baseline (for comparison)
  - Metric-optimal **portfolio selection** at a fixed budget:
    - NPV‑optimal (max total NPV)
    - IRR‑optimal (max portfolio IRR on combined cashflows)
    - Payback‑optimal (min simple payback on combined cashflows)
    - PI‑optimal (max portfolio PI = PV/Cost)
- **Uncertainty analysis**:
  - Per-measure **low/base/high** ranges for *annual savings*
  - **Error-bar** plots (NPV, IRR, Payback, PI)
  - **Monte Carlo** (optional) for optimal total NPV distribution
  - **NPV-vs-budget** with shaded **uncertainty band** (Low↔High)

---

## 🧱 Repository Structure (suggested)

```
.
├── retrofit_optimization_uncertainty_metrics.py   # main script
├── README.md                                      # this file
└── figs/                                          # auto-saved plots (created on run)
```

---

## ⚙️ Installation

1. **Python**: 3.8+ recommended  
2. **Install dependencies**:
   ```bash
   pip install numpy matplotlib
   ```
   *(Optional)*: `pandas` if you later add CSV I/O.

---

## 🚀 Quick Start

1. Save the main script as:
   ```text
   retrofit_optimization_uncertainty_metrics.py
   ```
2. Open the script and adjust the **Inputs** (see next section).
3. Run:
   ```bash
   python retrofit_optimization_uncertainty_metrics.py
   ```
4. Check the console output and the plots saved under `./figs/` (if `save_figs=True`).

---

## ✍️ Inputs (What you need to edit)

Open the script and locate these sections near the top:

### 1) Global settings
```python
discount_rate = 0.05     # annual discount rate (e.g., 0.05 = 5%)
budget        = 40000    # fixed budget for fixed analysis
analysis_mode = "both"   # "fixed" | "sensitivity" | "both"
```

### 2) Uncertainty & plotting toggles
```python
enable_uncertainty = True
enable_monte_carlo = True   # set False to skip MC
mc_samples         = 500

min_budget = 0
max_budget = 90000
budget_step = 2000

save_figs = True
fig_dir   = "figs"
```

### 3) Measures (edit/add your projects)
Each measure needs:
- `name` (str),
- `cost` (USD upfront),
- `annual_savings_cost` (USD/year, **base** case),
- `annual_savings_range` `(low, high)` (USD/year),
- `lifetime` (years savings persist).

```python
measures = [
    {"name": "LED Lighting Upgrade",  "cost": 10000, "annual_savings_cost": 2000, "annual_savings_range": (1900, 2100), "lifetime": 10},
    {"name": "HVAC Modernization",    "cost": 25000, "annual_savings_cost": 3000, "annual_savings_range": (2700, 3300), "lifetime": 10},
    {"name": "Solar PV Installation", "cost": 30000, "annual_savings_cost": 4000, "annual_savings_range": (3600, 4400), "lifetime": 10},
    {"name": "Insulation Upgrade",    "cost":  8000, "annual_savings_cost":  600, "annual_savings_range":   (500,  700), "lifetime": 15},
    {"name": "Efficient Motors",      "cost": 12000, "annual_savings_cost": 1000, "annual_savings_range":   (900, 1100), "lifetime": 10},
    {"name": "Water Fixture Retrofit","cost":  5000, "annual_savings_cost":  800, "annual_savings_range":   (700,  900), "lifetime": 10},
]
```

> 💡 **Tip:** If you need to load measures from CSV later, keep the same field names and map them into dictionaries like above.

---

## 🧪 What the script produces

- **Console (fixed budget):**  
  - NPV‑optimal selection (exact DP)
  - IRR‑optimal, Payback‑optimal, PI‑optimal selections (portfolio-level)
  - If Monte Carlo is enabled: **selection robustness** (% times each measure is chosen)

- **Plots saved to `./figs/` (if enabled):**
  1) **Per‑measure uncertainty** (4 charts): NPV, IRR, Payback, PI (with error bars)
  2) **Monte Carlo** histogram of *optimal total NPV*
  3) **NPV vs Budget**: Optimal (DP) vs Greedy (ROI), with **Low–High band**

---

## 🧠 Theory / Methods

### Financial metrics
- **Annuity factor** for an $n$‑year level saving at discount rate $r$:
  \[
  \text{AF}(r,n) = \frac{1 - (1+r)^{-n}}{r} \quad (\text{if } r>0);\ \text{AF} = n \text{ if } r=0.
  \]
- **Present Value of savings**: $\text{PV} = S \times \text{AF}(r,n)$  
- **NPV**: $\text{NPV} = \text{PV} - \text{Cost}$  
- **IRR (per measure)**: solve for $r$ where NPV = 0; implemented by **bisection** on the annuity PV.  
- **Payback**:
  - **Simple**: `Cost / AnnualSavings` (years; ∞ if never)  
  - **Discounted**: sum of discounted annual savings reaches Cost; linear interpolation within the crossing year.  
- **Profitability Index (PI)**: $\text{PI} = \text{PV}/\text{Cost}$

### Portfolio metrics (fixed budget)
- **NPV‑optimal**: exact **0/1 knapsack (Dynamic Programming)** maximizing total NPV subject to total cost ≤ budget.  
- **IRR‑optimal**: brute force over feasible combos; compute **combined cashflows** and solve **portfolio IRR**. Tie‑break by higher NPV.  
- **Payback‑optimal**: brute force; minimize **simple payback** on combined cashflows (tie‑break by higher NPV).  
- **PI‑optimal**: brute force; maximize combined **PI = PV_total / Cost_total** (tie‑break by higher NPV).  
- **Greedy baseline**: rank by **undiscounted ROI** = (TotalLifetimeSavings / Cost), select until budget is used (skip ROI<1). For comparison only; may be suboptimal.

### Uncertainty
- **Per‑measure low/base/high** ranges for **annual_savings_cost** → propagate to NPV/IRR/Payback/PI (error bars).  
- **NPV‑vs‑budget**:
  - **Base**: DP curve (optimal total NPV)
  - **Low & High scenarios**: DP curves → shaded **uncertainty band**  
- **Monte Carlo (optional)**: sample annual savings within ranges; for each sample run DP → histogram of optimal total NPV and **selection robustness** (% selected).

---

## 🧭 How to Use (Step‑by‑Step)

1. **Edit measures and settings** in the script (see *Inputs*).  
2. **Choose a mode**:
   - `"fixed"`: fixed budget; prints metric‑optimal selections + uncertainty charts (+ MC if enabled)
   - `"sensitivity"`: budget sweep; plots optimal vs greedy + uncertainty band
   - `"both"`: do both in one run
3. **Run the script** and check `./figs/` for outputs.
4. **Decide** using the **metric‑optimal** results and **uncertainty plots**:  
   - *From NPV*: choose the mix that maximizes total value.  
   - *From IRR / Payback*: use if capital is scarce / liquidity risk matters.  
   - *From PI*: best “value per dollar”; size‑agnostic efficiency.  
   - Use **selection robustness** and **low‑case NPVs** to avoid fragile choices.

---

## 🧯 Troubleshooting

- **IndexError in payback**: fixed by making `combo_cashflows(..., years=50)` extend flows to the requested horizon **and** clamping `combo_payback(..., horizon)` to available length.  
- **`ValueError: 'yerr' must not contain negative values'`**: fixed via `make_asymm_yerr(...)` which builds **non‑negative** asymmetric error bars and masks NaN/∞.

---

## 📈 Extending

- Add **CSV I/O** to load measures.  
- Add **energy price escalation** or **O&M** costs.  
- Switch to **MILP/ILP** (e.g., `pulp`/`ortools`) for larger instances / additional constraints (mutual exclusivity, dependencies, minimum counts, etc.).  
- Build a **Streamlit** front‑end for interactive what‑ifs.

---

## 📝 License

MIT (or your preferred license).

---

## 🙌 Acknowledgments

Developed for building retrofit decision support and classroom demonstrations of capital budgeting under uncertainty.
