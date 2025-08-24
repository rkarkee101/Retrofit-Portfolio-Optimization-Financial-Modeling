# retrofit_optimization_uncertainty_metrics.py
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional

# =========================
# User Inputs / Configuration
# =========================
discount_rate = 0.05          # 5% annual discount rate
budget        = 40000         # Fixed budget for "fixed" analysis
analysis_mode = "both"        # "fixed", "sensitivity", or "both"

# Uncertainty toggles
enable_uncertainty   = True
enable_monte_carlo   = True   # Monte Carlo at fixed budget
mc_samples           = 500

# Sensitivity sweep (used if analysis_mode includes "sensitivity")
min_budget = 0
max_budget = 90000            # up to or beyond sum of costs
budget_step = 2000

# Save figures?
save_figs = True
fig_dir   = "figs"

# =========================
# Measures with uncertainty
# =========================
# annual_savings_range = (low, high)
measures = [
    {"name": "LED Lighting Upgrade",  "cost": 10000, "annual_savings_cost": 2000, "annual_savings_range": (1900, 2100), "lifetime": 10},
    {"name": "HVAC Modernization",    "cost": 25000, "annual_savings_cost": 3000, "annual_savings_range": (2700, 3300), "lifetime": 10},
    {"name": "Solar PV Installation", "cost": 30000, "annual_savings_cost": 4000, "annual_savings_range": (3600, 4400), "lifetime": 10},
    {"name": "Insulation Upgrade",    "cost":  8000, "annual_savings_cost":  600, "annual_savings_range":   (500,  700), "lifetime": 15},
    {"name": "Efficient Motors",      "cost": 12000, "annual_savings_cost": 1000, "annual_savings_range":   (900, 1100), "lifetime": 10},
    {"name": "Water Fixture Retrofit","cost":  5000, "annual_savings_cost":  800, "annual_savings_range":   (700,  900), "lifetime": 10},
]

# =========================
# Helpers
# =========================
def ensure_dir(d: str):
    if save_figs:
        os.makedirs(d, exist_ok=True)

def annuity_factor(r: float, n: int) -> float:
    return n if r == 0 else (1 - (1 + r) ** -n) / r

def pv_of_savings(annual_savings: float, n: int, r: float) -> float:
    return annual_savings * annuity_factor(r, n)

def compute_npv(cost: float, annual_savings: float, n: int, r: float) -> float:
    return pv_of_savings(annual_savings, n, r) - cost

def simple_payback(cost: float, annual_savings: float) -> float:
    if annual_savings <= 0:
        return np.inf
    return cost / annual_savings  # fractional years allowed

def discounted_payback(cost: float, annual_savings: float, r: float, n: int) -> float:
    """Discounted payback (linear interpolation within the payback year). Returns inf if never within n years."""
    if annual_savings <= 0:
        return np.inf
    cum = 0.0
    for year in range(1, n+1):
        inc = annual_savings / ((1 + r) ** year)
        cum += inc
        if cum >= cost:
            prev_cum = cum - inc
            remaining = cost - prev_cum
            frac = remaining / inc if inc != 0 else 0.0
            return (year - 1) + frac
    return np.inf

def irr_for_annuity(cost: float, annual_savings: float, n: int) -> float:
    """
    Solve IRR for cashflow: -cost at t=0, +annual_savings for years 1..n.
    Returns np.nan if no positive IRR exists.
    """
    if annual_savings <= 0:
        return np.nan
    if annual_savings * n - cost <= 0:
        return np.nan

    def npv_at(r):
        return annual_savings * annuity_factor(r, n) - cost

    lo, hi = 0.0, 1.0
    f_lo = npv_at(lo)
    f_hi = npv_at(hi)
    iters = 0
    while f_lo * f_hi > 0 and hi < 10.0:
        hi *= 2.0
        f_hi = npv_at(hi)
        iters += 1
        if iters > 50:
            break
    if f_lo * f_hi > 0:
        return np.nan

    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = npv_at(mid)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid < 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return 0.5 * (lo + hi)

def compute_measure_metrics(m: Dict, r: float) -> Dict:
    """Return base/low/high for NPV, IRR, Payback (simple & discounted), PI."""
    S_base = m["annual_savings_cost"]
    S_low, S_high = m["annual_savings_range"]
    c, n = m["cost"], m["lifetime"]

    def one_case(S):
        npv = compute_npv(c, S, n, r)
        pv  = pv_of_savings(S, n, r)
        pi  = pv / c if c > 0 else np.inf
        irr = irr_for_annuity(c, S, n)
        pb  = simple_payback(c, S)
        dpb = discounted_payback(c, S, r, n)
        return npv, irr, pb, dpb, pi

    base = one_case(S_base)
    low  = one_case(S_low)
    high = one_case(S_high)

    keys = ["NPV", "IRR", "PB_simple", "PB_discounted", "PI"]
    out  = {}
    for i, k in enumerate(keys):
        out[f"{k}_base"] = base[i]
        out[f"{k}_low"]  = low[i]
        out[f"{k}_high"] = high[i]
    return out

def optimize_dp(costs: List[int], values: List[float], capacity: int) -> Tuple[float, List[int]]:
    """0/1 knapsack DP: returns (best_value, chosen_indices)"""
    n = len(costs)
    W = capacity
    dp = [[0.0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        c = costs[i-1]; v = values[i-1]
        for w in range(W+1):
            if c <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w-c] + v)
            else:
                dp[i][w] = dp[i-1][w]
    # backtrack
    chosen = []
    w = W
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            chosen.append(i-1)
            w -= costs[i-1]
    chosen.reverse()
    return dp[n][W], chosen

def greedy_by_roi(items: List[Dict], capacity: int, roi_key="ROI_undisc") -> Tuple[float, List[int]]:
    """Greedy: pick by ROI (default: total lifetime savings / cost), skipping ROI<1."""
    order = sorted(
        [(i, items[i][roi_key]) for i in range(len(items))],
        key=lambda x: x[1],
        reverse=True
    )
    remaining = capacity
    total_npv = 0.0
    chosen = []
    for idx, _ in order:
        m = items[idx]
        if m["ROI_undisc"] < 1.0:
            continue
        if m["cost"] <= remaining:
            remaining -= m["cost"]
            total_npv += m["NPV_base"]
            chosen.append(idx)
    return total_npv, chosen

def combo_cashflows(measure_indices: List[int], measures: List[Dict], years: int = None) -> List[float]:
    """
    Combined cashflows: [t0, t1, ..., tT]
    t0 = -sum(cost); for t=1..lifetime, add annual savings; zeros thereafter.
    If `years` provided, ensure at least that horizon.
    """
    total_cost = -sum(measures[i]["cost"] for i in measure_indices)
    max_life = max((measures[i]["lifetime"] for i in measure_indices), default=0)
    T = max_life
    if years is not None:
        T = max(T, int(years))
    flows = [0.0] * (T + 1)
    flows[0] = total_cost
    for i in measure_indices:
        S = measures[i]["annual_savings_cost"]
        n = measures[i]["lifetime"]
        for t in range(1, n + 1):
            flows[t] += S
    return flows

def irr_from_flows(flows: List[float]) -> float:
    """IRR for general cashflows (bisection); returns np.nan if no positive IRR."""
    def npv_at(r):
        return sum(cf/((1+r)**t) for t, cf in enumerate(flows))
    if npv_at(0.0) <= 0:
        return np.nan
    lo, hi = 0.0, 1.0
    f_lo = npv_at(lo); f_hi = npv_at(hi)
    iters = 0
    while f_lo * f_hi > 0 and hi < 10.0:
        hi *= 2
        f_hi = npv_at(hi)
        iters += 1
        if iters > 60:
            break
    if f_lo * f_hi > 0:
        return np.nan
    for _ in range(100):
        mid = 0.5*(lo+hi)
        f_mid = npv_at(mid)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid < 0:
            hi = mid; f_hi = f_mid
        else:
            lo = mid; f_lo = f_mid
    return 0.5*(lo+hi)

def combo_payback(flows: List[float], r: float, discounted: bool, horizon: int = None) -> float:
    """
    Payback time for combined cashflows.
    Clamps horizon to len(flows)-1 to avoid OOB. Returns inf if never paid back.
    """
    max_t = len(flows) - 1
    T = max_t if horizon is None else min(int(horizon), max_t)

    target = -flows[0]
    if target <= 0:
        return 0.0

    cum = 0.0
    for t in range(1, T + 1):
        inc = flows[t] / ((1 + r) ** t) if discounted else flows[t]
        cum += inc
        if cum >= target:
            prev = cum - inc
            frac = 0.0 if inc == 0 else (target - prev) / inc
            return (t - 1) + frac
    return np.inf

def make_asymm_yerr(low_vals, base_vals, high_vals, scale=1.0):
    """
    Build (2,N) yerr with non-negative magnitudes.
    - Uses abs(base-low) and abs(high-base).
    - Masks NaN/inf by setting error to 0 so matplotlib is happy.
    - Optional scale factor (e.g., 100 for IRR %).
    """
    base = np.array(base_vals, dtype=float)
    low  = np.array(low_vals,  dtype=float)
    high = np.array(high_vals, dtype=float)

    lower = np.abs(base - low) * scale
    upper = np.abs(high - base) * scale

    finite_lower = np.isfinite(base) & np.isfinite(low)
    finite_upper = np.isfinite(base) & np.isfinite(high)

    lower = np.where(finite_lower, lower, 0.0)
    upper = np.where(finite_upper, upper, 0.0)

    return np.vstack([lower, upper])

# =========================
# Pre-compute metrics (low/base/high)
# =========================
for m in measures:
    metrics = compute_measure_metrics(m, discount_rate)
    m.update(metrics)
    m["NPV"] = m["NPV_base"]
    m["ROI_undisc"] = (m["annual_savings_cost"] * m["lifetime"]) / m["cost"] if m["cost"]>0 else np.inf

# =========================
# Fixed-budget: metric-optimal selections + uncertainty plots
# =========================
ensure_dir(fig_dir)

if analysis_mode in ["fixed","both"]:
    costs = [m["cost"] for m in measures]

    # ---------- NPV Optimal (DP) ----------
    base_values = [m["NPV_base"] for m in measures]
    best_npv, chosen_npv = optimize_dp(costs, base_values, budget)

    # ---------- IRR Optimal (brute force on combined cashflows) ----------
    best_irr, best_irr_combo, best_irr_npv = -1.0, [], -np.inf
    for rcount in range(1, len(measures)+1):
        for combo in itertools.combinations(range(len(measures)), rcount):
            tot_cost = sum(measures[i]["cost"] for i in combo)
            if tot_cost <= budget:
                flows = combo_cashflows(list(combo), measures, years=50)
                irr = irr_from_flows(flows)
                if np.isnan(irr):
                    continue
                npv_combo = sum(measures[i]["NPV_base"] for i in combo)
                if irr > best_irr + 1e-9 or (abs(irr-best_irr)<=1e-9 and npv_combo>best_irr_npv):
                    best_irr, best_irr_combo, best_irr_npv = irr, list(combo), npv_combo

    # ---------- Payback Optimal (min simple payback on combined cashflows) ----------
    best_pb, best_pb_combo, best_pb_npv = np.inf, [], -np.inf
    for rcount in range(1, len(measures)+1):
        for combo in itertools.combinations(range(len(measures)), rcount):
            tot_cost = sum(measures[i]["cost"] for i in combo)
            if tot_cost <= budget:
                flows = combo_cashflows(list(combo), measures, years=50)
                pb = combo_payback(flows, r=0.0, discounted=False, horizon=50)
                if np.isfinite(pb):
                    npv_combo = sum(measures[i]["NPV_base"] for i in combo)
                    if pb < best_pb - 1e-9 or (abs(pb-best_pb)<=1e-9 and npv_combo>best_pb_npv):
                        best_pb, best_pb_combo, best_pb_npv = pb, list(combo), npv_combo

    # ---------- PI Optimal (max PV/Cost on combined) ----------
    best_pi, best_pi_combo, best_pi_npv = -np.inf, [], -np.inf
    for rcount in range(1, len(measures)+1):
        for combo in itertools.combinations(range(len(measures)), rcount):
            tot_cost = sum(measures[i]["cost"] for i in combo)
            if tot_cost <= budget and tot_cost>0:
                total_pv = sum(pv_of_savings(measures[i]["annual_savings_cost"], measures[i]["lifetime"], discount_rate) for i in combo)
                pi = total_pv / tot_cost
                npv_combo = total_pv - tot_cost
                if pi > best_pi + 1e-12 or (abs(pi-best_pi)<=1e-12 and npv_combo>best_pi_npv):
                    best_pi, best_pi_combo, best_pi_npv = pi, list(combo), npv_combo

    # ---------- Print summary ----------
    def names(combo): return [measures[i]["name"] for i in combo]
    print("\n=== FIXED BUDGET ANALYSIS (with multi-metric recommendations) ===")
    print(f"Budget: ${budget:,}")

    print(f"\nNPV-optimal:     {names(chosen_npv)}  | Total NPV = ${best_npv:,.2f}")
    if best_irr_combo:
        print(f"IRR-optimal:     {names(best_irr_combo)}  | Portfolio IRR ≈ {best_irr*100:,.2f}%  | NPV = ${best_irr_npv:,.2f}")
    else:
        print("IRR-optimal:     No feasible combo with positive IRR.")
    if best_pb_combo:
        print(f"Payback-optimal: {names(best_pb_combo)}  | Simple Payback ≈ {best_pb:,.2f} years  | NPV = ${best_pb_npv:,.2f}")
    else:
        print("Payback-optimal: No feasible combo pays back within horizon.")
    if best_pi_combo:
        print(f"PI-optimal:      {names(best_pi_combo)}  | Portfolio PI ≈ {best_pi:,.3f}  | NPV = ${best_pi_npv:,.2f}")
    else:
        print("PI-optimal:      No feasible combo with finite PI.")

    # ---------- Per-measure uncertainty plots ----------
    if enable_uncertainty:
        names_m = [m["name"] for m in measures]
        xpos = np.arange(len(names_m))

        # 1) NPV error bars
        npv_b = [m["NPV_base"] for m in measures]
        npv_l = [m["NPV_low"]  for m in measures]
        npv_h = [m["NPV_high"] for m in measures]
        plt.figure(figsize=(9,5))
        plt.bar(names_m, npv_b)
        yerr = make_asymm_yerr(npv_l, npv_b, npv_h)
        plt.errorbar(x=xpos, y=npv_b, yerr=yerr, fmt='none', capsize=5, linewidth=1)
        plt.axhline(0, linewidth=0.8)
        plt.xticks(rotation=45, ha='right'); plt.ylabel("NPV (USD)")
        plt.title("Per-Measure NPV with Annual-Savings Uncertainty")
        plt.tight_layout(); 
        if save_figs: plt.savefig(os.path.join(fig_dir, "uncertainty_npv_per_measure.png"), dpi=150)
        plt.show()

        # 2) IRR error bars (%)
        irr_b = [m["IRR_base"] for m in measures]
        irr_l = [m["IRR_low"]  for m in measures]
        irr_h = [m["IRR_high"] for m in measures]
        irr_b_pct = np.array(irr_b, dtype=float) * 100.0
        irr_b_pct = np.where(np.isfinite(irr_b_pct), irr_b_pct, np.nan)
        plt.figure(figsize=(9,5))
        plt.bar(names_m, irr_b_pct)
        yerr = make_asymm_yerr(irr_l, irr_b, irr_h, scale=100.0)
        plt.errorbar(x=xpos, y=irr_b_pct, yerr=yerr, fmt='none', capsize=5, linewidth=1)
        plt.xticks(rotation=45, ha='right'); plt.ylabel("IRR (%)")
        plt.title("Per-Measure IRR with Annual-Savings Uncertainty")
        plt.tight_layout(); 
        if save_figs: plt.savefig(os.path.join(fig_dir, "uncertainty_irr_per_measure.png"), dpi=150)
        plt.show()

        # 3) Simple Payback error bars
        pb_b = np.array([m["PB_simple_base"] for m in measures], dtype=float)
        pb_l = np.array([m["PB_simple_low"]  for m in measures], dtype=float)
        pb_h = np.array([m["PB_simple_high"] for m in measures], dtype=float)
        pb_b_plot = np.where(np.isfinite(pb_b), pb_b, np.nan)
        plt.figure(figsize=(9,5))
        plt.bar(names_m, pb_b_plot)
        yerr = make_asymm_yerr(pb_l, pb_b, pb_h)
        plt.errorbar(x=xpos, y=pb_b_plot, yerr=yerr, fmt='none', capsize=5, linewidth=1)
        plt.xticks(rotation=45, ha='right'); plt.ylabel("Simple Payback (years)")
        plt.title("Per-Measure Simple Payback with Uncertainty")
        plt.tight_layout(); 
        if save_figs: plt.savefig(os.path.join(fig_dir, "uncertainty_payback_per_measure.png"), dpi=150)
        plt.show()

        # 4) Profitability Index error bars
        pi_b = np.array([m["PI_base"] for m in measures], dtype=float)
        pi_l = np.array([m["PI_low"]  for m in measures], dtype=float)
        pi_h = np.array([m["PI_high"] for m in measures], dtype=float)
        pi_b_plot = np.where(np.isfinite(pi_b), pi_b, np.nan)
        plt.figure(figsize=(9,5))
        plt.bar(names_m, pi_b_plot)
        yerr = make_asymm_yerr(pi_l, pi_b, pi_h)
        plt.errorbar(x=xpos, y=pi_b_plot, yerr=yerr, fmt='none', capsize=5, linewidth=1)
        plt.xticks(rotation=45, ha='right'); plt.ylabel("Profitability Index (PV/Cost)")
        plt.title("Per-Measure Profitability Index with Uncertainty")
        plt.tight_layout(); 
        if save_figs: plt.savefig(os.path.join(fig_dir, "uncertainty_pi_per_measure.png"), dpi=150)
        plt.show()

    # ---------- Monte Carlo at fixed budget ----------
    if enable_uncertainty and enable_monte_carlo:
        rng = np.random.default_rng(42)
        totals = []
        pick_counts = np.zeros(len(measures), dtype=int)
        for _ in range(mc_samples):
            sampled_values = []
            for m in measures:
                lo, hi = m["annual_savings_range"]
                s = rng.uniform(lo, hi)
                sampled_values.append(compute_npv(m["cost"], s, m["lifetime"], discount_rate))
            total, chosen = optimize_dp([m["cost"] for m in measures], sampled_values, budget)
            totals.append(total)
            for idx in chosen:
                pick_counts[idx] += 1
        totals = np.array(totals)

        plt.figure(figsize=(7,5))
        plt.hist(totals, bins=25)
        plt.xlabel("Optimal Total NPV (USD)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Optimal Total NPV (Monte Carlo, n={mc_samples})")
        plt.tight_layout()
        if save_figs: plt.savefig(os.path.join(fig_dir, "mc_opt_total_npv_hist.png"), dpi=150)
        plt.show()

        print("\nSelection robustness across Monte Carlo (percent of runs selected):")
        for i, m in enumerate(measures):
            print(f"  {m['name']}: {100.0*pick_counts[i]/mc_samples:5.1f}%")

# =========================
# Budget sensitivity: Optimal vs Greedy + Uncertainty Band
# =========================
if analysis_mode in ["sensitivity","both"]:
    budgets = list(range(min_budget, max_budget+1, budget_step))
    costs   = [m["cost"] for m in measures]

    # Base DP table up to max budget
    n = len(costs); W = max_budget
    base_values = [m["NPV_base"] for m in measures]
    dp = [[0.0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        c = costs[i-1]; v = base_values[i-1]
        for w in range(W+1):
            dp[i][w] = max(dp[i-1][w], dp[i-1][w-c] + v) if c <= w else dp[i-1][w]
    optimal_npvs = [dp[n][B] for B in budgets]

    # Greedy (ROI) curve (using base scenario for aggregation)
    greedy_npvs = []
    for B in budgets:
        total_g, _ = greedy_by_roi(measures, B, roi_key="ROI_undisc")
        greedy_npvs.append(total_g)

    # Uncertainty band from Low & High scenarios
    low_values  = [m["NPV_low"]  for m in measures]
    high_values = [m["NPV_high"] for m in measures]
    dp_low = [[0.0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        c = costs[i-1]; v = low_values[i-1]
        for w in range(W+1):
            dp_low[i][w] = max(dp_low[i-1][w], dp_low[i-1][w-c] + v) if c <= w else dp_low[i-1][w]
    low_npvs = [dp_low[n][B] for B in budgets]

    dp_high = [[0.0]*(W+1) for _ in range(n+1)]
    for i in range(1, n+1):
        c = costs[i-1]; v = high_values[i-1]
        for w in range(W+1):
            dp_high[i][w] = max(dp_high[i-1][w], dp_high[i-1][w-c] + v) if c <= w else dp_high[i-1][w]
    high_npvs = [dp_high[n][B] for B in budgets]

    # Plot sensitivity with band
    plt.figure(figsize=(8,5))
    plt.plot(budgets, optimal_npvs, linewidth=3, label="Optimal (Max NPV, Base)")
    plt.plot(budgets, greedy_npvs, linewidth=2, linestyle='--', label="Greedy (High ROI first)")
    plt.fill_between(budgets, low_npvs, high_npvs, alpha=0.2, label="Uncertainty band (Low–High)")
    plt.axvline(x=budget, linestyle=':', label=f"Reference Budget ${budget:,}")
    plt.xlabel("Budget (USD)"); plt.ylabel("Total NPV (USD)")
    plt.title("Optimal Total NPV vs Budget\n(with Greedy Comparison & Savings-Uncertainty Band)")
    plt.legend()
    plt.tight_layout()
    if save_figs: plt.savefig(os.path.join(fig_dir, "sensitivity_npv_vs_budget_band.png"), dpi=150)
    plt.show()
