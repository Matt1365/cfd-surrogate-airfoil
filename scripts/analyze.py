#!/usr/bin/env python3
"""Aerodynamic analysis of the CFD sweep: Cl-alpha, Cd-alpha, drag polar, lift slope.

Reads data/sweep_results.csv, writes figures to results/figures/ and
key numbers to results/aero_summary.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data" / "sweep_results.csv")
# A case counts as force-converged when Cl varied < 1e-4 over its last 50 iterations
df["force_converged"] = df["Cl_range_last50"] < 1e-4
print(f"{len(df)} cases, residual-converged: {df.converged.sum()}, force-converged: {df.force_converged.sum()}")

speeds = sorted(df.U_inf.unique())
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(speeds)))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for u, c in zip(speeds, colors):
    d = df[df.U_inf == u].sort_values("alpha_deg")
    lab = f"Re = {d.Re.iloc[0]:.2e}"
    ax[0].plot(d.alpha_deg, d.Cl, "o-", color=c, label=lab)
    ax[1].plot(d.alpha_deg, d.Cd, "o-", color=c, label=lab)
    ax[2].plot(d.Cd, d.Cl, "o-", color=c, label=lab)
ax[0].set(xlabel="angle of attack [deg]", ylabel="Cl", title="Lift curve")
ax[1].set(xlabel="angle of attack [deg]", ylabel="Cd", title="Drag curve")
ax[2].set(xlabel="Cd", ylabel="Cl", title="Drag polar")
for a in ax:
    a.grid(alpha=0.3)
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "aero_curves.png", dpi=150)

# Lift slope and zero-lift angle from the linear range (-4..6 deg)
rows = []
for u in speeds:
    d = df[(df.U_inf == u) & (df.alpha_deg.between(-4, 6))]
    k, b = np.polyfit(d.alpha_deg, d.Cl, 1)
    full = df[df.U_inf == u]
    best = full.loc[(full.Cl / full.Cd).idxmax()]
    rows.append({
        "U_inf": u, "Re": full.Re.iloc[0],
        "dCl_dalpha_per_deg": k, "dCl_dalpha_per_rad": np.degrees(k),
        "alpha_zero_lift_deg": -b / k,
        "max_L_over_D": best.Cl / best.Cd, "alpha_at_max_L_over_D": best.alpha_deg,
    })
summary = pd.DataFrame(rows)
summary.to_csv(ROOT / "results" / "aero_summary.csv", index=False)
print(summary.round(4).to_string(index=False))
print(f"figures -> {FIG.relative_to(ROOT)}")
