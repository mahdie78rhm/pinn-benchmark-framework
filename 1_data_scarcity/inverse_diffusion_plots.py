"""Plotting script for the inverse-diffusion results of this phase.

The summary numbers below were copied from the corresponding experiment output.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================================================
# Summary results (copied from the experiment output)
# =========================================================
data_l2 = """Percent,Model,Mean_Error,Std_Error
10.0,Hybrid,0.004223,0.002435
10.0,Pure PINN,0.139529,0.094991
20.0,Hybrid,0.002026,0.001065
20.0,Pure PINN,0.139529,0.094991
50.0,Hybrid,0.001599,0.000614
50.0,Pure PINN,0.139529,0.094991
80.0,Hybrid,0.002462,0.001380
80.0,Pure PINN,0.139529,0.094991
100.0,Hybrid,0.003632,0.002293
100.0,Pure PINN,0.139529,0.094991
"""

data_c = """Percent,Model,Mean_C_Error,Std_C_Error,Mean_C_Pred
10.0,Hybrid,0.001408,0.000733,0.998592
10.0,Pure PINN,0.188989,0.146314,1.188989
20.0,Hybrid,0.001245,0.000450,0.998755
20.0,Pure PINN,0.188989,0.146314,1.188989
50.0,Hybrid,0.001031,0.000490,0.998969
50.0,Pure PINN,0.188989,0.146314,1.188989
80.0,Hybrid,0.001341,0.000687,0.998659
80.0,Pure PINN,0.188989,0.146314,1.188989
100.0,Hybrid,0.001428,0.000309,0.998572
100.0,Pure PINN,0.188989,0.146314,1.188989
"""

runtime_data = """Model,Mean_Runtime_min,Std_Runtime_min
Pure PINN,5.52,0.08
Hybrid,5.72,0.12
"""

df = pd.read_csv(io.StringIO(data_l2))
df_c = pd.read_csv(io.StringIO(data_c))
df_runtime = pd.read_csv(io.StringIO(runtime_data))

# Model filter
df_dh = df[df["Model"].isin(["Pure PINN", "Hybrid"])].copy()

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
colors = {"Pure PINN": "#D9534F", "Hybrid": "#0275D8"}

# =========================================================
# Plot 1: Shaded Error Bounds (Mean L2)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Pure PINN", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name].sort_values("Percent")

    plt.plot(
        sub["Percent"],
        sub["Mean_Error"],
        label=model_name,
        color=colors[model_name],
        linewidth=2.5,
        marker="o",
        markersize=7,
    )

    lower_bound = np.maximum(sub["Mean_Error"] - sub["Std_Error"], 1e-6)
    upper_bound = sub["Mean_Error"] + sub["Std_Error"]

    plt.fill_between(
        sub["Percent"],
        lower_bound,
        upper_bound,
        color=colors[model_name],
        alpha=0.25,
        label=f"{model_name} ± Std",
    )

plt.yscale("log")
plt.title(
    "Shaded Error Bounds: Pure PINN vs Hybrid (L2 Error)", fontsize=14, fontweight="bold", pad=15
)
plt.xlabel("Observation Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(df_dh["Percent"].unique())
plt.legend(fontsize=11, loc="upper right")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("1_shaded_line_plot_L2.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: Line Plot with Error Bars (Log Scale)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Pure PINN", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name].sort_values("Percent")
    plt.errorbar(
        sub["Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model_name,
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2.2,
        markersize=8,
        color=colors[model_name],
    )

plt.yscale("log")
plt.title("Mean L2 Error Comparison (Log Scale)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Observation Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(df_dh["Percent"].unique())
plt.legend(fontsize=12, loc="upper right", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_line_plot_log_scale_L2.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: Grouped Bar Chart
# =========================================================
plt.figure(figsize=(11, 6), dpi=300)

percentages = sorted(df_dh["Percent"].unique())
x = np.arange(len(percentages))
width = 0.35

dd_sub = df_dh[df_dh["Model"] == "Pure PINN"].sort_values("Percent")
hy_sub = df_dh[df_dh["Model"] == "Hybrid"].sort_values("Percent")

plt.bar(
    x - width / 2,
    dd_sub["Mean_Error"],
    width,
    yerr=dd_sub["Std_Error"],
    label="Pure PINN",
    color=colors["Pure PINN"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy_sub["Mean_Error"],
    width,
    yerr=hy_sub["Std_Error"],
    label="Hybrid",
    color=colors["Hybrid"],
    capsize=4,
    alpha=0.9,
)

plt.yscale("log")
plt.xlabel("Observation Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.title("Error Comparison across Data Percentages", fontsize=14, fontweight="bold", pad=15)
plt.xticks(x, [f"{int(p)}%" for p in percentages])
plt.legend(fontsize=12)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("3_grouped_bar_chart_L2.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 4: Improvement Factor (Pure PINN / Hybrid)
# =========================================================
pivot_df = df_dh.pivot(index="Percent", columns="Model", values="Mean_Error").reset_index()
pivot_df["Improvement_Factor"] = pivot_df["Pure PINN"] / pivot_df["Hybrid"]

plt.figure(figsize=(10, 5.5), dpi=300)
bars = plt.bar(
    [f"{int(p)}%" for p in pivot_df["Percent"]],
    pivot_df["Improvement_Factor"],
    color="#2E4053",
    alpha=0.85,
    edgecolor="black",
    linewidth=0.8,
)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval * 1.05,
        f"{yval:.0f}×" if yval >= 10 else f"{yval:.1f}×",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.yscale("log")
plt.title(
    "How Many Times Hybrid Performs Better Than Pure PINN", fontsize=13, fontweight="bold", pad=15
)
plt.xlabel("Observation Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Pure PINN / Hybrid)", fontsize=11)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_improvement_factor_L2.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 5: Parameter (C) identification error
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Pure PINN", "Hybrid"]:
    sub = df_c[df_c["Model"] == model_name].sort_values("Percent")
    plt.errorbar(
        sub["Percent"],
        sub["Mean_C_Error"],
        yerr=sub["Std_C_Error"],
        label=model_name,
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2.2,
        markersize=8,
        color=colors[model_name],
    )

plt.yscale("log")
plt.title("Parameter Identification Error |C_pred − 1|", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Observation Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean |C_pred − 1| (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(df_c["Percent"].unique())
plt.legend(fontsize=12, loc="upper right")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("5_parameter_C_error.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 6: Runtime comparison
# =========================================================
plt.figure(figsize=(8, 5.5), dpi=300)

models = df_runtime["Model"]
means = df_runtime["Mean_Runtime_min"]
stds = df_runtime["Std_Runtime_min"]
bar_colors = [colors[m] for m in models]

bars = plt.bar(
    models,
    means,
    yerr=stds,
    color=bar_colors,
    capsize=8,
    width=0.55,
    alpha=0.9,
    edgecolor="black",
    linewidth=1.0,
)

for bar, mean in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        mean + 0.15,
        f"{mean:.2f} min",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

plt.ylabel("Mean Runtime per Run (minutes)", fontsize=12, fontweight="bold")
plt.title("Runtime Comparison: Pure PINN vs Hybrid", fontsize=14, fontweight="bold", pad=15)
plt.ylim(0, max(means) * 1.25)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("6_runtime_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

print("All figures saved:")
print("   1_shaded_line_plot_L2.png")
print("   2_line_plot_log_scale_L2.png")
print("   3_grouped_bar_chart_L2.png")
print("   4_improvement_factor_L2.png")
print("   5_parameter_C_error.png")
print("   6_runtime_comparison.png")
