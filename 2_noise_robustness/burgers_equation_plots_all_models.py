"""Plotting script for the Burgers-equation results of this phase.

The summary numbers below were copied from the corresponding experiment output.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# =========================================================
data_csv = """Noise_Percent,Model,Mean_Error,Std_Error,Mean_Time
0.0,Hybrid_PID,0.000781,0.000855,243.51
0.0,Pure_DataDriven,0.002693,0.001065,63.25
0.0,Pure_PINN,0.080099,0.190665,255.50
0.5,Hybrid_PID,0.000646,0.000155,248.95
0.5,Pure_DataDriven,0.003674,0.000453,55.04
0.5,Pure_PINN,0.080099,0.190665,255.50
1.0,Hybrid_PID,0.001133,0.000976,231.81
1.0,Pure_DataDriven,0.003862,0.001084,57.94
1.0,Pure_PINN,0.080099,0.190665,255.50
2.0,Hybrid_PID,0.001123,0.000147,248.42
2.0,Pure_DataDriven,0.017263,0.014989,213.53
2.0,Pure_PINN,0.080099,0.190665,255.50
5.0,Hybrid_PID,0.002079,0.000520,227.17
5.0,Pure_DataDriven,0.049098,0.055997,161.10
5.0,Pure_PINN,0.080099,0.190665,255.50
10.0,Hybrid_PID,0.004401,0.000672,215.18
10.0,Pure_DataDriven,0.354320,0.169884,329.49
10.0,Pure_PINN,0.080099,0.190665,255.50
15.0,Hybrid_PID,0.007374,0.003927,219.63
15.0,Pure_DataDriven,0.643455,0.090523,379.42
15.0,Pure_PINN,0.080099,0.190665,255.50
"""

df = pd.read_csv(io.StringIO(data_csv))

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.15)
colors = {"Hybrid_PID": "#0275D8", "Pure_DataDriven": "#D9534F", "Pure_PINN": "#5CB85C"}
models = ["Hybrid_PID", "Pure_DataDriven", "Pure_PINN"]
markers = {"Hybrid_PID": "s", "Pure_DataDriven": "o", "Pure_PINN": "D"}

noise_levels = sorted(df["Noise_Percent"].unique())
x = np.arange(len(noise_levels))

# =========================================================
# Plot 1: Shaded Uncertainty Plot (all three models)
# =========================================================
plt.figure(figsize=(11, 6.5), dpi=300)

for model in models:
    sub = df[df["Model"] == model].sort_values("Noise_Percent")
    plt.plot(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        label=model,
        color=colors[model],
        linewidth=2.4,
        marker=markers[model],
        markersize=7,
    )
    lower = np.maximum(sub["Mean_Error"] - sub["Std_Error"], 1e-7)
    upper = sub["Mean_Error"] + sub["Std_Error"]
    plt.fill_between(sub["Noise_Percent"], lower, upper, color=colors[model], alpha=0.22)

plt.yscale("log")
plt.title(
    "Shaded Error Bounds – Noise Robustness (Burgers)", fontsize=14, fontweight="bold", pad=12
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(noise_levels)
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("1_shaded_error_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: Line Plot with Error Bars (all three models)
# =========================================================
plt.figure(figsize=(11, 6.5), dpi=300)

for model in models:
    sub = df[df["Model"] == model].sort_values("Noise_Percent")
    plt.errorbar(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model,
        fmt="-" + markers[model],
        capsize=4,
        capthick=1.4,
        linewidth=2.1,
        markersize=7,
        color=colors[model],
    )

plt.yscale("log")
plt.title(
    "Mean L2 Error with Error Bars – Noise Robustness", fontsize=14, fontweight="bold", pad=12
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(noise_levels)
plt.legend(fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_errorbar_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: Grouped Bar Chart (error, all three models)
# =========================================================
plt.figure(figsize=(13, 6.5), dpi=300)

width = 0.25

for i, model in enumerate(models):
    sub = df[df["Model"] == model].sort_values("Noise_Percent")
    plt.bar(
        x + (i - 1) * width,
        sub["Mean_Error"],
        width,
        yerr=sub["Std_Error"],
        label=model,
        color=colors[model],
        capsize=3,
        alpha=0.9,
    )

plt.yscale("log")
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.title(
    "Error Comparison across Noise Levels – All Models", fontsize=14, fontweight="bold", pad=12
)
plt.xticks(x, [str(n) for n in noise_levels])
plt.legend(fontsize=11)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("3_grouped_bar_error_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 4: Improvement Factor (Hybrid vs Data-Driven)
# =========================================================
pivot = (
    df[df["Model"].isin(["Hybrid_PID", "Pure_DataDriven"])]
    .pivot(index="Noise_Percent", columns="Model", values="Mean_Error")
    .reset_index()
)
pivot["Improvement"] = pivot["Pure_DataDriven"] / pivot["Hybrid_PID"]

plt.figure(figsize=(10, 5.5), dpi=300)
bars = plt.bar(
    [str(n) + "%" for n in pivot["Noise_Percent"]],
    pivot["Improvement"],
    color="#2E4053",
    alpha=0.85,
)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval * 1.08,
        f"{yval:.0f}×" if yval >= 10 else f"{yval:.1f}×",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.yscale("log")
plt.title(
    "Improvement Factor: Hybrid_PID vs Pure_DataDriven", fontsize=13, fontweight="bold", pad=12
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (DataDriven / Hybrid)", fontsize=11)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_improvement_factor_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 5: Training time comparison (all three models)
# =========================================================
plt.figure(figsize=(12, 6), dpi=300)

width = 0.25

for i, model in enumerate(models):
    sub = df[df["Model"] == model].sort_values("Noise_Percent")
    plt.bar(
        x + (i - 1) * width, sub["Mean_Time"], width, label=model, color=colors[model], alpha=0.9
    )

plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Training Time (seconds)", fontsize=12, fontweight="bold")
plt.title(
    "Training Time Comparison – All Three Models\n(Noise Robustness – Burgers)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xticks(x, [str(n) for n in noise_levels])
plt.legend(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("5_training_time_noise.png", dpi=300, bbox_inches="tight")
plt.show()

print("All 5 figures saved:")
print("   1_shaded_error_noise.png")
print("   2_errorbar_noise.png")
print("   3_grouped_bar_error_noise.png")
print("   4_improvement_factor_noise.png")
print("   5_training_time_noise.png")
