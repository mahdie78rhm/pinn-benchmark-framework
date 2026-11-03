"""Plotting script for the Burgers-equation results of this phase.

The summary numbers below were copied from the corresponding experiment output.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================================================
# Summary results (copied from the experiment output)
# =========================================================
data = {
    "T_Train": [0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.6, 0.6, 0.6],
    "Model": [
        "Data-Driven",
        "PINN_NoObs",
        "PINN_WithObs",
        "Data-Driven",
        "PINN_NoObs",
        "PINN_WithObs",
        "Data-Driven",
        "PINN_NoObs",
        "PINN_WithObs",
    ],
    "L2_Interp_mean": [
        0.001714,
        0.012948,
        0.010358,
        0.003358,
        0.024414,
        0.007948,
        0.005594,
        0.047127,
        0.006002,
    ],
    "L2_Interp_std": [
        0.000444,
        0.006971,
        0.015273,
        0.001529,
        0.016249,
        0.004527,
        0.003900,
        0.029197,
        0.003541,
    ],
    "L2_Extrap_mean": [
        0.499426,
        0.075453,
        0.058549,
        0.367757,
        0.090630,
        0.087243,
        0.140314,
        0.098759,
        0.007802,
    ],
    "L2_Extrap_std": [
        0.069738,
        0.050801,
        0.107033,
        0.065511,
        0.060929,
        0.167433,
        0.075822,
        0.070097,
        0.005028,
    ],
    "RMSE_Extrap_mean": [
        0.293864,
        0.044397,
        0.034450,
        0.201539,
        0.049668,
        0.047811,
        0.070035,
        0.049294,
        0.003894,
    ],
    "RMSE_Extrap_std": [
        0.041034,
        0.029891,
        0.062979,
        0.035901,
        0.033390,
        0.091757,
        0.037845,
        0.034988,
        0.002510,
    ],
    "Time_mean": [46.5818, 206.523, 222.809, 48.4437, 206.701, 224.199, 47.5900, 206.682, 209.581],
    "Time_std": [1.66645, 7.65448, 34.2193, 4.43987, 8.68622, 33.5888, 3.11114, 7.62136, 8.19232],
}

df = pd.DataFrame(data)

avg_time = {
    "Data-Driven": 47.54,
    "PINN_NoObs": 206.64,
    "PINN_WithObs": 218.86,
}
avg_time_std = {
    "Data-Driven": 3.17,
    "PINN_NoObs": 7.52,
    "PINN_WithObs": 27.24,
}

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.15)
colors = {
    "Data-Driven": "#D9534F",
    "PINN_NoObs": "#5CB85C",
    "PINN_WithObs": "#0275D8",
}
markers = {"Data-Driven": "o", "PINN_NoObs": "s", "PINN_WithObs": "D"}
models = ["Data-Driven", "PINN_NoObs", "PINN_WithObs"]
t_values = [0.2, 0.4, 0.6]

# =========================================================
# Plot 1: L2 Error Interpolation
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model in models:
    sub = df[df["Model"] == model].sort_values("T_Train")
    plt.errorbar(
        sub["T_Train"],
        sub["L2_Interp_mean"],
        yerr=sub["L2_Interp_std"],
        label=model,
        fmt=f"-{markers[model]}",
        color=colors[model],
        linewidth=2.2,
        markersize=8,
        capsize=5,
        capthick=1.5,
    )

plt.yscale("log")
plt.xlabel("T_Train", fontsize=13, fontweight="bold")
plt.ylabel("L2 Error (Interpolation) – Log Scale", fontsize=12, fontweight="bold")
plt.title("L2 Error – Interpolation Region", fontsize=14, fontweight="bold", pad=12)
plt.xticks(t_values)
plt.legend(fontsize=11, frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("1_L2_Interpolation.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: L2 Error Extrapolation
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model in models:
    sub = df[df["Model"] == model].sort_values("T_Train")
    plt.errorbar(
        sub["T_Train"],
        sub["L2_Extrap_mean"],
        yerr=sub["L2_Extrap_std"],
        label=model,
        fmt=f"-{markers[model]}",
        color=colors[model],
        linewidth=2.2,
        markersize=8,
        capsize=5,
        capthick=1.5,
    )

plt.yscale("log")
plt.xlabel("T_Train", fontsize=13, fontweight="bold")
plt.ylabel("L2 Error (Extrapolation) – Log Scale", fontsize=12, fontweight="bold")
plt.title("L2 Error – Extrapolation Region", fontsize=14, fontweight="bold", pad=12)
plt.xticks(t_values)
plt.legend(fontsize=11, frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("2_L2_Extrapolation.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: RMSE Extrapolation
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model in models:
    sub = df[df["Model"] == model].sort_values("T_Train")
    plt.errorbar(
        sub["T_Train"],
        sub["RMSE_Extrap_mean"],
        yerr=sub["RMSE_Extrap_std"],
        label=model,
        fmt=f"-{markers[model]}",
        color=colors[model],
        linewidth=2.2,
        markersize=8,
        capsize=5,
        capthick=1.5,
    )

plt.yscale("log")
plt.xlabel("T_Train", fontsize=13, fontweight="bold")
plt.ylabel("RMSE (Extrapolation) – Log Scale", fontsize=12, fontweight="bold")
plt.title("RMSE – Extrapolation Region", fontsize=14, fontweight="bold", pad=12)
plt.xticks(t_values)
plt.legend(fontsize=11, frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("3_RMSE_Extrapolation.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 4: Training time comparison (Grouped Bar)
# =========================================================
plt.figure(figsize=(11, 6), dpi=300)

x = np.arange(len(t_values))
width = 0.25

for i, model in enumerate(models):
    sub = df[df["Model"] == model].sort_values("T_Train")
    plt.bar(
        x + (i - 1) * width,
        sub["Time_mean"],
        width,
        yerr=sub["Time_std"],
        label=model,
        color=colors[model],
        capsize=4,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )

plt.xlabel("T_Train", fontsize=13, fontweight="bold")
plt.ylabel("Training Time (seconds)", fontsize=12, fontweight="bold")
plt.title("Training Time Comparison across T_Train Levels", fontsize=14, fontweight="bold", pad=12)
plt.xticks(x, t_values)
plt.legend(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("4_Time_Comparison_Grouped.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 5: Overall average training time (Overall Average)
# =========================================================
plt.figure(figsize=(8, 5.5), dpi=300)

model_names = list(avg_time.keys())
means = [avg_time[m] for m in model_names]
stds = [avg_time_std[m] for m in model_names]
bar_colors = [colors[m] for m in model_names]

bars = plt.bar(
    model_names,
    means,
    yerr=stds,
    color=bar_colors,
    capsize=6,
    alpha=0.9,
    edgecolor="black",
    linewidth=0.8,
    width=0.55,
)

for bar, mean in zip(bars, means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 8,
        f"{mean:.1f}s",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.ylabel("Average Training Time (seconds)", fontsize=12, fontweight="bold")
plt.title("Overall Average Training Time per Model", fontsize=14, fontweight="bold", pad=12)
plt.grid(axis="y", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("5_Overall_Average_Time.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 6: Improvement factor (extrapolation) (Data-Driven / PINN_WithObs)
# =========================================================
pivot = df.pivot(index="T_Train", columns="Model", values="L2_Extrap_mean").reset_index()
pivot["Improvement"] = pivot["Data-Driven"] / pivot["PINN_WithObs"]

plt.figure(figsize=(9, 5.5), dpi=300)
bars = plt.bar(
    [str(t) for t in pivot["T_Train"]],
    pivot["Improvement"],
    color="#2E4053",
    alpha=0.88,
    edgecolor="black",
    width=0.5,
)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height * 1.05,
        f"{height:.1f}×",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

plt.yscale("log")
plt.xlabel("T_Train", fontsize=13, fontweight="bold")
plt.ylabel("Improvement Factor (Data-Driven / PINN_WithObs)", fontsize=11, fontweight="bold")
plt.title(
    "How Many Times PINN_WithObs is Better in Extrapolation", fontsize=13, fontweight="bold", pad=12
)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("6_Improvement_Factor_Extrap.png", dpi=300, bbox_inches="tight")
plt.show()

print("All 6 figures saved successfully:")
print("   1_L2_Interpolation.png")
print("   2_L2_Extrapolation.png")
print("   3_RMSE_Extrapolation.png")
print("   4_Time_Comparison_Grouped.png")
print("   5_Overall_Average_Time.png")
print("   6_Improvement_Factor_Extrap.png")
