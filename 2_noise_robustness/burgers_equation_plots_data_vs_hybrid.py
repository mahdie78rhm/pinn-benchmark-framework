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

# Two models only for the error plots
df_dh = df[df["Model"].isin(["Hybrid_PID", "Pure_DataDriven"])].copy()

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.15)
colors = {"Hybrid_PID": "#0275D8", "Pure_DataDriven": "#D9534F", "Pure_PINN": "#5CB85C"}

noise_levels = sorted(df_dh["Noise_Percent"].unique())
x = np.arange(len(noise_levels))

# =========================================================
# Plot 1: Shaded Uncertainty Plot (Hybrid and DataDriven only)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Hybrid_PID", "Pure_DataDriven"]:
    sub = df_dh[df_dh["Model"] == model_name].sort_values("Noise_Percent")
    plt.plot(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        label=model_name,
        color=colors[model_name],
        linewidth=2.5,
        marker="o",
        markersize=7,
    )
    lower = np.maximum(sub["Mean_Error"] - sub["Std_Error"], 1e-7)
    upper = sub["Mean_Error"] + sub["Std_Error"]
    plt.fill_between(sub["Noise_Percent"], lower, upper, color=colors[model_name], alpha=0.25)

plt.yscale("log")
plt.title(
    "Shaded Error Bounds: Hybrid_PID vs Pure_DataDriven\n(Noise Robustness – Burgers)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(noise_levels)
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("1_shaded_error_DH_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: Line Plot with Error Bars (two models only)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Hybrid_PID", "Pure_DataDriven"]:
    sub = df_dh[df_dh["Model"] == model_name].sort_values("Noise_Percent")
    plt.errorbar(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model_name,
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2.2,
        markersize=7,
        color=colors[model_name],
    )

plt.yscale("log")
plt.title(
    "Mean L2 Error with Error Bars (Log Scale)\nHybrid_PID vs Pure_DataDriven",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(noise_levels)
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_errorbar_DH_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: Grouped Bar Chart (error, two models only)
# =========================================================
plt.figure(figsize=(12, 6), dpi=300)

width = 0.35
hy = df_dh[df_dh["Model"] == "Hybrid_PID"].sort_values("Noise_Percent")
dd = df_dh[df_dh["Model"] == "Pure_DataDriven"].sort_values("Noise_Percent")

plt.bar(
    x - width / 2,
    hy["Mean_Error"],
    width,
    yerr=hy["Std_Error"],
    label="Hybrid_PID",
    color=colors["Hybrid_PID"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    dd["Mean_Error"],
    width,
    yerr=dd["Std_Error"],
    label="Pure_DataDriven",
    color=colors["Pure_DataDriven"],
    capsize=4,
    alpha=0.9,
)

plt.yscale("log")
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.title(
    "Error Comparison across Noise Levels\nHybrid_PID vs Pure_DataDriven",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xticks(x, [str(n) for n in noise_levels])
plt.legend(fontsize=12)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("3_grouped_bar_error_DH_noise.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# =========================================================
pivot = df_dh.pivot(index="Noise_Percent", columns="Model", values="Mean_Error").reset_index()
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

models_time = ["Hybrid_PID", "Pure_DataDriven", "Pure_PINN"]
width = 0.25

for i, model in enumerate(models_time):
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
plt.savefig("5_training_time_all_models_noise.png", dpi=300, bbox_inches="tight")
plt.show()

print("All 5 figures saved:")
print("   1_shaded_error_DH_noise.png")
print("   2_errorbar_DH_noise.png")
print("   3_grouped_bar_error_DH_noise.png")
print("   4_improvement_factor_noise.png")
print("   5_training_time_all_models_noise.png")
