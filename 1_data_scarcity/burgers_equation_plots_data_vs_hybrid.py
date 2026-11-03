"""Plotting script for the Burgers-equation results of this phase.

The summary numbers below were copied from the corresponding experiment output.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# Summary results (copied from the experiment output)
# =========================================================
data_csv = """Percent,Model,Mean_Error,Std_Error,Mean_Time,Std_Time
2.0,Data-Driven,0.211681,0.047006,43.565103,0.623318
2.0,Hybrid,0.018762,0.031488,199.299523,19.099846
2.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
5.0,Data-Driven,0.135917,0.024025,44.623930,1.044354
5.0,Hybrid,0.002882,0.001916,198.473946,9.852133
5.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
10.0,Data-Driven,0.106626,0.059074,45.951133,2.093608
10.0,Hybrid,0.001199,0.000866,198.307832,10.596193
10.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
20.0,Data-Driven,0.048236,0.023417,47.028870,3.600666
20.0,Hybrid,0.000990,0.000557,194.040846,4.773302
20.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
50.0,Data-Driven,0.012223,0.006192,47.309838,3.626046
50.0,Hybrid,0.000467,0.000165,196.271674,6.086445
50.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
80.0,Data-Driven,0.007218,0.004322,48.282001,4.020263
80.0,Hybrid,0.000510,0.000170,194.943851,6.541908
80.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
100.0,Data-Driven,0.004369,0.001243,51.572495,4.417543
100.0,Hybrid,0.000589,0.000132,197.216309,13.134618
100.0,Pure PINN,0.117814,0.275912,216.453083,60.315403
"""

df = pd.read_csv(io.StringIO(data_csv))

# Two models only for the error plots
df_dh = df[df["Model"].isin(["Data-Driven", "Hybrid"])].copy()

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.15)
colors = {"Data-Driven": "#D9534F", "Hybrid": "#0275D8", "Pure PINN": "#5CB85C"}

# =========================================================
# Plot 1: Shaded Uncertainty Plot (Data-Driven and Hybrid only)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Data-Driven", "Hybrid"]:
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
    lower = np.maximum(sub["Mean_Error"] - sub["Std_Error"], 1e-7)
    upper = sub["Mean_Error"] + sub["Std_Error"]
    plt.fill_between(sub["Percent"], lower, upper, color=colors[model_name], alpha=0.25)

plt.yscale("log")
plt.xscale("log")
plt.title(
    "Shaded Error Bounds: Data-Driven vs Hybrid\n(Phase 1 – Data Scarcity)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Data Percentage (%) – Log Scale", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks([2, 5, 10, 20, 50, 80, 100], labels=["2", "5", "10", "20", "50", "80", "100"])
plt.legend(fontsize=11, loc="upper right")
plt.grid(True, which="both", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("1_shaded_error_DH.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: Line Plot with Error Bars (two models only)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Data-Driven", "Hybrid"]:
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
        markersize=7,
        color=colors[model_name],
    )

plt.yscale("log")
plt.xscale("log")
plt.title(
    "Mean L2 Error with Error Bars (Log Scale)\nData-Driven vs Hybrid",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Data Percentage (%) – Log Scale", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks([2, 5, 10, 20, 50, 80, 100], labels=["2", "5", "10", "20", "50", "80", "100"])
plt.legend(fontsize=12)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_errorbar_DH.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: Grouped Bar Chart (error, two models only)
# =========================================================
plt.figure(figsize=(12, 6), dpi=300)

percentages = sorted(df_dh["Percent"].unique())
x = np.arange(len(percentages))
width = 0.35

dd = df_dh[df_dh["Model"] == "Data-Driven"].sort_values("Percent")
hy = df_dh[df_dh["Model"] == "Hybrid"].sort_values("Percent")

plt.bar(
    x - width / 2,
    dd["Mean_Error"],
    width,
    yerr=dd["Std_Error"],
    label="Data-Driven",
    color=colors["Data-Driven"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy["Mean_Error"],
    width,
    yerr=hy["Std_Error"],
    label="Hybrid",
    color=colors["Hybrid"],
    capsize=4,
    alpha=0.9,
)

plt.yscale("log")
plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.title(
    "Error Comparison across Data Percentages\nData-Driven vs Hybrid",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xticks(x, [str(int(p)) for p in percentages])
plt.legend(fontsize=12)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("3_grouped_bar_error_DH.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# =========================================================
pivot = df_dh.pivot(index="Percent", columns="Model", values="Mean_Error").reset_index()
pivot["Improvement"] = pivot["Data-Driven"] / pivot["Hybrid"]

plt.figure(figsize=(10, 5.5), dpi=300)
bars = plt.bar(
    [str(int(p)) + "%" for p in pivot["Percent"]], pivot["Improvement"], color="#2E4053", alpha=0.85
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
    "Improvement Factor: How Many Times Hybrid is Better than Data-Driven",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Data-Driven / Hybrid)", fontsize=11)
plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_improvement_factor.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 5: Training time comparison (all three models)
# =========================================================
plt.figure(figsize=(12, 6), dpi=300)

models_time = ["Data-Driven", "Hybrid", "Pure PINN"]
width = 0.25

for i, model in enumerate(models_time):
    sub = df[df["Model"] == model].sort_values("Percent")
    plt.bar(
        x + (i - 1) * width,
        sub["Mean_Time"],
        width,
        yerr=sub["Std_Time"],
        label=model,
        color=colors[model],
        capsize=3,
        alpha=0.9,
    )

plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Training Time (seconds)", fontsize=12, fontweight="bold")
plt.title(
    "Training Time Comparison – All Three Models\n(Phase 1 – Data Scarcity)",
    fontsize=14,
    fontweight="bold",
    pad=12,
)
plt.xticks(x, [str(int(p)) for p in percentages])
plt.legend(fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.55)
plt.tight_layout()
plt.savefig("5_training_time_all_models.png", dpi=300, bbox_inches="tight")
plt.show()

print("All figures saved:")
print("   1_shaded_error_DH.png")
print("   2_errorbar_DH.png")
print("   3_grouped_bar_error_DH.png")
print("   4_improvement_factor.png")
print("   5_training_time_all_models.png")
