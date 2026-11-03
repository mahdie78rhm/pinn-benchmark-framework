"""Plotting script for the heat-equation results of this phase.

The summary numbers below were copied from the corresponding experiment output.
"""

import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =========================================================
# Raw per-run results
# =========================================================
data_str = """Percent,Model,Run,L2_Error
100.0,Pure PINN,0,0.0005410091739890549
100.0,Data-Driven,0,0.003365011143428579
100.0,Hybrid,0,0.000563259251440313
100.0,Pure PINN,1,0.0003054009837840193
100.0,Data-Driven,1,0.002184964338038215
100.0,Hybrid,1,0.0008824324006762757
100.0,Pure PINN,2,0.0006281139846282418
100.0,Data-Driven,2,0.003904358152803775
100.0,Hybrid,2,0.0009929857765932047
100.0,Pure PINN,3,0.00040312235042818105
100.0,Data-Driven,3,0.003833119743629233
100.0,Hybrid,3,0.00033285222078747464
100.0,Pure PINN,4,0.0008806217948897246
100.0,Data-Driven,4,0.002528652794991313
100.0,Hybrid,4,0.0006397589782441695
100.0,Pure PINN,5,0.000766624282724269
100.0,Data-Driven,5,0.003955419512109587
100.0,Hybrid,5,0.00026055655543129796
80.0,Pure PINN,0,0.0005410091739890549
80.0,Data-Driven,0,0.003037013964833055
80.0,Hybrid,0,0.0005972188757588405
80.0,Pure PINN,1,0.0003054009837840193
80.0,Data-Driven,1,0.002433944808801225
80.0,Hybrid,1,0.0008485643301829656
80.0,Pure PINN,2,0.0006281139846282418
80.0,Data-Driven,2,0.001828492601425069
80.0,Hybrid,2,0.000960376928710905
80.0,Pure PINN,3,0.00040312235042818105
80.0,Data-Driven,3,0.0049494116244974025
80.0,Hybrid,3,0.000530948842648175
80.0,Pure PINN,4,0.0008806217948897246
80.0,Data-Driven,4,0.00280778075431095
80.0,Hybrid,4,0.0005529956125158578
80.0,Pure PINN,5,0.000766624282724269
80.0,Data-Driven,5,0.003943056049795008
80.0,Hybrid,5,0.00021865403570783953
50.0,Pure PINN,0,0.0005410091739890549
50.0,Data-Driven,0,0.0020363147006079497
50.0,Hybrid,0,0.0005682272804363655
50.0,Pure PINN,1,0.0003054009837840193
50.0,Data-Driven,1,0.006540620142630248
50.0,Hybrid,1,0.0007398079582350915
50.0,Pure PINN,2,0.0006281139846282418
50.0,Data-Driven,2,0.0049467048278304775
50.0,Hybrid,2,0.0009688716140861152
50.0,Pure PINN,3,0.00040312235042818105
50.0,Data-Driven,3,0.004591024847161657
50.0,Hybrid,3,0.0005019590211959568
50.0,Pure PINN,4,0.0008806217948897246
50.0,Data-Driven,4,0.002361381629072626
50.0,Hybrid,4,0.0005319632843018259
50.0,Pure PINN,5,0.000766624282724269
50.0,Data-Driven,5,0.0026084140482076808
50.0,Hybrid,5,0.000337071851079677
20.0,Pure PINN,0,0.0005410091739890549
20.0,Data-Driven,0,0.002317395464777499
20.0,Hybrid,0,0.00048411151902531075
20.0,Pure PINN,1,0.0003054009837840193
20.0,Data-Driven,1,0.005239569109311666
20.0,Hybrid,1,0.0009212983672654565
20.0,Pure PINN,2,0.0006281139846282418
20.0,Data-Driven,2,0.008936315112111322
20.0,Hybrid,2,0.0009618912388751568
20.0,Pure PINN,3,0.00040312235042818105
20.0,Data-Driven,3,0.0033062227714981316
20.0,Hybrid,3,0.0005664304777504151
20.0,Pure PINN,4,0.0008806217948897246
20.0,Data-Driven,4,0.005942520011305015
20.0,Hybrid,4,0.0004413022278683747
20.0,Pure PINN,5,0.000766624282724269
20.0,Data-Driven,5,0.0020666705643166074
20.0,Hybrid,5,0.00025904461687760547
10.0,Pure PINN,0,0.0005410091739890549
10.0,Data-Driven,0,0.0074495358342339425
10.0,Hybrid,0,0.0003864167471148205
10.0,Pure PINN,1,0.0003054009837840193
10.0,Data-Driven,1,0.013939128775308387
10.0,Hybrid,1,0.0008672922977659205
10.0,Pure PINN,2,0.0006281139846282418
10.0,Data-Driven,2,0.014874851993414873
10.0,Hybrid,2,0.0011747689440737468
10.0,Pure PINN,3,0.00040312235042818105
10.0,Data-Driven,3,0.0058469111034884814
10.0,Hybrid,3,0.000500883474592398
10.0,Pure PINN,4,0.0008806217948897246
10.0,Data-Driven,4,0.007650922687324717
10.0,Hybrid,4,0.0005542237087381752
10.0,Pure PINN,5,0.000766624282724269
10.0,Data-Driven,5,0.014827424752059401
10.0,Hybrid,5,0.00029729927347437693
5.0,Pure PINN,0,0.0005410091739890549
5.0,Data-Driven,0,0.09242672848130103
5.0,Hybrid,0,0.0003420083009057597
5.0,Pure PINN,1,0.0003054009837840193
5.0,Data-Driven,1,0.018986439809705682
5.0,Hybrid,1,0.0008530086725171759
5.0,Pure PINN,2,0.0006281139846282418
5.0,Data-Driven,2,0.013052873091877349
5.0,Hybrid,2,0.0009331607157422384
5.0,Pure PINN,3,0.00040312235042818105
5.0,Data-Driven,3,0.06130408353240601
5.0,Hybrid,3,0.0005309773471809155
5.0,Pure PINN,4,0.0008806217948897246
5.0,Data-Driven,4,0.014522301675969426
5.0,Hybrid,4,0.000345628245832972
5.0,Pure PINN,5,0.000766624282724269
5.0,Data-Driven,5,0.04417742210174852
5.0,Hybrid,5,0.00023414324273379714
2.0,Pure PINN,0,0.0005410091739890549
2.0,Data-Driven,0,0.09745193945377618
2.0,Hybrid,0,0.00035865443923911174
2.0,Pure PINN,1,0.0003054009837840193
2.0,Data-Driven,1,0.040404266650099045
2.0,Hybrid,1,0.0008131881959978554
2.0,Pure PINN,2,0.0006281139846282418
2.0,Data-Driven,2,0.047017176310326966
2.0,Hybrid,2,0.0010135258260676211
2.0,Pure PINN,3,0.00040312235042818105
2.0,Data-Driven,3,0.2186355218788763
2.0,Hybrid,3,0.0005476714339145412
2.0,Pure PINN,4,0.0008806217948897246
2.0,Data-Driven,4,0.03463458446465847
2.0,Hybrid,4,0.0006132633237805567
2.0,Pure PINN,5,0.000766624282724269
2.0,Data-Driven,5,0.04345519963665923
2.0,Hybrid,5,0.0002842436429837443
"""

df_raw = pd.read_csv(io.StringIO(data_str))

# =========================================================
# Mean and std per percent and model
# =========================================================
df = (
    df_raw.groupby(["Percent", "Model"])["L2_Error"]
    .agg(Mean_Error="mean", Std_Error="std")
    .reset_index()
)

df = df.sort_values(["Percent", "Model"]).reset_index(drop=True)

print("Summary statistics computed from the raw results:")
print(df.to_string(index=False))
print()

df_dh = df[df["Model"].isin(["Data-Driven", "Hybrid"])].copy()

# Plot style
sns.set_theme(style="whitegrid", font_scale=1.1)
colors = {
    "Data-Driven": "#D9534F",
    "Hybrid": "#0275D8",
    "Pure PINN": "#5CB85C",
}

# =========================================================
# Plot 1: line plot with shaded band (Shaded Uncertainty Plot)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Data-Driven", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name]
    plt.plot(
        sub["Percent"],
        sub["Mean_Error"],
        label=model_name,
        color=colors[model_name],
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    plt.fill_between(
        sub["Percent"],
        sub["Mean_Error"] - sub["Std_Error"],
        sub["Mean_Error"] + sub["Std_Error"],
        color=colors[model_name],
        alpha=0.25,
        label=f"{model_name} ± Std",
    )

plt.title(
    "Shaded Error Bounds: Data-Driven vs Hybrid Model",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error", fontsize=12, fontweight="bold")
plt.legend(fontsize=11, loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("1_shaded_line_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 2: log-scale line plot with error bars (Log Scale + Error Bars)
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Data-Driven", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name]
    plt.errorbar(
        sub["Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model_name,
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        color=colors[model_name],
    )

plt.xscale("log")
plt.title(
    "Mean Error Comparison (Log Scale)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Data Percentage (%) - Log Scale", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error", fontsize=12, fontweight="bold")
plt.xticks(
    df["Percent"].unique(),
    labels=[str(int(p) if p.is_integer() else p) for p in df["Percent"].unique()],
)
plt.legend(fontsize=12, loc="upper right", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_line_plot_log_scale.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 3: grouped bar chart (Grouped Bar Chart)
# =========================================================
plt.figure(figsize=(12, 6), dpi=300)

percentages = sorted(df["Percent"].unique())
x = np.arange(len(percentages))
width = 0.35

dd_means = [
    df_dh[(df_dh["Model"] == "Data-Driven") & (df_dh["Percent"] == p)]["Mean_Error"].values[0]
    for p in percentages
]
dd_stds = [
    df_dh[(df_dh["Model"] == "Data-Driven") & (df_dh["Percent"] == p)]["Std_Error"].values[0]
    for p in percentages
]
hy_means = [
    df_dh[(df_dh["Model"] == "Hybrid") & (df_dh["Percent"] == p)]["Mean_Error"].values[0]
    for p in percentages
]
hy_stds = [
    df_dh[(df_dh["Model"] == "Hybrid") & (df_dh["Percent"] == p)]["Std_Error"].values[0]
    for p in percentages
]

plt.bar(
    x - width / 2,
    dd_means,
    width,
    yerr=dd_stds,
    label="Data-Driven",
    color=colors["Data-Driven"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy_means,
    width,
    yerr=hy_stds,
    label="Hybrid",
    color=colors["Hybrid"],
    capsize=4,
    alpha=0.9,
)

plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error", fontsize=12, fontweight="bold")
plt.title(
    "Error Comparison across Data Percentages",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xticks(x, [str(int(p) if p.is_integer() else p) for p in percentages])
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("3_grouped_bar_chart.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 4: Improvement factor of Hybrid over Data-Driven
# =========================================================
pivot_df = df_dh.pivot(index="Percent", columns="Model", values="Mean_Error").reset_index()
pivot_df["Improvement_Factor"] = pivot_df["Data-Driven"] / pivot_df["Hybrid"]

plt.figure(figsize=(10, 5), dpi=300)
bars = plt.bar(
    [f"{int(p) if p.is_integer() else p}%" for p in pivot_df["Percent"]],
    pivot_df["Improvement_Factor"],
    color="#2E4053",
    alpha=0.85,
)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.3,
        f"{yval:.1f}×",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

plt.title(
    "How Many Times Hybrid Performs Better Than Data-Driven",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Data-Driven / Hybrid)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_improvement_factor_bar.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================================================
# Plot 5: all three models with the Pure PINN line
# =========================================================
plt.figure(figsize=(10, 6), dpi=300)

for model_name in ["Data-Driven", "Hybrid", "Pure PINN"]:
    sub = df[df["Model"] == model_name]
    plt.errorbar(
        sub["Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model_name,
        fmt="-o",
        capsize=4,
        linewidth=2,
        markersize=6,
        color=colors[model_name],
    )

plt.yscale("log")
plt.title(
    "All Models Comparison (Log Y-Scale)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Data Percentage (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (Log Scale)", fontsize=12, fontweight="bold")
plt.xticks(
    percentages,
    labels=[str(int(p) if p.is_integer() else p) for p in percentages],
)
plt.legend(fontsize=11, loc="upper right")
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("5_all_models_log.png", dpi=300, bbox_inches="tight")
plt.show()

print("All figures saved.")
