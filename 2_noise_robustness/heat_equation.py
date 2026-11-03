"""Phase 2 - Noise robustness study on the 1D heat equation.

Data-Driven vs Hybrid vs Pure PINN with increasing Gaussian noise on the labeled data.
The noise RNG is independent of the network initialization seed, and the same labeled
sample is reused across noise levels within a run.
"""

import os

os.environ["DDE_BACKEND"] = "pytorch"
import gc
import time
import warnings

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_all_seeds(seed: int):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_cuda():
    """Force GPU to finish pending async ops before/after a timing boundary."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def train_with_lbfgs(model, loss_weights=None, tag=""):
    try:
        if loss_weights is not None:
            model.compile("L-BFGS", metrics=["l2 relative error"], loss_weights=loss_weights)
        else:
            model.compile("L-BFGS", metrics=["l2 relative error"])
        model.train(verbose=0)
        return True
    except Exception as e:
        print(f"L-BFGS failed for {tag}: {repr(e)}")
        return False


# ==================== Settings ====================
L, a, n = 1.0, 0.4, 1

NUM_DOMAIN = 2000
NUM_BOUNDARY = 100
NUM_INITIAL = 200

NUM_SAMPLE_POINTS = 2500  # fixed number of labeled/observation points
HOLDOUT_FRACTION = 0.2  # fixed, independent evaluation set

noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
num_runs = 6

results = []


def heat_eq_exact_solution(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx


geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: np.sin(n * np.pi * x[:, 0:1] / L), lambda _, on_initial: on_initial
)

x_dim, t_dim = 256, 201
t_space = np.linspace(0, 1, t_dim).reshape(-1, 1)
x_space = np.linspace(0, L, x_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = heat_eq_exact_solution(X_full).reshape(-1, 1)

TOTAL_POINTS = len(X_full)
print(f"Total mesh points: {TOTAL_POINTS}")

# ==================== Fixed, independent holdout + pool ====================
holdout_rng = np.random.RandomState(9999)
n_holdout = int(TOTAL_POINTS * HOLDOUT_FRACTION)

all_idx = np.arange(TOTAL_POINTS)
holdout_idx = holdout_rng.choice(all_idx, size=n_holdout, replace=False)
pool_idx = np.setdiff1d(all_idx, holdout_idx)

X_eval = X_full[holdout_idx]
y_eval = y_full[holdout_idx]

X_pool = X_full[pool_idx]
y_pool = y_full[pool_idx]

num_labeled = min(NUM_SAMPLE_POINTS, len(X_pool))
actual_percent = 100.0 * num_labeled / len(X_pool)
print(f"Holdout points: {len(X_eval)} | Pool points: {len(X_pool)}")
print(f"Fixed labeled sample size: {num_labeled} ({actual_percent:.2f}% of pool)")

y_pred_hybrid_for_heatmap = None

# ==================== GPU/backend warm-up (excluded from all timing) ====================
if torch.cuda.is_available():
    print("Warming up GPU/backend (excluded from timing)...")
    _net_warm = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
    _data_warm = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=5000,
        solution=heat_eq_exact_solution,
    )
    _model_warm = dde.Model(_data_warm, _net_warm)
    _model_warm.compile("adam", lr=1e-3)
    _model_warm.train(iterations=10, verbose=0)
    sync_cuda()
    del _net_warm, _data_warm, _model_warm
    clean_memory()

# ==================== Pure PINN baseline (independent of noise) ====================
print("\nTraining Pure PINN baseline (per run)...")
pi_predictions = {}
pi_lbfgs_success = {}
pi_time = {}
for run in tqdm(range(num_runs), desc="Pure PINN runs"):
    run_seed = 1234 + run * 100
    set_all_seeds(run_seed)

    net_pi = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
    data_pi = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=5000,
        solution=heat_eq_exact_solution,
    )
    model_pi = dde.Model(data_pi, net_pi)

    sync_cuda()
    t0 = time.time()
    model_pi.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model_pi.train(iterations=15000, verbose=0)
    success = train_with_lbfgs(model_pi, tag=f"Pure_PI run={run}")
    sync_cuda()
    t1 = time.time()

    pi_lbfgs_success[run] = success
    pi_time[run] = t1 - t0
    pi_predictions[run] = model_pi.predict(X_eval)

    del net_pi, data_pi, model_pi
    clean_memory()

# ==================== Fixed labeled sample per run (reused across noise levels) ====================
run_samples = {}
for run in range(num_runs):
    run_seed = 1234 + run * 100
    rng = np.random.RandomState(run_seed)
    idx = rng.choice(len(X_pool), num_labeled, replace=False)
    run_samples[run] = (X_pool[idx], y_pool[idx])

# ==================== Main loop: noise levels x runs ====================
print("\nRunning Data-Driven and Hybrid experiments across noise levels...")
for run in tqdm(range(num_runs), desc="Runs"):
    run_base_seed = 1234 + run * 100
    X_data, y_clean = run_samples[run]
    std_y = np.std(y_clean) if np.std(y_clean) > 0 else 1.0

    error_pi = dde.metrics.l2_relative_error(y_eval, pi_predictions[run])

    for noise in noise_levels:
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Pure_PI",
                "Run": run,
                "L2_Error": float(error_pi),
                "Time_sec": pi_time[run],
                "LBFGS_Success": pi_lbfgs_success[run],
            }
        )

    for noise in noise_levels:
        noise_rng = np.random.RandomState(9000 + run * 100 + int(round(noise * 10000)))
        y_noise = y_clean + noise * std_y * noise_rng.randn(num_labeled, 1)

        # ---------------- Data-Driven ----------------
        set_all_seeds(run_base_seed)
        net_dd = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        data_dd = dde.data.DataSet(X_train=X_data, y_train=y_noise, X_test=X_eval, y_test=y_eval)
        model_dd = dde.Model(data_dd, net_dd)

        sync_cuda()
        t0 = time.time()
        model_dd.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        model_dd.train(iterations=15000, verbose=0)
        success_dd = train_with_lbfgs(model_dd, tag=f"Data-Driven run={run} noise={noise}")
        sync_cuda()
        t1 = time.time()

        error_dd = dde.metrics.l2_relative_error(y_eval, model_dd.predict(X_eval))
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Data-Driven",
                "Run": run,
                "L2_Error": float(error_dd),
                "Time_sec": t1 - t0,
                "LBFGS_Success": success_dd,
            }
        )

        del net_dd, data_dd, model_dd
        clean_memory()

        # ---------------- Hybrid ----------------
        set_all_seeds(run_base_seed)
        net_hy = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        observe_bc = dde.icbc.PointSetBC(X_data, y_noise)
        data_hybrid = dde.data.TimePDE(
            geomtime,
            pde,
            [bc, ic, observe_bc],
            num_domain=NUM_DOMAIN,
            num_boundary=NUM_BOUNDARY,
            num_initial=NUM_INITIAL,
            num_test=5000,
            solution=heat_eq_exact_solution,
        )
        model_hybrid = dde.Model(data_hybrid, net_hy)
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        sync_cuda()
        t0 = time.time()
        model_hybrid.compile(
            "adam", lr=1e-3, loss_weights=loss_weights, metrics=["l2 relative error"]
        )
        model_hybrid.train(iterations=15000, verbose=0)
        success_hy = train_with_lbfgs(
            model_hybrid, loss_weights=loss_weights, tag=f"Hybrid run={run} noise={noise}"
        )
        sync_cuda()
        t1 = time.time()

        error_hybrid = dde.metrics.l2_relative_error(y_eval, model_hybrid.predict(X_eval))
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Hybrid",
                "Run": run,
                "L2_Error": float(error_hybrid),
                "Time_sec": t1 - t0,
                "LBFGS_Success": success_hy,
            }
        )

        if noise == 0.05 and run == (num_runs - 1):
            y_pred_hybrid_for_heatmap = model_hybrid.predict(X_full)

        del net_hy, data_hybrid, model_hybrid
        clean_memory()

# ==================== Save results ====================
df = pd.DataFrame(results)
df.to_csv("Phase2_Noise_Robustness_Results.csv", index=False)

if not df["LBFGS_Success"].all():
    n_failed = (~df["LBFGS_Success"]).sum()
    print(f"Note: {n_failed} of {len(df)} rows had L-BFGS failures (see LBFGS_Success column).")

summary = (
    df.groupby(["Noise_Percent", "Model"])
    .agg(
        Mean_Error=("L2_Error", "mean"),
        Std_Error=("L2_Error", "std"),
        Mean_Time=("Time_sec", "mean"),
        Std_Time=("Time_sec", "std"),
    )
    .reset_index()
)
summary.to_csv("Phase2_Summary.csv", index=False)

# ==================== Printed tables ====================
try:
    from tabulate import tabulate

    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


def print_table(dataframe, title):
    print(f"\n{title}")
    print("=" * len(title))
    if HAS_TABULATE:
        print(tabulate(dataframe, headers="keys", tablefmt="github", showindex=False))
    else:
        print(dataframe.to_string(index=False))


# Full raw results (every run, every noise level, every model)
df_sorted = df.sort_values(["Model", "Noise_Percent", "Run"]).reset_index(drop=True).copy()
df_sorted["Noise_Percent"] = df_sorted["Noise_Percent"].round(2)
df_sorted["L2_Error"] = df_sorted["L2_Error"].round(6)
df_sorted["Time_sec"] = df_sorted["Time_sec"].round(2)
print_table(df_sorted, "Full raw results (all runs)")

# Summary pivoted for readability: Mean_Error (Std_Error) per Model x Noise level
error_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Mean_Error")
error_std_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Std_Error")
error_display = error_pivot.copy()
for col in error_display.columns:
    error_display[col] = [
        f"{m:.6f} ± {s:.6f}" for m, s in zip(error_pivot[col], error_std_pivot[col])
    ]
error_display = error_display.reset_index()
error_display["Noise_Percent"] = error_display["Noise_Percent"].round(2)
print_table(error_display, "Mean L2 Error ± Std, by Noise Level and Model")

time_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Mean_Time").reset_index()
time_pivot["Noise_Percent"] = time_pivot["Noise_Percent"].round(2)
for col in time_pivot.columns:
    if col != "Noise_Percent":
        time_pivot[col] = time_pivot[col].round(2)
print_table(time_pivot, "Mean Training Time (seconds), by Noise Level and Model")

# ==================== Sanity / validity checks ====================
print("\nSanity checks")
print("=============")

checks = []

# 1. Holdout and pool must never overlap (structural guarantee, verified anyway)
overlap = np.intersect1d(holdout_idx, pool_idx)
checks.append(("Holdout/Pool disjoint", len(overlap) == 0, f"{len(overlap)} overlapping indices"))

# 2. Expected row count
expected_rows = num_runs * len(noise_levels) * 3
checks.append(
    (
        "Expected result row count",
        len(df) == expected_rows,
        f"got {len(df)}, expected {expected_rows}",
    )
)

# 3. Sample size matches request
checks.append(
    (
        "Labeled sample size == requested",
        num_labeled == min(NUM_SAMPLE_POINTS, len(X_pool)),
        f"num_labeled={num_labeled}",
    )
)

# 4. L-BFGS success rate
lbfgs_rate = df["LBFGS_Success"].mean()
checks.append(("L-BFGS success rate >= 90%", lbfgs_rate >= 0.9, f"{lbfgs_rate*100:.1f}% succeeded"))

# 5. Pure_PI error should be ~constant across noise levels (it never sees labeled/noisy data)
pi_errs = df[df["Model"] == "Pure_PI"].groupby("Run")["L2_Error"].std()
checks.append(
    (
        "Pure_PI error constant within each run across noise levels",
        bool((pi_errs < 1e-9).all()),
        f"max within-run std = {pi_errs.max():.2e}",
    )
)

# 6. Core hypothesis: at the highest noise level, Hybrid should on average beat Data-Driven
max_noise = max(noise_levels) * 100
dd_high = df[(df["Model"] == "Data-Driven") & (df["Noise_Percent"] == max_noise)]["L2_Error"].mean()
hy_high = df[(df["Model"] == "Hybrid") & (df["Noise_Percent"] == max_noise)]["L2_Error"].mean()
checks.append(
    (
        f"Hybrid error < Data-Driven error at {max_noise:.1f}% noise",
        hy_high < dd_high,
        f"Hybrid={hy_high:.6f}, Data-Driven={dd_high:.6f}",
    )
)

# 7. Data-Driven error should generally increase (or at least not collapse) with noise
dd_low = df[(df["Model"] == "Data-Driven") & (df["Noise_Percent"] == 0.0)]["L2_Error"].mean()
checks.append(
    (
        "Data-Driven error at max noise >= error at 0% noise",
        dd_high >= dd_low,
        f"0%={dd_low:.6f}, {max_noise:.1f}%={dd_high:.6f}",
    )
)

for name, passed, detail in checks:
    status = "PASS" if passed else "CHECK"
    print(f"[{status}] {name}  ({detail})")

n_failed_checks = sum(1 for _, passed, _ in checks if not passed)
if n_failed_checks == 0:
    print("\nAll sanity checks passed. Results appear methodologically consistent.")
else:
    print(
        f"\n{n_failed_checks} check(s) did not pass -- inspect before trusting the results. "
        "Note items 6 and 7 reflect the scientific hypothesis being tested, not a code bug: "
        "if they fail, the underlying training (e.g. iteration count, network capacity) may need "
        "review rather than the evaluation logic itself."
    )

# ==================== Plot 1: line + shaded std (log y) ====================
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    }
)
colors = {"Data-Driven": "#d62728", "Hybrid": "#1f77b4", "Pure_PI": "#2ca02c"}

plt.figure(figsize=(10.5, 6.8))
for model in ["Data-Driven", "Hybrid", "Pure_PI"]:
    sub = df[df["Model"] == model]
    grouped = sub.groupby("Noise_Percent")["L2_Error"].agg(["mean", "std"]).reset_index()
    plt.plot(
        grouped["Noise_Percent"],
        grouped["mean"],
        marker="o",
        color=colors[model],
        lw=2.8,
        label=model.replace("_", " "),
    )
    plt.fill_between(
        grouped["Noise_Percent"],
        grouped["mean"] - grouped["std"],
        grouped["mean"] + grouped["std"],
        color=colors[model],
        alpha=0.18,
    )
plt.yscale("log")
plt.xlabel("Noise Level (%)")
plt.ylabel("Relative $L^2$ Error (log scale, held-out points)")
plt.title(
    f"Noise Robustness on {actual_percent:.1f}% Labeled Data (Held-Out Evaluation)",
    pad=20,
    fontweight="bold",
)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.savefig("Fig_Noise_Robustness_Line.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 2: boxplot ====================
plt.figure(figsize=(12, 6.5))
ax = sns.boxplot(data=df, x="Noise_Percent", y="L2_Error", hue="Model", palette=colors, width=0.65)
plt.yscale("log")
plt.title(
    f"Error Distribution across {num_runs} Independent Runs ({actual_percent:.1f}% Data, Held-Out)",
    pad=15,
    fontweight="bold",
)
plt.xlabel("Noise Level (%)")
plt.ylabel("Relative $L^2$ Error (log scale)")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, [l.replace("_", " ") for l in labels], title="Model Architecture")
plt.grid(True, which="both", axis="y", ls="--", alpha=0.6)
plt.savefig("Fig_Noise_Robustness_Boxplot.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 3: error heatmap (illustrative only) ====================
if y_pred_hybrid_for_heatmap is not None:
    error_map = np.abs(y_full.flatten() - y_pred_hybrid_for_heatmap.flatten()).reshape(t_dim, x_dim)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(error_map.T, cmap="inferno", origin="lower", extent=[0, 1, 0, L], aspect="auto")
    cbar = plt.colorbar()
    cbar.set_label(
        "Absolute Error $|u_{exact} - u_{pred}|$ (full grid, illustrative only)",
        rotation=270,
        labelpad=20,
    )
    plt.xlabel("Time ($t$)")
    plt.ylabel("Space ($x$)")
    plt.title("Absolute Pointwise Error Map - Hybrid (5% Noise Level)", pad=15, fontweight="bold")
    plt.grid(False)
    plt.savefig("Fig_Hybrid_Error_Heatmap.png", dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()

# ==============================================================================
# Comparison plots (Shaded, Log-Errorbar, Grouped Bar, Improvement Factor, Time)
# ==============================================================================
colors2 = {"Data-Driven": "#D9534F", "Hybrid": "#0275D8", "Pure_PI": "#5CB85C"}
df_dh = summary[summary["Model"].isin(["Data-Driven", "Hybrid"])].copy()
df_dh = df_dh.sort_values("Noise_Percent")
uniq_noise = sorted(df_dh["Noise_Percent"].unique())

# ---- 1. Shaded line ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Data-Driven", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name]
    plt.plot(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        label=model_name,
        color=colors2[model_name],
        linewidth=2.5,
        marker="o",
        markersize=6,
    )
    plt.fill_between(
        sub["Noise_Percent"],
        sub["Mean_Error"] - sub["Std_Error"],
        sub["Mean_Error"] + sub["Std_Error"],
        color=colors2[model_name],
        alpha=0.25,
    )
plt.title(
    "Shaded Error Bounds vs Noise Level: Data-Driven vs Hybrid",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error", fontsize=12, fontweight="bold")
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("1_shaded_line_plot.png", dpi=300)
plt.show()
plt.close()

# ---- 2. Log-scale errorbar ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Data-Driven", "Hybrid"]:
    sub = df_dh[df_dh["Model"] == model_name]
    plt.errorbar(
        sub["Noise_Percent"],
        sub["Mean_Error"],
        yerr=sub["Std_Error"],
        label=model_name,
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        color=colors2[model_name],
    )
plt.yscale("log")
plt.title("Mean Error vs Noise Level (Log Scale)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error (log scale)", fontsize=12, fontweight="bold")
plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_line_plot_log_scale.png", dpi=300)
plt.show()
plt.close()

# ---- 3. Grouped bar chart ----
plt.figure(figsize=(12, 6), dpi=300)
x = np.arange(len(uniq_noise))
width = 0.35
dd = df_dh[df_dh["Model"] == "Data-Driven"].set_index("Noise_Percent").loc[uniq_noise]
hy = df_dh[df_dh["Model"] == "Hybrid"].set_index("Noise_Percent").loc[uniq_noise]

plt.bar(
    x - width / 2,
    dd["Mean_Error"],
    width,
    yerr=dd["Std_Error"],
    label="Data-Driven",
    color=colors2["Data-Driven"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy["Mean_Error"],
    width,
    yerr=hy["Std_Error"],
    label="Hybrid",
    color=colors2["Hybrid"],
    capsize=4,
    alpha=0.9,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error", fontsize=12, fontweight="bold")
plt.title("Error Comparison across Noise Levels", fontsize=14, fontweight="bold", pad=15)
plt.xticks(x, uniq_noise)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("3_grouped_bar_chart.png", dpi=300)
plt.show()
plt.close()

# ---- 4. Improvement factor ----
pivot_df = df_dh.pivot(index="Noise_Percent", columns="Model", values="Mean_Error").reset_index()
pivot_df = pivot_df.sort_values("Noise_Percent")
pivot_df["Improvement_Factor"] = pivot_df["Data-Driven"] / pivot_df["Hybrid"]

plt.figure(figsize=(10, 5), dpi=300)
bars = plt.bar(
    [str(p) + "%" for p in pivot_df["Noise_Percent"]],
    pivot_df["Improvement_Factor"],
    color="#2E4053",
    alpha=0.85,
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval + 0.2,
        f"{yval:.1f}x",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
plt.title(
    "How Many Times Hybrid Model Performs Better Than Data-Driven",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Data-Driven / Hybrid)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_improvement_factor_bar.png", dpi=300)
plt.show()
plt.close()

# ---- 5. Time comparison ----
time_summary = (
    df.groupby("Model")
    .agg(Mean_Time=("Time_sec", "mean"), Std_Time=("Time_sec", "std"))
    .reset_index()
)
model_order = ["Pure_PI", "Data-Driven", "Hybrid"]
time_summary = time_summary.set_index("Model").loc[model_order].reset_index()

plt.figure(figsize=(8, 5), dpi=300)
bar_colors = [colors2.get(m, "#888888") for m in time_summary["Model"]]
bars = plt.bar(
    time_summary["Model"],
    time_summary["Mean_Time"],
    yerr=time_summary["Std_Time"],
    capsize=6,
    color=bar_colors,
    alpha=0.9,
)
for bar, val in zip(bars, time_summary["Mean_Time"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height(),
        f"{val:.1f}s",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
plt.title(
    "Average Training Time per Model (all noise levels & runs)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.ylabel("Time (seconds)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("5_time_comparison_bar.png", dpi=300)
plt.show()
plt.close()

print("\nDone. All figures and CSV files saved.")
