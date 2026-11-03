"""Phase 2 - Noise robustness study on the 1D Burgers equation.

Data-Driven vs Hybrid vs Pure PINN with increasing noise on the labeled data, using the
same protocol as the heat-equation study.
"""

import os

os.environ["DDE_BACKEND"] = "pytorch"
import gc
import time
import warnings
from io import BytesIO

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
from scipy import stats
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


def train_with_lbfgs(model, compile_kwargs=None, tag=""):
    """Compile + train with L-BFGS, logging (not silently swallowing) failures."""
    try:
        kwargs = compile_kwargs or {}
        model.compile("L-BFGS", **kwargs)
        model.train(verbose=0)
        return True
    except Exception as e:
        print(f"L-BFGS failed for {tag}: {repr(e)}")
        lbfgs_failures.append({"tag": tag, "error_type": type(e).__name__, "error_msg": str(e)})
        return False


# ==================== Equation & Domain (Burgers-specific, NOT transferred from Heat) ====================
def download_burgers():
    url = "https://github.com/lululxvi/deepxde/raw/master/examples/dataset/Burgers.npz"
    print("Downloading Burgers.npz ...")
    response = requests.get(url)
    response.raise_for_status()
    return np.load(BytesIO(response.content))


data_npz = download_burgers()
nu = 0.01 / np.pi


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu * dy_xx


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)


def gen_testdata():
    t = data_npz["t"]
    x = data_npz["x"]
    exact = data_npz["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y


X_full, y_full = gen_testdata()
t_dim = len(data_npz["t"].flatten())
x_dim = len(data_npz["x"].flatten())

# ==================== Settings (transferred from Heat code) ====================
NUM_DOMAIN = 2000
NUM_BOUNDARY = 100
NUM_INITIAL = 200

NUM_SAMPLE_POINTS = 2500  # fixed number of labeled/observation points
HOLDOUT_FRACTION = 0.2  # fixed, independent evaluation set

noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
num_runs = 6

results = []
lbfgs_failures = []
y_pred_hybrid_for_heatmap = None

TOTAL_POINTS = len(X_full)
print(f"Total mesh points: {TOTAL_POINTS}")

# ==================== Fixed, independent holdout + pool (transferred from Heat) ====================
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
        solution=None,
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
pi_residual = {}
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
        solution=None,
    )
    model_pi = dde.Model(data_pi, net_pi)

    sync_cuda()
    t0 = time.time()
    model_pi.compile("adam", lr=1e-3)
    model_pi.train(iterations=15000, verbose=0)
    success = train_with_lbfgs(model_pi, compile_kwargs=None, tag=f"Pure_PINN run={run}")
    sync_cuda()
    t1 = time.time()

    pi_lbfgs_success[run] = success
    pi_time[run] = t1 - t0
    pi_predictions[run] = model_pi.predict(X_eval)
    pi_residual[run] = float(np.mean(np.abs(model_pi.predict(X_eval, operator=pde))))

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
                "Model": "Pure_PINN",
                "Run": run,
                "L2_Error": float(error_pi),
                "Time_sec": pi_time[run],
                "PDE_Residual": pi_residual[run],
                "LBFGS_Success": pi_lbfgs_success[run],
            }
        )

    for noise in noise_levels:
        noise_rng = np.random.RandomState(9000 + run * 100 + int(round(noise * 10000)))
        y_noise = y_clean + noise * std_y * noise_rng.randn(num_labeled, 1)

        # ---------------- Pure Data-Driven ----------------
        set_all_seeds(run_base_seed)
        net_dd = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        data_dd = dde.data.DataSet(X_train=X_data, y_train=y_noise, X_test=X_eval, y_test=y_eval)
        model_dd = dde.Model(data_dd, net_dd)

        sync_cuda()
        t0 = time.time()
        model_dd.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        model_dd.train(iterations=15000, verbose=0)
        success_dd = train_with_lbfgs(
            model_dd,
            compile_kwargs={"metrics": ["l2 relative error"]},
            tag=f"Data-Driven run={run} noise={noise}",
        )
        sync_cuda()
        t1 = time.time()

        pred_dd_eval = model_dd.predict(X_eval)
        error_dd = dde.metrics.l2_relative_error(y_eval, pred_dd_eval)
        res_dd = float(np.mean(np.abs(model_dd.predict(X_eval, operator=pde))))
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Pure_DataDriven",
                "Run": run,
                "L2_Error": float(error_dd),
                "Time_sec": t1 - t0,
                "PDE_Residual": res_dd,
                "LBFGS_Success": success_dd,
            }
        )

        del net_dd, data_dd, model_dd
        clean_memory()

        # ---------------- Hybrid ----------------
        set_all_seeds(run_base_seed)
        net_hy = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        observe_bc = dde.icbc.PointSetBC(X_data, y_noise, component=0)
        data_hybrid = dde.data.TimePDE(
            geomtime,
            pde,
            [bc, ic, observe_bc],
            num_domain=NUM_DOMAIN,
            num_boundary=NUM_BOUNDARY,
            num_initial=NUM_INITIAL,
            num_test=5000,
            solution=None,
        )
        model_hybrid = dde.Model(data_hybrid, net_hy)
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        sync_cuda()
        t0 = time.time()
        model_hybrid.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model_hybrid.train(iterations=15000, verbose=0)
        success_hy = train_with_lbfgs(
            model_hybrid,
            compile_kwargs={"loss_weights": loss_weights},
            tag=f"Hybrid run={run} noise={noise}",
        )
        sync_cuda()
        t1 = time.time()

        pred_hy_eval = model_hybrid.predict(X_eval)
        error_hybrid = dde.metrics.l2_relative_error(y_eval, pred_hy_eval)
        res_hy = float(np.mean(np.abs(model_hybrid.predict(X_eval, operator=pde))))
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Hybrid_PID",
                "Run": run,
                "L2_Error": float(error_hybrid),
                "Time_sec": t1 - t0,
                "PDE_Residual": res_hy,
                "LBFGS_Success": success_hy,
            }
        )

        if noise == 0.05 and run == (num_runs - 1):
            y_pred_hybrid_for_heatmap = model_hybrid.predict(X_full)

        del net_hy, data_hybrid, model_hybrid
        clean_memory()

# ==================== Save results ====================
df = pd.DataFrame(results)
df.to_csv("Phase2_Burgers_Noise_Robustness_Results.csv", index=False)

if lbfgs_failures:
    pd.DataFrame(lbfgs_failures).to_csv("Phase2_Burgers_LBFGS_Failures_Log.csv", index=False)
    print(
        f"\n{len(lbfgs_failures)} L-BFGS failure(s) logged -> Phase2_Burgers_LBFGS_Failures_Log.csv"
    )

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
summary.to_csv("Phase2_Burgers_Summary.csv", index=False)

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


df_sorted = df.sort_values(["Model", "Noise_Percent", "Run"]).reset_index(drop=True).copy()
df_sorted["Noise_Percent"] = df_sorted["Noise_Percent"].round(2)
df_sorted["L2_Error"] = df_sorted["L2_Error"].round(6)
df_sorted["Time_sec"] = df_sorted["Time_sec"].round(2)
print_table(df_sorted, "Full raw results (all runs)")

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

# ==================== Sanity / validity checks (transferred from Heat) ====================
print("\nSanity checks")
print("=============")

checks = []

overlap = np.intersect1d(holdout_idx, pool_idx)
checks.append(("Holdout/Pool disjoint", len(overlap) == 0, f"{len(overlap)} overlapping indices"))

expected_rows = num_runs * len(noise_levels) * 3
checks.append(
    (
        "Expected result row count",
        len(df) == expected_rows,
        f"got {len(df)}, expected {expected_rows}",
    )
)

checks.append(
    (
        "Labeled sample size == requested",
        num_labeled == min(NUM_SAMPLE_POINTS, len(X_pool)),
        f"num_labeled={num_labeled}",
    )
)

lbfgs_rate = df["LBFGS_Success"].mean()
checks.append(("L-BFGS success rate >= 90%", lbfgs_rate >= 0.9, f"{lbfgs_rate*100:.1f}% succeeded"))

pi_errs = df[df["Model"] == "Pure_PINN"].groupby("Run")["L2_Error"].std()
checks.append(
    (
        "Pure_PINN error constant within each run across noise levels",
        bool((pi_errs < 1e-9).all()),
        f"max within-run std = {pi_errs.max():.2e}",
    )
)

max_noise = max(noise_levels) * 100
dd_high = df[(df["Model"] == "Pure_DataDriven") & (df["Noise_Percent"] == max_noise)][
    "L2_Error"
].mean()
hy_high = df[(df["Model"] == "Hybrid_PID") & (df["Noise_Percent"] == max_noise)]["L2_Error"].mean()
checks.append(
    (
        f"Hybrid_PID error < Pure_DataDriven error at {max_noise:.1f}% noise",
        hy_high < dd_high,
        f"Hybrid_PID={hy_high:.6f}, Pure_DataDriven={dd_high:.6f}",
    )
)

dd_low = df[(df["Model"] == "Pure_DataDriven") & (df["Noise_Percent"] == 0.0)]["L2_Error"].mean()
checks.append(
    (
        "Pure_DataDriven error at max noise >= error at 0% noise",
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
        "Note the last two items reflect the scientific hypothesis being tested, not a code bug: "
        "if they fail, the underlying training (e.g. iteration count, network capacity) may need "
        "review rather than the evaluation logic itself."
    )

# ==================== Statistical test: Wilcoxon signed-rank (kept from Burgers script) ====================
print("\nWilcoxon signed-rank test (Hybrid_PID vs the rest, per noise level)...")
stat_rows = []
for noise in noise_levels:
    npct = noise * 100
    hybrid_err = (
        df[(df["Model"] == "Hybrid_PID") & (df["Noise_Percent"] == npct)]
        .sort_values("Run")["L2_Error"]
        .values
    )
    dd_err = (
        df[(df["Model"] == "Pure_DataDriven") & (df["Noise_Percent"] == npct)]
        .sort_values("Run")["L2_Error"]
        .values
    )
    pinn_err = (
        df[(df["Model"] == "Pure_PINN") & (df["Noise_Percent"] == npct)]
        .sort_values("Run")["L2_Error"]
        .values
    )

    for other_name, other_err in [("Pure_DataDriven", dd_err), ("Pure_PINN", pinn_err)]:
        diff = hybrid_err - other_err
        if np.allclose(diff, 0):
            wstat, pval = np.nan, 1.0
        else:
            try:
                wstat, pval = stats.wilcoxon(hybrid_err, other_err, alternative="less")
            except ValueError:
                wstat, pval = np.nan, np.nan
        stat_rows.append(
            {
                "Noise_Percent": npct,
                "Comparison": f"Hybrid_PID < {other_name}",
                "Hybrid_mean": float(np.mean(hybrid_err)),
                "Other_mean": float(np.mean(other_err)),
                "Wilcoxon_stat": wstat,
                "p_value_one_sided": pval,
                "Significant_at_0.05": bool(pval < 0.05) if pval == pval else False,
                "n_pairs": len(hybrid_err),
            }
        )

df_stats = pd.DataFrame(stat_rows)
df_stats.to_csv("Phase2_Burgers_Statistical_Tests_Wilcoxon.csv", index=False)
print(df_stats.to_string(index=False))
print(
    "\n(Note: n=6 pairs per test; Wilcoxon has limited power at small n. "
    "For a stronger claim, increasing num_runs is recommended.)"
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
model_order = ["Pure_DataDriven", "Hybrid_PID", "Pure_PINN"]
colors = {"Pure_DataDriven": "#d62728", "Hybrid_PID": "#1f77b4", "Pure_PINN": "#2ca02c"}

plt.figure(figsize=(10.5, 6.8))
for model in model_order:
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
    f"Noise Robustness on {actual_percent:.1f}% Labeled Data (Held-Out Evaluation) - Burgers",
    pad=20,
    fontweight="bold",
)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.savefig("Fig_Burgers_Noise_Robustness_Line.png", dpi=400, bbox_inches="tight")
plt.close()

# ==================== Plot 2: boxplot ====================
plt.figure(figsize=(12, 6.5))
df_plot = df.copy()
df_plot["Model"] = pd.Categorical(df_plot["Model"], categories=model_order, ordered=True)
ax = sns.boxplot(
    data=df_plot,
    x="Noise_Percent",
    y="L2_Error",
    hue="Model",
    palette=colors,
    hue_order=model_order,
    width=0.65,
)
plt.yscale("log")
plt.title(
    f"Error Distribution across {num_runs} Independent Runs ({actual_percent:.1f}% Data, Held-Out) - Burgers",
    pad=15,
    fontweight="bold",
)
plt.xlabel("Noise Level (%)")
plt.ylabel("Relative $L^2$ Error (log scale)")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, [l.replace("_", " ") for l in labels], title="Model Architecture")
plt.grid(True, which="both", axis="y", ls="--", alpha=0.6)
plt.savefig("Fig_Burgers_Noise_Robustness_Boxplot.png", dpi=400, bbox_inches="tight")
plt.close()

# ==================== Plot 3: error heatmap (illustrative only) ====================
if y_pred_hybrid_for_heatmap is not None:
    t_vals = data_npz["t"].flatten()
    x_vals = data_npz["x"].flatten()
    exact_grid = data_npz["usol"].T
    pred_grid = y_pred_hybrid_for_heatmap.reshape(len(t_vals), len(x_vals))
    error_map = np.abs(pred_grid - exact_grid)

    plt.figure(figsize=(11, 5.5))
    im = plt.pcolormesh(x_vals, t_vals, error_map, shading="auto", cmap="inferno")
    cbar = plt.colorbar(im)
    cbar.set_label(
        "Absolute Error $|u_{exact} - u_{pred}|$ (full grid, illustrative only)",
        rotation=270,
        labelpad=20,
    )
    plt.xlabel("Space ($x$)")
    plt.ylabel("Time ($t$)")
    plt.title(
        "Absolute Pointwise Error Map - Hybrid_PID (5% Noise Level) - Burgers",
        pad=15,
        fontweight="bold",
    )
    plt.grid(False)
    plt.savefig("Fig_Burgers_Hybrid_Error_Heatmap.png", dpi=400, bbox_inches="tight")
    plt.close()

# ==============================================================================
# Comparison plots (Shaded, Log-Errorbar, Grouped Bar, Improvement Factor, Time)
# ==============================================================================
colors2 = {"Pure_DataDriven": "#D9534F", "Hybrid_PID": "#0275D8", "Pure_PINN": "#5CB85C"}
df_dh = summary[summary["Model"].isin(["Pure_DataDriven", "Hybrid_PID"])].copy()
df_dh = df_dh.sort_values("Noise_Percent")
uniq_noise = sorted(df_dh["Noise_Percent"].unique())

# ---- 1. Shaded line ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Pure_DataDriven", "Hybrid_PID"]:
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
    "Shaded Error Bounds vs Noise Level: Data-Driven vs Hybrid (Burgers)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error", fontsize=12, fontweight="bold")
plt.legend(fontsize=11, loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("1_burgers_shaded_line_plot.png", dpi=300)
plt.close()

# ---- 2. Log-scale errorbar ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Pure_DataDriven", "Hybrid_PID"]:
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
plt.title("Mean Error vs Noise Level (Log Scale) - Burgers", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error (log scale)", fontsize=12, fontweight="bold")
plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("2_burgers_line_plot_log_scale.png", dpi=300)
plt.close()

# ---- 3. Grouped bar chart ----
plt.figure(figsize=(12, 6), dpi=300)
x = np.arange(len(uniq_noise))
width = 0.35
dd = df_dh[df_dh["Model"] == "Pure_DataDriven"].set_index("Noise_Percent").loc[uniq_noise]
hy = df_dh[df_dh["Model"] == "Hybrid_PID"].set_index("Noise_Percent").loc[uniq_noise]

plt.bar(
    x - width / 2,
    dd["Mean_Error"],
    width,
    yerr=dd["Std_Error"],
    label="Pure_DataDriven",
    color=colors2["Pure_DataDriven"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy["Mean_Error"],
    width,
    yerr=hy["Std_Error"],
    label="Hybrid_PID",
    color=colors2["Hybrid_PID"],
    capsize=4,
    alpha=0.9,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean Error", fontsize=12, fontweight="bold")
plt.title("Error Comparison across Noise Levels - Burgers", fontsize=14, fontweight="bold", pad=15)
plt.xticks(x, uniq_noise)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("3_burgers_grouped_bar_chart.png", dpi=300)
plt.close()

# ---- 4. Improvement factor ----
pivot_df = df_dh.pivot(index="Noise_Percent", columns="Model", values="Mean_Error").reset_index()
pivot_df = pivot_df.sort_values("Noise_Percent")
pivot_df["Improvement_Factor"] = pivot_df["Pure_DataDriven"] / pivot_df["Hybrid_PID"]

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
    "How Many Times Hybrid_PID Performs Better Than Pure_DataDriven (Burgers)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Data-Driven / Hybrid)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4_burgers_improvement_factor_bar.png", dpi=300)
plt.close()

# ---- 5. Time comparison ----
time_summary = (
    df.groupby("Model")
    .agg(Mean_Time=("Time_sec", "mean"), Std_Time=("Time_sec", "std"))
    .reset_index()
)
model_order_time = ["Pure_PINN", "Pure_DataDriven", "Hybrid_PID"]
time_summary = time_summary.set_index("Model").loc[model_order_time].reset_index()

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
    "Average Training Time per Model (all noise levels & runs) - Burgers",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.ylabel("Time (seconds)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("5_burgers_time_comparison_bar.png", dpi=300)
plt.close()

print("\nDone. All figures and CSV files saved.")
