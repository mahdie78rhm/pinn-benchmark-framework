"""Phase 1 - Data scarcity study on the 1D Burgers equation.

Same protocol as the heat-equation study (fixed holdout, fixed base pool, Pure PINN vs
Data-Driven vs Hybrid over decreasing labeled-data fractions) applied to Burgers.
"""

import os

os.environ["DDE_BACKEND"] = "pytorch"
import time
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import torch
import gc
import requests
from io import BytesIO
from scipy import stats

warnings.filterwarnings("ignore")
print("Phase 1: Data Scarcity Study - Burgers Equation\n")

SCRIPT_START = time.perf_counter()

# ==================== Settings ====================
BASE_SEED = 1234
HOLDOUT_FRACTION = 0.2
NUM_POOL_BASE_POINTS = 2500  # fixed; aligned with NUM_SAMPLE_POINTS of later phases
percentages = [1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02]
num_runs = 6
MIN_LABELED_POINTS = 4
nu = 0.01 / np.pi

dde.config.set_random_seed(BASE_SEED)
np.random.seed(BASE_SEED)


def reset_seed(seed):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==================== Full reference grid (Burgers) ====================
def download_burgers():
    url = "https://github.com/lululxvi/deepxde/raw/master/examples/dataset/Burgers.npz"
    response = requests.get(url)
    response.raise_for_status()
    return np.load(BytesIO(response.content))


data_npz = download_burgers()
t_arr, x_arr = data_npz["t"], data_npz["x"]
exact = data_npz["usol"].T
xx, tt = np.meshgrid(x_arr, t_arr)
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = exact.flatten()[:, None]

print(f"Reference grid size: {len(X_full)} points")

# ==================== Fixed, independent holdout (seed=9999) ====================
holdout_rng = np.random.RandomState(9999)
n_total = len(X_full)
n_holdout = int(n_total * HOLDOUT_FRACTION)

all_idx = np.arange(n_total)
holdout_idx = holdout_rng.choice(all_idx, size=n_holdout, replace=False)
pool_idx = np.setdiff1d(all_idx, holdout_idx)

X_test_holdout = X_full[holdout_idx]
y_test_holdout = y_full[holdout_idx]
X_pool = X_full[pool_idx]
y_pool = y_full[pool_idx]

print(f"Holdout test set: {len(X_test_holdout)} points ({HOLDOUT_FRACTION*100:.0f}% of the grid)")
print(f"Pool: {len(X_pool)} points")

# ==================== X_pool_base: fixed subset (~2500 points, seed=4242) ====================
base_rng = np.random.RandomState(4242)
n_pool_base = min(NUM_POOL_BASE_POINTS, len(X_pool))
pool_local_idx = np.arange(len(X_pool))
pool_base_idx = base_rng.choice(pool_local_idx, size=n_pool_base, replace=False)

X_pool_base = X_pool[pool_base_idx]
y_pool_base = y_pool[pool_base_idx]

BASE_POINTS = len(X_pool_base)
print(
    f"X_pool_base (fixed, target={NUM_POOL_BASE_POINTS}): {BASE_POINTS} points "
    f"({BASE_POINTS/len(X_pool)*100:.1f}% of the pool)"
)
print(f"BASE_POINTS = {BASE_POINTS} -> percent=100% uses all of X_pool_base\n")

# ==================== Geometry and PDE (Burgers) ====================
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu * dy_xx


def build_net():
    return dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")


def evaluate(model):
    return dde.metrics.l2_relative_error(y_test_holdout, model.predict(X_test_holdout))


def check_finite(error, tag):
    if not np.isfinite(error):
        print(f"WARNING: NaN/Inf in {tag}")
        return False
    return True


def train_with_retry(fn, seed, tag, max_retries=1):
    attempt, current_seed, elapsed = 0, seed, np.nan
    while attempt <= max_retries:
        error, elapsed = fn(current_seed)
        if check_finite(error, tag):
            return error, elapsed, attempt
        attempt += 1
        current_seed = seed + 100000 + attempt
    print(f"ERROR: {tag}: NaN persisted after {max_retries+1} attempts")
    return np.nan, elapsed, attempt


results = []
nan_events = []

# ==================== Pure PINN (trained once per run, outside the percent loop) ====================
pinn_errors_per_run = {}
pinn_times_per_run = {}
print("Training Pure PINN...")
for run in tqdm(range(num_runs), desc="Pure PINN"):
    seed = BASE_SEED + run * 11

    def _train_pinn(s):
        reset_seed(s)
        t0 = time.perf_counter()
        net = build_net()
        data_pinn = dde.data.TimePDE(
            geomtime,
            pde,
            [bc, ic],
            num_domain=2000,
            num_boundary=100,
            num_initial=200,
            num_test=2000,
        )
        model = dde.Model(data_pinn, net)
        model.compile("adam", lr=1e-3)
        model.train(iterations=15000, verbose=0)
        model.compile("L-BFGS")
        model.train(verbose=0)
        elapsed = time.perf_counter() - t0
        err = evaluate(model)
        del model, data_pinn, net
        clean_memory()
        return err, elapsed

    error, elapsed, retries = train_with_retry(_train_pinn, seed, f"PurePINN-run{run}")
    pinn_errors_per_run[run] = error
    pinn_times_per_run[run] = elapsed
    if retries > 0:
        nan_events.append({"Model": "Pure PINN", "Run": run, "Percent": "N/A", "Retries": retries})

# ==================== Main loop: Data-Driven + Hybrid ====================
for percent in tqdm(percentages, desc="Data percent"):
    for run in range(num_runs):
        seed = BASE_SEED + run * 11

        reset_seed(seed)
        num_labeled = max(MIN_LABELED_POINTS, int(BASE_POINTS * percent))
        num_labeled = min(num_labeled, len(X_pool_base))
        idx = np.random.choice(len(X_pool_base), num_labeled, replace=False)
        X_obs, y_obs = X_pool_base[idx], y_pool_base[idx]

        results.append(
            {
                "Percent": percent * 100,
                "Model": "Pure PINN",
                "Run": run,
                "L2_Error": pinn_errors_per_run[run],
                "Train_Time_Sec": pinn_times_per_run[run],
            }
        )

        # ---------------- Data-Driven ----------------
        def _train_dd(s):
            reset_seed(s)
            t0 = time.perf_counter()
            net = build_net()
            data_dd = dde.data.DataSet(
                X_train=X_obs, y_train=y_obs, X_test=X_test_holdout, y_test=y_test_holdout
            )
            model_dd = dde.Model(data_dd, net)
            model_dd.compile("adam", lr=1e-3, metrics=["l2 relative error"])
            model_dd.train(iterations=15000, verbose=0)
            model_dd.compile("L-BFGS", metrics=["l2 relative error"])
            model_dd.train(verbose=0)
            elapsed = time.perf_counter() - t0
            err = evaluate(model_dd)
            del model_dd, data_dd, net
            clean_memory()
            return err, elapsed

        error_dd, time_dd, retries_dd = train_with_retry(_train_dd, seed, f"DD-p{percent}-run{run}")
        results.append(
            {
                "Percent": percent * 100,
                "Model": "Data-Driven",
                "Run": run,
                "L2_Error": error_dd,
                "Train_Time_Sec": time_dd,
            }
        )
        if retries_dd > 0:
            nan_events.append(
                {"Model": "Data-Driven", "Run": run, "Percent": percent, "Retries": retries_dd}
            )

        def _train_hybrid(s):
            reset_seed(s)
            t0 = time.perf_counter()
            net = build_net()
            observe_bc = dde.icbc.PointSetBC(X_obs, y_obs, component=0)
            data_hybrid = dde.data.TimePDE(
                geomtime,
                pde,
                [bc, ic, observe_bc],
                num_domain=2000,
                num_boundary=100,
                num_initial=200,
                num_test=2000,
            )
            model_hybrid = dde.Model(data_hybrid, net)
            loss_weights = [1.0, 1.0, 1.0, 1.0]
            model_hybrid.compile("adam", lr=1e-3, loss_weights=loss_weights)
            model_hybrid.train(iterations=15000, verbose=0)
            model_hybrid.compile("L-BFGS", loss_weights=loss_weights)
            model_hybrid.train(verbose=0)
            elapsed = time.perf_counter() - t0
            err = evaluate(model_hybrid)
            del model_hybrid, data_hybrid, net
            clean_memory()
            return err, elapsed

        error_hy, time_hy, retries_hy = train_with_retry(
            _train_hybrid, seed, f"Hybrid-p{percent}-run{run}"
        )
        results.append(
            {
                "Percent": percent * 100,
                "Model": "Hybrid",
                "Run": run,
                "L2_Error": error_hy,
                "Train_Time_Sec": time_hy,
            }
        )
        if retries_hy > 0:
            nan_events.append(
                {"Model": "Hybrid", "Run": run, "Percent": percent, "Retries": retries_hy}
            )

# ==================== Save and summarize ====================
df = pd.DataFrame(results)
df.to_csv("Phase1_Burgers_Data_Scarcity_Results.csv", index=False)

if nan_events:
    pd.DataFrame(nan_events).to_csv("Phase1_Burgers_NaN_Retry_Log.csv", index=False)
    print(f"\nWARNING: {len(nan_events)} retries due to NaN/Inf occurred.")

summary = (
    df.groupby(["Percent", "Model"])
    .agg(
        Mean_Error=("L2_Error", "mean"),
        Std_Error=("L2_Error", "std"),
        Mean_Time=("Train_Time_Sec", "mean"),
        Std_Time=("Train_Time_Sec", "std"),
    )
    .reset_index()
)

print("\nResults summary (independent holdout):")
print(summary.round(6))

# Statistical test: paired t-test
stat_rows = []
for percent in percentages:
    sub = df[df["Percent"] == percent * 100]
    pivot = sub.pivot(index="Run", columns="Model", values="L2_Error").dropna()
    if len(pivot) >= 2:
        for a, b in [
            ("Hybrid", "Pure PINN"),
            ("Hybrid", "Data-Driven"),
            ("Pure PINN", "Data-Driven"),
        ]:
            t_stat, p_val = stats.ttest_rel(pivot[a], pivot[b])
            stat_rows.append(
                {
                    "Percent": percent * 100,
                    "Comparison": f"{a} vs {b}",
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "N_pairs": len(pivot),
                }
            )

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv("Phase1_Burgers_Statistical_Tests.csv", index=False)
print("\nStatistical test:")
print(stat_df.round(4))

# ==================== Plots ====================
colors = {"Data-Driven": "#D32F2F", "Pure PINN": "#1976D2", "Hybrid": "#388E3C"}
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

ax = axes[0, 0]
for m in ["Data-Driven", "Pure PINN", "Hybrid"]:
    d = summary[summary["Model"] == m].sort_values("Percent")
    ax.errorbar(
        d["Percent"],
        d["Mean_Error"],
        yerr=d["Std_Error"],
        marker="o",
        linewidth=2.2,
        capsize=4,
        label=m,
        color=colors[m],
    )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Percentage of Labeled Data (%)")
ax.set_ylabel("Mean L2 Relative Error")
ax.set_title("(a) Error vs. data scarcity")
ax.legend()
ax.grid(True, which="both", linestyle="--")

ax = axes[0, 1]
for m in ["Data-Driven", "Pure PINN", "Hybrid"]:
    d = summary[summary["Model"] == m].sort_values("Percent")
    ax.plot(d["Percent"], d["Mean_Time"], marker="s", linewidth=2.2, label=m, color=colors[m])
    ax.fill_between(
        d["Percent"],
        d["Mean_Time"] - d["Std_Time"],
        d["Mean_Time"] + d["Std_Time"],
        color=colors[m],
        alpha=0.15,
    )
ax.set_xscale("log")
ax.set_xlabel("Percentage of Labeled Data (%)")
ax.set_ylabel("Mean Training Time (s)")
ax.set_title("(b) Training time vs. data scarcity")
ax.legend()
ax.grid(True, which="both", linestyle="--")

ax = axes[1, 0]
model_list = ["Data-Driven", "Pure PINN", "Hybrid"]
width = 0.25
x_base = np.arange(len(percentages))
for i, m in enumerate(model_list):
    box_data = [
        df[(df["Percent"] == p * 100) & (df["Model"] == m)]["L2_Error"].dropna().values
        for p in percentages
    ]
    bp = ax.boxplot(
        box_data,
        positions=x_base + (i - 1) * width,
        widths=width * 0.9,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(colors[m])
        patch.set_alpha(0.6)
ax.set_xticks(x_base)
ax.set_xticklabels([f"{p*100:g}%" for p in percentages], rotation=45)
ax.set_yscale("log")
ax.set_xlabel("Percentage of Labeled Data")
ax.set_ylabel("L2 Relative Error (distribution)")
ax.set_title("(c) Error spread across runs")
handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.6) for m in model_list]
ax.legend(handles, model_list)
ax.grid(True, axis="y", linestyle="--")

ax = axes[1, 1]
pair_colors = {"Hybrid vs Pure PINN": "#6A1B9A", "Hybrid vs Data-Driven": "#EF6C00"}
for pair_name, c in pair_colors.items():
    sub = stat_df[stat_df["Comparison"] == pair_name].sort_values("Percent")
    if len(sub):
        ax.plot(
            sub["Percent"],
            -np.log10(sub["p_value"].clip(lower=1e-300)),
            marker="^",
            linewidth=2,
            label=pair_name,
            color=c,
        )
ax.axhline(-np.log10(0.05), color="red", linestyle="--", label="p=0.05")
ax.set_xscale("log")
ax.set_xlabel("Percentage of Labeled Data (%)")
ax.set_ylabel("-log10(p-value)")
ax.set_title("(d) Statistical significance of the Hybrid advantage")
ax.legend(fontsize=8)
ax.grid(True, which="both", linestyle="--")

plt.tight_layout()
plt.savefig("Phase1_Burgers_Dashboard.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nTotal runtime: {(time.perf_counter()-SCRIPT_START)/60:.1f} min")
