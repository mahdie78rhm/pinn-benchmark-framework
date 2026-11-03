"""Phase 1 - Data scarcity study on the 1D heat equation.

Compares Pure PINN, Data-Driven and Hybrid (PINN + labeled data) models while the
fraction of labeled points is reduced. Labeled points are drawn from a fixed pool that
is disjoint from a fixed holdout set used only for evaluation. Every setting is repeated
over independent runs; raw and summary results are saved as CSV.
"""

# ==============================================================================
# ==============================================================================
#
#
# ==============================================================================

import os

os.environ["DDE_BACKEND"] = "pytorch"
import deepxde as dde
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import torch

warnings.filterwarnings("ignore")

print("Phase 1: Data Scarcity Study (fixed 6% base subset of the pool)\n")

dde.config.set_random_seed(1234)
np.random.seed(1234)

# ==================== Settings ====================
L = 1.0
a = 0.4
n = 1
percentages = [1.0, 0.8, 0.5, 0.2, 0.1, 0.05, 0.02]
num_runs = 6
HOLDOUT_FRACTION = 0.2  # 20% of the grid, held out for evaluation only
POOL_BASE_FRACTION = 0.06  # 6% of the pool used as the full-data base of the scarcity scenario
MIN_LABELED_POINTS = 4  # minimum number of labeled points

results = []


def heat_eq_exact_solution(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx


# ==================== Full reference grid ====================
x_dim, t_dim = (256, 201)
t_space = np.linspace(0, 1, t_dim).reshape(-1, 1)
x_space = np.linspace(0, L, x_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = heat_eq_exact_solution(X_full).reshape(-1, 1)

print(f"Reference grid size: {len(X_full)} points")

# ==================== Fixed, independent holdout test set ====================
holdout_rng = np.random.RandomState(9999)  # separate seed
n_total = len(X_full)
n_holdout = int(n_total * HOLDOUT_FRACTION)

all_idx = np.arange(n_total)
holdout_idx = holdout_rng.choice(all_idx, size=n_holdout, replace=False)
pool_idx = np.setdiff1d(all_idx, holdout_idx)  # complement of the holdout

X_test_holdout = X_full[holdout_idx]
y_test_holdout = y_full[holdout_idx]

X_pool = X_full[pool_idx]
y_pool = y_full[pool_idx]

print(
    f"Holdout test set (fixed, never seen in training): {len(X_test_holdout)} points "
    f"({HOLDOUT_FRACTION*100:.0f}% of the grid)"
)
print(f"Pool (source of labeled samples): {len(X_pool)} points")

# ==================== Fixed base subset of the pool (6%) ====================
base_rng = np.random.RandomState(4242)  # separate seed for the base subset
n_pool_base = int(len(X_pool) * POOL_BASE_FRACTION)

pool_local_idx = np.arange(len(X_pool))
pool_base_idx = base_rng.choice(pool_local_idx, size=n_pool_base, replace=False)

X_pool_base = X_pool[pool_base_idx]
y_pool_base = y_pool[pool_base_idx]

print(
    f"Fixed base subset (={POOL_BASE_FRACTION*100:.0f}% of the pool, basis of the data-scarcity scenario): "
    f"{len(X_pool_base)} points"
)

# ==================== BASE_POINTS equals the size of X_pool_base ====================
BASE_POINTS = len(X_pool_base)
print(f"BASE_POINTS = {BASE_POINTS} (size of the fixed base subset, not the whole pool)")
print("   => percent=100% uses all points of X_pool_base.\n")

geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: np.sin(n * np.pi * x[:, 0:1] / L), lambda _, on_initial: on_initial
)

# ==================== Pure PINN (trained once per run, outside the percent loop) ====================
pinn_errors_per_run = {}
print("Training Pure PINN models (once per run)...")
for run in tqdm(range(num_runs), desc="Pure PINN runs"):
    seed = 1234 + run * 11
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_pinn = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic],
        num_domain=2000,
        num_boundary=100,
        num_initial=200,
        solution=heat_eq_exact_solution,
        num_test=2000,
    )
    net = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
    model_pinn = dde.Model(data_pinn, net)

    model_pinn.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model_pinn.train(iterations=15000, verbose=0)
    model_pinn.compile("L-BFGS", metrics=["l2 relative error"])
    model_pinn.train(verbose=0)

    error = dde.metrics.l2_relative_error(y_test_holdout, model_pinn.predict(X_test_holdout))
    pinn_errors_per_run[run] = error

# ==================== Main loop (Data-Driven + Hybrid) ====================
for percent in tqdm(percentages, desc="Data percent"):
    for run in range(num_runs):
        seed = 1234 + run * 11
        dde.config.set_random_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        num_labeled = max(MIN_LABELED_POINTS, int(BASE_POINTS * percent))
        num_labeled = min(num_labeled, len(X_pool_base))  # never exceed the base size

        idx = np.random.choice(len(X_pool_base), num_labeled, replace=False)
        X_obs = X_pool_base[idx]
        y_obs = y_pool_base[idx]

        results.append(
            {
                "Percent": percent * 100,
                "Model": "Pure PINN",
                "Run": run,
                "L2_Error": pinn_errors_per_run[run],
            }
        )

        # ---------------- Data-Driven ----------------
        data_dd = dde.data.DataSet(
            X_train=X_obs, y_train=y_obs, X_test=X_test_holdout, y_test=y_test_holdout
        )
        net = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        model_dd = dde.Model(data_dd, net)
        model_dd.compile("adam", lr=1e-3, metrics=["l2 relative error"])
        model_dd.train(iterations=15000, verbose=0)
        model_dd.compile("L-BFGS", metrics=["l2 relative error"])
        model_dd.train(verbose=0)

        error_dd = dde.metrics.l2_relative_error(y_test_holdout, model_dd.predict(X_test_holdout))
        results.append(
            {"Percent": percent * 100, "Model": "Data-Driven", "Run": run, "L2_Error": error_dd}
        )

        # ---------------- Hybrid ----------------
        observe_bc = dde.icbc.PointSetBC(X_obs, y_obs, component=0)
        data_hybrid = dde.data.TimePDE(
            geomtime,
            pde,
            [bc, ic, observe_bc],
            num_domain=2000,
            num_boundary=100,
            num_initial=200,
            solution=heat_eq_exact_solution,
            num_test=2000,
        )
        net = dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")
        model_hybrid = dde.Model(data_hybrid, net)

        loss_weights = [1.0, 1.0, 1.0, 1.0]  # PDE, BC, IC, Data
        model_hybrid.compile(
            "adam", lr=1e-3, loss_weights=loss_weights, metrics=["l2 relative error"]
        )
        model_hybrid.train(iterations=15000, verbose=0)
        model_hybrid.compile("L-BFGS", loss_weights=loss_weights, metrics=["l2 relative error"])
        model_hybrid.train(verbose=0)

        error_hybrid = dde.metrics.l2_relative_error(
            y_test_holdout, model_hybrid.predict(X_test_holdout)
        )
        results.append(
            {"Percent": percent * 100, "Model": "Hybrid", "Run": run, "L2_Error": error_hybrid}
        )

# ==================== Save and display results ====================
df = pd.DataFrame(results)
df.to_csv("Phase1_Data_Scarcity_Results_Q1_Fixed_v2.csv", index=False)

summary = (
    df.groupby(["Percent", "Model"])
    .agg(Mean_Error=("L2_Error", "mean"), Std_Error=("L2_Error", "std"))
    .reset_index()
)

print("\nResults summary (evaluated on the shared independent holdout):")
print(summary.round(6))

# Plot
plt.figure(figsize=(14, 8))
colors = {"Data-Driven": "#D32F2F", "Pure PINN": "#1976D2", "Hybrid": "#388E3C"}
for model_name in ["Data-Driven", "Pure PINN", "Hybrid"]:
    data = summary[summary["Model"] == model_name]
    plt.errorbar(
        data["Percent"],
        data["Mean_Error"],
        yerr=data["Std_Error"],
        marker="o",
        linewidth=2.5,
        capsize=5,
        label=model_name,
        color=colors[model_name],
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel(
    f"Percentage of Labeled Data (% of {BASE_POINTS} points, "
    f"a fixed {POOL_BASE_FRACTION*100:.0f}% subset of the pool)"
)
plt.ylabel("Mean L2 Relative Error (evaluated on independent holdout set)")
plt.title("Data Scarcity Study - Three Learning Paradigms (Leakage-Free Evaluation)")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("Data_Scarcity_Study_Fixed_v2.png", dpi=300, bbox_inches="tight")
plt.show()
