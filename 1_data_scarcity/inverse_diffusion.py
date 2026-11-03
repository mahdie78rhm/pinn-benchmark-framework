"""Phase 1 - Data scarcity study for the inverse diffusion problem.

Pure PINN vs Hybrid (PINN + observations) while jointly recovering the unknown
diffusion coefficient C. Reports both the solution L2 error on a fixed holdout set and
the parameter identification error |C_pred - C_true|.
"""

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

print("Phase 1: Inverse Problem Data Scarcity Study (Pure PINN vs Hybrid)\n")

dde.config.set_random_seed(1234)
np.random.seed(1234)

# ==================== Settings ====================
C_TRUE = 1.0  # true parameter value (y = exp(-t) sin(pi x) satisfies the PDE with C = 1)
C_INIT = 2.0  # initial guess of C (as in the DeepXDE example)
percentages = [0.10, 0.20, 0.50, 0.80, 1.00]
num_runs = 6
HOLDOUT_FRACTION = 0.2
POOL_BASE_SIZE = 100  # fixed count, aligned with the collocation points
MIN_LABELED_POINTS = 4
ADAM_ITERATIONS = 50000  # as in the DeepXDE example

results = []
c_results = []


# ==================== Equation and reference solution ====================
def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


def make_pde(C):
    """Build a PDE residual function that receives C through a closure
    (each run/percent needs a fresh, independent dde.Variable)."""

    def pde(x, y):
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return (
            dy_t
            - C * dy_xx
            + torch.exp(-x[:, 1:])
            * (torch.sin(np.pi * x[:, 0:1]) - np.pi**2 * torch.sin(np.pi * x[:, 0:1]))
        )

    return pde


def get_C_value(C):
    """Extract the numeric value of a dde.Variable independent of the backend."""
    try:
        return float(C.detach().cpu().numpy())
    except AttributeError:
        return float(C)


# ==================== Full reference grid ====================
x_dim, t_dim = (256, 201)
x_space = np.linspace(-1, 1, x_dim).reshape(-1, 1)
t_space = np.linspace(0, 1, t_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = func(X_full)

print(f"Reference grid size: {len(X_full)} points")

# ==================== Fixed, independent holdout test set ====================
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

print(
    f"Holdout test set (fixed, never seen in training): {len(X_test_holdout)} points "
    f"({HOLDOUT_FRACTION*100:.0f}% of the grid)"
)
print(f"Pool: {len(X_pool)} points")

# ==================== Fixed base subset of the pool (source of observation points) ====================
base_rng = np.random.RandomState(4242)
n_pool_base = POOL_BASE_SIZE

pool_local_idx = np.arange(len(X_pool))
pool_base_idx = base_rng.choice(pool_local_idx, size=n_pool_base, replace=False)

X_pool_base = X_pool[pool_base_idx]
y_pool_base = y_pool[pool_base_idx]

BASE_POINTS = len(X_pool_base)
print(
    f"Base (source of observation points, fixed count aligned with the collocation points): {BASE_POINTS} points"
)
print("   => percent=100% uses all 100 base points as observations in Hybrid.\n")

# ==================== Geometry and boundary/initial conditions (following the DeepXDE example) ====================
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# ==================== Pure PINN (no observations, trained once per run) ====================
pinn_l2_per_run = {}
pinn_C_per_run = {}
print("Training Pure PINN models (no observations, PDE+BC+IC only)...")
for run in tqdm(range(num_runs), desc="Pure PINN runs"):
    seed = 1234 + run * 11
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    C_pure = dde.Variable(C_INIT)
    pde_pure = make_pde(C_pure)

    data_pure = dde.data.TimePDE(
        geomtime,
        pde_pure,
        [bc, ic],
        num_domain=40,
        num_boundary=20,
        num_initial=10,
        solution=func,
        num_test=10000,
    )
    net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
    model_pure = dde.Model(data_pure, net)

    model_pure.compile(
        "adam",
        lr=0.001,
        loss_weights=[1, 1, 1],
        metrics=["l2 relative error"],
        external_trainable_variables=C_pure,
    )
    model_pure.train(iterations=ADAM_ITERATIONS, verbose=0)

    error = dde.metrics.l2_relative_error(y_test_holdout, model_pure.predict(X_test_holdout))
    pinn_l2_per_run[run] = error
    pinn_C_per_run[run] = get_C_value(C_pure)

# ==================== Main loop: Hybrid with varying observation percentages ====================
for percent in tqdm(percentages, desc="Observation percent"):
    for run in range(num_runs):
        seed = 1234 + run * 11
        dde.config.set_random_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        num_labeled = max(MIN_LABELED_POINTS, int(BASE_POINTS * percent))
        num_labeled = min(num_labeled, len(X_pool_base))

        idx = np.random.choice(len(X_pool_base), num_labeled, replace=False)
        X_obs = X_pool_base[idx]
        y_obs = y_pool_base[idx]

        results.append(
            {
                "Percent": percent * 100,
                "Model": "Pure PINN",
                "Run": run,
                "L2_Error": pinn_l2_per_run[run],
            }
        )
        c_results.append(
            {
                "Percent": percent * 100,
                "Model": "Pure PINN",
                "Run": run,
                "C_Predicted": pinn_C_per_run[run],
                "C_Error": abs(pinn_C_per_run[run] - C_TRUE),
            }
        )

        C_hybrid = dde.Variable(C_INIT)
        pde_hybrid = make_pde(C_hybrid)
        observe_y = dde.icbc.PointSetBC(X_obs, y_obs, component=0)

        data_hybrid = dde.data.TimePDE(
            geomtime,
            pde_hybrid,
            [bc, ic, observe_y],
            num_domain=40,
            num_boundary=20,
            num_initial=10,
            anchors=X_obs,
            solution=func,
            num_test=10000,
        )
        net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
        model_hybrid = dde.Model(data_hybrid, net)

        model_hybrid.compile(
            "adam",
            lr=0.001,
            loss_weights=[1, 1, 1, 1],
            metrics=["l2 relative error"],
            external_trainable_variables=C_hybrid,
        )
        model_hybrid.train(iterations=ADAM_ITERATIONS, verbose=0)

        error_hybrid = dde.metrics.l2_relative_error(
            y_test_holdout, model_hybrid.predict(X_test_holdout)
        )
        C_hybrid_val = get_C_value(C_hybrid)

        results.append(
            {"Percent": percent * 100, "Model": "Hybrid", "Run": run, "L2_Error": error_hybrid}
        )
        c_results.append(
            {
                "Percent": percent * 100,
                "Model": "Hybrid",
                "Run": run,
                "C_Predicted": C_hybrid_val,
                "C_Error": abs(C_hybrid_val - C_TRUE),
            }
        )

# ==================== Save and display results ====================
df = pd.DataFrame(results)
df.to_csv("Phase2_Inverse_Diffusion_L2_Results.csv", index=False)

df_c = pd.DataFrame(c_results)
df_c.to_csv("Phase2_Inverse_Diffusion_C_Results.csv", index=False)

summary_l2 = (
    df.groupby(["Percent", "Model"])
    .agg(Mean_L2=("L2_Error", "mean"), Std_L2=("L2_Error", "std"))
    .reset_index()
)

summary_c = (
    df_c.groupby(["Percent", "Model"])
    .agg(
        Mean_C_Error=("C_Error", "mean"),
        Std_C_Error=("C_Error", "std"),
        Mean_C_Pred=("C_Predicted", "mean"),
    )
    .reset_index()
)

print("\nSolution L2 error summary (independent holdout):")
print(summary_l2.round(6))
print("\nParameter C identification error summary (|C_pred - 1|):")
print(summary_c.round(6))

plt.figure(figsize=(14, 8))
colors = {"Pure PINN": "#1976D2", "Hybrid": "#388E3C"}
for model_name in ["Pure PINN", "Hybrid"]:
    data = summary_l2[summary_l2["Model"] == model_name]
    plt.errorbar(
        data["Percent"],
        data["Mean_L2"],
        yerr=data["Std_L2"],
        marker="o",
        linewidth=2.5,
        capsize=5,
        label=model_name,
        color=colors[model_name],
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"Percentage of Observation Points (% of {BASE_POINTS} points)")
plt.ylabel("Mean L2 Relative Error (evaluated on independent holdout set)")
plt.title("Inverse Diffusion Equation — Solution Error vs Data Scarcity")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("Phase2_L2_Error.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=(14, 8))
for model_name in ["Pure PINN", "Hybrid"]:
    data = summary_c[summary_c["Model"] == model_name]
    plt.errorbar(
        data["Percent"],
        data["Mean_C_Error"],
        yerr=data["Std_C_Error"],
        marker="o",
        linewidth=2.5,
        capsize=5,
        label=model_name,
        color=colors[model_name],
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel(f"Percentage of Observation Points (% of {BASE_POINTS} points)")
plt.ylabel(f"Mean |C_predicted - {C_TRUE}|")
plt.title("Inverse Diffusion Equation — Parameter Identification Error vs Data Scarcity")
plt.legend()
plt.grid(True, which="both", linestyle="--")
plt.savefig("Phase2_C_Error.png", dpi=300, bbox_inches="tight")
plt.show()
