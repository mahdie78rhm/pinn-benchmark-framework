"""Phase 2 - Noise robustness study for the inverse diffusion problem.

Hybrid vs Pure PINN under increasing observation noise; reports the solution L2 error and
the identification error of the diffusion coefficient C.
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


# ==================== Settings ====================
C_TRUE = 1.0  # true parameter value (y = exp(-t) sin(pi x) satisfies the PDE with C = 1)
C_INIT = 2.0  # initial guess of C (as in the DeepXDE example)

NUM_DOMAIN = 40
NUM_BOUNDARY = 20
NUM_INITIAL = 10

NUM_SAMPLE_POINTS = 100  # aligned with the collocation points
HOLDOUT_FRACTION = 0.2  # fixed, independent evaluation set
ADAM_ITERATIONS = 50000  # as in the DeepXDE example

noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
num_runs = 6

results = []


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


def make_pde(C):
    """Build a PDE residual function that receives C through a closure
    (each model needs a fresh dde.Variable so nothing leaks between trainings)."""

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
    try:
        return float(C.detach().cpu().numpy())
    except AttributeError:
        return float(C)


geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

x_dim, t_dim = 256, 201
x_space = np.linspace(-1, 1, x_dim).reshape(-1, 1)
t_space = np.linspace(0, 1, t_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = func(X_full)

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
print(
    f"Fixed labeled sample size: {num_labeled} ({actual_percent:.4f}% of pool, "
    f"aligned with collocation count ~{NUM_DOMAIN + NUM_BOUNDARY + NUM_INITIAL})"
)

y_pred_hybrid_for_heatmap = None

# ==================== GPU/backend warm-up (excluded from all timing) ====================
if torch.cuda.is_available():
    print("Warming up GPU/backend (excluded from timing)...")
    _C_warm = dde.Variable(C_INIT)
    _pde_warm = make_pde(_C_warm)
    _net_warm = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
    _data_warm = dde.data.TimePDE(
        geomtime,
        _pde_warm,
        [bc, ic],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=2000,
        solution=func,
    )
    _model_warm = dde.Model(_data_warm, _net_warm)
    _model_warm.compile("adam", lr=1e-3, external_trainable_variables=_C_warm)
    _model_warm.train(iterations=10, verbose=0)
    sync_cuda()
    del _C_warm, _pde_warm, _net_warm, _data_warm, _model_warm
    clean_memory()

# ==================== Pure PINN baseline (independent of noise, no data at all) ====================
print("\nTraining Pure PINN baseline (per run)...")
pi_predictions = {}
pi_C_value = {}
pi_time = {}
for run in tqdm(range(num_runs), desc="Pure PINN runs"):
    run_seed = 1234 + run * 11
    set_all_seeds(run_seed)

    C_pure = dde.Variable(C_INIT)
    pde_pure = make_pde(C_pure)
    net_pi = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
    data_pi = dde.data.TimePDE(
        geomtime,
        pde_pure,
        [bc, ic],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=2000,
        solution=func,
    )
    model_pi = dde.Model(data_pi, net_pi)

    sync_cuda()
    t0 = time.time()
    model_pi.compile(
        "adam",
        lr=0.001,
        loss_weights=[1, 1, 1],
        metrics=["l2 relative error"],
        external_trainable_variables=C_pure,
    )
    model_pi.train(iterations=ADAM_ITERATIONS, verbose=0)
    sync_cuda()
    t1 = time.time()

    pi_time[run] = t1 - t0
    pi_predictions[run] = model_pi.predict(X_eval)
    pi_C_value[run] = get_C_value(C_pure)

    del C_pure, pde_pure, net_pi, data_pi, model_pi
    clean_memory()

# ==================== Fixed labeled sample per run (reused across noise levels) ====================
run_samples = {}
for run in range(num_runs):
    run_seed = 1234 + run * 11
    rng = np.random.RandomState(run_seed)
    idx = rng.choice(len(X_pool), num_labeled, replace=False)
    run_samples[run] = (X_pool[idx], y_pool[idx])

# ==================== Main loop: noise levels x runs (Hybrid only; Pure PINN cached) ====================
print("\nRunning Hybrid experiments across noise levels (Pure PINN reused from baseline)...")
for run in tqdm(range(num_runs), desc="Runs"):
    run_base_seed = 1234 + run * 11
    X_data, y_clean = run_samples[run]
    std_y = np.std(y_clean) if np.std(y_clean) > 0 else 1.0

    error_pi = dde.metrics.l2_relative_error(y_eval, pi_predictions[run])
    C_error_pi = abs(pi_C_value[run] - C_TRUE)

    for noise in noise_levels:
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Pure_PI",
                "Run": run,
                "L2_Error": float(error_pi),
                "C_Predicted": pi_C_value[run],
                "C_Error": C_error_pi,
                "Time_sec": pi_time[run],
            }
        )

    for noise in noise_levels:
        noise_rng = np.random.RandomState(9000 + run * 11 + int(round(noise * 10000)))
        y_noise = y_clean + noise * std_y * noise_rng.randn(num_labeled, 1)

        # ---------------- Hybrid ----------------
        set_all_seeds(run_base_seed)
        C_hybrid = dde.Variable(C_INIT)
        pde_hybrid = make_pde(C_hybrid)
        net_hy = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
        observe_bc = dde.icbc.PointSetBC(X_data, y_noise, component=0)
        data_hybrid = dde.data.TimePDE(
            geomtime,
            pde_hybrid,
            [bc, ic, observe_bc],
            num_domain=NUM_DOMAIN,
            num_boundary=NUM_BOUNDARY,
            num_initial=NUM_INITIAL,
            anchors=X_data,
            num_test=2000,
            solution=func,
        )
        model_hybrid = dde.Model(data_hybrid, net_hy)
        loss_weights = [1.0, 1.0, 1.0, 1.0]

        sync_cuda()
        t0 = time.time()
        model_hybrid.compile(
            "adam",
            lr=0.001,
            loss_weights=loss_weights,
            metrics=["l2 relative error"],
            external_trainable_variables=C_hybrid,
        )
        model_hybrid.train(iterations=ADAM_ITERATIONS, verbose=0)
        sync_cuda()
        t1 = time.time()

        error_hybrid = dde.metrics.l2_relative_error(y_eval, model_hybrid.predict(X_eval))
        C_hybrid_val = get_C_value(C_hybrid)
        results.append(
            {
                "Noise_Percent": noise * 100,
                "Model": "Hybrid",
                "Run": run,
                "L2_Error": float(error_hybrid),
                "C_Predicted": C_hybrid_val,
                "C_Error": abs(C_hybrid_val - C_TRUE),
                "Time_sec": t1 - t0,
            }
        )

        if noise == 0.05 and run == (num_runs - 1):
            y_pred_hybrid_for_heatmap = model_hybrid.predict(X_full)

        del C_hybrid, pde_hybrid, net_hy, data_hybrid, model_hybrid
        clean_memory()

# ==================== Save results ====================
df = pd.DataFrame(results)
df.to_csv("Phase3_Inverse_Diffusion_Noise_Robustness_Results.csv", index=False)

summary = (
    df.groupby(["Noise_Percent", "Model"])
    .agg(
        Mean_L2=("L2_Error", "mean"),
        Std_L2=("L2_Error", "std"),
        Mean_C_Error=("C_Error", "mean"),
        Std_C_Error=("C_Error", "std"),
        Mean_C_Pred=("C_Predicted", "mean"),
        Mean_Time=("Time_sec", "mean"),
        Std_Time=("Time_sec", "std"),
    )
    .reset_index()
)
summary.to_csv("Phase3_Summary.csv", index=False)

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
df_sorted["C_Predicted"] = df_sorted["C_Predicted"].round(4)
df_sorted["C_Error"] = df_sorted["C_Error"].round(4)
df_sorted["Time_sec"] = df_sorted["Time_sec"].round(2)
print_table(df_sorted, "Full raw results (all runs)")

error_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Mean_L2")
error_std_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Std_L2")
error_display = error_pivot.copy()
for col in error_display.columns:
    error_display[col] = [
        f"{m:.6f} ± {s:.6f}" for m, s in zip(error_pivot[col], error_std_pivot[col])
    ]
error_display = error_display.reset_index()
error_display["Noise_Percent"] = error_display["Noise_Percent"].round(2)
print_table(error_display, "Mean L2 Error ± Std, by Noise Level and Model")

c_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Mean_C_Error")
c_std_pivot = summary.pivot(index="Noise_Percent", columns="Model", values="Std_C_Error")
c_display = c_pivot.copy()
for col in c_display.columns:
    c_display[col] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(c_pivot[col], c_std_pivot[col])]
c_display = c_display.reset_index()
c_display["Noise_Percent"] = c_display["Noise_Percent"].round(2)
print_table(c_display, "Mean |C_pred - C_true| ± Std, by Noise Level and Model")

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

overlap = np.intersect1d(holdout_idx, pool_idx)
checks.append(("Holdout/Pool disjoint", len(overlap) == 0, f"{len(overlap)} overlapping indices"))

expected_rows = num_runs * len(noise_levels) * 2  # Pure_PI + Hybrid only
checks.append(
    (
        "Expected result row count",
        len(df) == expected_rows,
        f"got {len(df)}, expected {expected_rows}",
    )
)

checks.append(
    (
        "Labeled sample size == requested (100, not 2500)",
        num_labeled == min(NUM_SAMPLE_POINTS, len(X_pool)),
        f"num_labeled={num_labeled}",
    )
)

pi_l2_std = df[df["Model"] == "Pure_PI"].groupby("Run")["L2_Error"].std()
checks.append(
    (
        "Pure_PI L2 error constant within each run across noise levels",
        bool((pi_l2_std < 1e-9).all()),
        f"max within-run std = {pi_l2_std.max():.2e}",
    )
)

pi_c_std = df[df["Model"] == "Pure_PI"].groupby("Run")["C_Error"].std()
checks.append(
    (
        "Pure_PI C-identification error constant within each run across noise levels",
        bool((pi_c_std < 1e-9).all()),
        f"max within-run std = {pi_c_std.max():.2e}",
    )
)

hy_low_l2 = df[(df["Model"] == "Hybrid") & (df["Noise_Percent"] == 0.0)]["L2_Error"].mean()
pi_low_l2 = df[(df["Model"] == "Pure_PI") & (df["Noise_Percent"] == 0.0)]["L2_Error"].mean()
checks.append(
    (
        "Hybrid L2 error < Pure_PI L2 error at 0% noise (data should help solution accuracy)",
        hy_low_l2 < pi_low_l2,
        f"Hybrid={hy_low_l2:.6f}, Pure_PI={pi_low_l2:.6f}",
    )
)

hy_low_c = df[(df["Model"] == "Hybrid") & (df["Noise_Percent"] == 0.0)]["C_Error"].mean()
pi_low_c = df[(df["Model"] == "Pure_PI") & (df["Noise_Percent"] == 0.0)]["C_Error"].mean()
checks.append(
    (
        "Hybrid C-identification error < Pure_PI C error at 0% noise (parameter needs data to be identifiable)",
        hy_low_c < pi_low_c,
        f"Hybrid={hy_low_c:.4f}, Pure_PI={pi_low_c:.4f}",
    )
)

max_noise = max(noise_levels) * 100
hy_high_c = df[(df["Model"] == "Hybrid") & (df["Noise_Percent"] == max_noise)]["C_Error"].mean()
checks.append(
    (
        f"Hybrid C error at {max_noise:.1f}% noise >= C error at 0% noise (degrades with noise)",
        hy_high_c >= hy_low_c,
        f"0%={hy_low_c:.4f}, {max_noise:.1f}%={hy_high_c:.4f}",
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
        "Note the noise-degradation checks reflect the scientific hypothesis being tested, "
        "not necessarily a code bug."
    )

# ==================== Plot 1: L2 error, line + shaded std (log y) ====================
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
colors = {"Hybrid": "#1f77b4", "Pure_PI": "#2ca02c"}

plt.figure(figsize=(10.5, 6.8))
for model in ["Hybrid", "Pure_PI"]:
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
    f"Solution Error vs Noise on {num_labeled}-point Fixed Sample (Held-Out Evaluation)",
    pad=20,
    fontweight="bold",
)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.savefig("Fig1_L2_Noise_Robustness_Line.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 1b: C-identification error, line + shaded std (log y) ====================
plt.figure(figsize=(10.5, 6.8))
for model in ["Hybrid", "Pure_PI"]:
    sub = df[df["Model"] == model]
    grouped = sub.groupby("Noise_Percent")["C_Error"].agg(["mean", "std"]).reset_index()
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
plt.ylabel(f"Mean $|C_{{pred}} - {C_TRUE}|$ (log scale)")
plt.title(
    f"Parameter Identification Error vs Noise on {num_labeled}-point Fixed Sample",
    pad=20,
    fontweight="bold",
)
plt.legend(loc="upper left", frameon=True)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.savefig("Fig1b_C_Error_Noise_Robustness_Line.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 2: L2 error boxplot ====================
plt.figure(figsize=(12, 6.5))
ax = sns.boxplot(data=df, x="Noise_Percent", y="L2_Error", hue="Model", palette=colors, width=0.55)
plt.yscale("log")
plt.title(
    f"L2 Error Distribution across {num_runs} Independent Runs ({num_labeled}-point Sample, Held-Out)",
    pad=15,
    fontweight="bold",
)
plt.xlabel("Noise Level (%)")
plt.ylabel("Relative $L^2$ Error (log scale)")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, [l.replace("_", " ") for l in labels], title="Model Architecture")
plt.grid(True, which="both", axis="y", ls="--", alpha=0.6)
plt.savefig("Fig2_L2_Noise_Robustness_Boxplot.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 2b: C-error boxplot ====================
plt.figure(figsize=(12, 6.5))
ax = sns.boxplot(data=df, x="Noise_Percent", y="C_Error", hue="Model", palette=colors, width=0.55)
plt.yscale("log")
plt.title(
    f"Parameter (C) Error Distribution across {num_runs} Independent Runs ({num_labeled}-point Sample)",
    pad=15,
    fontweight="bold",
)
plt.xlabel("Noise Level (%)")
plt.ylabel(f"$|C_{{pred}} - {C_TRUE}|$ (log scale)")
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, [l.replace("_", " ") for l in labels], title="Model Architecture")
plt.grid(True, which="both", axis="y", ls="--", alpha=0.6)
plt.savefig("Fig2b_C_Error_Noise_Robustness_Boxplot.png", dpi=400, bbox_inches="tight")
plt.show()
plt.close()

# ==================== Plot 3: error heatmap (illustrative only) ====================
if y_pred_hybrid_for_heatmap is not None:
    error_map = np.abs(y_full.flatten() - y_pred_hybrid_for_heatmap.flatten()).reshape(t_dim, x_dim)
    plt.figure(figsize=(11, 5.5))
    plt.imshow(error_map.T, cmap="inferno", origin="lower", extent=[0, 1, -1, 1], aspect="auto")
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
    plt.savefig("Fig3_Hybrid_Error_Heatmap.png", dpi=400, bbox_inches="tight")
    plt.show()
    plt.close()

# ==============================================================================
# ==============================================================================
colors2 = {"Hybrid": "#0275D8", "Pure_PI": "#5CB85C"}
df_models = summary.sort_values("Noise_Percent")
uniq_noise = sorted(df_models["Noise_Percent"].unique())

# ---- 4a. Log-scale errorbar (L2) ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Pure_PI", "Hybrid"]:
    sub = df_models[df_models["Model"] == model_name]
    plt.errorbar(
        sub["Noise_Percent"],
        sub["Mean_L2"],
        yerr=sub["Std_L2"],
        label=model_name.replace("_", " "),
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        color=colors2[model_name],
    )
plt.yscale("log")
plt.title("Mean L2 Error vs Noise Level (Log Scale)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error (log scale)", fontsize=12, fontweight="bold")
plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4a_L2_errorbar_log.png", dpi=300)
plt.show()
plt.close()

# ---- 4b. Log-scale errorbar (C error) ----
plt.figure(figsize=(10, 6), dpi=300)
for model_name in ["Pure_PI", "Hybrid"]:
    sub = df_models[df_models["Model"] == model_name]
    plt.errorbar(
        sub["Noise_Percent"],
        sub["Mean_C_Error"],
        yerr=sub["Std_C_Error"],
        label=model_name.replace("_", " "),
        fmt="-o",
        capsize=5,
        capthick=1.5,
        linewidth=2,
        markersize=7,
        color=colors2[model_name],
    )
plt.yscale("log")
plt.title(
    "Mean Parameter (C) Error vs Noise Level (Log Scale)", fontsize=14, fontweight="bold", pad=15
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel(f"Mean $|C_{{pred}}-{C_TRUE}|$ (log scale)", fontsize=12, fontweight="bold")
plt.legend(fontsize=12, loc="upper left", frameon=True)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("4b_C_error_errorbar_log.png", dpi=300)
plt.show()
plt.close()

# ---- 5a. Grouped bar chart (L2) ----
plt.figure(figsize=(12, 6), dpi=300)
x = np.arange(len(uniq_noise))
width = 0.35
pi_bar = df_models[df_models["Model"] == "Pure_PI"].set_index("Noise_Percent").loc[uniq_noise]
hy_bar = df_models[df_models["Model"] == "Hybrid"].set_index("Noise_Percent").loc[uniq_noise]

plt.bar(
    x - width / 2,
    pi_bar["Mean_L2"],
    width,
    yerr=pi_bar["Std_L2"],
    label="Pure PI",
    color=colors2["Pure_PI"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy_bar["Mean_L2"],
    width,
    yerr=hy_bar["Std_L2"],
    label="Hybrid",
    color=colors2["Hybrid"],
    capsize=4,
    alpha=0.9,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Mean L2 Error", fontsize=12, fontweight="bold")
plt.title("L2 Error Comparison across Noise Levels", fontsize=14, fontweight="bold", pad=15)
plt.xticks(x, uniq_noise)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("5a_L2_grouped_bar.png", dpi=300)
plt.show()
plt.close()

# ---- 5b. Grouped bar chart (C error) ----
plt.figure(figsize=(12, 6), dpi=300)
plt.bar(
    x - width / 2,
    pi_bar["Mean_C_Error"],
    width,
    yerr=pi_bar["Std_C_Error"],
    label="Pure PI",
    color=colors2["Pure_PI"],
    capsize=4,
    alpha=0.9,
)
plt.bar(
    x + width / 2,
    hy_bar["Mean_C_Error"],
    width,
    yerr=hy_bar["Std_C_Error"],
    label="Hybrid",
    color=colors2["Hybrid"],
    capsize=4,
    alpha=0.9,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel(f"Mean $|C_{{pred}}-{C_TRUE}|$", fontsize=12, fontweight="bold")
plt.title(
    "Parameter (C) Error Comparison across Noise Levels", fontsize=14, fontweight="bold", pad=15
)
plt.xticks(x, uniq_noise)
plt.legend(fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("5b_C_error_grouped_bar.png", dpi=300)
plt.show()
plt.close()

# ---- 6a. Improvement factor: Pure_PI / Hybrid on L2 error ----
pivot_l2 = df_models.pivot(index="Noise_Percent", columns="Model", values="Mean_L2").reset_index()
pivot_l2 = pivot_l2.sort_values("Noise_Percent")
pivot_l2["Improvement_Factor"] = pivot_l2["Pure_PI"] / pivot_l2["Hybrid"]

plt.figure(figsize=(10, 5), dpi=300)
bars = plt.bar(
    [str(p) + "%" for p in pivot_l2["Noise_Percent"]],
    pivot_l2["Improvement_Factor"],
    color="#2E4053",
    alpha=0.85,
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval,
        f"{yval:.1f}x",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
plt.title(
    "How Many Times Hybrid Model Outperforms Pure PINN (Solution Error)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Pure_PI L2 / Hybrid L2)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("6a_L2_improvement_factor_bar.png", dpi=300)
plt.show()
plt.close()

# ---- 6b. Improvement factor: Pure_PI / Hybrid on C error ----
pivot_c = df_models.pivot(
    index="Noise_Percent", columns="Model", values="Mean_C_Error"
).reset_index()
pivot_c = pivot_c.sort_values("Noise_Percent")
pivot_c["Improvement_Factor"] = pivot_c["Pure_PI"] / pivot_c["Hybrid"]

plt.figure(figsize=(10, 5), dpi=300)
bars = plt.bar(
    [str(p) + "%" for p in pivot_c["Noise_Percent"]],
    pivot_c["Improvement_Factor"],
    color="#6C3483",
    alpha=0.85,
)
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        yval,
        f"{yval:.1f}x",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )
plt.title(
    "How Many Times Hybrid Model Outperforms Pure PINN (Parameter Identification)",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Noise Level (%)", fontsize=12, fontweight="bold")
plt.ylabel("Improvement Factor (Pure_PI C-error / Hybrid C-error)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("6b_C_error_improvement_factor_bar.png", dpi=300)
plt.show()
plt.close()

# ---- 7. Time comparison ----
time_summary = (
    df.groupby("Model")
    .agg(Mean_Time=("Time_sec", "mean"), Std_Time=("Time_sec", "std"))
    .reset_index()
)
model_order = ["Pure_PI", "Hybrid"]
time_summary = time_summary.set_index("Model").loc[model_order].reset_index()

plt.figure(figsize=(8, 5), dpi=300)
bar_colors = [colors2.get(m, "#888888") for m in time_summary["Model"]]
bars = plt.bar(
    [m.replace("_", " ") for m in time_summary["Model"]],
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
plt.savefig("7_time_comparison_bar.png", dpi=300)
plt.show()
plt.close()

print("\nDone. All figures and CSV files saved.")
