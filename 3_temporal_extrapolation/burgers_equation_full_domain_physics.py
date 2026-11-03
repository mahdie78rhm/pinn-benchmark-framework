"""Phase 3 - Temporal extrapolation on the 1D Burgers equation (full-domain physics).

PDE collocation points of the PINN models span the entire time domain, including the
extrapolation region.
"""

import os

os.environ["DDE_BACKEND"] = "pytorch"
import gc
import glob
import re
import time
import urllib.request
import warnings

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------- Settings ----------------------
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints_burgers")
os.makedirs(CKPT_DIR, exist_ok=True)

gpu_available = torch.cuda.is_available()
print(
    f"GPU available: {gpu_available}"
    + (f" -> {torch.cuda.get_device_name(0)}" if gpu_available else "")
)
if gpu_available:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ===================== Problem / experiment settings =====================
nu = 0.01 / np.pi
L_min, L_max = -1.0, 1.0
T_MAX = 1.0
t_train_levels = [0.2, 0.4, 0.6]
NUM_RUNS = 6
ADAM_ITERS = 15000

NUM_DOMAIN = 2000
NUM_BOUNDARY = 100
NUM_INITIAL = 200
NUM_SAMPLE_POINTS = 2500

FAIR_CHECKPOINT = True
PINN_MONITOR = "train loss" if FAIR_CHECKPOINT else "test loss"

MODELS = ["Data-Driven", "PINN_NoObs", "PINN_WithObs"]
COLORS = {"Data-Driven": "tab:blue", "PINN_NoObs": "tab:green", "PINN_WithObs": "tab:orange"}
LABELS = {
    "Data-Driven": "Data-Driven Only",
    "PINN_NoObs": "Pure PINN (No Obs)",
    "PINN_WithObs": "Hybrid PINN (With Obs)",
}

RESULTS_CSV = os.path.join(OUTPUT_DIR, "Phase3_Burgers_Extrapolation_FullDomainPhysics.csv")
BINS_CSV = os.path.join(OUTPUT_DIR, "Phase3_Burgers_DistanceBins_FullDomainPhysics.csv")
DIST_BINS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]

DATA_SEED_OFFSET = 0
DD_SEED_OFFSET = 10_000
NO_SEED_OFFSET = 20_000
HYB_SEED_OFFSET = 30_000


def set_seed(seed):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Resume
if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    results = df_existing.to_dict("records")
    done_runs = set(zip(df_existing["T_Train"], df_existing["Model"], df_existing["Run"]))
    print(f"Found {len(results)} existing records.")
else:
    results = []
    done_runs = set()

if os.path.exists(BINS_CSV):
    bin_results = pd.read_csv(BINS_CSV).to_dict("records")
else:
    bin_results = []


def save_all():
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    pd.DataFrame(bin_results).to_csv(BINS_CSV, index=False)


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================== Reference solution / PDE =====================
def load_burgers_exact():
    file_path = "Burgers.npz"
    if not os.path.exists(file_path):
        url = (
            "https://raw.githubusercontent.com/lululxvi/deepxde/master/examples/dataset/Burgers.npz"
        )
        urllib.request.urlretrieve(url, file_path)

    data = np.load(file_path)
    t, x, exact = data["t"].flatten(), data["x"].flatten(), data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y, x, t, exact.T


X_full, y_full, x_ref, t_ref, Exact_mat = load_burgers_exact()

interpolator = scipy.interpolate.RegularGridInterpolator(
    (x_ref, t_ref), Exact_mat, bounds_error=False, fill_value=None
)


def burgers_solution(x_t):
    return interpolator(x_t)[:, None]


def pde_burgers(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - nu * dy_xx


def create_net():
    return dde.nn.FNN([2] + [50] * 5 + [1], "tanh", "Glorot normal")


# ===================== Evaluation helpers =====================
def region_errors(y_true, y_pred, t_values, t_train):
    interp_mask = (t_values <= t_train).flatten()
    extrap_mask = ~interp_mask

    def rel_l2(yt, yp):
        if len(yt) == 0:
            return np.nan
        return float(np.linalg.norm(yp - yt) / (np.linalg.norm(yt) + 1e-12))

    def rmse(yt, yp):
        if len(yt) == 0:
            return np.nan
        return float(np.sqrt(np.mean((yp - yt) ** 2)))

    return {
        "l2_interp": rel_l2(y_true[interp_mask], y_pred[interp_mask]),
        "l2_extrap": rel_l2(y_true[extrap_mask], y_pred[extrap_mask]),
        "rmse_interp": rmse(y_true[interp_mask], y_pred[interp_mask]),
        "rmse_extrap": rmse(y_true[extrap_mask], y_pred[extrap_mask]),
    }


def region_pde_residual(residual_full, t_values, t_train):
    interp_mask = (t_values <= t_train).flatten()
    extrap_mask = ~interp_mask

    def mean_abs(r):
        return float(np.mean(np.abs(r))) if len(r) else np.nan

    return {
        "pde_res_interp": mean_abs(residual_full[interp_mask]),
        "pde_res_extrap": mean_abs(residual_full[extrap_mask]),
    }


def distance_binned_errors(y_true, y_pred, t_values, t_train, bins):
    dist = t_values.flatten() - t_train
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (dist > lo) & (dist <= hi)
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        err_rel = float(np.linalg.norm(yp - yt) / (np.linalg.norm(yt) + 1e-12))
        err_rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        out.append((lo, hi, err_rel, err_rmse, int(mask.sum())))
    return out


def best_checkpoint_path(prefix):
    files = glob.glob(prefix + "-*.pt")
    if not files:
        return None

    def step_of(f):
        m = re.search(r"-(\d+)\.pt$", f)
        return int(m.group(1)) if m else -1

    return max(files, key=step_of)


def record_result(model_name, t_train, run, y_pred, residual_full, final_loss, time_sec):
    errs = region_errors(y_full, y_pred, X_full[:, 1:2], t_train)
    pde_errs = region_pde_residual(residual_full, X_full[:, 1:2], t_train)
    results.append(
        {
            "T_Train": t_train,
            "Model": model_name,
            "Run": run,
            "L2_Error_Global": float(dde.metrics.l2_relative_error(y_full, y_pred)),
            "L2_Error_Interp": errs["l2_interp"],
            "L2_Error_Extrap": errs["l2_extrap"],
            "RMSE_Interp": errs["rmse_interp"],
            "RMSE_Extrap": errs["rmse_extrap"],
            "PDE_Residual_Global": float(np.mean(np.abs(residual_full))),
            "PDE_Residual_Interp": pde_errs["pde_res_interp"],
            "PDE_Residual_Extrap": pde_errs["pde_res_extrap"],
            "Final_Train_Loss": final_loss,
            "Time_sec": time_sec,
        }
    )
    for lo, hi, err_rel, err_rmse, npts in distance_binned_errors(
        y_full, y_pred, X_full[:, 1:2], t_train, DIST_BINS
    ):
        bin_results.append(
            {
                "T_Train": t_train,
                "Model": model_name,
                "Run": run,
                "Dist_Lo": lo,
                "Dist_Hi": hi,
                "L2_Error": err_rel,
                "RMSE": err_rmse,
                "N_Points": npts,
            }
        )


def train_with_best_checkpoint(model, ckpt_prefix, adam_iters, monitor="test loss"):
    cb_ckpt = dde.callbacks.ModelCheckpoint(
        ckpt_prefix, save_better_only=True, monitor=monitor, period=1000
    )

    sync_cuda()
    t0 = time.time()
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    losshistory, _ = model.train(iterations=adam_iters, verbose=0, callbacks=[cb_ckpt])

    model.compile("L-BFGS", metrics=["l2 relative error"])
    try:
        model.train(verbose=0, callbacks=[cb_ckpt])
    except Exception as e:
        print(f"[WARN] L-BFGS failed for {ckpt_prefix}: {e}. Falling back to best Adam checkpoint.")

    sync_cuda()
    t1 = time.time()

    best_ckpt = best_checkpoint_path(ckpt_prefix)
    if best_ckpt:
        model.restore(best_ckpt, verbose=0)

    for f in glob.glob(ckpt_prefix + "-*.pt"):
        os.remove(f)

    return losshistory, (t1 - t0)


# ==================== GPU/backend warm-up (excluded from all timing) ====================
if gpu_available:
    print("Warming up GPU/backend (excluded from timing)...")
    geom_w = dde.geometry.Interval(L_min, L_max)
    timedomain_w = dde.geometry.TimeDomain(0, T_MAX)
    geomtime_w = dde.geometry.GeometryXTime(geom_w, timedomain_w)
    bc_w = dde.icbc.DirichletBC(geomtime_w, lambda x: 0, lambda x, on_b: on_b)
    ic_w = dde.icbc.IC(
        geomtime_w,
        lambda x: -np.sin(np.pi * x[:, 0:1]),
        lambda x, on_i: on_i and np.isclose(x[1], 0),
    )
    _net_warm = create_net()
    _data_warm = dde.data.TimePDE(
        geomtime_w,
        pde_burgers,
        [bc_w, ic_w],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=2000,
        solution=burgers_solution,
    )
    _model_warm = dde.Model(_data_warm, _net_warm)
    _model_warm.compile("adam", lr=1e-3)
    _model_warm.train(iterations=10, verbose=0)
    sync_cuda()
    del _net_warm, _data_warm, _model_warm, geom_w, timedomain_w, geomtime_w, bc_w, ic_w
    clean_memory()

# ===================== Main loop =====================
for t_train in tqdm(t_train_levels, desc="t_train levels"):
    for run in tqdm(range(NUM_RUNS), desc="Runs"):
        if all((t_train, m, run) in done_runs for m in MODELS):
            continue

        base_seed = 1234 + run * 11

        set_seed(base_seed + DATA_SEED_OFFSET)

        idx_past = X_full[:, 1] <= t_train
        X_past = X_full[idx_past]
        y_past = y_full[idx_past]

        available = len(X_past)
        num_data = min(NUM_SAMPLE_POINTS, available)
        idx_select = (
            np.arange(available)
            if available <= num_data
            else np.random.choice(available, num_data, replace=False)
        )

        X_observed = X_past[idx_select]
        y_observed = y_past[idx_select]

        n_obs = len(X_observed)
        perm = np.random.permutation(n_obs)
        n_val = max(1, int(0.15 * n_obs))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]

        X_fit, y_fit = X_observed[fit_idx], y_observed[fit_idx]
        X_val, y_val = X_observed[val_idx], y_observed[val_idx]

        # 1. Data-Driven
        if (t_train, "Data-Driven", run) not in done_runs:
            set_seed(base_seed + DD_SEED_OFFSET)

            net_dd = create_net()
            data_dd = dde.data.DataSet(X_train=X_fit, y_train=y_fit, X_test=X_val, y_test=y_val)
            model_dd = dde.Model(data_dd, net_dd)
            ckpt_prefix = os.path.join(CKPT_DIR, f"dd_t{t_train}_run{run}")
            losshistory_dd, time_dd = train_with_best_checkpoint(
                model_dd, ckpt_prefix, ADAM_ITERS, monitor="test loss"
            )

            y_pred_dd = model_dd.predict(X_full)
            residual_full_dd = model_dd.predict(X_full, operator=pde_burgers)
            record_result(
                "Data-Driven",
                t_train,
                run,
                y_pred_dd,
                residual_full_dd,
                float(np.sum(losshistory_dd.loss_train[-1])),
                time_dd,
            )
            save_all()
            clean_memory()

        geom = dde.geometry.Interval(L_min, L_max)
        timedomain = dde.geometry.TimeDomain(0, T_MAX)
        geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda x, on_b: on_b)
        ic = dde.icbc.IC(
            geomtime,
            lambda x: -np.sin(np.pi * x[:, 0:1]),
            lambda x, on_i: on_i and np.isclose(x[1], 0),
        )

        # 2. Pure PINN
        if (t_train, "PINN_NoObs", run) not in done_runs:
            set_seed(base_seed + NO_SEED_OFFSET)

            net_no = create_net()
            data_no = dde.data.TimePDE(
                geomtime,
                pde_burgers,
                [bc, ic],
                num_domain=NUM_DOMAIN,
                num_boundary=NUM_BOUNDARY,
                num_initial=NUM_INITIAL,
                num_test=5000,
                solution=burgers_solution,
            )
            model_no = dde.Model(data_no, net_no)
            ckpt_prefix_no = os.path.join(CKPT_DIR, f"no_t{t_train}_run{run}")
            losshist_no, time_no = train_with_best_checkpoint(
                model_no, ckpt_prefix_no, ADAM_ITERS, monitor=PINN_MONITOR
            )

            y_pred_no = model_no.predict(X_full)
            residual_full_no = model_no.predict(X_full, operator=pde_burgers)
            record_result(
                "PINN_NoObs",
                t_train,
                run,
                y_pred_no,
                residual_full_no,
                float(np.sum(losshist_no.loss_train[-1])),
                time_no,
            )
            save_all()
            clean_memory()

        # 3. Hybrid PINN
        if (t_train, "PINN_WithObs", run) not in done_runs:
            set_seed(base_seed + HYB_SEED_OFFSET)

            observe_bc = dde.icbc.PointSetBC(X_fit, y_fit, component=0)
            net_hyb = create_net()
            data_hyb = dde.data.TimePDE(
                geomtime,
                pde_burgers,
                [bc, ic, observe_bc],
                num_domain=NUM_DOMAIN,
                num_boundary=NUM_BOUNDARY,
                num_initial=NUM_INITIAL,
                num_test=5000,
                solution=burgers_solution,
            )
            model_hyb = dde.Model(data_hyb, net_hyb)
            ckpt_prefix_hyb = os.path.join(CKPT_DIR, f"hyb_t{t_train}_run{run}")
            losshist_hyb, time_hyb = train_with_best_checkpoint(
                model_hyb, ckpt_prefix_hyb, ADAM_ITERS, monitor=PINN_MONITOR
            )

            y_pred_hyb = model_hyb.predict(X_full)
            residual_full_hyb = model_hyb.predict(X_full, operator=pde_burgers)
            record_result(
                "PINN_WithObs",
                t_train,
                run,
                y_pred_hyb,
                residual_full_hyb,
                float(np.sum(losshist_hyb.loss_train[-1])),
                time_hyb,
            )
            save_all()
            clean_memory()

# ===================== Final tables =====================
df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)
df_bins = pd.DataFrame(bin_results)
df_bins.to_csv(BINS_CSV, index=False)

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


summary = (
    df.groupby(["T_Train", "Model"])
    .agg(
        {
            "L2_Error_Interp": ["mean", "std", "median"],
            "L2_Error_Extrap": ["mean", "std", "median"],
            "RMSE_Interp": ["mean", "std", "median"],
            "RMSE_Extrap": ["mean", "std", "median"],
            "PDE_Residual_Global": ["mean", "std"],
            "PDE_Residual_Interp": ["mean", "std"],
            "PDE_Residual_Extrap": ["mean", "std"],
            "Time_sec": ["mean", "std"],
        }
    )
    .round(6)
)
print_table(summary.reset_index(), "Summary (Interp + Extrap + RMSE + PDE Residual + Time)")

per_run = df[
    [
        "T_Train",
        "Model",
        "Run",
        "L2_Error_Extrap",
        "RMSE_Extrap",
        "PDE_Residual_Extrap",
        "Time_sec",
        "Final_Train_Loss",
    ]
].sort_values(["T_Train", "Model", "Run"])
print_table(per_run.round(6), "Per-run results (spot-check for divergent runs)")

time_summary = df.groupby("Model")["Time_sec"].agg(["mean", "std"]).reset_index()
print_table(
    time_summary.round(2), "Average training time per model (seconds, across all t_train & runs)"
)

# ===================== Original 2x2 diagnostic figure =====================
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
ax1, ax2, ax3, ax4 = axes.flatten()

for model in MODELS:
    sub = df[df["Model"] == model]
    grouped = sub.groupby("T_Train").agg(
        {"L2_Error_Extrap": ["mean", "std"], "RMSE_Extrap": ["mean", "std"]}
    )
    ax1.errorbar(
        t_train_levels,
        grouped[("L2_Error_Extrap", "mean")],
        yerr=grouped[("L2_Error_Extrap", "std")],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )
    ax2.errorbar(
        t_train_levels,
        grouped[("RMSE_Extrap", "mean")],
        yerr=grouped[("RMSE_Extrap", "std")],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )

example_t = t_train_levels[-1]
for model in MODELS:
    sub = df_bins[(df_bins["Model"] == model) & (df_bins["T_Train"] == example_t)]
    grouped_bins = (
        sub.groupby(["Dist_Lo", "Dist_Hi"])
        .agg({"L2_Error": ["mean", "std"], "RMSE": ["mean", "std"]})
        .reset_index()
    )
    x_centers = (grouped_bins["Dist_Lo"] + grouped_bins["Dist_Hi"]) / 2
    ax3.errorbar(
        x_centers,
        grouped_bins[("L2_Error", "mean")],
        yerr=grouped_bins[("L2_Error", "std")],
        marker="s",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )
    ax4.errorbar(
        x_centers,
        grouped_bins[("RMSE", "mean")],
        yerr=grouped_bins[("RMSE", "std")],
        marker="s",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )

ax1.set_title("Burgers: Extrapolation L² Relative Error", fontsize=14, fontweight="bold")
ax1.set_xlabel("Training Time Horizon $t_{train}$")
ax1.set_yscale("log")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title("Burgers: Extrapolation RMSE", fontsize=14, fontweight="bold")
ax2.set_xlabel("Training Time Horizon $t_{train}$")
ax2.set_yscale("log")
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.set_title(
    f"Burgers: Error Growth vs Distance from $t_{{train}}$={example_t} (L²)",
    fontsize=14,
    fontweight="bold",
)
ax3.set_xlabel("Time Distance ($t - t_{train}$)")
ax3.set_yscale("log")
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4.set_title(
    f"Burgers: Error Growth vs Distance from $t_{{train}}$={example_t} (RMSE)",
    fontsize=14,
    fontweight="bold",
)
ax4.set_xlabel("Time Distance ($t - t_{train}$)")
ax4.set_yscale("log")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(
    "Phase 3 (Burgers): Data-Driven vs Pure PINN vs Hybrid PINN — Extrapolation Performance",
    fontsize=16,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(
    os.path.join(OUTPUT_DIR, "Phase3_Burgers_Full_Comparison_FullDomainPhysics.png"),
    dpi=350,
    bbox_inches="tight",
)
plt.show()

# ===================== PDE residual: global vs extrapolation-region =====================
fig2, (axg, axe) = plt.subplots(1, 2, figsize=(14, 5.5))

for model in MODELS:
    sub = df[df["Model"] == model]
    grouped = sub.groupby("T_Train").agg(
        {
            "PDE_Residual_Global": ["mean", "std"],
            "PDE_Residual_Extrap": ["mean", "std"],
        }
    )
    axg.errorbar(
        t_train_levels,
        grouped[("PDE_Residual_Global", "mean")],
        yerr=grouped[("PDE_Residual_Global", "std")],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )
    axe.errorbar(
        t_train_levels,
        grouped[("PDE_Residual_Extrap", "mean")],
        yerr=grouped[("PDE_Residual_Extrap", "std")],
        marker="o",
        linewidth=2.5,
        markersize=8,
        label=LABELS[model],
        color=COLORS[model],
        capsize=5,
    )

axg.set_title("PDE Residual (Global, whole grid)", fontsize=14, fontweight="bold")
axg.set_xlabel("Training Time Horizon $t_{train}$")
axg.set_yscale("log")
axg.legend()
axg.grid(True, alpha=0.3)

axe.set_title(
    "PDE Residual (Extrapolation region only, $t > t_{train}$)", fontsize=14, fontweight="bold"
)
axe.set_xlabel("Training Time Horizon $t_{train}$")
axe.set_yscale("log")
axe.legend()
axe.grid(True, alpha=0.3)

plt.suptitle(
    "Burgers — Physics Consistency: Global vs Extrapolation-Region PDE Residual",
    fontsize=15,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(
    os.path.join(OUTPUT_DIR, "Phase3_Burgers_PDE_Residual_Extrap_FullDomainPhysics.png"),
    dpi=350,
    bbox_inches="tight",
)
plt.show()
plt.close()

print("\nDone. All figures and CSV files saved.")
