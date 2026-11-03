"""Phase 4 - Hybrid-model ablation for the inverse diffusion problem.

Full-factorial study over the loss weight lambda (architecture and activation are fixed
here) with several independent runs per setting.
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
print(
    "Phase 4 - Inverse Diffusion Hybrid Ablation (single-stage: 24 combos x 6 runs each = 144 runs)"
)

# ===================== Settings =====================
C_TRUE = 1.0
C_INIT = 2.0
ADAM_ITERS = 50000  # Adam only, no L-BFGS

# Aligned with the confirmed Inverse-Diffusion configuration (page-matched
# collocation, 100 fixed observation points, NOT heat's 2500)
NUM_DOMAIN = 40
NUM_BOUNDARY = 20
NUM_INITIAL = 10
NUM_SAMPLE_POINTS = 100
NUM_TEST = 2000
HOLDOUT_FRACTION = 0.2

lambda_values = [0.01, 0.1, 1.0, 10.0]

architectures = [[32] * 3]  # only the main architecture
activations = ["tanh"]  # only tanh
sampling_methods = ["uniform"]

NUM_RUNS = 6  # every combination gets this many independent repeats
TOP_N_REPORT = 5  # how many best combos to highlight in the summary/plot

# Fixed filenames (no timestamp) so resume can find prior progress
RESULTS_CSV = "Phase4_Diffusion_Hybrid_Results.csv"
SUMMARY_CSV = "Phase4_Diffusion_Hybrid_Summary.csv"

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


def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def func(x):
    return np.sin(np.pi * x[:, 0:1]) * np.exp(-x[:, 1:])


def make_pde(C):
    """Build a PDE residual function that receives C through a closure
    (each combo/run needs a fresh dde.Variable so nothing leaks between trainings)."""

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


x_dim, t_dim = 256, 201
x_space = np.linspace(-1, 1, x_dim).reshape(-1, 1)
t_space = np.linspace(0, 1, t_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = func(X_full)

# ==================== Fixed Holdout / Pool split ====================
POOL_SPLIT_SEED = 9999
holdout_rng = np.random.RandomState(POOL_SPLIT_SEED)
n_total = len(X_full)
n_holdout = int(n_total * HOLDOUT_FRACTION)
all_idx = np.arange(n_total)
holdout_idx = holdout_rng.choice(all_idx, size=n_holdout, replace=False)
pool_idx = np.setdiff1d(all_idx, holdout_idx)
X_holdout, y_holdout = X_full[holdout_idx], y_full[holdout_idx]
X_pool, y_pool = X_full[pool_idx], y_full[pool_idx]
print(
    f"Holdout: {len(X_holdout)} points | Pool: {len(X_pool)} points "
    f"(labeled sample size per run: {min(NUM_SAMPLE_POINTS, len(X_pool))})"
)

geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, func, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, func, lambda _, on_initial: on_initial)

# ==================== GPU/backend warm-up (excluded from all timing) ====================
if gpu_available:
    print("Warming up GPU/backend (excluded from timing)...")
    C_w = dde.Variable(C_INIT)
    pde_w = make_pde(C_w)
    net_w = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot normal")
    observe_w = dde.icbc.PointSetBC(X_pool[:20], y_pool[:20], component=0)
    data_w = dde.data.TimePDE(
        geomtime,
        pde_w,
        [bc, ic, observe_w],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=NUM_TEST,
        solution=func,
    )
    model_w = dde.Model(data_w, net_w)
    model_w.compile("adam", lr=1e-3, external_trainable_variables=C_w)
    model_w.train(iterations=10, verbose=0)
    sync_cuda()
    del C_w, pde_w, net_w, observe_w, data_w, model_w
    clean_memory()


# =============================================================================
# Core trainer used for every (combo, run) pair.
# =============================================================================
def train_one_combo(lam, arch, act, samp, run, data_seed, init_seed):
    """Train + evaluate exactly one Hybrid configuration.

    data_seed:   selects X_obs from X_pool via its own RandomState instance --
                 never touches global numpy/torch RNG state, so it can't leak
                 into (or be leaked into by) network/variable initialization.
    init_seed:   seeds dde/numpy/torch globally for network weight init and
                 the "pseudo" (true random) domain/boundary/initial sampling.
    samp:        kept for schema/logging consistency; always "uniform" here.
    """
    data_rng = np.random.RandomState(data_seed)
    num_data = min(NUM_SAMPLE_POINTS, len(X_pool))
    idx = data_rng.choice(len(X_pool), num_data, replace=False)
    X_obs, y_obs = X_pool[idx], y_pool[idx]
    observe = dde.icbc.PointSetBC(X_obs, y_obs, component=0)

    dde.config.set_random_seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)

    C_var = dde.Variable(C_INIT)
    pde_var = make_pde(C_var)
    net = dde.nn.FNN([2] + arch + [1], act, "Glorot normal")

    data = dde.data.TimePDE(
        geomtime,
        pde_var,
        [bc, ic, observe],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=NUM_TEST,
        solution=func,
    )

    model = dde.Model(data, net)
    loss_weights = [lam, 1.0, 1.0, 1.0]  # PDE, BC, IC, Data

    sync_cuda()
    t0 = time.time()
    model.compile(
        "adam",
        lr=1e-3,
        loss_weights=loss_weights,
        metrics=["l2 relative error"],
        external_trainable_variables=C_var,
    )
    model.train(iterations=ADAM_ITERS, verbose=0)
    sync_cuda()
    t1 = time.time()

    # Evaluate ONLY on the held-out set -- disjoint from X_obs.
    y_pred = model.predict(X_holdout)
    l2 = float(dde.metrics.l2_relative_error(y_holdout, y_pred))
    pde_res = float(np.mean(np.abs(model.predict(X_holdout, operator=pde_var))))
    c_pred = get_C_value(C_var)
    c_err = abs(c_pred - C_TRUE)

    del C_var, pde_var, net, data, model
    clean_memory()

    return {
        "Lambda": lam,
        "Architecture": str(arch),
        "Activation": act,
        "Sampling": samp,
        "Run": run,
        "Model": "Hybrid",
        "L2_Error": l2,
        "PDE_Residual": pde_res,
        "C_Predicted": c_pred,
        "C_Error": c_err,
        "Time_sec": t1 - t0,
        "DataSeed": data_seed,
        "InitSeed": init_seed,
    }


# =============================================================================
# Single stage: every combination gets NUM_RUNS independent repeats
# =============================================================================
if os.path.exists(RESULTS_CSV):
    df_existing = pd.read_csv(RESULTS_CSV)
    results = df_existing.to_dict("records")
    done_runs = set(
        zip(
            df_existing["Lambda"],
            df_existing["Architecture"],
            df_existing["Activation"],
            df_existing["Sampling"],
            df_existing["Run"],
        )
    )
    print(f"Found {len(results)} existing records.")
else:
    results = []
    done_runs = set()


def save_all():
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)


total_runs = (
    len(lambda_values) * len(architectures) * len(activations) * len(sampling_methods) * NUM_RUNS
)
print(
    f"Total runs: {total_runs} ({len(lambda_values)*len(architectures)*len(activations)*len(sampling_methods)} "
    f"combos x {NUM_RUNS} repeats)"
)
pbar = tqdm(total=total_runs, desc="Hybrid Progress")
pbar.update(len(results))

for lam_idx, lam in enumerate(lambda_values):
    for arch_idx, arch in enumerate(architectures):
        for act_idx, act in enumerate(activations):
            for samp_idx, samp in enumerate(sampling_methods):
                for run in range(NUM_RUNS):
                    key = (lam, str(arch), act, samp, run)
                    if key in done_runs:
                        continue

                    data_seed = 50_000 + run
                    init_seed = (
                        1_000 + lam_idx * 512 + arch_idx * 128 + act_idx * 32 + samp_idx * 8 + run
                    )

                    result = train_one_combo(lam, arch, act, samp, run, data_seed, init_seed)
                    results.append(result)
                    save_all()
                    pbar.update(1)

pbar.close()

df = pd.DataFrame(results)
df.to_csv(RESULTS_CSV, index=False)

summary = (
    df.groupby(["Lambda", "Architecture", "Activation", "Sampling"])
    .agg(
        Mean_L2=("L2_Error", "mean"),
        Std_L2=("L2_Error", "std"),
        Mean_C_Error=("C_Error", "mean"),
        Std_C_Error=("C_Error", "std"),
        Mean_Residual=("PDE_Residual", "mean"),
        Mean_Time=("Time_sec", "mean"),
        N_Repeats=("L2_Error", "count"),
    )
    .reset_index()
    .sort_values("Mean_L2")
)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nDone. {len(df)} runs saved to {RESULTS_CSV}.")
print(
    f"Final ranking by solution accuracy (mean +/- std over {NUM_RUNS} independent repeats per combo):"
)
print(summary.round(6).to_string())

summary_by_c = summary.sort_values("Mean_C_Error")
print(f"\nFinal ranking by parameter-identification accuracy (Mean_C_Error, lower is better):")
print(summary_by_c.round(6).to_string())
print(f"\nResults: {RESULTS_CSV} | Summary: {SUMMARY_CSV}")

# =============================================================================
# Statistical significance: is the L2 ranking actually distinguishable from noise?
# =============================================================================
from scipy.stats import friedmanchisquare, wilcoxon

df["Combo"] = df.apply(lambda r: f"lam={r.Lambda},{r.Architecture},{r.Activation}", axis=1)
pivot_all = df.pivot(index="Run", columns="Combo", values="L2_Error")

stat_rows = []
if pivot_all.isnull().values.any() or len(pivot_all) < 3:
    print(
        "\n[Stats] Skipping Friedman test: incomplete grid (missing runs for some combo) "
        "or fewer than 3 runs -- fill in all combos/runs first."
    )
else:
    fr_stat, fr_p = friedmanchisquare(*[pivot_all[c].values for c in pivot_all.columns])
    print(
        f"\n[Stats] Friedman test across all {pivot_all.shape[1]} combos "
        f"(n={pivot_all.shape[0]} paired runs): chi2={fr_stat:.3f}, p={fr_p:.4g}"
    )
    print(
        "[Stats] "
        + (
            "Significant -> combos are NOT all equivalent; see pairwise comparisons below."
            if fr_p < 0.05
            else "Not significant at alpha=0.05 -> cannot conclude the combos differ from each other."
        )
    )

    top_labels = (
        summary.head(TOP_N_REPORT)
        .apply(lambda r: f"lam={r.Lambda},{r.Architecture},{r.Activation}", axis=1)
        .tolist()
    )
    pair_results = []
    for i in range(len(top_labels)):
        for j in range(i + 1, len(top_labels)):
            a, b = top_labels[i], top_labels[j]
            diff = pivot_all[a].values - pivot_all[b].values
            if np.allclose(diff, 0):
                stat, p = np.nan, 1.0
            else:
                stat, p = wilcoxon(pivot_all[a].values, pivot_all[b].values)
            pair_results.append(
                {
                    "Rank_A": i + 1,
                    "Combo_A": a,
                    "Rank_B": j + 1,
                    "Combo_B": b,
                    "W_stat": stat,
                    "p_raw": p,
                }
            )

    m = len(pair_results)
    order = np.argsort([r["p_raw"] for r in pair_results])
    p_holm = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pair_results[idx]["p_raw"]
        running_max = max(running_max, adj)
        p_holm[idx] = min(running_max, 1.0)
    for r, p_adj in zip(pair_results, p_holm):
        r["p_holm"] = p_adj
        r["Significant_alpha05"] = p_adj < 0.05

    stat_df = pd.DataFrame(pair_results)
    stat_df.to_csv("Phase4_Diffusion_Statistical_Tests.csv", index=False)
    print(f"\n[Stats] Pairwise Wilcoxon among top {TOP_N_REPORT} combos by L2 (Holm-corrected):")
    print(stat_df.round(4).to_string(index=False))
    print(
        "[Stats] NOTE: with n=6 paired runs, power is limited -- a non-significant pair "
        "means 'not distinguishable from noise here', not 'proven equivalent'."
    )


sns.set_style("whitegrid")

# 1. Effect of lambda on L2 error (mean +/- std, log scale)
plt.figure(figsize=(9, 6), dpi=300)
grp_l2 = df.groupby("Lambda")["L2_Error"].agg(["mean", "std"]).reindex(lambda_values)
plt.bar(
    [str(v) for v in grp_l2.index],
    grp_l2["mean"],
    yerr=grp_l2["std"],
    capsize=5,
    color="tab:orange",
    alpha=0.85,
)
plt.yscale("log")
plt.title(
    f"Effect of PDE Loss Weight (lambda) on Solution Accuracy\n"
    f"(Architecture=[32,32,32], Activation=tanh, n={NUM_RUNS} reps/lambda)",
    fontweight="bold",
)
plt.xlabel("Lambda (PDE loss weight)")
plt.ylabel("Mean L2 Error (log)")
plt.tight_layout()
plt.savefig("Phase4_Diffusion_1_lambda_effect_L2.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 1b. Effect of lambda on C-identification error (mean +/- std, log scale)
plt.figure(figsize=(9, 6), dpi=300)
grp_c = df.groupby("Lambda")["C_Error"].agg(["mean", "std"]).reindex(lambda_values)
plt.bar(
    [str(v) for v in grp_c.index],
    grp_c["mean"],
    yerr=grp_c["std"],
    capsize=5,
    color="tab:purple",
    alpha=0.85,
)
plt.yscale("log")
plt.title(
    f"Effect of PDE Loss Weight (lambda) on Parameter Identification\n"
    f"(Architecture=[32,32,32], Activation=tanh, n={NUM_RUNS} reps/lambda)",
    fontweight="bold",
)
plt.xlabel("Lambda (PDE loss weight)")
plt.ylabel(f"Mean |C_pred - {C_TRUE}| (log)")
plt.tight_layout()
plt.savefig("Phase4_Diffusion_1b_lambda_effect_Cerror.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 1c. Combined view: L2 and C-error vs lambda side by side (linear-x, log-y),
#     so the two criteria can be read together at a glance.
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
axes[0].errorbar(
    grp_l2.index.astype(str),
    grp_l2["mean"],
    yerr=grp_l2["std"],
    fmt="-o",
    capsize=5,
    color="tab:orange",
    linewidth=2,
    markersize=8,
)
axes[0].set_yscale("log")
axes[0].set_title("Solution Accuracy (L2 Error) vs Lambda", fontweight="bold")
axes[0].set_xlabel("Lambda")
axes[0].set_ylabel("Mean L2 Error (log)")
axes[0].grid(True, which="both", ls="--", alpha=0.5)

axes[1].errorbar(
    grp_c.index.astype(str),
    grp_c["mean"],
    yerr=grp_c["std"],
    fmt="-o",
    capsize=5,
    color="tab:purple",
    linewidth=2,
    markersize=8,
)
axes[1].set_yscale("log")
axes[1].set_title("Parameter Identification (C Error) vs Lambda", fontweight="bold")
axes[1].set_xlabel("Lambda")
axes[1].set_ylabel(f"Mean |C_pred - {C_TRUE}| (log)")
axes[1].grid(True, which="both", ls="--", alpha=0.5)
plt.suptitle(
    "Effect of PDE Loss Weight (Architecture=[32,32,32], Activation=tanh)", fontweight="bold"
)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("Phase4_Diffusion_1c_lambda_effect_combined.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 5. Top combinations by L2 accuracy (now just a ranking of the 4 lambda values)
top_plot = summary.head(TOP_N_REPORT).copy()
top_plot["Label"] = top_plot.apply(
    lambda r: f"lam={r.Lambda},{r.Architecture},{r.Activation},{r.Sampling}", axis=1
)
plt.figure(figsize=(10, 7), dpi=300)
plt.barh(
    top_plot["Label"][::-1],
    top_plot["Mean_L2"][::-1],
    xerr=top_plot["Std_L2"][::-1],
    capsize=3,
    color="tab:green",
    alpha=0.85,
)
plt.xscale("log")
plt.xlabel(f"Mean L2 Error (log), +/- std over {NUM_RUNS} independent repeats")
plt.title(
    f"Top {len(top_plot)} Hybrid Configurations by Solution Accuracy (n={NUM_RUNS} reps/combo)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("Phase4_Diffusion_5_top_combos_L2.png", dpi=300)
plt.show()
plt.close()

# 5b. Top combinations by parameter-identification accuracy
top_plot_c = summary_by_c.head(TOP_N_REPORT).copy()
top_plot_c["Label"] = top_plot_c.apply(
    lambda r: f"lam={r.Lambda},{r.Architecture},{r.Activation},{r.Sampling}", axis=1
)
plt.figure(figsize=(10, 7), dpi=300)
plt.barh(
    top_plot_c["Label"][::-1],
    top_plot_c["Mean_C_Error"][::-1],
    xerr=top_plot_c["Std_C_Error"][::-1],
    capsize=3,
    color="tab:purple",
    alpha=0.85,
)
plt.xscale("log")
plt.xlabel(f"Mean |C_pred - {C_TRUE}| (log), +/- std over {NUM_RUNS} independent repeats")
plt.title(
    f"Top {len(top_plot_c)} Hybrid Configurations by Parameter Identification (n={NUM_RUNS} reps/combo)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig("Phase4_Diffusion_5b_top_combos_Cerror.png", dpi=300)
plt.show()
plt.close()

# 6. PDE residual vs L2 error, colored by Lambda
plt.figure(figsize=(7, 6), dpi=300)
plt.scatter(df["PDE_Residual"], df["L2_Error"], alpha=0.5, c=df["Lambda"], cmap="plasma")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("PDE Residual (log)")
plt.ylabel("L2 Error (log)")
plt.colorbar(label="Lambda")
plt.title(
    "PDE Residual vs L2 Error, colored by Lambda (all combos, n=6 reps each)", fontweight="bold"
)
plt.tight_layout()
plt.savefig("Phase4_Diffusion_6_residual_vs_l2.png", dpi=300)
plt.show()
plt.close()

# 7. L2 error vs C-identification error, colored by Lambda (do the two criteria agree?)
plt.figure(figsize=(7, 6), dpi=300)
plt.scatter(df["L2_Error"], df["C_Error"], alpha=0.5, c=df["Lambda"], cmap="plasma")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("L2 Error (log)")
plt.ylabel(f"|C_pred - {C_TRUE}| (log)")
plt.colorbar(label="Lambda")
plt.title("Solution Accuracy vs Parameter Identification, colored by Lambda", fontweight="bold")
plt.tight_layout()
plt.savefig("Phase4_Diffusion_7_l2_vs_Cerror.png", dpi=300)
plt.show()
plt.close()

print("\nAll figures saved.")
