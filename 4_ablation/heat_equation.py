"""Phase 4 - Hybrid-model ablation on the 1D heat equation.

Single-stage full-factorial study over the loss weight lambda, network depth and
activation function. Every combination is trained for NUM_RUNS independent runs; the data
sample of a run is shared by all combinations. Results are resume-safe (CSV checkpoints).
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
print("Phase 4 - Hybrid ablation (single-stage: 24 combinations x 6 runs each)")

# ===================== Settings =====================
L = 1.0
a = 0.4
n = 1
T_MAX = 1.0
ADAM_ITERS = 15000  # total Adam budget per run, same for every sampling method

# Aligned with Phase 3: same collocation counts, fixed (not %-based) observation points
NUM_DOMAIN = 2000
NUM_BOUNDARY = 100
NUM_INITIAL = 200
NUM_SAMPLE_POINTS = 2500
NUM_TEST = 5000
HOLDOUT_FRACTION = 0.2

lambda_values = [0.01, 0.1, 1.0, 10.0]
architectures = [[50] * 3, [50] * 5, [50] * 8]
activations = ["tanh", "GELU"]
sampling_methods = ["uniform"]

NUM_RUNS = 6  # every combination gets this many independent repeats
TOP_N_REPORT = 5  # how many best combos to highlight in the summary/plot

# Fixed filenames (no timestamp) so resume can find prior progress
RESULTS_CSV = "Phase4_Hybrid_Results.csv"
SUMMARY_CSV = "Phase4_Hybrid_Summary.csv"

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


def heat_eq_exact_solution(X):
    x = X[:, 0:1]
    t = X[:, 1:2]
    return np.exp(-(n**2 * np.pi**2 * a * t) / (L**2)) * np.sin(n * np.pi * x / L)


def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t - a * dy_xx


x_dim, t_dim = 256, 201
t_space = np.linspace(0, T_MAX, t_dim).reshape(-1, 1)
x_space = np.linspace(0, L, x_dim).reshape(-1, 1)
xx, tt = np.meshgrid(x_space.flatten(), t_space.flatten())
X_full = np.vstack((np.ravel(xx), np.ravel(tt))).T
y_full = heat_eq_exact_solution(X_full).reshape(-1, 1)

# ==================== Fixed Holdout / Pool split ====================
# Independent seed, done once, never touched by any per-combo RNG below. X_obs
# is always drawn from X_pool; L2_Error is always measured on X_holdout, so no
# combination is ever evaluated on a point it directly trained on.
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

geom = dde.geometry.Interval(0, L)
timedomain = dde.geometry.TimeDomain(0, T_MAX)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(
    geomtime, lambda x: np.sin(n * np.pi * x[:, 0:1] / L), lambda _, on_initial: on_initial
)

# ==================== GPU/backend warm-up (excluded from all timing) ====================
if gpu_available:
    print("Warming up GPU/backend (excluded from timing)...")
    net_w = dde.nn.FNN([2] + [50] * 3 + [1], "tanh", "Glorot normal")
    observe_w = dde.icbc.PointSetBC(X_pool[:100], y_pool[:100], component=0)
    data_w = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe_w],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=NUM_TEST,
        solution=heat_eq_exact_solution,
    )
    model_w = dde.Model(data_w, net_w)
    model_w.compile("adam", lr=1e-3)
    model_w.train(iterations=10, verbose=0)
    sync_cuda()
    del net_w, observe_w, data_w, model_w
    clean_memory()


# =============================================================================
# Core trainer used for every (combo, run) pair.
# =============================================================================
def train_one_combo(lam, arch, act, samp, run, data_seed, init_seed, colloc_seed):
    """Train + evaluate exactly one Hybrid configuration.

    data_seed:   selects X_obs from X_pool via its own RandomState instance --
                 never touches global numpy/torch RNG state, so it can't leak
                 into (or be leaked into by) network initialization.
    init_seed:   seeds dde/numpy/torch globally for network weight init and
                 the "pseudo" (true random) domain/boundary/initial sampling.
    samp:        kept for schema/logging consistency; always "uniform"
                 (other sampling methods were dropped to cut runtime).
    """
    data_rng = np.random.RandomState(data_seed)
    num_data = min(NUM_SAMPLE_POINTS, len(X_pool))
    idx = data_rng.choice(len(X_pool), num_data, replace=False)
    X_obs, y_obs = X_pool[idx], y_pool[idx]
    observe = dde.icbc.PointSetBC(X_obs, y_obs, component=0)

    dde.config.set_random_seed(init_seed)
    np.random.seed(init_seed)
    torch.manual_seed(init_seed)
    net = dde.nn.FNN([2] + arch + [1], act, "Glorot normal")

    # Explicit "pseudo" so this is true random sampling, not DeepXDE's default
    # Hammersley (a QMC sequence), which would make "uniform" a misleading label.
    data = dde.data.TimePDE(
        geomtime,
        pde,
        [bc, ic, observe],
        num_domain=NUM_DOMAIN,
        num_boundary=NUM_BOUNDARY,
        num_initial=NUM_INITIAL,
        num_test=NUM_TEST,
        solution=heat_eq_exact_solution,
    )

    model = dde.Model(data, net)
    loss_weights = [lam, 1.0, 1.0, 1.0]  # PDE, BC, IC, Data

    sync_cuda()
    t0 = time.time()
    model.compile("adam", lr=1e-3, loss_weights=loss_weights, metrics=["l2 relative error"])
    model.train(iterations=ADAM_ITERS, verbose=0)

    # L-BFGS can diverge/raise; without this guard one bad combination kills
    # the entire sweep.
    model.compile("L-BFGS", loss_weights=loss_weights, metrics=["l2 relative error"])
    try:
        model.train(verbose=0)
    except Exception as e:
        print(
            f"[WARN] L-BFGS failed for lam={lam}, arch={arch}, act={act}, samp={samp}, "
            f"run={run}: {e}. Using post-Adam weights."
        )

    sync_cuda()
    t1 = time.time()

    # Evaluate ONLY on the held-out set -- disjoint from X_obs.
    y_pred = model.predict(X_holdout)
    l2 = float(dde.metrics.l2_relative_error(y_holdout, y_pred))
    pde_res = float(np.mean(np.abs(model.predict(X_holdout, operator=pde))))

    del net, data, model
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

                    # data_seed depends only on `run`: all combos share the labeled sample within a run,
                    # while samples differ across runs.
                    data_seed = 50_000 + run
                    init_seed = (
                        1_000 + lam_idx * 512 + arch_idx * 128 + act_idx * 32 + samp_idx * 8 + run
                    )
                    colloc_seed = 20_000 + samp_idx * 1000 + run

                    result = train_one_combo(
                        lam, arch, act, samp, run, data_seed, init_seed, colloc_seed
                    )
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
        Mean_Residual=("PDE_Residual", "mean"),
        Mean_Time=("Time_sec", "mean"),
        N_Repeats=("L2_Error", "count"),
    )
    .reset_index()
    .sort_values("Mean_L2")
)
summary.to_csv(SUMMARY_CSV, index=False)

print(f"\nDone. {len(df)} runs saved to {RESULTS_CSV}.")
print(f"Final ranking (mean +/- std over {NUM_RUNS} independent repeats per combo):")
print(summary.round(6).to_string())
print(f"\nResults: {RESULTS_CSV} | Summary: {SUMMARY_CSV}")

# =============================================================================
# Statistical significance: is the ranking actually distinguishable from noise?
# =============================================================================
# Combos share the labeled sample within a run (paired samples), so the omnibus test is
# Friedman rather than Kruskal-Wallis.
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

    # Pairwise Wilcoxon signed-rank among the TOP_N_REPORT best combos, paired
    # by Run (same rationale as above), with Holm-Bonferroni correction for
    # the multiple comparisons.
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

    # Holm-Bonferroni step-down correction (no extra dependency beyond scipy/numpy)
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
    stat_df.to_csv("Phase4_Statistical_Tests.csv", index=False)
    print(f"\n[Stats] Pairwise Wilcoxon among top {TOP_N_REPORT} combos (Holm-corrected):")
    print(stat_df.round(4).to_string(index=False))
    print(
        "[Stats] NOTE: with n=6 paired runs, power is limited -- a non-significant pair "
        "means 'not distinguishable from noise here', not 'proven equivalent'."
    )

# ===================== Plots =====================
sns.set_style("whitegrid")

# 1. Effect of each factor (mean L2, log scale), averaged over the other factors
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
factor_specs = [
    ("Lambda", axes[0], [str(v) for v in lambda_values]),
    ("Architecture", axes[1], [str(a) for a in architectures]),
    ("Activation", axes[2], activations),
]
for factor, ax, order in factor_specs:
    grp = df.groupby(factor)["L2_Error"].agg(["mean", "std"])
    if factor != "Lambda":
        grp = grp.reindex(order)
    ax.bar(
        grp.index.astype(str),
        grp["mean"],
        yerr=grp["std"],
        capsize=4,
        color="tab:orange",
        alpha=0.85,
    )
    ax.set_yscale("log")
    ax.set_title(f"Mean L2 Error by {factor} (n={NUM_RUNS} reps/combo)", fontweight="bold")
    ax.set_ylabel("Mean L2 Error (log)")
    ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig("Phase4_1_factor_effects.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# 2. Heatmap: Lambda x Architecture (averaged over activation, Screening)
pivot_la = df.pivot_table(index="Lambda", columns="Architecture", values="L2_Error", aggfunc="mean")
plt.figure(figsize=(8, 5), dpi=300)
sns.heatmap(pivot_la, annot=True, fmt=".4f", cmap="viridis_r")
plt.title("Mean L2 Error: Lambda x Architecture (all combos, n=6 reps each)", fontweight="bold")
plt.tight_layout()
plt.savefig("Phase4_2_heatmap_lambda_arch.png", dpi=300)
plt.show()
plt.close()

# 3. Heatmap: Lambda x Activation (averaged over architecture, Screening)
#    Replaces the old "Activation x Sampling" heatmap, which is degenerate now
#    that only one sampling method (uniform) remains.
pivot_act = df.pivot_table(index="Lambda", columns="Activation", values="L2_Error", aggfunc="mean")
plt.figure(figsize=(7, 5), dpi=300)
sns.heatmap(pivot_act, annot=True, fmt=".4f", cmap="viridis_r")
plt.title("Mean L2 Error: Lambda x Activation (all combos, n=6 reps each)", fontweight="bold")
plt.tight_layout()
plt.savefig("Phase4_3_heatmap_lambda_activation.png", dpi=300)
plt.show()
plt.close()

# 3b. Lambda x Architecture heatmap per activation (shared color scale)
vmin, vmax = df["L2_Error"].min(), df["L2_Error"].max()
fig, axes = plt.subplots(1, len(activations), figsize=(7 * len(activations), 5), squeeze=False)
for j, act in enumerate(activations):
    ax = axes[0, j]
    sub = df[df["Activation"] == act]
    pivot_facet = sub.pivot_table(
        index="Lambda", columns="Architecture", values="L2_Error", aggfunc="mean"
    )
    sns.heatmap(
        pivot_facet,
        annot=True,
        fmt=".4f",
        cmap="viridis_r",
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=(j == len(activations) - 1),
    )
    ax.set_title(f"Activation = {act}", fontweight="bold")
plt.suptitle(
    "Mean L2 Error: Lambda x Architecture, faceted by Activation "
    "(reveals 3-way interaction hidden by the averaged heatmaps above)",
    fontsize=12,
    fontweight="bold",
)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("Phase4_3b_heatmap_lambda_arch_by_activation.png", dpi=300)
plt.show()
plt.close()

# 4. Training time by architecture (Screening; deeper/wider nets cost more per
#    iteration -- replaces the old "time by sampling method" plot, which is
#    degenerate now that only one sampling method remains).
plt.figure(figsize=(8, 5), dpi=300)
time_by_arch = (
    df.groupby("Architecture")["Time_sec"]
    .agg(["mean", "std"])
    .reindex([str(a) for a in architectures])
)
plt.bar(
    time_by_arch.index,
    time_by_arch["mean"],
    yerr=time_by_arch["std"],
    capsize=5,
    color="tab:blue",
    alpha=0.85,
)
plt.title("Mean Training Time by Architecture (all combos, n=6 reps each)", fontweight="bold")
plt.ylabel("Time (seconds)")
plt.tight_layout()
plt.savefig("Phase4_4_time_by_architecture.png", dpi=300)
plt.show()
plt.close()

# 5. Top combinations (mean +/- std over the NUM_RUNS independent repeats
#    every combo already received -- no separate confirmation pass needed).
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
plt.title(f"Top {len(top_plot)} Hybrid Configurations (n={NUM_RUNS} reps/combo)", fontweight="bold")
plt.tight_layout()
plt.savefig("Phase4_5_top_combos.png", dpi=300)
plt.show()
plt.close()

# 6. PDE residual vs L2 error (convergence quality check, Screening: broad coverage)
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
plt.savefig("Phase4_6_residual_vs_l2.png", dpi=300)
plt.show()
plt.close()

print("\nAll figures saved.")
