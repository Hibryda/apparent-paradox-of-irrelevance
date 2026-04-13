"""Generate all 5 paper figures from unified_analysis.json.

Sources all data from results/unified_analysis.json (authoritative corrected data)
except Figure 1 which uses results/bl2_bridge_lemma.json for per-pair detail.

Outputs PNG files to output/figures/ for inclusion in the paper.
Style: clean academic, matching PRE conventions.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.special import ndtr
from pathlib import Path

OUT = Path("output/figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'degree': '#e74c3c',
    'kcore': '#2980b9',
    'eigenvector': '#27ae60',
    'clustering': '#f39c12',
    'random': '#95a5a6',
}

CORPUS_COLORS = {
    'dev': '#2980b9',
    'held-out-1': '#e74c3c',
    'expanded': '#27ae60',
}


def load_unified():
    return json.load(open("results/unified_analysis.json"))


# ── Figure 1: BL2 scatter ──────────────────────────────────────

def fig1_bl2_scatter():
    """BL2-predicted vs observed Var(sym) from unified data."""
    data = load_unified()

    predicted = []
    observed = []
    features = []

    for net in data["per_network"]:
        for feat_name in ["degree", "kcore", "eigenvector", "clustering", "random"]:
            bl2 = net["features"][feat_name].get("bl2", {})
            p = bl2.get("var_predicted")
            o = bl2.get("var_observed")
            if p is not None and o is not None and o > 0:
                predicted.append(p)
                observed.append(o)
                features.append(feat_name)

    predicted = np.array(predicted)
    observed = np.array(observed)
    r2 = np.corrcoef(predicted, observed)[0, 1] ** 2

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for feat, color in COLORS.items():
        mask = np.array([f == feat for f in features])
        if mask.any():
            ax.scatter(predicted[mask], observed[mask], c=color, s=18, alpha=0.7,
                      label=feat, edgecolors='none')

    lims = [0, max(predicted.max(), observed.max()) * 1.1]
    ax.plot(lims, lims, '--', color='grey', linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("BL2-Predicted Var(sym)")
    ax.set_ylabel("Observed Var(sym)")
    ax.set_title(f"R² = {r2:.3f}")
    ax.legend(frameon=False, loc='lower right')
    ax.set_aspect('equal')
    fig.savefig(OUT / "fig1_bl2_scatter.png")
    plt.close()
    print(f"  F1: {len(predicted)} points, R²={r2:.3f}")


# ── Figure 2: Mixture AUC decomposition (vertical) ───────────

def fig2_mixture_auc():
    """Horizontal stacked bar: tie + continuous contributions per network, sorted by total AUC."""
    data = load_unified()

    networks = []
    tie_contrib = []
    cont_contrib = []

    for net in data["per_network"]:
        mix = net["features"]["kcore"]["mixture"]
        tie_auc = mix["auc_tie"]
        total_auc = mix["auc_observed"]
        cont_auc = total_auc - tie_auc

        name = net["network"]
        # Shorten long names
        if len(name) > 25:
            name = name[:22] + "..."
        networks.append(name)
        tie_contrib.append(tie_auc)
        cont_contrib.append(cont_auc)

    # Sort by total AUC
    total = np.array(tie_contrib) + np.array(cont_contrib)
    idx = np.argsort(total)
    networks = [networks[i] for i in idx]
    tie_contrib = np.array(tie_contrib)[idx]
    cont_contrib = np.array(cont_contrib)[idx]

    n = len(networks)
    fig, ax = plt.subplots(figsize=(6, 9))  # fits within a single page
    y = np.arange(n)

    ax.barh(y, tie_contrib, color='#2980b9', label='Tie contribution', height=0.8)
    ax.barh(y, cont_contrib, left=tie_contrib, color='#f39c12',
            label='Continuous contribution', height=0.8)
    ax.axvline(x=0.5, color='grey', linestyle='--', linewidth=0.8, alpha=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(networks, fontsize=5.5)
    ax.set_xlabel("AUC")
    ax.set_title("K-core Similarity: Mixture AUC Decomposition")
    ax.legend(frameon=False, loc='lower right')
    ax.set_xlim(0, 1.0)

    fig.savefig(OUT / "fig2_mixture_auc.png")
    plt.close()
    print(f"  F2: {n} networks")


# ── Figure 3: d' vs AUC (single panel, all 65 networks) ──────

def fig3_dprime_vs_auc():
    """Signed d' vs AUC for kcore and degree across all 65 networks.
    Matches paper description: pooled rho, Gaussian AUC curve overlay."""
    data = load_unified()

    dprime_vals = []
    auc_vals = []
    feat_types = []

    for net in data["per_network"]:
        for feat in ["kcore", "degree"]:
            fd = net["features"][feat]
            dp = fd.get("signed_dprime")
            auc = fd.get("auc")
            if dp is not None and auc is not None:
                dprime_vals.append(dp)
                auc_vals.append(auc)
                feat_types.append(feat)

    rho, _ = sp_stats.spearmanr(dprime_vals, auc_vals)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for feat, color, label in [("kcore", '#2980b9', 'k-core'), ("degree", '#e74c3c', 'degree')]:
        mask = [f == feat for f in feat_types]
        dp = np.array(dprime_vals)[mask]
        au = np.array(auc_vals)[mask]
        ax.scatter(dp, au, c=color, s=25, alpha=0.7, label=label, edgecolors='none')

    # Gaussian AUC curve: AUC = Phi(d'/sqrt(2))
    dp_range = np.linspace(min(dprime_vals) - 0.3, max(dprime_vals) + 0.3, 200)
    gauss_auc = ndtr(dp_range / np.sqrt(2))
    ax.plot(dp_range, gauss_auc, '--', color='grey', linewidth=1.2, alpha=0.6,
            label=f"Gaussian: Φ(d'/√2)")

    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Signed d'")
    ax.set_ylabel("AUC")
    ax.set_title(f"Signed d' vs AUC (pooled ρ = {rho:.3f}, n = {len(dprime_vals)})")
    ax.legend(frameon=False, loc='lower right')

    fig.savefig(OUT / "fig3_dprime_vs_auc.png")
    plt.close()
    print(f"  F3: {len(dprime_vals)} points, rho={rho:.3f}")


# ── Figure 4: Metric independence ─────────────────────────────

def fig4_metric_independence():
    """AUC by metric for k-core and degree across 65 networks.
    Three continuous metrics (cosine dropped — degenerate for scalars) + binary match."""
    data = load_unified()

    metrics = ["sym", "jaccard", "exp_diff", "binary_match"]
    metric_labels = ["sym", "Jaccard", "exp-diff", "binary match"]
    metric_colors = ['#2980b9', '#27ae60', '#f39c12', '#95a5a6']
    metric_styles = ['-', '-', '-', '--']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))

    for feat_name, ax, title in [("kcore", ax1, "K-core"), ("degree", ax2, "Degree")]:
        for metric, label, color, ls in zip(metrics, metric_labels, metric_colors, metric_styles):
            aucs = []
            for net in data["per_network"]:
                auc = net["features"][feat_name]["metric_aucs"].get(metric)
                if auc is not None:
                    aucs.append(auc)

            if aucs:
                x = np.arange(len(aucs))
                ax.plot(x, sorted(aucs), color=color, label=label, linewidth=1.5,
                       alpha=0.9 if metric != "binary_match" else 0.6,
                       linestyle=ls)

        ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlabel("Network (sorted by AUC)")
        ax.set_ylabel("AUC")
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_metric_independence.png")
    plt.close()
    print(f"  F4: 3 continuous + binary match, cosine omitted")


# ── Figure 5: Cumulative validation (single panel, by provenance) ─

def fig5_cumulative_validation():
    """Signed d' vs AUC colored by corpus provenance (dev, held-out, expanded)."""
    data = load_unified()

    dprime_vals = []
    auc_vals = []
    corpus_types = []
    feat_types = []

    for net in data["per_network"]:
        corpus = net["corpus"]
        for feat in ["kcore", "degree"]:
            fd = net["features"][feat]
            dp = fd.get("signed_dprime")
            auc = fd.get("auc")
            if dp is not None and auc is not None:
                dprime_vals.append(dp)
                auc_vals.append(auc)
                corpus_types.append(corpus)
                feat_types.append(feat)

    rho, _ = sp_stats.spearmanr(dprime_vals, auc_vals)

    corpus_labels = {
        'dev': f'Development (31)',
        'held-out-1': f'Held-out (10)',
        'expanded': f'Expanded (24)',
    }

    fig, ax = plt.subplots(figsize=(5.5, 5))

    for corpus, color in CORPUS_COLORS.items():
        mask = [c == corpus for c in corpus_types]
        if any(mask):
            dp = np.array(dprime_vals)[mask]
            au = np.array(auc_vals)[mask]
            ax.scatter(dp, au, c=color, s=25, alpha=0.7,
                      label=corpus_labels.get(corpus, corpus), edgecolors='none')

    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='grey', linestyle=':', linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Signed d'")
    ax.set_ylabel("AUC")
    ax.set_title(f"Pooled ρ = {rho:.3f} (n = {len(dprime_vals)})")
    ax.legend(frameon=False, loc='lower right')

    fig.savefig(OUT / "fig5_cumulative_validation.png")
    plt.close()
    print(f"  F5: {len(dprime_vals)} points, rho={rho:.3f}")


# ── Main ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating paper figures from unified_analysis.json...")
    fig1_bl2_scatter()
    fig2_mixture_auc()
    fig3_dprime_vs_auc()
    fig4_metric_independence()
    fig5_cumulative_validation()
    print(f"\nAll figures saved to {OUT}/")
    for f in sorted(OUT.glob("*.png")):
        sz = f.stat().st_size / 1024
        print(f"  {f.name} ({sz:.0f}KB)")
