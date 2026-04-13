#!/usr/bin/env python3
"""Corrected and comprehensive tie effect analysis.

Incorporates 6 corrections from peer review:
  1. MIXTURE AUC FORMULA — replaces invalid d'_tie + d'_continuous decomposition
  2. SCORE-LEVEL HERFINDAHL — on sym scores, not raw feature values
  3. PARAMETER-MATCHED MODEL COMPARISON — Gaussian vs Restricted vs Full mixture
  4. NETWORK-LEVEL BOOTSTRAP — CIs for rho(d', AUC) and rho(tie_enrichment, AUC)
  5. BOOTSTRAP CIs ON FOUR-QUADRANT — tie fraction with 95% CI
  6. CROSS-FEATURE GENERALIZATION — test mechanism beyond kcore

Features: degree, kcore, eigenvector (fallback: degree_centrality),
          clustering, random (control).

Runtime target: <10 min.
"""
import sys
import json
import time
import warnings
import numpy as np
import networkx as nx
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

SM_ROOT = Path(__file__).resolve().parents[1] / ".." / "symmetry-mechanism"
sys.path.insert(0, str(SM_ROOT / "src"))

from phase6_benchmark import (
    load_netzschleuder_csv, load_airport,
    PHASE6_CONFIGS, BIO_DIR,
)
from multi_network_analysis import (
    NETWORK_CONFIGS,
    load_edge_list as mna_load_edge_list,
    load_bitcoin, load_string_network,
    load_epinions, load_slashdot, load_wiki_rfa,
)

BENCHMARK_PATH = SM_ROOT / "data" / "phase6_benchmark.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

MAX_EDGES = 4000
MAX_NONEDGES = 4000
TIE_TOL = 1e-6
RNG_SEED = 42
FEATURES = ["degree", "kcore", "eigenvector", "clustering", "random"]
BOOTSTRAP_B = 500
HERFINDAHL_BINS = 100


# ══════════════════════════════════════════════════════════════════════
# LOADER
# ══════════════════════════════════════════════════════════════════════

def load_network_by_name(name):
    """Load network by name from benchmark configs (handles all 32 networks)."""
    for cfg in NETWORK_CONFIGS:
        if cfg["name"] == name:
            loader = cfg.get("loader", "edge_list")
            max_n = cfg.get("max_nodes", 3000)
            if loader == "edge_list":
                return mna_load_edge_list(cfg["path"], max_nodes=max_n)
            elif loader == "bitcoin":
                return load_bitcoin(cfg["path"], max_nodes=max_n)
            elif loader == "string":
                return load_string_network(cfg["path"], max_nodes=max_n)
            elif loader == "epinions":
                return load_epinions(cfg["path"], max_nodes=max_n)
            elif loader == "slashdot":
                return load_slashdot(cfg["path"], max_nodes=max_n)
            elif loader == "wiki_rfa":
                return load_wiki_rfa(cfg["path"], max_nodes=max_n)
            else:
                return mna_load_edge_list(cfg["path"], max_nodes=max_n)
    for cfg in PHASE6_CONFIGS:
        if cfg["name"] == name:
            if cfg["loader"] == "netzschleuder":
                return load_netzschleuder_csv(
                    cfg["path"], weight_col=cfg.get("weight_col", "weight"),
                    max_nodes=cfg.get("max_nodes", 3000),
                    preserve_sign=cfg.get("preserve_sign", False))
            elif cfg["loader"] == "airport":
                return load_airport(cfg["path"], max_nodes=cfg.get("max_nodes", 3000))
    return None


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def sym_vec(a, b):
    """Vectorised sym(a, b) = 1 - |a-b|/(a+b). Returns NaN where a+b==0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, 1.0 - np.abs(a - b) / s, np.nan)


def compute_features(G, rng):
    """Compute node features: degree, kcore, eigenvector, clustering, random."""
    nodes = list(G.nodes())
    deg = dict(G.degree())
    kcore = nx.core_number(G)
    try:
        eigv = nx.eigenvector_centrality(G, max_iter=300, tol=1e-3)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigv = nx.degree_centrality(G)
    clust = nx.clustering(G)
    random_feat = {n: abs(rng.normal(0, 1)) + 0.01 for n in nodes}
    return {
        "degree": deg,
        "kcore": kcore,
        "eigenvector": eigv,
        "clustering": clust,
        "random": random_feat,
    }


def sample_pairs(G, rng):
    """Sample edges and non-edges."""
    edges = list(G.edges())
    if len(edges) > MAX_EDGES:
        idx = rng.choice(len(edges), MAX_EDGES, replace=False)
        edges = [edges[i] for i in idx]

    nodes = list(G.nodes())
    edge_set = set(G.edges())
    non_edges = []
    target = min(MAX_NONEDGES, len(edges))
    attempts = 0
    while len(non_edges) < target and attempts < target * 20:
        u, v = rng.choice(nodes, 2, replace=False)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            non_edges.append((u, v))
        attempts += 1
    return edges, non_edges


def compute_sym_arrays(edges, non_edges, feat_dict):
    """Compute sym arrays for edges and non-edges given a feature dict.

    Adds tiny epsilon (0.001) to avoid zero-sum pairs for integer features.
    Returns (sym_edge, sym_nonedge) as 1-D float arrays with NaN filtered out.
    """
    fu_e = np.array([float(feat_dict.get(u, 0)) + 0.001 for u, _ in edges])
    fv_e = np.array([float(feat_dict.get(v, 0)) + 0.001 for _, v in edges])
    fu_ne = np.array([float(feat_dict.get(u, 0)) + 0.001 for u, _ in non_edges])
    fv_ne = np.array([float(feat_dict.get(v, 0)) + 0.001 for _, v in non_edges])

    sym_e = sym_vec(fu_e, fv_e)
    sym_ne = sym_vec(fu_ne, fv_ne)

    sym_e = sym_e[~np.isnan(sym_e)]
    sym_ne = sym_ne[~np.isnan(sym_ne)]
    return sym_e, sym_ne


def auc_mann_whitney(pos, neg):
    """Mann-Whitney AUC. Returns NaN if either array is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u_stat, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u_stat / (len(pos) * len(neg)))


def safe_spearman(x, y):
    """Spearman on finite pairs only. Returns (rho, p, n)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def dprime(sym_e, sym_ne):
    """Fisher discriminant d' = |mu_e - mu_ne| / sqrt(0.5*(var_e + var_ne)).

    Uses pooled variance (root-mean of variances) in denominator.
    Returns NaN if insufficient data or zero variance.
    """
    if len(sym_e) < 5 or len(sym_ne) < 5:
        return np.nan
    mu_e, mu_ne = np.mean(sym_e), np.mean(sym_ne)
    var_e = np.var(sym_e, ddof=1)
    var_ne = np.var(sym_ne, ddof=1)
    denom = np.sqrt(0.5 * (var_e + var_ne))
    if denom < 1e-15:
        return np.nan
    return float(abs(mu_e - mu_ne) / denom)


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 1: MIXTURE AUC FORMULA
# ══════════════════════════════════════════════════════════════════════

def compute_mixture_auc(sym_e, sym_ne):
    """Compute the mixture AUC decomposition.

    For a given feature:
      p_e  = P(sym=1 | edge)
      p_ne = P(sym=1 | non-edge)
      AUC_continuous = AUC on non-tied pairs only
      AUC_mix = p_e*(1-p_ne) + 0.5*p_e*p_ne + (1-p_e)*(1-p_ne)*AUC_continuous

    Returns dict with all components, or None if insufficient data.
    """
    if len(sym_e) < 10 or len(sym_ne) < 10:
        return None

    # Tie fractions
    tie_e = np.abs(sym_e - 1.0) < TIE_TOL
    tie_ne = np.abs(sym_ne - 1.0) < TIE_TOL
    p_e = float(np.mean(tie_e))
    p_ne = float(np.mean(tie_ne))

    # AUC on non-tied subset only
    cont_e = sym_e[~tie_e]
    cont_ne = sym_ne[~tie_ne]
    if len(cont_e) >= 2 and len(cont_ne) >= 2:
        auc_continuous = auc_mann_whitney(cont_e, cont_ne)
    else:
        # All pairs tied or nearly so — continuous AUC is undefined, use 0.5
        auc_continuous = 0.5

    # Mixture formula
    # Term 1: tied edge vs untied non-edge — edge wins (sym=1 > sym<1)
    # Term 2: tied edge vs tied non-edge — draw (both sym=1)
    # Term 3: untied edge vs untied non-edge — use continuous AUC
    # Missing implicit: untied edge vs tied non-edge — non-edge wins (0 contribution to AUC)
    auc_mix = (p_e * (1.0 - p_ne)
               + 0.5 * p_e * p_ne
               + (1.0 - p_e) * (1.0 - p_ne) * auc_continuous)

    # Observed AUC (full sample)
    auc_observed = auc_mann_whitney(sym_e, sym_ne)

    # Gaussian prediction: Phi(d'/sqrt(2))
    d = dprime(sym_e, sym_ne)
    if np.isfinite(d):
        auc_gaussian = float(stats.norm.cdf(d / np.sqrt(2)))
    else:
        auc_gaussian = np.nan

    return {
        "p_e": p_e,
        "p_ne": p_ne,
        "auc_continuous": float(auc_continuous) if np.isfinite(auc_continuous) else None,
        "auc_mix": float(auc_mix),
        "auc_observed": float(auc_observed) if np.isfinite(auc_observed) else None,
        "auc_gaussian": float(auc_gaussian) if np.isfinite(auc_gaussian) else None,
        "mix_error": float(abs(auc_mix - auc_observed)) if np.isfinite(auc_observed) else None,
        "gaussian_error": (float(abs(auc_gaussian - auc_observed))
                           if np.isfinite(auc_gaussian) and np.isfinite(auc_observed) else None),
        "dprime": float(d) if np.isfinite(d) else None,
    }


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 2: SCORE-LEVEL HERFINDAHL
# ══════════════════════════════════════════════════════════════════════

def herfindahl_scores(sym_all):
    """Herfindahl on sym SCORES (not raw feature values).

    Bin sym scores into 100 bins of width 0.01 in [0,1].
    H_score = sum(p_bin^2).
    """
    if len(sym_all) == 0:
        return np.nan
    # Bin into [0, 0.01), [0.01, 0.02), ..., [0.99, 1.0]
    bin_edges = np.linspace(0, 1, HERFINDAHL_BINS + 1)
    counts, _ = np.histogram(sym_all, bins=bin_edges)
    fracs = counts / len(sym_all)
    H = float(np.sum(fracs ** 2))
    return H


def herfindahl_feature(feat_dict, nodes):
    """Herfindahl on raw feature values (original approach)."""
    vals = np.array([float(feat_dict.get(n, 0)) for n in nodes])
    if len(vals) == 0:
        return np.nan
    unique, counts = np.unique(vals, return_counts=True)
    fracs = counts / len(vals)
    return float(np.sum(fracs ** 2))


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 3: PARAMETER-MATCHED MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════

def model_comparison(rows):
    """Three models predicting AUC for each (network, feature):

    Model G: Gaussian — AUC_pred = Phi(d'/sqrt(2))
    Model R: Restricted mixture — uses cross-network median AUC_continuous
    Model F: Full mixture — exact mixture formula with per-network AUC_continuous

    Returns per-model RMSE and paired Wilcoxon G vs R.
    """
    # Filter rows with all needed fields
    valid = [r for r in rows
             if (r.get("mixture") is not None
                 and r["mixture"].get("auc_observed") is not None
                 and r["mixture"].get("auc_continuous") is not None
                 and r["mixture"].get("dprime") is not None)]

    if len(valid) < 5:
        return None

    observed = np.array([r["mixture"]["auc_observed"] for r in valid])
    pred_gaussian = np.array([r["mixture"]["auc_gaussian"]
                              if r["mixture"]["auc_gaussian"] is not None else np.nan
                              for r in valid])
    pred_full = np.array([r["mixture"]["auc_mix"] for r in valid])

    # Cross-network median of AUC_continuous for restricted model
    auc_conts = [r["mixture"]["auc_continuous"] for r in valid
                 if r["mixture"]["auc_continuous"] is not None]
    if len(auc_conts) < 3:
        return None
    auc_cont_global = float(np.median(auc_conts))

    pred_restricted = np.array([
        (r["mixture"]["p_e"] * (1.0 - r["mixture"]["p_ne"])
         + 0.5 * r["mixture"]["p_e"] * r["mixture"]["p_ne"]
         + (1.0 - r["mixture"]["p_e"]) * (1.0 - r["mixture"]["p_ne"]) * auc_cont_global)
        for r in valid
    ])

    # Per-row squared errors
    mask_g = np.isfinite(pred_gaussian) & np.isfinite(observed)
    mask_r = np.isfinite(pred_restricted) & np.isfinite(observed)
    mask_f = np.isfinite(pred_full) & np.isfinite(observed)

    se_g = (pred_gaussian - observed) ** 2
    se_r = (pred_restricted - observed) ** 2
    se_f = (pred_full - observed) ** 2

    rmse_g = float(np.sqrt(np.mean(se_g[mask_g]))) if mask_g.sum() > 0 else np.nan
    rmse_r = float(np.sqrt(np.mean(se_r[mask_r]))) if mask_r.sum() > 0 else np.nan
    rmse_f = float(np.sqrt(np.mean(se_f[mask_f]))) if mask_f.sum() > 0 else np.nan

    # Paired Wilcoxon: Model G vs Model R (are their errors significantly different?)
    mask_both = mask_g & mask_r
    n_both = int(mask_both.sum())
    if n_both >= 5:
        diffs = se_g[mask_both] - se_r[mask_both]
        # Remove zeros (Wilcoxon requires non-zero differences)
        nonzero = diffs[np.abs(diffs) > 1e-15]
        if len(nonzero) >= 5:
            try:
                w_stat, w_p = stats.wilcoxon(nonzero, alternative="greater")
                wilcoxon_result = {
                    "statistic": float(w_stat),
                    "p_value": float(w_p),
                    "n_nonzero": int(len(nonzero)),
                    "interpretation": ("Gaussian has LARGER errors" if w_p < 0.05
                                       else "No significant difference"),
                }
            except ValueError:
                wilcoxon_result = {"error": "insufficient non-zero differences"}
        else:
            wilcoxon_result = {"error": f"only {len(nonzero)} non-zero differences"}
    else:
        wilcoxon_result = {"error": f"only {n_both} paired observations"}

    return {
        "n_observations": len(valid),
        "auc_continuous_global_median": auc_cont_global,
        "rmse_gaussian": rmse_g,
        "rmse_restricted_mixture": rmse_r,
        "rmse_full_mixture": rmse_f,
        "wilcoxon_gaussian_vs_restricted": wilcoxon_result,
    }


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 4: NETWORK-LEVEL BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════

def bootstrap_spearman_network_level(network_metrics, key_x, key_y, B=BOOTSTRAP_B):
    """Bootstrap CIs for Spearman correlation by resampling NETWORKS.

    network_metrics: list of dicts, one per network, each with key_x and key_y.
    Returns median rho, 95% CI, observed rho, p.
    """
    # Filter valid
    valid = [(m[key_x], m[key_y]) for m in network_metrics
             if (m.get(key_x) is not None and m.get(key_y) is not None
                 and np.isfinite(m[key_x]) and np.isfinite(m[key_y]))]
    if len(valid) < 5:
        return None

    xs = np.array([v[0] for v in valid])
    ys = np.array([v[1] for v in valid])
    n = len(xs)

    observed_rho, observed_p = stats.spearmanr(xs, ys)

    rng = np.random.default_rng(RNG_SEED + 99)
    boot_rhos = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        # Need at least 5 unique values for meaningful Spearman
        if len(np.unique(idx)) < 5:
            continue
        r, _ = stats.spearmanr(xs[idx], ys[idx])
        if np.isfinite(r):
            boot_rhos.append(r)

    if len(boot_rhos) < 50:
        return None

    boot_rhos = np.array(boot_rhos)
    ci_lo = float(np.percentile(boot_rhos, 2.5))
    ci_hi = float(np.percentile(boot_rhos, 97.5))

    return {
        "observed_rho": float(observed_rho),
        "observed_p": float(observed_p),
        "n_networks": n,
        "bootstrap_B": len(boot_rhos),
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "bootstrap_median_rho": float(np.median(boot_rhos)),
    }


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 5: BOOTSTRAP CIs ON FOUR-QUADRANT
# ══════════════════════════════════════════════════════════════════════

def bootstrap_quadrant(sym_kcore_e, sym_kcore_ne, sym_deg_e, sym_deg_ne, rng, B=BOOTSTRAP_B):
    """Bootstrap CIs on tie fraction by resampling edge-nonedge pairs.

    For each bootstrap iteration:
      - Resample edges (with replacement) from sym_kcore_e / sym_deg_e
      - Resample non-edges (with replacement) from sym_kcore_ne / sym_deg_ne
      - Recompute quadrant decomposition

    Returns dict with median tie fraction, 95% CI, count > 100%.
    """
    n_e = len(sym_kcore_e)
    n_ne = len(sym_kcore_ne)
    if n_e < 10 or n_ne < 10:
        return None

    def compute_tie_fraction(sk_e, sk_ne, sd_e, sd_ne):
        """Compute tie fraction of AUC gap for given arrays."""
        # Use random pair sampling if cross-product too large
        max_pairs = 200_000
        actual = len(sk_e) * len(sk_ne)
        if actual > max_pairs:
            n_samp = max_pairs
            ie = rng.integers(0, len(sk_e), n_samp)
            ine = rng.integers(0, len(sk_ne), n_samp)
            ske = sk_e[ie]
            skne = sk_ne[ine]
            sde = sd_e[ie]
            sdne = sd_ne[ine]
        else:
            ske = np.repeat(sk_e, len(sk_ne))
            skne = np.tile(sk_ne, len(sk_e))
            sde = np.repeat(sd_e, len(sk_ne))
            sdne = np.tile(sd_ne, len(sk_e))

        n_total = len(ske)
        kcore_tie_e = np.abs(ske - 1.0) < TIE_TOL
        kcore_tie_ne = np.abs(skne - 1.0) < TIE_TOL

        q1 = np.sum(kcore_tie_e & ~kcore_tie_ne)
        q2 = np.sum(~kcore_tie_e & kcore_tie_ne)
        net_tie_adv = (q1 - q2) / n_total

        kcore_auc = (np.sum(ske > skne) + 0.5 * np.sum(np.abs(ske - skne) < TIE_TOL)) / n_total
        deg_auc = (np.sum(sde > sdne) + 0.5 * np.sum(np.abs(sde - sdne) < TIE_TOL)) / n_total
        gap = kcore_auc - deg_auc

        if abs(gap) > 1e-9:
            return net_tie_adv / gap
        return np.nan

    # Observed
    observed = compute_tie_fraction(sym_kcore_e, sym_kcore_ne, sym_deg_e, sym_deg_ne)

    # Bootstrap
    boot_fracs = []
    for _ in range(B):
        idx_e = rng.choice(n_e, n_e, replace=True)
        idx_ne = rng.choice(n_ne, n_ne, replace=True)
        frac = compute_tie_fraction(
            sym_kcore_e[idx_e], sym_kcore_ne[idx_ne],
            sym_deg_e[idx_e], sym_deg_ne[idx_ne])
        if np.isfinite(frac):
            boot_fracs.append(frac)

    if len(boot_fracs) < 50:
        return None

    boot_fracs = np.array(boot_fracs)
    return {
        "observed_tie_fraction": float(observed) if np.isfinite(observed) else None,
        "bootstrap_median": float(np.median(boot_fracs)),
        "ci_95_lo": float(np.percentile(boot_fracs, 2.5)),
        "ci_95_hi": float(np.percentile(boot_fracs, 97.5)),
        "n_boot": len(boot_fracs),
        "frac_above_100pct": float(np.mean(boot_fracs > 1.0)),
    }


# ══════════════════════════════════════════════════════════════════════
# CORRECTION 6: CROSS-FEATURE GENERALIZATION
# ══════════════════════════════════════════════════════════════════════

def classify_feature_discreteness(fname):
    """Classify features by discreteness for cross-feature test."""
    if fname == "kcore":
        return "discrete"
    elif fname == "clustering":
        return "semi-discrete"  # many nodes share same clustering value
    elif fname in ("degree", "eigenvector", "random"):
        return "continuous"
    return "unknown"


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def run():
    t0 = time.time()
    print("=" * 78)
    print("TIE EFFECT — CORRECTED ANALYSIS (6 CORRECTIONS)")
    print("=" * 78)

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    rng = np.random.default_rng(RNG_SEED)

    # ── Collectors ────────────────────────────────────────────────────
    all_rows = []          # Per (network, feature) — flat
    network_summaries = {} # Per network: aggregated metrics for bootstrap
    quadrant_bootstraps = {} # Per network: bootstrap on tie fraction

    loaded = 0

    for key, net_info in benchmark.items():
        name = net_info["name"]
        try:
            G = load_network_by_name(name)
        except Exception:
            continue
        if G is None:
            continue
        if nx.number_of_selfloops(G) > 0:
            G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_nodes() < 30 or G.number_of_edges() < 30:
            continue

        loaded += 1
        nodes = list(G.nodes())
        features = compute_features(G, rng)
        edges, non_edges = sample_pairs(G, rng)

        print(f"\n[{loaded:2d}] {name} (n={G.number_of_nodes()}, m={G.number_of_edges()}, "
              f"sampled {len(edges)}e/{len(non_edges)}ne)")

        # Pre-compute sym arrays for all features
        sym_arrays = {}
        for fname in FEATURES:
            fvals = features[fname]
            sym_e, sym_ne = compute_sym_arrays(edges, non_edges, fvals)
            sym_arrays[fname] = (sym_e, sym_ne)

        # Track per-network summaries for kcore (for bootstrap)
        net_summary = {"network": name}

        for fname in FEATURES:
            sym_e, sym_ne = sym_arrays[fname]
            if len(sym_e) < 10 or len(sym_ne) < 10:
                continue

            # ── Basic metrics ──
            auc_obs = auc_mann_whitney(sym_e, sym_ne)
            d = dprime(sym_e, sym_ne)

            # Tie enrichment
            tie_e_mask = np.abs(sym_e - 1.0) < TIE_TOL
            tie_ne_mask = np.abs(sym_ne - 1.0) < TIE_TOL
            p_tie_edge = float(np.mean(tie_e_mask))
            p_tie_nonedge = float(np.mean(tie_ne_mask))
            if p_tie_nonedge > 0:
                tie_enrichment = p_tie_edge / p_tie_nonedge
            elif p_tie_edge > 0:
                tie_enrichment = np.inf
            else:
                tie_enrichment = 1.0

            # ── CORRECTION 1: Mixture AUC ──
            mixture = compute_mixture_auc(sym_e, sym_ne)

            # ── CORRECTION 2: Score-level Herfindahl ──
            sym_all = np.concatenate([sym_e, sym_ne])
            h_score = herfindahl_scores(sym_all)
            h_feature = herfindahl_feature(features[fname], nodes)

            # ── CORRECTION 6: Feature classification ──
            discreteness = classify_feature_discreteness(fname)

            row = {
                "network": name,
                "feature": fname,
                "discreteness": discreteness,
                "auc": float(auc_obs) if np.isfinite(auc_obs) else None,
                "dprime": float(d) if np.isfinite(d) else None,
                "tie_rate_edge": p_tie_edge,
                "tie_rate_nonedge": p_tie_nonedge,
                "tie_enrichment": float(tie_enrichment) if np.isfinite(tie_enrichment) else None,
                "herfindahl_score": float(h_score) if np.isfinite(h_score) else None,
                "herfindahl_feature": float(h_feature) if np.isfinite(h_feature) else None,
                "mixture": mixture,
            }
            all_rows.append(row)

            # Print summary line
            mix_err = mixture["mix_error"] if mixture and mixture["mix_error"] is not None else np.nan
            gauss_err = mixture["gaussian_error"] if mixture and mixture["gaussian_error"] is not None else np.nan
            auc_str = f"AUC={auc_obs:.3f}" if np.isfinite(auc_obs) else "AUC=N/A"
            enr_str = (f"enrich={tie_enrichment:.2f}"
                       if np.isfinite(tie_enrichment) else "enrich=N/A")
            print(f"     {fname:12s}  {auc_str}  p_e={p_tie_edge:.3f}  p_ne={p_tie_nonedge:.3f}  "
                  f"{enr_str}  H_score={h_score:.4f}  mix_err={mix_err:.4f}  "
                  f"gauss_err={gauss_err:.4f}")

            # Store kcore metrics in network summary for bootstrap
            if fname == "kcore":
                net_summary["kcore_auc"] = float(auc_obs) if np.isfinite(auc_obs) else None
                net_summary["kcore_dprime"] = float(d) if np.isfinite(d) else None
                net_summary["kcore_tie_enrichment"] = (float(tie_enrichment)
                                                       if np.isfinite(tie_enrichment) else None)

        # ── CORRECTION 5: Bootstrap on four-quadrant ──
        if "kcore" in sym_arrays and "degree" in sym_arrays:
            sk_e, sk_ne = sym_arrays["kcore"]
            sd_e, sd_ne = sym_arrays["degree"]
            min_e = min(len(sk_e), len(sd_e))
            min_ne = min(len(sk_ne), len(sd_ne))
            if min_e >= 10 and min_ne >= 10:
                qb = bootstrap_quadrant(
                    sk_e[:min_e], sk_ne[:min_ne],
                    sd_e[:min_e], sd_ne[:min_ne], rng, B=BOOTSTRAP_B)
                if qb:
                    quadrant_bootstraps[name] = qb
                    print(f"     QUADRANT: tie_frac={qb['observed_tie_fraction']:.1%}"
                          f"  CI=[{qb['ci_95_lo']:.1%}, {qb['ci_95_hi']:.1%}]"
                          f"  P(>100%)={qb['frac_above_100pct']:.1%}"
                          if qb['observed_tie_fraction'] is not None
                          else f"     QUADRANT: tie_frac=N/A")

        network_summaries[name] = net_summary

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("AGGREGATE ANALYSIS")
    print("=" * 78)

    # ── CORRECTION 1 AGGREGATE: Mixture vs Gaussian accuracy ──
    print("\n--- CORRECTION 1: MIXTURE AUC FORMULA ---")
    mix_errors = []
    gauss_errors = []
    for r in all_rows:
        m = r.get("mixture")
        if m and m.get("mix_error") is not None:
            mix_errors.append(m["mix_error"])
        if m and m.get("gaussian_error") is not None:
            gauss_errors.append(m["gaussian_error"])

    if mix_errors:
        print(f"  Mixture prediction RMSE:   {np.sqrt(np.mean(np.array(mix_errors)**2)):.4f}  "
              f"(mean abs error: {np.mean(mix_errors):.4f}, n={len(mix_errors)})")
    if gauss_errors:
        print(f"  Gaussian prediction RMSE:  {np.sqrt(np.mean(np.array(gauss_errors)**2)):.4f}  "
              f"(mean abs error: {np.mean(gauss_errors):.4f}, n={len(gauss_errors)})")
    if mix_errors and gauss_errors:
        # How many have mix_error < 0.01?
        n_close = sum(1 for e in mix_errors if e < 0.01)
        print(f"  Mixture predictions within 0.01 of observed: {n_close}/{len(mix_errors)} "
              f"({n_close/len(mix_errors):.0%})")

    # ── CORRECTION 2 AGGREGATE: Score-level Herfindahl ──
    print("\n--- CORRECTION 2: SCORE-LEVEL HERFINDAHL ---")
    aucs_arr = np.array([r["auc"] if r["auc"] is not None else np.nan for r in all_rows])
    h_scores_arr = np.array([r["herfindahl_score"] if r["herfindahl_score"] is not None else np.nan
                             for r in all_rows])
    h_features_arr = np.array([r["herfindahl_feature"] if r["herfindahl_feature"] is not None
                               else np.nan for r in all_rows])
    tie_enr_arr = np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                            for r in all_rows])

    rho_hs, p_hs, n_hs = safe_spearman(h_scores_arr, aucs_arr)
    rho_hf, p_hf, n_hf = safe_spearman(h_features_arr, aucs_arr)
    rho_hs_te, p_hs_te, n_hs_te = safe_spearman(h_scores_arr, tie_enr_arr)

    print(f"  Spearman(H_score, AUC):           rho={rho_hs:+.3f}, p={p_hs:.2e}, n={n_hs}")
    print(f"  Spearman(H_feature, AUC):         rho={rho_hf:+.3f}, p={p_hf:.2e}, n={n_hf}")
    print(f"  Spearman(H_score, tie_enrichment): rho={rho_hs_te:+.3f}, p={p_hs_te:.2e}, n={n_hs_te}")

    # ── CORRECTION 3: Model comparison ──
    print("\n--- CORRECTION 3: PARAMETER-MATCHED MODEL COMPARISON ---")
    mc = model_comparison(all_rows)
    if mc:
        print(f"  n observations: {mc['n_observations']}")
        print(f"  AUC_continuous global median: {mc['auc_continuous_global_median']:.4f}")
        print(f"  RMSE Model G (Gaussian):            {mc['rmse_gaussian']:.4f}")
        print(f"  RMSE Model R (Restricted mixture):  {mc['rmse_restricted_mixture']:.4f}")
        print(f"  RMSE Model F (Full mixture):        {mc['rmse_full_mixture']:.4f}")
        w = mc["wilcoxon_gaussian_vs_restricted"]
        if "error" not in w:
            print(f"  Wilcoxon (G vs R, greater): W={w['statistic']:.1f}, "
                  f"p={w['p_value']:.4f}, n={w['n_nonzero']}")
            print(f"  Interpretation: {w['interpretation']}")
        else:
            print(f"  Wilcoxon: {w['error']}")
    else:
        print("  Insufficient data for model comparison.")

    # ── CORRECTION 4: Network-level bootstrap ──
    print("\n--- CORRECTION 4: NETWORK-LEVEL BOOTSTRAP ---")
    net_metrics = list(network_summaries.values())

    boot_dprime = bootstrap_spearman_network_level(net_metrics, "kcore_dprime", "kcore_auc")
    boot_tie = bootstrap_spearman_network_level(net_metrics, "kcore_tie_enrichment", "kcore_auc")

    if boot_dprime:
        print(f"  rho(d', AUC) for kcore across networks:")
        print(f"    observed rho={boot_dprime['observed_rho']:+.3f}, "
              f"p={boot_dprime['observed_p']:.2e}, n={boot_dprime['n_networks']}")
        print(f"    bootstrap 95% CI: [{boot_dprime['ci_95_lo']:+.3f}, "
              f"{boot_dprime['ci_95_hi']:+.3f}]")
    else:
        print("  rho(d', AUC): insufficient data for bootstrap")

    if boot_tie:
        print(f"  rho(tie_enrichment, AUC) for kcore across networks:")
        print(f"    observed rho={boot_tie['observed_rho']:+.3f}, "
              f"p={boot_tie['observed_p']:.2e}, n={boot_tie['n_networks']}")
        print(f"    bootstrap 95% CI: [{boot_tie['ci_95_lo']:+.3f}, "
              f"{boot_tie['ci_95_hi']:+.3f}]")
    else:
        print("  rho(tie_enrichment, AUC): insufficient data for bootstrap")

    # ── CORRECTION 5: Bootstrap on four-quadrant ──
    print("\n--- CORRECTION 5: BOOTSTRAP CIs ON FOUR-QUADRANT ---")
    if quadrant_bootstraps:
        obs_fracs = [qb["observed_tie_fraction"] for qb in quadrant_bootstraps.values()
                     if qb["observed_tie_fraction"] is not None]
        medians = [qb["bootstrap_median"] for qb in quadrant_bootstraps.values()]
        above_100 = [qb["frac_above_100pct"] for qb in quadrant_bootstraps.values()]

        # Count networks where tie fraction > 100%
        n_above_100 = sum(1 for f in obs_fracs if f > 1.0)

        print(f"  Networks with bootstrap: {len(quadrant_bootstraps)}")
        if obs_fracs:
            print(f"  Median tie fraction (observed): {np.median(obs_fracs):.1%}")
            print(f"  Mean tie fraction (observed):   {np.mean(obs_fracs):.1%}")
            print(f"  Networks with tie fraction > 100%: {n_above_100}/{len(obs_fracs)}")
        print(f"  Mean P(>100%) across networks: {np.mean(above_100):.1%}")

        # Per-network table
        print(f"\n  {'Network':<35s} {'Obs':>7s} {'Median':>7s} {'CI_lo':>7s} {'CI_hi':>7s} {'P>100%':>7s}")
        print("  " + "-" * 73)
        for name, qb in sorted(quadrant_bootstraps.items()):
            obs = qb["observed_tie_fraction"]
            obs_s = f"{obs:.1%}" if obs is not None else "N/A"
            print(f"  {name:<35s} {obs_s:>7s} {qb['bootstrap_median']:>7.1%} "
                  f"{qb['ci_95_lo']:>7.1%} {qb['ci_95_hi']:>7.1%} {qb['frac_above_100pct']:>7.1%}")
    else:
        print("  No quadrant bootstrap results.")

    # ── CORRECTION 6: Cross-feature generalization ──
    print("\n--- CORRECTION 6: CROSS-FEATURE GENERALIZATION ---")
    print("  Testing whether tie enrichment → AUC works beyond kcore:")
    print()

    # Overall correlation: tie_enrichment vs AUC
    rho_all, p_all, n_all = safe_spearman(tie_enr_arr, aucs_arr)
    print(f"  ALL features: Spearman(tie_enrichment, AUC): "
          f"rho={rho_all:+.3f}, p={p_all:.2e}, n={n_all}")

    # Per-feature type
    for disc_type in ["discrete", "semi-discrete", "continuous"]:
        mask = [r["discreteness"] == disc_type for r in all_rows]
        te_sub = np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                           for r, m in zip(all_rows, mask) if m])
        auc_sub = np.array([r["auc"] if r["auc"] is not None else np.nan
                            for r, m in zip(all_rows, mask) if m])
        fnames_sub = [r["feature"] for r, m in zip(all_rows, mask) if m]
        if len(te_sub) >= 5:
            rho_s, p_s, n_s = safe_spearman(te_sub, auc_sub)
            unique_features = sorted(set(fnames_sub))
            print(f"  {disc_type:15s} ({', '.join(unique_features)}): "
                  f"rho={rho_s:+.3f}, p={p_s:.2e}, n={n_s}")

    # Per-feature breakdown
    print()
    print(f"  {'Feature':<15s} {'rho(TE,AUC)':>12s} {'p-value':>12s} {'n':>5s} "
          f"{'Mean TE':>10s} {'Mean AUC':>10s}")
    print("  " + "-" * 67)
    for fname in FEATURES:
        rows_f = [r for r in all_rows if r["feature"] == fname]
        te_f = np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                         for r in rows_f])
        auc_f = np.array([r["auc"] if r["auc"] is not None else np.nan for r in rows_f])
        rho_f, p_f, n_f = safe_spearman(te_f, auc_f)
        mean_te = np.nanmean(te_f) if np.any(np.isfinite(te_f)) else np.nan
        mean_auc = np.nanmean(auc_f) if np.any(np.isfinite(auc_f)) else np.nan
        rho_s = f"{rho_f:+.3f}" if np.isfinite(rho_f) else "N/A"
        p_s = f"{p_f:.2e}" if np.isfinite(p_f) else "N/A"
        print(f"  {fname:<15s} {rho_s:>12s} {p_s:>12s} {n_f:>5d} "
              f"{mean_te:>10.3f} {mean_auc:>10.3f}")

    # Mean tie enrichment by feature
    print()
    print(f"  Mean tie rates by feature:")
    for fname in FEATURES:
        rows_f = [r for r in all_rows if r["feature"] == fname]
        te_vals = [r["tie_rate_edge"] for r in rows_f if r["tie_rate_edge"] is not None]
        tne_vals = [r["tie_rate_nonedge"] for r in rows_f if r["tie_rate_nonedge"] is not None]
        enr_vals = [r["tie_enrichment"] for r in rows_f
                    if r["tie_enrichment"] is not None and np.isfinite(r["tie_enrichment"])]
        if te_vals and tne_vals and enr_vals:
            print(f"    {fname:12s}  P(sym=1|edge)={np.mean(te_vals):.4f}  "
                  f"P(sym=1|nonedge)={np.mean(tne_vals):.4f}  "
                  f"enrichment={np.median(enr_vals):.3f} (median)")

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)

    dprimes_arr = np.array([r["dprime"] if r["dprime"] is not None else np.nan for r in all_rows])
    rho_dp, p_dp, n_dp = safe_spearman(dprimes_arr, aucs_arr)

    summary_rows = [
        ("tie_enrichment vs AUC", rho_all, p_all, n_all),
        ("d' vs AUC", rho_dp, p_dp, n_dp),
        ("H_score vs AUC", rho_hs, p_hs, n_hs),
        ("H_feature vs AUC", rho_hf, p_hf, n_hf),
        ("H_score vs tie_enrich", rho_hs_te, p_hs_te, n_hs_te),
    ]

    def verdict_str(rho, p):
        if np.isnan(rho) or np.isnan(p):
            return "INSUFFICIENT DATA"
        if p < 0.01:
            return "CONFIRMED"
        elif p < 0.05:
            return "SUPPORTED"
        elif p < 0.10:
            return "TREND"
        else:
            return "NOT SIGNIFICANT"

    print(f"{'Metric':<25s} {'rho':>8s} {'p-value':>12s} {'n':>5s} {'Verdict':<20s}")
    print("-" * 75)
    for label, rho, p, n in summary_rows:
        rho_s = f"{rho:+.3f}" if np.isfinite(rho) else "N/A"
        p_s = f"{p:.2e}" if np.isfinite(p) else "N/A"
        v = verdict_str(rho, p)
        print(f"{label:<25s} {rho_s:>8s} {p_s:>12s} {n:>5d} {v:<20s}")

    if mc:
        print(f"\nModel comparison (RMSE):  Gaussian={mc['rmse_gaussian']:.4f}  "
              f"Restricted={mc['rmse_restricted_mixture']:.4f}  "
              f"Full={mc['rmse_full_mixture']:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "metadata": {
            "description": "Corrected tie effect analysis with 6 corrections",
            "corrections": [
                "C1: Mixture AUC formula (replaces invalid d'_tie + d'_continuous)",
                "C2: Score-level Herfindahl (on sym scores, not raw features)",
                "C3: Parameter-matched model comparison (Gaussian vs mixture)",
                "C4: Network-level bootstrap (resample networks, not rows)",
                "C5: Bootstrap CIs on four-quadrant tie fraction",
                "C6: Cross-feature generalization (discrete vs continuous)",
            ],
            "n_networks_loaded": loaded,
            "max_edges": MAX_EDGES,
            "max_nonedges": MAX_NONEDGES,
            "tie_tolerance": TIE_TOL,
            "rng_seed": RNG_SEED,
            "bootstrap_B": BOOTSTRAP_B,
            "herfindahl_bins": HERFINDAHL_BINS,
            "runtime_sec": round(time.time() - t0, 1),
        },
        "per_network_feature": all_rows,
        "correction_1_mixture_auc": {
            "rmse_mixture": (float(np.sqrt(np.mean(np.array(mix_errors)**2)))
                             if mix_errors else None),
            "rmse_gaussian": (float(np.sqrt(np.mean(np.array(gauss_errors)**2)))
                              if gauss_errors else None),
            "n_within_0.01": sum(1 for e in mix_errors if e < 0.01) if mix_errors else 0,
            "n_total": len(mix_errors),
        },
        "correction_2_herfindahl": {
            "h_score_vs_auc": {"rho": rho_hs, "p": p_hs, "n": n_hs},
            "h_feature_vs_auc": {"rho": rho_hf, "p": p_hf, "n": n_hf},
            "h_score_vs_tie_enrichment": {"rho": rho_hs_te, "p": p_hs_te, "n": n_hs_te},
        },
        "correction_3_model_comparison": mc,
        "correction_4_bootstrap": {
            "dprime_vs_auc": boot_dprime,
            "tie_enrichment_vs_auc": boot_tie,
        },
        "correction_5_quadrant_bootstrap": quadrant_bootstraps,
        "correction_6_cross_feature": {
            "all_features": {"rho": rho_all, "p": p_all, "n": n_all},
            "per_feature": {
                fname: {
                    "rho": safe_spearman(
                        np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]),
                        np.array([r["auc"] if r["auc"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]))[0],
                    "p": safe_spearman(
                        np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]),
                        np.array([r["auc"] if r["auc"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]))[1],
                    "n": safe_spearman(
                        np.array([r["tie_enrichment"] if r["tie_enrichment"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]),
                        np.array([r["auc"] if r["auc"] is not None else np.nan
                                  for r in all_rows if r["feature"] == fname]))[2],
                    "discreteness": classify_feature_discreteness(fname),
                }
                for fname in FEATURES
            },
        },
        "summary": [
            {"metric": label, "rho": float(rho) if np.isfinite(rho) else None,
             "p": float(p) if np.isfinite(p) else None,
             "n": n, "verdict": verdict_str(rho, p)}
            for label, rho, p, n in summary_rows
        ],
    }

    out_path = RESULTS_DIR / "tie_effect_corrected.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print(f"Runtime: {time.time() - t0:.1f}s")
    print(f"Networks loaded: {loaded}")


if __name__ == "__main__":
    run()
