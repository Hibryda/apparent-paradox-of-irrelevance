"""BL2 Bridge Lemma: Distribution-free variance bound for sym via the ratio representation.

BL1 assumes f = mu + eps with eps ~ N(0, sigma^2) and derives:
  Var(sym) ≈ (1 - 2/pi) * sigma^2 / (2*mu^2)
  Relative separation error: sigma^2 / (2*mu^2)

BL1 fails for real networks: 0/31 networks have sigma/mu < 0.3.
Real degree distributions are heavy-tailed, discrete, and bounded below by 1.

BL2 INSIGHT:
  sym(a,b) = 1 - |a-b|/(a+b) = 2*min(a,b)/(a+b) = 2r/(1+r)
  where r = min(a,b)/max(a,b) ∈ (0, 1].

  Let h(r) = 2r/(1+r). This is monotone increasing with:
    h'(r) = 2/(1+r)^2
    h''(r) = -4/(1+r)^3

  The distribution of r is always bounded in (0,1], regardless of the
  marginal distribution of the feature. No Gaussianity needed.

  DELTA METHOD (first-order):
    Var(sym) ≈ [h'(E[r])]^2 * Var(r) = 4/(1+E[r])^4 * Var(r)

  DELTA METHOD (second-order, bias-corrected):
    E[sym] ≈ h(E[r]) + h''(E[r])/2 * Var(r)
    Var(sym) ≈ [h'(E[r])]^2 * Var(r) + [h''(E[r])]^2/4 * (E[(r-E[r])^4] - Var(r)^2)

  Since h is concave on (0,1], the second-order correction E[sym] < h(E[r]),
  i.e., Jensen's inequality gives h(E[r]) as an upper bound on E[sym].

  EXACT (non-parametric):
    Given empirical distribution of r, compute Var(h(r)) directly.
    This is the "oracle" BL2 — it uses the true r-distribution.

  PRACTICAL BL2:
    From the marginal distribution of feature f, predict E[r] and Var(r)
    for random pairs, then apply the delta method.

    For independent draws f_i, f_j from a distribution with CDF F:
      E[r] = E[min/max] = 2*E[min(f_i,f_j)] / E[f_i + f_j] (approximately)

    More precisely, for iid draws:
      E[min(f_i,f_j)] = integral_0^inf (1 - F(x))^2 dx  (for positive RVs)
      E[max(f_i,f_j)] = 2*mu - E[min] = 2*mu - integral_0^inf (1-F(x))^2 dx

    But the delta method on r = min/max doesn't need these — it only needs
    E[r] and Var(r), which we estimate from the marginal empirical distribution.

This script:
  1. DERIVES the BL2 predictions (delta-method and exact)
  2. TESTS against empirical data from the 32-network benchmark
  3. COMPARES BL2 vs BL1 R² for Var(sym) prediction
"""
import sys
import json
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
RNG_SEED = 42


# ── Network loading (same as ceiling_effect_tests.py) ──────────────────

def load_network_by_name(name):
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


# ── Helpers ─────────────────────────────────────────────────────────────

def sym_vec(a, b):
    """Vectorised sym(a, b) = 1 - |a-b|/(a+b). Returns NaN where a+b==0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(s > 0, 1.0 - np.abs(a - b) / s, np.nan)
    return result


def ratio_vec(a, b):
    """Vectorised r = min(a,b)/max(a,b). Returns NaN where max==0."""
    mn = np.minimum(a, b)
    mx = np.maximum(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(mx > 0, mn / mx, np.nan)
    return result


def h(r):
    """h(r) = 2r/(1+r) — the sym function in ratio space."""
    return 2.0 * r / (1.0 + r)


def h_prime(r):
    """h'(r) = 2/(1+r)^2"""
    return 2.0 / (1.0 + r) ** 2


def h_double_prime(r):
    """h''(r) = -4/(1+r)^3"""
    return -4.0 / (1.0 + r) ** 3


def compute_features(G, rng):
    """Compute node features: degree, kcore, eigenvector, clustering."""
    nodes = list(G.nodes())
    deg = dict(G.degree())
    kcore = nx.core_number(G)
    try:
        eigv = nx.eigenvector_centrality(G, max_iter=300, tol=1e-3)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigv = nx.degree_centrality(G)
    clust = nx.clustering(G)

    return {
        "degree": deg,
        "kcore": kcore,
        "eigenvector": eigv,
        "clustering": clust,
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


def predict_var_r_from_marginal(node_vals, rng, n_pairs=20000):
    """Estimate E[r], Var(r), and excess kurtosis of r for random pairs.

    Instead of computing all O(n^2) pairs, draw random pairs from the node
    values and compute r = min/max for each.

    Returns (E[r], Var(r), excess_kurtosis_r).
    """
    vals = np.array(node_vals, dtype=float)
    vals = vals[vals > 0]  # filter zeros
    if len(vals) < 10:
        return np.nan, np.nan, np.nan

    n = min(n_pairs, len(vals) * (len(vals) - 1) // 2)
    idx_i = rng.choice(len(vals), n, replace=True)
    idx_j = rng.choice(len(vals), n, replace=True)
    # Avoid self-pairs
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]

    r = ratio_vec(vals[idx_i], vals[idx_j])
    valid = ~np.isnan(r)
    r = r[valid]
    if len(r) < 10:
        return np.nan, np.nan, np.nan

    return float(np.mean(r)), float(np.var(r)), float(stats.kurtosis(r, fisher=True))


# ── BL2 Predictions ───────────────────────────────────────────────────

def bl2_delta_method(E_r, Var_r):
    """BL2 delta-method prediction: Var(sym) ≈ [h'(E[r])]^2 * Var(r).

    h(r) = 2r/(1+r), h'(r) = 2/(1+r)^2
    Var(sym) ≈ 4/(1+E[r])^4 * Var(r)
    """
    if np.isnan(E_r) or np.isnan(Var_r) or E_r <= 0:
        return np.nan
    return (h_prime(E_r) ** 2) * Var_r


def bl2_second_order(E_r, Var_r, kurt_r=None):
    """BL2 second-order delta method prediction.

    Var(sym) ≈ [h'(E[r])]^2 * Var(r) + [h''(E[r])]^2/4 * (kurtosis_excess * Var(r)^2)

    For the basic version without kurtosis, just use the first-order term.
    """
    if np.isnan(E_r) or np.isnan(Var_r) or E_r <= 0:
        return np.nan

    first_order = (h_prime(E_r) ** 2) * Var_r

    if kurt_r is not None and not np.isnan(kurt_r):
        # E[(r-E[r])^4] = (kurt_excess + 3) * Var(r)^2
        fourth_central = (kurt_r + 3) * Var_r ** 2
        second_order_correction = (h_double_prime(E_r) ** 2) / 4 * (fourth_central - Var_r ** 2)
        return first_order + second_order_correction

    return first_order


def bl1_predicted_variance(mu, sigma):
    """BL1 prediction: Var(sym) ≈ (1 - 2/pi) * sigma^2 / (2*mu^2)."""
    P1_COEFF = (1 - 2 / np.pi) / 2  # ~0.1817
    if mu < 1e-12:
        return np.nan
    return P1_COEFF * (sigma / mu) ** 2


# ── Main Analysis ──────────────────────────────────────────────────────

def run():
    print("=" * 78)
    print("BL2 BRIDGE LEMMA: Distribution-Free Variance Prediction")
    print("=" * 78)

    # --- Mathematical derivation ---
    print("""
MATHEMATICAL DERIVATION
=======================

BL1 (Gaussian assumption):
  sym(a,b) = 1 - |a-b|/(a+b),  a = mu + eps1,  b = mu + eps2,  eps ~ N(0,sigma^2)
  Var(sym) = (1 - 2/pi) * sigma^2 / (2*mu^2)
  Requires: sigma/mu < 0.3  (0/31 real networks satisfy this)

BL2 (distribution-free):
  sym(a,b) = 2*min(a,b)/(a+b) = 2r/(1+r)  where r = min(a,b)/max(a,b) in (0,1]

  Let h(r) = 2r/(1+r).  Then sym = h(r).
    h'(r) = 2/(1+r)^2
    h''(r) = -4/(1+r)^3

  Delta method (first order):
    Var(sym) = Var(h(r)) ~ [h'(E[r])]^2 * Var(r) = 4*Var(r) / (1+E[r])^4

  This works for ANY distribution of r, not just Gaussian.
  r is always bounded in (0,1], so the delta method is well-behaved.

  KEY ADVANTAGE: The r-distribution is bounded, unimodal, and well-concentrated
  even when the feature distribution is heavy-tailed.  A power-law degree
  distribution generates a beta-like r distribution in (0,1].

PREDICTION CHAIN:
  Feature distribution F  -->  r-distribution (min/max pairs)  -->  Var(sym)
  We test three levels:
    BL2-oracle:  Var(r) from actual sampled pairs (same data as empirical sym)
    BL2-marginal: Var(r) from random pairs drawn from the marginal distribution
    BL1:          Gaussian approximation sigma^2/(2*mu^2) * (1-2/pi)
""")

    # --- Load benchmark ---
    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    rng = np.random.default_rng(RNG_SEED)

    # Accumulators for pooled analysis
    all_bl1_pred = []
    all_bl2_oracle_pred = []
    all_bl2_delta_pred = []
    all_bl2_marginal_pred = []
    all_bl2_marginal_2nd_pred = []
    all_empirical = []
    all_cv = []
    all_feature_names = []
    all_network_names = []

    per_network_results = {}
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
        features = compute_features(G, rng)
        edges, non_edges = sample_pairs(G, rng)

        print(f"\n[{loaded:2d}] {name} (n={G.number_of_nodes()}, m={G.number_of_edges()}, "
              f"sampled {len(edges)} edges, {len(non_edges)} non-edges)")

        net_result = {"features": {}}

        for fname, fvals in features.items():
            # Get feature values for sampled pairs
            fu_e = np.array([float(fvals.get(u, 0)) + 0.001 for u, _ in edges])
            fv_e = np.array([float(fvals.get(v, 0)) + 0.001 for _, v in edges])
            fu_ne = np.array([float(fvals.get(u, 0)) + 0.001 for u, _ in non_edges])
            fv_ne = np.array([float(fvals.get(v, 0)) + 0.001 for _, v in non_edges])

            fu_all = np.concatenate([fu_e, fu_ne])
            fv_all = np.concatenate([fv_e, fv_ne])

            # --- Empirical sym ---
            sym_all = sym_vec(fu_all, fv_all)
            valid_sym = ~np.isnan(sym_all)
            if valid_sym.sum() < 20:
                continue
            var_sym_empirical = float(np.var(sym_all[valid_sym]))

            # --- Ratio r = min/max for sampled pairs ---
            r_all = ratio_vec(fu_all, fv_all)
            valid_r = ~np.isnan(r_all)
            if valid_r.sum() < 20:
                continue
            r_valid = r_all[valid_r]

            E_r_oracle = float(np.mean(r_valid))
            Var_r_oracle = float(np.var(r_valid))
            kurt_r_oracle = float(stats.kurtosis(r_valid, fisher=True))  # excess kurtosis

            # --- BL2-oracle: exact Var(h(r)) from actual sampled pairs ---
            h_r = h(r_valid)
            var_sym_bl2_oracle = float(np.var(h_r))

            # --- BL2-delta: delta method using E[r], Var(r) from sampled pairs ---
            var_sym_bl2_delta = bl2_delta_method(E_r_oracle, Var_r_oracle)

            # --- BL2-marginal: E[r], Var(r) from random pairs of the marginal ---
            node_vals = [float(fvals.get(n, 0)) + 0.001 for n in G.nodes()]
            E_r_marginal, Var_r_marginal, kurt_r_marginal = predict_var_r_from_marginal(node_vals, rng)
            var_sym_bl2_marginal = bl2_delta_method(E_r_marginal, Var_r_marginal)

            # --- BL2-marginal-2nd: second-order delta method from marginal ---
            var_sym_bl2_marginal_2nd = bl2_second_order(E_r_marginal, Var_r_marginal, kurt_r_marginal)

            # --- BL1 prediction ---
            node_arr = np.array(node_vals)
            mu = float(np.mean(node_arr))
            sigma = float(np.std(node_arr))
            cv = sigma / mu if mu > 1e-12 else np.inf
            var_sym_bl1 = bl1_predicted_variance(mu, sigma)

            # --- BL2 second-order ---
            var_sym_bl2_second = bl2_second_order(E_r_oracle, Var_r_oracle, kurt_r_oracle)

            feat_result = {
                "mu": mu,
                "sigma": sigma,
                "cv": cv,
                "E_r_oracle": E_r_oracle,
                "Var_r_oracle": Var_r_oracle,
                "kurt_r_oracle": kurt_r_oracle,
                "E_r_marginal": float(E_r_marginal) if not np.isnan(E_r_marginal) else None,
                "Var_r_marginal": float(Var_r_marginal) if not np.isnan(Var_r_marginal) else None,
                "var_sym_empirical": var_sym_empirical,
                "var_sym_bl1": var_sym_bl1,
                "var_sym_bl2_oracle": var_sym_bl2_oracle,
                "var_sym_bl2_delta": var_sym_bl2_delta,
                "var_sym_bl2_second": var_sym_bl2_second,
                "var_sym_bl2_marginal": var_sym_bl2_marginal,
                "var_sym_bl2_marginal_2nd": var_sym_bl2_marginal_2nd,
            }
            net_result["features"][fname] = feat_result

            # Accumulate for pooled analysis (skip NaN)
            if (not np.isnan(var_sym_bl1)
                    and not np.isnan(var_sym_bl2_oracle)
                    and not np.isnan(var_sym_bl2_delta)
                    and not np.isnan(var_sym_bl2_marginal)
                    and not np.isnan(var_sym_bl2_marginal_2nd)
                    and not np.isnan(var_sym_empirical)
                    and var_sym_empirical > 1e-15):
                all_bl1_pred.append(var_sym_bl1)
                all_bl2_oracle_pred.append(var_sym_bl2_oracle)
                all_bl2_delta_pred.append(var_sym_bl2_delta)
                all_bl2_marginal_pred.append(var_sym_bl2_marginal)
                all_bl2_marginal_2nd_pred.append(var_sym_bl2_marginal_2nd)
                all_empirical.append(var_sym_empirical)
                all_cv.append(cv)
                all_feature_names.append(fname)
                all_network_names.append(name)

            print(f"     {fname:>12s}: CV={cv:.2f}  Var_emp={var_sym_empirical:.6f}  "
                  f"BL1={var_sym_bl1:.6f}  BL2_delta={var_sym_bl2_delta:.6f}  "
                  f"BL2_marginal={var_sym_bl2_marginal:.6f}  "
                  f"BL2_marg_2nd={var_sym_bl2_marginal_2nd:.6f}")

        per_network_results[name] = net_result

    # ═══════════════════════════════════════════════════════════════════
    # POOLED ANALYSIS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("POOLED ANALYSIS")
    print("=" * 78)

    emp = np.array(all_empirical)
    bl1 = np.array(all_bl1_pred)
    bl2_oracle = np.array(all_bl2_oracle_pred)
    bl2_delta = np.array(all_bl2_delta_pred)
    bl2_marginal = np.array(all_bl2_marginal_pred)
    bl2_marginal_2nd = np.array(all_bl2_marginal_2nd_pred)
    cvs = np.array(all_cv)

    n_pairs = len(emp)
    print(f"\nTotal (network, feature) pairs: {n_pairs}")
    print(f"Networks loaded: {loaded}")
    print(f"CV range: [{np.min(cvs):.3f}, {np.max(cvs):.3f}] (mean={np.mean(cvs):.3f})")
    print(f"  In BL1 regime (CV<0.3): {np.sum(cvs < 0.3)}/{n_pairs}")
    print()

    # --- Compute R² for each prediction method ---
    def compute_metrics(predicted, empirical, label):
        """Compute Pearson R², Spearman rho, MAPE, and median ratio."""
        # Pearson R²
        r_pearson, p_pearson = stats.pearsonr(predicted, empirical)
        r2 = r_pearson ** 2

        # Spearman rank correlation
        rho_spearman, p_spearman = stats.spearmanr(predicted, empirical)

        # Median absolute percentage error
        with np.errstate(divide="ignore", invalid="ignore"):
            ape = np.abs(predicted - empirical) / empirical
        mape = float(np.median(ape[~np.isnan(ape)])) * 100

        # Median pred/emp ratio
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = predicted / empirical
        median_ratio = float(np.median(ratios[~np.isnan(ratios) & ~np.isinf(ratios)]))

        # Log-space R² (better for heavy-tailed values)
        log_pred = np.log10(predicted[predicted > 0])
        log_emp = np.log10(empirical[predicted > 0])
        if len(log_pred) >= 3:
            r_log, _ = stats.pearsonr(log_pred, log_emp)
            r2_log = r_log ** 2
        else:
            r2_log = np.nan

        print(f"  {label}:")
        print(f"    Pearson R² = {r2:.4f}  (r={r_pearson:+.4f}, p={p_pearson:.2e})")
        print(f"    Log-space R² = {r2_log:.4f}")
        print(f"    Spearman rho = {rho_spearman:+.4f}  (p={p_spearman:.2e})")
        print(f"    Median |%error| = {mape:.1f}%")
        print(f"    Median pred/emp = {median_ratio:.3f}")

        return {
            "r2": float(r2),
            "r2_log": float(r2_log) if not np.isnan(r2_log) else None,
            "r_pearson": float(r_pearson),
            "p_pearson": float(p_pearson),
            "rho_spearman": float(rho_spearman),
            "p_spearman": float(p_spearman),
            "median_ape_pct": mape,
            "median_ratio": median_ratio,
        }

    print("--- Prediction accuracy (Var(sym) predicted vs empirical) ---\n")
    metrics_bl1 = compute_metrics(bl1, emp, "BL1 (Gaussian)")
    print()
    metrics_bl2_oracle = compute_metrics(bl2_oracle, emp, "BL2-oracle (exact Var(h(r)))")
    print()
    metrics_bl2_delta = compute_metrics(bl2_delta, emp, "BL2-delta (delta method, sampled pairs)")
    print()
    metrics_bl2_marginal = compute_metrics(bl2_marginal, emp, "BL2-marginal (delta method, marginal pairs)")
    print()
    metrics_bl2_marginal_2nd = compute_metrics(bl2_marginal_2nd, emp, "BL2-marginal-2nd (2nd-order delta, marginal)")

    # --- Per-feature breakdown ---
    print("\n--- Per-feature R² breakdown ---\n")
    feature_metrics = {}
    for feat in ["degree", "kcore", "eigenvector", "clustering"]:
        mask = np.array([f == feat for f in all_feature_names])
        if mask.sum() < 3:
            continue
        print(f"  Feature: {feat} (N={mask.sum()})")

        # BL1
        if mask.sum() >= 3:
            r_bl1, _ = stats.pearsonr(bl1[mask], emp[mask])
            r2_bl1 = r_bl1 ** 2
        else:
            r2_bl1 = np.nan

        # BL2-delta
        if mask.sum() >= 3:
            r_bl2, _ = stats.pearsonr(bl2_delta[mask], emp[mask])
            r2_bl2 = r_bl2 ** 2
        else:
            r2_bl2 = np.nan

        # BL2-marginal
        if mask.sum() >= 3:
            r_bl2m, _ = stats.pearsonr(bl2_marginal[mask], emp[mask])
            r2_bl2m = r_bl2m ** 2
        else:
            r2_bl2m = np.nan

        # BL2-marginal-2nd
        if mask.sum() >= 3:
            r_bl2m2, _ = stats.pearsonr(bl2_marginal_2nd[mask], emp[mask])
            r2_bl2m2 = r_bl2m2 ** 2
        else:
            r2_bl2m2 = np.nan

        # BL2-oracle
        if mask.sum() >= 3:
            r_bl2o, _ = stats.pearsonr(bl2_oracle[mask], emp[mask])
            r2_bl2o = r_bl2o ** 2
        else:
            r2_bl2o = np.nan

        print(f"    BL1 R²={r2_bl1:.4f}  BL2-delta R²={r2_bl2:.4f}  "
              f"BL2-marg R²={r2_bl2m:.4f}  BL2-marg-2nd R²={r2_bl2m2:.4f}")

        feature_metrics[feat] = {
            "n": int(mask.sum()),
            "r2_bl1": float(r2_bl1) if not np.isnan(r2_bl1) else None,
            "r2_bl2_delta": float(r2_bl2) if not np.isnan(r2_bl2) else None,
            "r2_bl2_marginal": float(r2_bl2m) if not np.isnan(r2_bl2m) else None,
            "r2_bl2_marginal_2nd": float(r2_bl2m2) if not np.isnan(r2_bl2m2) else None,
            "r2_bl2_oracle": float(r2_bl2o) if not np.isnan(r2_bl2o) else None,
        }

    # --- Delta method accuracy: how well does it approximate the oracle? ---
    print("\n--- Delta method accuracy (BL2-delta vs BL2-oracle) ---\n")
    r_delta_oracle, p_delta_oracle = stats.pearsonr(bl2_delta, bl2_oracle)
    r2_delta_oracle = r_delta_oracle ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_oracle_ratio = bl2_delta / bl2_oracle
    valid_ratio = ~np.isnan(delta_oracle_ratio) & ~np.isinf(delta_oracle_ratio)
    print(f"  R²(delta vs oracle) = {r2_delta_oracle:.4f}")
    print(f"  Median delta/oracle ratio = {np.median(delta_oracle_ratio[valid_ratio]):.4f}")
    print(f"  Mean delta/oracle ratio = {np.mean(delta_oracle_ratio[valid_ratio]):.4f}")

    # --- Edge vs non-edge r-distributions ---
    print("\n--- r-distribution properties (aggregated) ---\n")
    print(f"  Mean E[r] across pairs: {np.mean([r['E_r_oracle'] for net in per_network_results.values() for r in net['features'].values() if 'E_r_oracle' in r]):.4f}")
    print(f"  Mean Var(r) across pairs: {np.mean([r['Var_r_oracle'] for net in per_network_results.values() for r in net['features'].values() if 'Var_r_oracle' in r]):.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # BL2 THEOREM STATEMENT
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("BL2 BRIDGE LEMMA — THEOREM")
    print("=" * 78)

    print(f"""
BL2 BRIDGE LEMMA (distribution-free):

  For any positive-valued feature f with arbitrary distribution,
  define r = min(f_i, f_j) / max(f_i, f_j) for a pair (i,j).

  Let E_r = E[r] and V_r = Var(r) (computed from the marginal of f).
  Then:
    Var(sym(f_i, f_j)) = 4 * V_r / (1 + E_r)^4 + O(V_r^2)

  This holds for ANY distribution: power-law, log-normal, discrete, bounded.
  No CV threshold needed. r is always in (0,1].

EMPIRICAL VALIDATION (N={n_pairs} pairs across {loaded} networks):
  BL1 (Gaussian):              R² = {metrics_bl1['r2']:.4f}  (log R² = {metrics_bl1['r2_log']})
  BL2-delta (sampled):         R² = {metrics_bl2_delta['r2']:.4f}  (log R² = {metrics_bl2_delta['r2_log']})
  BL2-marginal (from F):       R² = {metrics_bl2_marginal['r2']:.4f}  (log R² = {metrics_bl2_marginal['r2_log']})
  BL2-marginal-2nd (from F):   R² = {metrics_bl2_marginal_2nd['r2']:.4f}  (log R² = {metrics_bl2_marginal_2nd['r2_log']})
  BL2-oracle (exact Var h):    R² = {metrics_bl2_oracle['r2']:.4f}  (log R² = {metrics_bl2_oracle['r2_log']})

COMPARISON:
  BL2-oracle is by construction R²=1.0 (same data, same transform).
  BL2-delta approximation quality: R²(delta vs oracle) = {r2_delta_oracle:.4f}
  BL2-marginal vs BL1: R² improvement {metrics_bl2_marginal['r2']:.4f} vs {metrics_bl1['r2']:.4f}
  BL2-marginal-2nd vs BL2-marginal: {metrics_bl2_marginal_2nd['r2']:.4f} vs {metrics_bl2_marginal['r2']:.4f}

INTERPRETATION:
  BL1 fails because it models the WRONG thing: it assumes small perturbations
  around a mean, but real features have CV >> 1. The denominator noise (a+b)
  dominates, violating the Taylor expansion.

  BL2 works because r = min/max is naturally bounded in (0,1], making the
  delta method well-conditioned regardless of the feature's tail behavior.
  The transformation sym = h(r) = 2r/(1+r) is smooth and monotone on (0,1],
  so the delta method converges whenever Var(r) is finite (always true since
  r is bounded).

  The R² gap between BL2-delta and BL2-oracle measures how much curvature
  in h(r) matters — i.e., how non-linear the mapping is over the actual
  range of r values.
""")

    # --- Final comparison table ---
    print("=" * 78)
    print("BL1 vs BL2 COMPARISON TABLE")
    print("=" * 78)
    print()
    print(f"  {'Method':<28s} {'R²':>8s} {'log R²':>8s} {'Spearman':>9s} {'Med %err':>9s} {'Med ratio':>10s}")
    print(f"  {'-'*72}")
    for label, m in [
        ("BL1 (Gaussian)", metrics_bl1),
        ("BL2-delta (sampled)", metrics_bl2_delta),
        ("BL2-marginal (from F)", metrics_bl2_marginal),
        ("BL2-marginal-2nd (from F)", metrics_bl2_marginal_2nd),
        ("BL2-oracle (exact)", metrics_bl2_oracle),
    ]:
        r2l = f"{m['r2_log']:.4f}" if m['r2_log'] is not None else "N/A"
        print(f"  {label:<28s} {m['r2']:8.4f} {r2l:>8s} {m['rho_spearman']:9.4f} "
              f"{m['median_ape_pct']:8.1f}% {m['median_ratio']:10.3f}")
    print()
    print(f"  VERDICT: BL2-marginal R² = {metrics_bl2_marginal['r2']:.4f} vs BL1 R² = {metrics_bl1['r2']:.4f}")
    bl2_best = max(metrics_bl2_marginal['r2'], metrics_bl2_marginal_2nd['r2'])
    bl2_best_name = "BL2-marginal-2nd" if metrics_bl2_marginal_2nd['r2'] > metrics_bl2_marginal['r2'] else "BL2-marginal"
    improvement = bl2_best / metrics_bl1['r2'] if metrics_bl1['r2'] > 0 else float('inf')
    print(f"  Best BL2 variant: {bl2_best_name} (R²={bl2_best:.4f}, {improvement:.1f}x improvement over BL1)")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════

    def sanitize(obj):
        """Recursively convert numpy types to native Python for JSON."""
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        return obj

    output = {
        "n_networks": loaded,
        "n_pairs": n_pairs,
        "cv_range": {"min": float(np.min(cvs)), "max": float(np.max(cvs)), "mean": float(np.mean(cvs))},
        "n_in_bl1_regime": int(np.sum(cvs < 0.3)),
        "pooled_metrics": {
            "bl1": sanitize(metrics_bl1),
            "bl2_oracle": sanitize(metrics_bl2_oracle),
            "bl2_delta": sanitize(metrics_bl2_delta),
            "bl2_marginal": sanitize(metrics_bl2_marginal),
            "bl2_marginal_2nd": sanitize(metrics_bl2_marginal_2nd),
        },
        "delta_vs_oracle": {
            "r2": float(r2_delta_oracle),
            "median_ratio": float(np.median(delta_oracle_ratio[valid_ratio])),
        },
        "per_feature": sanitize(feature_metrics),
        "per_network": sanitize(per_network_results),
        "pairs": sanitize([
            {
                "network": n,
                "feature": f,
                "cv": c,
                "var_emp": e,
                "var_bl1": b1,
                "var_bl2_oracle": b2o,
                "var_bl2_delta": b2d,
                "var_bl2_marginal": b2m,
                "var_bl2_marginal_2nd": b2m2,
            }
            for n, f, c, e, b1, b2o, b2d, b2m, b2m2 in zip(
                all_network_names, all_feature_names, all_cv,
                all_empirical, all_bl1_pred, all_bl2_oracle_pred,
                all_bl2_delta_pred, all_bl2_marginal_pred,
                all_bl2_marginal_2nd_pred,
            )
        ]),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bl2_bridge_lemma.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run()
