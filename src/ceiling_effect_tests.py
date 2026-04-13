"""Pre-registered statistical test suite for the ceiling effect.

Tests P1 (variance formula), T8 (separation shrinkage), and BL1 (bridge lemma)
across all benchmark networks using five independent tests:

  1. WITHIN-NETWORK DEGREE-BIN TEST — bin pairs by min(deg), test Var/Sep vs bin
  2. CROSS-FEATURE TEST — CV predicts sym variance across features within each network
  3. PREDICTED VARIANCE TEST — P1 formula vs empirical Var(sym)
  4. BL1 RELATIVE ERROR VERIFICATION — separation approximation error bounded

Verdicts: CONFIRMED (p<0.01), SUPPORTED (p<0.05), TREND (p<0.10),
          NOT CONFIRMED (p>=0.10), REFUTED (significant in wrong direction).
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
N_BINS = 5
RNG_SEED = 42


# ── Helpers ──────────────────────────────────────────────────────────

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


def sym_vec(a, b):
    """Vectorised sym(a, b) = 1 - |a-b|/(a+b). Returns NaN where a+b==0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(s > 0, 1.0 - np.abs(a - b) / s, np.nan)
    return result


def auc_from_scores(pos_scores, neg_scores, rng, max_compare=5000):
    """Mann-Whitney AUC between positive and negative score arrays."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    if n_pos * n_neg > 1e8:
        ip = rng.choice(n_pos, min(n_pos, max_compare), replace=False)
        ine = rng.choice(n_neg, min(n_neg, max_compare), replace=False)
        p, n = pos_scores[ip], neg_scores[ine]
    else:
        p, n = pos_scores, neg_scores
    u = np.sum(p[:, None] > n[None, :]) + 0.5 * np.sum(p[:, None] == n[None, :])
    return float(u / (len(p) * len(n)))


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
    """Sample edges and non-edges, return arrays of (u, v, is_edge)."""
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


def verdict(p, direction_correct):
    """Return verdict string given p-value and whether direction matches prediction."""
    if np.isnan(p):
        return "INSUFFICIENT DATA"
    if direction_correct:
        if p < 0.01:
            return "CONFIRMED"
        elif p < 0.05:
            return "SUPPORTED"
        elif p < 0.10:
            return "TREND"
        else:
            return "NOT CONFIRMED"
    else:
        if p < 0.05:
            return "REFUTED"
        else:
            return "NOT CONFIRMED"


def sign_test_verdict(n_correct, n_total, alpha=0.05):
    """Binomial sign test: is n_correct/n_total significantly above 0.5?"""
    if n_total == 0:
        return "INSUFFICIENT DATA", 1.0
    p = stats.binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue
    if p < 0.01:
        return "CONFIRMED", p
    elif p < 0.05:
        return "SUPPORTED", p
    elif p < 0.10:
        return "TREND", p
    else:
        return "NOT CONFIRMED", p


# ── Test 1: Within-network degree-bin test ───────────────────────────

def test_within_degree_bins(G, edges, non_edges, deg, rng):
    """Bin pairs by min(deg_i, deg_j), test Var and Sep across bins."""
    all_pairs_u = []
    all_pairs_v = []
    all_is_edge = []
    for u, v in edges:
        all_pairs_u.append(u)
        all_pairs_v.append(v)
        all_is_edge.append(1)
    for u, v in non_edges:
        all_pairs_u.append(u)
        all_pairs_v.append(v)
        all_is_edge.append(0)

    all_is_edge = np.array(all_is_edge)

    # Compute min(deg_u, deg_v) for each pair
    min_deg = np.array([min(deg.get(u, 0), deg.get(v, 0))
                        for u, v in zip(all_pairs_u, all_pairs_v)], dtype=float)

    # Compute sym(deg_u, deg_v)
    deg_u = np.array([float(deg.get(u, 0)) + 0.001 for u in all_pairs_u])
    deg_v = np.array([float(deg.get(v, 0)) + 0.001 for v in all_pairs_v])
    sym_vals = sym_vec(deg_u, deg_v)

    # Filter NaN
    valid = ~np.isnan(sym_vals)
    min_deg = min_deg[valid]
    sym_vals = sym_vals[valid]
    is_edge = all_is_edge[valid]

    if len(min_deg) < N_BINS * 20:
        return None

    # Quantile bins
    try:
        bin_edges = np.quantile(min_deg, np.linspace(0, 1, N_BINS + 1))
        # Ensure unique bin edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return None
        actual_bins = len(bin_edges) - 1
    except Exception:
        return None

    bin_indices = np.digitize(min_deg, bin_edges[1:-1])  # 0-indexed bins

    bin_results = []
    for b in range(actual_bins):
        mask = bin_indices == b
        if mask.sum() < 20:
            continue
        s = sym_vals[mask]
        ie = is_edge[mask]
        md = min_deg[mask]

        var_sym = float(np.var(s))
        mean_sym = float(np.mean(s))
        mean_deg = float(np.mean(md))

        edge_s = s[ie == 1]
        nonedge_s = s[ie == 0]
        sep = float(np.mean(edge_s) - np.mean(nonedge_s)) if len(edge_s) > 0 and len(nonedge_s) > 0 else np.nan
        auc = auc_from_scores(edge_s, nonedge_s, rng) if len(edge_s) > 5 and len(nonedge_s) > 5 else np.nan

        bin_results.append({
            "bin": b,
            "mean_deg": mean_deg,
            "n_pairs": int(mask.sum()),
            "var_sym": var_sym,
            "mean_sym": mean_sym,
            "sep": sep,
            "auc": auc,
        })

    if len(bin_results) < 3:
        return None

    # Spearman correlations
    mean_degs = [r["mean_deg"] for r in bin_results]
    vars_sym = [r["var_sym"] for r in bin_results]
    seps = [r["sep"] for r in bin_results if not np.isnan(r["sep"])]
    mean_degs_sep = [r["mean_deg"] for r in bin_results if not np.isnan(r["sep"])]

    rho_var, p_var = stats.spearmanr(mean_degs, vars_sym)
    if np.isnan(rho_var):
        rho_var, p_var = 0.0, 1.0

    if len(seps) >= 3:
        rho_sep, p_sep = stats.spearmanr(mean_degs_sep, seps)
        if np.isnan(rho_sep):
            rho_sep, p_sep = 0.0, 1.0
    else:
        rho_sep, p_sep = np.nan, np.nan

    return {
        "bins": bin_results,
        "rho_var_vs_deg": float(rho_var),
        "p_var_vs_deg": float(p_var),
        "var_negative": rho_var < 0,
        "rho_sep_vs_deg": float(rho_sep) if not np.isnan(rho_sep) else None,
        "p_sep_vs_deg": float(p_sep) if not np.isnan(p_sep) else None,
        "sep_negative": bool(rho_sep < 0) if not np.isnan(rho_sep) else None,
    }


# ── Test 2: Cross-feature test ──────────────────────────────────────

def test_cross_feature(G, edges, non_edges, features, rng):
    """Test whether CV predicts sym variance across features within a network."""
    feat_cv = []
    feat_var = []
    feat_names = []

    for fname, fvals in features.items():
        # Compute feature values for nodes
        node_vals = np.array([float(fvals.get(n, 0)) for n in G.nodes()])
        mu = np.mean(node_vals)
        sigma = np.std(node_vals)
        if mu < 1e-12:
            continue
        cv = sigma / mu

        # Compute sym for all pairs using this feature
        fu = np.array([float(fvals.get(u, 0)) + 0.001 for u, _ in edges] +
                      [float(fvals.get(u, 0)) + 0.001 for u, _ in non_edges])
        fv = np.array([float(fvals.get(v, 0)) + 0.001 for _, v in edges] +
                      [float(fvals.get(v, 0)) + 0.001 for _, v in non_edges])
        s = sym_vec(fu, fv)
        valid = ~np.isnan(s)
        if valid.sum() < 20:
            continue

        var_s = float(np.var(s[valid]))
        feat_cv.append(cv)
        feat_var.append(var_s)
        feat_names.append(fname)

    if len(feat_cv) < 3:
        return None

    rho, p = stats.spearmanr(feat_cv, feat_var)
    if np.isnan(rho):
        return None

    return {
        "features": {fn: {"cv": float(c), "var_sym": float(v)}
                     for fn, c, v in zip(feat_names, feat_cv, feat_var)},
        "rho_cv_vs_var": float(rho),
        "p_cv_vs_var": float(p),
        "cv_predicts_var": rho > 0,  # prediction: higher CV -> higher Var(sym)
    }


# ── Test 3: Predicted variance test ─────────────────────────────────

def test_predicted_variance(G, edges, non_edges, features, rng):
    """Test P1: Var(sym) = sigma^2*(1-2/pi)/(2*mu^2) against empirical."""
    P1_COEFF = (1 - 2 / np.pi) / 2  # ~0.1817

    results = []
    for fname, fvals in features.items():
        # Compute pairwise feature means (mu for each pair ≈ (f_u + f_v)/2)
        fu = np.array([float(fvals.get(u, 0)) + 0.001 for u, _ in edges] +
                      [float(fvals.get(u, 0)) + 0.001 for u, _ in non_edges])
        fv = np.array([float(fvals.get(v, 0)) + 0.001 for _, v in edges] +
                      [float(fvals.get(v, 0)) + 0.001 for _, v in non_edges])

        s = sym_vec(fu, fv)
        valid = ~np.isnan(s)
        if valid.sum() < 20:
            continue

        # For P1, sigma and mu are node-level feature parameters
        node_vals = np.array([float(fvals.get(n, 0)) + 0.001 for n in G.nodes()])
        mu = float(np.mean(node_vals))
        sigma = float(np.std(node_vals))

        if mu < 1e-12:
            continue

        var_empirical = float(np.var(s[valid]))
        var_predicted = P1_COEFF * (sigma / mu) ** 2

        results.append({
            "feature": fname,
            "mu": mu,
            "sigma": sigma,
            "cv": sigma / mu,
            "var_empirical": var_empirical,
            "var_predicted": var_predicted,
        })

    if len(results) < 3:
        return None

    emp = [r["var_empirical"] for r in results]
    pred = [r["var_predicted"] for r in results]
    rho, p = stats.spearmanr(pred, emp)
    if np.isnan(rho):
        rho, p = 0.0, 1.0

    # R^2 (Pearson)
    r_pearson, _ = stats.pearsonr(pred, emp) if len(pred) >= 3 else (np.nan, np.nan)
    r2 = r_pearson ** 2 if not np.isnan(r_pearson) else np.nan

    # Systematic bias: is predicted > or < empirical on average?
    ratios = [r["var_predicted"] / r["var_empirical"]
              for r in results if r["var_empirical"] > 1e-15]
    mean_ratio = float(np.mean(ratios)) if ratios else np.nan

    return {
        "per_feature": results,
        "rho_pred_vs_emp": float(rho),
        "p_pred_vs_emp": float(p),
        "r2": float(r2) if not np.isnan(r2) else None,
        "mean_pred_emp_ratio": float(mean_ratio) if not np.isnan(mean_ratio) else None,
    }


# ── Test 4: BL1 relative error verification ─────────────────────────

def test_bl1(G, edges, non_edges, deg, rng):
    """Verify BL1: relative separation error bounded by sigma^2/(2*mu^2).

    BL1 says: for pairs (a,b) = (mu+eps1, mu+eps2) with eps ~ N(0, sigma^2),
    |sep_exact - sep_asymp| / |sep_asymp| = sigma^2 / (2*mu^2).

    The exact sep uses sym(a,b) = 1 - |a-b|/(a+b).
    The asymptotic sep uses sym ≈ 1 - |a-b|/(2*mu).

    BL1 is valid when sigma/mu < ~0.3 (the mu >> sigma regime).
    """
    # Compute exact and asymptotic sym for degree feature
    deg_u_e = np.array([float(deg.get(u, 0)) + 0.001 for u, _ in edges])
    deg_v_e = np.array([float(deg.get(v, 0)) + 0.001 for _, v in edges])
    deg_u_ne = np.array([float(deg.get(u, 0)) + 0.001 for u, _ in non_edges])
    deg_v_ne = np.array([float(deg.get(v, 0)) + 0.001 for _, v in non_edges])

    # Exact sym
    sym_edge_exact = sym_vec(deg_u_e, deg_v_e)
    sym_nonedge_exact = sym_vec(deg_u_ne, deg_v_ne)

    # Node-level mean and std
    node_vals = np.array([float(deg.get(n, 0)) + 0.001 for n in G.nodes()])
    mu = float(np.mean(node_vals))
    sigma = float(np.std(node_vals))
    cv = sigma / mu if mu > 1e-12 else np.inf

    # Asymptotic: sym ≈ 1 - |a-b|/(2*mu) where mu = node-level mean degree
    sym_edge_asymp = 1.0 - np.abs(deg_u_e - deg_v_e) / (2 * mu)
    sym_nonedge_asymp = 1.0 - np.abs(deg_u_ne - deg_v_ne) / (2 * mu)

    # Compute separations
    valid_e = ~np.isnan(sym_edge_exact)
    valid_ne = ~np.isnan(sym_nonedge_exact)

    if valid_e.sum() < 10 or valid_ne.sum() < 10:
        return None

    sep_exact = float(np.mean(sym_edge_exact[valid_e]) - np.mean(sym_nonedge_exact[valid_ne]))
    sep_asymp = float(np.mean(sym_edge_asymp[valid_e]) - np.mean(sym_nonedge_asymp[valid_ne]))

    if abs(sep_asymp) < 1e-12:
        return None

    # BL1 measures relative error against the ASYMPTOTIC separation
    rel_error = abs(sep_exact - sep_asymp) / abs(sep_asymp)
    bl1_bound = cv ** 2 / 2

    # Also compute per-pair mean error for E[sym]
    sym_all_exact = np.concatenate([sym_edge_exact[valid_e], sym_nonedge_exact[valid_ne]])
    sym_all_asymp = np.concatenate([sym_edge_asymp[valid_e], sym_nonedge_asymp[valid_ne]])
    mean_exact = float(np.mean(sym_all_exact))
    mean_asymp = float(np.mean(sym_all_asymp))

    return {
        "mu": mu,
        "sigma": sigma,
        "cv": cv,
        "in_regime": cv < 0.3,  # BL1 designed for sigma/mu < 0.3
        "sep_exact": sep_exact,
        "sep_asymp": sep_asymp,
        "rel_error": float(rel_error),
        "bl1_bound": float(bl1_bound),
        "bound_holds": rel_error <= bl1_bound,
        "mean_sym_exact": mean_exact,
        "mean_sym_asymp": mean_asymp,
    }


# ── Main ─────────────────────────────────────────────────────────────

def run():
    print("=" * 78)
    print("CEILING EFFECT PRE-REGISTERED TEST SUITE")
    print("=" * 78)

    with open(BENCHMARK_PATH) as f:
        benchmark = json.load(f)

    rng = np.random.default_rng(RNG_SEED)

    # Storage
    all_test1 = {}  # within-network degree bins
    all_test2 = {}  # cross-feature
    all_test3 = {}  # predicted variance
    all_test4 = {}  # BL1

    # Aggregate accumulators for predicted variance (test 3)
    all_pred_var = []
    all_emp_var = []

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
        deg = dict(G.degree())
        features = compute_features(G, rng)
        edges, non_edges = sample_pairs(G, rng)

        print(f"\n[{loaded:2d}] {name} (n={G.number_of_nodes()}, m={G.number_of_edges()}, "
              f"sampled {len(edges)} edges, {len(non_edges)} non-edges)")

        # Test 1: within-network degree bins
        r1 = test_within_degree_bins(G, edges, non_edges, deg, rng)
        if r1 is not None:
            all_test1[name] = r1
            print(f"     T1: rho(Var,deg)={r1['rho_var_vs_deg']:+.3f} (p={r1['p_var_vs_deg']:.3f})"
                  f"  rho(Sep,deg)={r1['rho_sep_vs_deg']:+.3f}" if r1['rho_sep_vs_deg'] is not None else
                  f"     T1: rho(Var,deg)={r1['rho_var_vs_deg']:+.3f} (p={r1['p_var_vs_deg']:.3f})"
                  f"  Sep: insufficient data")
        else:
            print(f"     T1: SKIPPED (insufficient data for binning)")

        # Test 2: cross-feature
        r2 = test_cross_feature(G, edges, non_edges, features, rng)
        if r2 is not None:
            all_test2[name] = r2
            print(f"     T2: rho(CV,Var)={r2['rho_cv_vs_var']:+.3f} (p={r2['p_cv_vs_var']:.3f})")
        else:
            print(f"     T2: SKIPPED")

        # Test 3: predicted variance
        r3 = test_predicted_variance(G, edges, non_edges, features, rng)
        if r3 is not None:
            all_test3[name] = r3
            print(f"     T3: rho(pred,emp)={r3['rho_pred_vs_emp']:+.3f} (p={r3['p_pred_vs_emp']:.3f})"
                  f"  R²={r3['r2']:.3f}" if r3['r2'] is not None else
                  f"     T3: rho(pred,emp)={r3['rho_pred_vs_emp']:+.3f} (p={r3['p_pred_vs_emp']:.3f})")
            for pf in r3["per_feature"]:
                all_pred_var.append(pf["var_predicted"])
                all_emp_var.append(pf["var_empirical"])
        else:
            print(f"     T3: SKIPPED")

        # Test 4: BL1
        r4 = test_bl1(G, edges, non_edges, deg, rng)
        if r4 is not None:
            all_test4[name] = r4
            regime_str = "IN-REGIME" if r4['in_regime'] else f"CV={r4['cv']:.2f}>0.3"
            print(f"     T4: rel_err={r4['rel_error']:.4f}, BL1_bound={r4['bl1_bound']:.4f}, "
                  f"{'HOLDS' if r4['bound_holds'] else 'VIOLATED'} ({regime_str})")
        else:
            print(f"     T4: SKIPPED")

    # ═══════════════════════════════════════════════════════════════════
    # AGGREGATE RESULTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("AGGREGATE RESULTS")
    print("=" * 78)

    # --- Test 1 aggregates ---
    print("\n--- TEST 1: Within-network degree-bin (Var and Sep vs degree bin) ---")
    n1 = len(all_test1)
    var_neg_count = sum(1 for r in all_test1.values() if r["var_negative"])
    sep_neg_count = sum(1 for r in all_test1.values()
                        if r["sep_negative"] is not None and r["sep_negative"])
    sep_total = sum(1 for r in all_test1.values() if r["sep_negative"] is not None)

    print(f"  Networks tested: {n1}")
    print(f"  Var decreases with degree bin: {var_neg_count}/{n1}")
    var_sign_verdict, var_sign_p = sign_test_verdict(var_neg_count, n1)
    print(f"  Sign test (Var): {var_sign_verdict} (p={var_sign_p:.4f})")

    print(f"  Sep decreases with degree bin: {sep_neg_count}/{sep_total}")
    sep_sign_verdict, sep_sign_p = sign_test_verdict(sep_neg_count, sep_total)
    print(f"  Sign test (Sep): {sep_sign_verdict} (p={sep_sign_p:.4f})")

    # Aggregate Spearman across networks: Fisher z-transform of rho values
    rho_vars = [r["rho_var_vs_deg"] for r in all_test1.values()]
    if rho_vars:
        mean_rho_var = float(np.mean(rho_vars))
        # One-sample t-test: is mean rho < 0?
        t_var, p_var_t = stats.ttest_1samp(rho_vars, 0, alternative="less")
        print(f"  Mean rho(Var,deg): {mean_rho_var:+.3f} (t={t_var:.2f}, p={p_var_t:.4f})")

    rho_seps = [r["rho_sep_vs_deg"] for r in all_test1.values()
                if r["rho_sep_vs_deg"] is not None and not np.isnan(r["rho_sep_vs_deg"])]
    if rho_seps:
        mean_rho_sep = float(np.mean(rho_seps))
        t_sep, p_sep_t = stats.ttest_1samp(rho_seps, 0, alternative="less")
        print(f"  Mean rho(Sep,deg): {mean_rho_sep:+.3f} (t={t_sep:.2f}, p={p_sep_t:.4f})")

    # --- Test 2 aggregates ---
    print("\n--- TEST 2: Cross-feature (CV predicts Var within network) ---")
    n2 = len(all_test2)
    cv_pos_count = sum(1 for r in all_test2.values() if r["cv_predicts_var"])
    print(f"  Networks tested: {n2}")
    print(f"  CV predicts Var (positive rho): {cv_pos_count}/{n2}")
    cv_sign_verdict, cv_sign_p = sign_test_verdict(cv_pos_count, n2)
    print(f"  Sign test: {cv_sign_verdict} (p={cv_sign_p:.4f})")

    rho_cvs = [r["rho_cv_vs_var"] for r in all_test2.values()]
    if rho_cvs:
        mean_rho_cv = float(np.mean(rho_cvs))
        t_cv, p_cv_t = stats.ttest_1samp(rho_cvs, 0, alternative="greater")
        print(f"  Mean rho(CV,Var): {mean_rho_cv:+.3f} (t={t_cv:.2f}, p={p_cv_t:.4f})")

    # --- Test 3 aggregates ---
    print("\n--- TEST 3: Predicted Var (P1 formula) vs Empirical ---")
    n3 = len(all_test3)
    print(f"  Networks tested: {n3}")

    # Per-network Spearman
    rho_preds = [r["rho_pred_vs_emp"] for r in all_test3.values()]
    if rho_preds:
        pos_count = sum(1 for r in rho_preds if r > 0)
        pred_sign_verdict, pred_sign_p = sign_test_verdict(pos_count, len(rho_preds))
        mean_rho_pred = float(np.mean(rho_preds))
        print(f"  Per-network rho > 0: {pos_count}/{len(rho_preds)}")
        print(f"  Sign test: {pred_sign_verdict} (p={pred_sign_p:.4f})")
        print(f"  Mean per-network rho: {mean_rho_pred:+.3f}")

    # Pooled across all (network, feature) pairs
    if len(all_pred_var) >= 5:
        rho_pooled, p_pooled = stats.spearmanr(all_pred_var, all_emp_var)
        r_pearson_pooled, _ = stats.pearsonr(all_pred_var, all_emp_var)
        r2_pooled = r_pearson_pooled ** 2
        print(f"  Pooled (N={len(all_pred_var)} pairs): rho={rho_pooled:+.3f} (p={p_pooled:.2e}), R²={r2_pooled:.3f}")

        # Systematic bias
        ratios = [p / e for p, e in zip(all_pred_var, all_emp_var) if e > 1e-15]
        if ratios:
            median_ratio = float(np.median(ratios))
            print(f"  Median pred/emp ratio: {median_ratio:.3f} "
                  f"({'over-predicts' if median_ratio > 1 else 'under-predicts'})")

    # --- Test 4 aggregates ---
    print("\n--- TEST 4: BL1 relative error bound ---")
    n4 = len(all_test4)
    holds_count = sum(1 for r in all_test4.values() if r["bound_holds"])
    in_regime = [(n, r) for n, r in all_test4.items() if r["in_regime"]]
    out_regime = [(n, r) for n, r in all_test4.items() if not r["in_regime"]]
    holds_in_regime = sum(1 for _, r in in_regime if r["bound_holds"])

    print(f"  Networks tested: {n4}")
    print(f"  In BL1 regime (sigma/mu < 0.3): {len(in_regime)}/{n4}")
    print(f"  BL1 bound holds (all): {holds_count}/{n4}")
    if in_regime:
        print(f"  BL1 bound holds (in-regime only): {holds_in_regime}/{len(in_regime)}")
    if out_regime:
        holds_out = sum(1 for _, r in out_regime if r["bound_holds"])
        print(f"  BL1 bound holds (out-of-regime): {holds_out}/{len(out_regime)}")

    rel_errors = [r["rel_error"] for r in all_test4.values()]
    bl1_bounds = [r["bl1_bound"] for r in all_test4.values()]
    cvs = [r["cv"] for r in all_test4.values()]
    if rel_errors:
        print(f"  Mean CV (sigma/mu): {np.mean(cvs):.3f} (min={np.min(cvs):.3f}, max={np.max(cvs):.3f})")
        print(f"  Mean rel_error: {np.mean(rel_errors):.4f}")
        print(f"  Mean BL1 bound: {np.mean(bl1_bounds):.4f}")

        # For in-regime networks, test bound
        if in_regime:
            diffs = [r["bl1_bound"] - r["rel_error"] for _, r in in_regime]
            if len(diffs) >= 2:
                t_bl1, p_bl1 = stats.ttest_1samp(diffs, 0, alternative="greater")
                print(f"  In-regime t-test (bound - error > 0): t={t_bl1:.2f}, p={p_bl1:.4f}")
            else:
                p_bl1 = np.nan
        else:
            p_bl1 = np.nan

    # ═══════════════════════════════════════════════════════════════════
    # VERDICTS
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)

    verdicts = {}

    # P1-within: Var decreases with degree bin
    if n1 > 0 and rho_vars:
        # Two-sided test, then check direction matches prediction
        _, p1w_p_two = stats.ttest_1samp(rho_vars, 0)
        p1w_correct = mean_rho_var < 0
        p1w_verdict = verdict(p1w_p_two, p1w_correct)
        p1w_p = p1w_p_two
    else:
        p1w_verdict = "INSUFFICIENT DATA"
        p1w_p = np.nan
    verdicts["P1-within"] = {
        "description": "Var decreases with degree bin (within-network)",
        "verdict": p1w_verdict,
        "stat": f"mean_rho={mean_rho_var:+.3f}" if rho_vars else "N/A",
        "p": float(p1w_p) if not np.isnan(p1w_p) else None,
        "n_networks": n1,
        "sign_test": f"{var_neg_count}/{n1}",
    }

    # P1-cross: CV predicts Var across features
    if n2 > 0 and rho_cvs:
        _, p1c_p_two = stats.ttest_1samp(rho_cvs, 0)
        p1c_correct = mean_rho_cv > 0
        p1c_verdict = verdict(p1c_p_two, p1c_correct)
        p1c_p = p1c_p_two
    else:
        p1c_verdict = "INSUFFICIENT DATA"
        p1c_p = np.nan
    verdicts["P1-cross"] = {
        "description": "CV predicts Var across features (within-network)",
        "verdict": p1c_verdict,
        "stat": f"mean_rho={mean_rho_cv:+.3f}" if rho_cvs else "N/A",
        "p": float(p1c_p) if not np.isnan(p1c_p) else None,
        "n_networks": n2,
        "sign_test": f"{cv_pos_count}/{n2}",
    }

    # P1-direct: Predicted Var from formula correlates with empirical Var
    if len(all_pred_var) >= 5:
        p1d_p = float(p_pooled)
        p1d_correct = rho_pooled > 0
        p1d_verdict = verdict(p1d_p, p1d_correct)
        p1d_stat = f"rho={rho_pooled:+.3f}, R²={r2_pooled:.3f}"
    else:
        p1d_verdict = "INSUFFICIENT DATA"
        p1d_p = np.nan
        p1d_stat = "N/A"
    verdicts["P1-direct"] = {
        "description": "Predicted Var from P1 formula correlates with empirical Var",
        "verdict": p1d_verdict,
        "stat": p1d_stat,
        "p": float(p1d_p) if not np.isnan(p1d_p) else None,
        "n_pairs": len(all_pred_var),
    }

    # T8-within: Separation decreases with degree bin
    if sep_total > 0 and rho_seps:
        # Use two-sided test, then check direction
        t8w_t, t8w_p = stats.ttest_1samp(rho_seps, 0)
        t8w_correct = mean_rho_sep < 0
        t8w_verdict = verdict(t8w_p, t8w_correct)
    else:
        t8w_verdict = "INSUFFICIENT DATA"
        t8w_p = np.nan
    verdicts["T8-within"] = {
        "description": "Separation decreases with degree bin (within-network)",
        "verdict": t8w_verdict,
        "stat": f"mean_rho={mean_rho_sep:+.3f}" if rho_seps else "N/A",
        "p": float(t8w_p) if not np.isnan(t8w_p) else None,
        "n_networks": sep_total,
        "sign_test": f"{sep_neg_count}/{sep_total}",
    }

    # BL1: Relative error bounded (evaluate within applicable regime)
    if len(in_regime) > 0:
        bl1_frac = holds_in_regime / len(in_regime)
        if bl1_frac >= 0.8:
            bl1_verdict = "CONFIRMED"
        elif bl1_frac >= 0.6:
            bl1_verdict = "SUPPORTED"
        elif bl1_frac >= 0.4:
            bl1_verdict = "NOT CONFIRMED"
        else:
            bl1_verdict = "REFUTED"
        bl1_stat = f"{holds_in_regime}/{len(in_regime)} in-regime ({holds_count}/{n4} all)"
    elif n4 > 0:
        bl1_verdict = "NOT APPLICABLE"
        bl1_stat = f"0 networks in sigma/mu<0.3 regime ({holds_count}/{n4} all)"
    else:
        bl1_verdict = "INSUFFICIENT DATA"
        bl1_stat = "N/A"
    verdicts["BL1"] = {
        "description": "Relative error bounded by sigma²/(2*mu²) [sigma/mu<0.3 regime]",
        "verdict": bl1_verdict,
        "stat": bl1_stat,
        "p": None,
        "n_networks": n4,
        "n_in_regime": len(in_regime),
    }

    # Print summary
    print(f"\n  {'Hypothesis':<14s} {'Verdict':<18s} {'Statistic':<32s} {'p-value':<12s} {'N'}")
    print(f"  {'-'*90}")
    for hyp, info in verdicts.items():
        p_str = f"{info['p']:.4f}" if info['p'] is not None else "N/A"
        n_str = str(info.get('n_networks', info.get('n_pairs', '')))
        sign = info.get('sign_test', '')
        stat_str = info['stat']
        if sign:
            stat_str += f" [{sign}]"
        print(f"  {hyp:<14s} {info['verdict']:<18s} {stat_str:<32s} {p_str:<12s} {n_str}")

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
        "n_networks_loaded": loaded,
        "verdicts": sanitize(verdicts),
        "test1_within_degree_bins": sanitize(all_test1),
        "test2_cross_feature": sanitize(all_test2),
        "test3_predicted_variance": sanitize(all_test3),
        "test4_bl1": sanitize(all_test4),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ceiling_effect_tests.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run()
