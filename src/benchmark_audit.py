#!/usr/bin/env python3
"""Benchmark audit: two critical pre-submission tests.

TEST 1 — BENCHMARK AUDIT: Classify all 32 networks as natively undirected
vs symmetrized-from-directed. Report proportions.

TEST 2 — PROTOCOL SENSITIVITY: Does d' predict AUC under 80/20 holdout
evaluation (not just closed-world)?  Compare Spearman rho(d'_holdout,
AUC_holdout) to closed-world rho = 0.956.

Runtime target: < 8 min.
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

MAX_EDGES = 3000
MAX_NONEDGES = 3000
RNG_SEED = 42
HOLDOUT_FRAC = 0.2
FEATURES = ["kcore", "degree"]


# ══════════════════════════════════════════════════════════════════════
# LOADER (same pattern as ceiling_effect_tests.py)
# ══════════════════════════════════════════════════════════════════════

def load_network_by_name(name):
    """Load network by name from benchmark configs (handles all 32 networks)."""
    G = None
    for cfg in NETWORK_CONFIGS:
        if cfg["name"] == name:
            loader = cfg.get("loader", "edge_list")
            max_n = cfg.get("max_nodes", 3000)
            if loader == "edge_list":
                G = mna_load_edge_list(cfg["path"], max_nodes=max_n)
            elif loader == "bitcoin":
                G = load_bitcoin(cfg["path"], max_nodes=max_n)
            elif loader == "string":
                G = load_string_network(cfg["path"], max_nodes=max_n)
            elif loader == "epinions":
                G = load_epinions(cfg["path"], max_nodes=max_n)
            elif loader == "slashdot":
                G = load_slashdot(cfg["path"], max_nodes=max_n)
            elif loader == "wiki_rfa":
                G = load_wiki_rfa(cfg["path"], max_nodes=max_n)
            else:
                G = mna_load_edge_list(cfg["path"], max_nodes=max_n)
            break
    if G is None:
        for cfg in PHASE6_CONFIGS:
            if cfg["name"] == name:
                if cfg["loader"] == "netzschleuder":
                    G = load_netzschleuder_csv(
                        cfg["path"], weight_col=cfg.get("weight_col", "weight"),
                        max_nodes=cfg.get("max_nodes", 3000),
                        preserve_sign=cfg.get("preserve_sign", False))
                elif cfg["loader"] == "airport":
                    G = load_airport(cfg["path"], max_nodes=cfg.get("max_nodes", 3000))
                break
    if G is not None:
        G.remove_edges_from(nx.selfloop_edges(G))
    return G


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def sym_vec(a, b):
    """Vectorised sym(a, b) = 1 - |a-b|/(a+b). Returns NaN where a+b==0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, 1.0 - np.abs(a - b) / s, np.nan)


def auc_from_scores(pos_scores, neg_scores):
    """Mann-Whitney AUC between positive and negative score arrays."""
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    # Cap comparison size
    rng = np.random.RandomState(RNG_SEED + 999)
    if n_pos * n_neg > 1e8:
        ip = rng.choice(n_pos, min(n_pos, 5000), replace=False)
        ine = rng.choice(n_neg, min(n_neg, 5000), replace=False)
        p, n = pos_scores[ip], neg_scores[ine]
    else:
        p, n = pos_scores, neg_scores
    u = np.sum(p[:, None] > n[None, :]) + 0.5 * np.sum(p[:, None] == n[None, :])
    return float(u / (len(p) * len(n)))


def compute_dprime(pos_scores, neg_scores):
    """Cohen's d' = (mu_pos - mu_neg) / pooled_sd."""
    if len(pos_scores) < 5 or len(neg_scores) < 5:
        return np.nan
    mu_p, mu_n = np.mean(pos_scores), np.mean(neg_scores)
    sd_p, sd_n = np.std(pos_scores, ddof=1), np.std(neg_scores, ddof=1)
    pooled = np.sqrt((sd_p**2 + sd_n**2) / 2)
    if pooled < 1e-15:
        return np.nan
    return float((mu_p - mu_n) / pooled)


# ══════════════════════════════════════════════════════════════════════
# TEST 1: BENCHMARK AUDIT — directed vs undirected classification
# ══════════════════════════════════════════════════════════════════════

def classify_directed():
    """Classify each of the 32 benchmark networks as directed or undirected."""

    # Build directed_source map from PHASE6_CONFIGS
    p6_directed = {}
    for cfg in PHASE6_CONFIGS:
        p6_directed[cfg["name"]] = cfg.get("directed_source", False)

    # Known directed from NETWORK_CONFIGS (by loader type / domain knowledge)
    known_directed_loaders = {"bitcoin", "epinions", "slashdot", "wiki_rfa"}

    benchmark = json.load(open(BENCHMARK_PATH))
    results = []

    for key in sorted(benchmark.keys()):
        entry = benchmark[key]
        name = entry["name"]
        domain = entry.get("domain", "unknown")
        ns = entry.get("network_stats", {})
        n_nodes = ns.get("n_nodes", 0)
        n_edges = ns.get("n_edges", 0)
        mean_deg = ns.get("mean_degree", 0)
        density = 2 * n_edges / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0

        # Determine directed status
        is_directed = False
        source = "undirected_default"

        # Check PHASE6_CONFIGS
        if name in p6_directed:
            is_directed = p6_directed[name]
            source = "phase6_config"
        else:
            # Check NETWORK_CONFIGS
            for cfg in NETWORK_CONFIGS:
                if cfg["name"] == name:
                    loader = cfg.get("loader", "edge_list")
                    if loader in known_directed_loaders:
                        is_directed = True
                        source = f"loader_{loader}"
                    else:
                        is_directed = False
                        source = "undirected_edge_list"
                    break

        results.append({
            "key": key,
            "name": name,
            "domain": domain,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "mean_degree": round(mean_deg, 2),
            "density": round(density, 6),
            "is_directed_source": is_directed,
            "classification_source": source,
        })

    n_directed = sum(1 for r in results if r["is_directed_source"])
    n_undirected = len(results) - n_directed
    frac_directed = n_directed / len(results) if results else 0
    concern = frac_directed > 0.30

    return {
        "networks": results,
        "n_total": len(results),
        "n_directed": n_directed,
        "n_undirected": n_undirected,
        "frac_directed": round(frac_directed, 3),
        "concern_above_30pct": concern,
    }


# ══════════════════════════════════════════════════════════════════════
# TEST 2: PROTOCOL SENSITIVITY — d' under holdout
# ══════════════════════════════════════════════════════════════════════

def holdout_evaluation(G, feature_name, rng):
    """80/20 edge holdout: compute AUC and d' on test set only.

    Steps:
    1. Split edges 80/20
    2. Build training subgraph
    3. Compute features on training subgraph
    4. Evaluate sym on test edges + sampled test non-edges
    5. Return AUC and d'
    """
    edges = list(G.edges())
    n_edges = len(edges)
    if n_edges < 50:
        return None

    # Shuffle and split
    idx = rng.permutation(n_edges)
    n_test = max(10, int(n_edges * HOLDOUT_FRAC))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    test_edges = [edges[i] for i in test_idx]
    train_edges = [edges[i] for i in train_idx]

    # Build training subgraph
    G_train = nx.Graph()
    G_train.add_nodes_from(G.nodes())
    G_train.add_edges_from(train_edges)

    # Remove isolated nodes (no training edges)
    # But keep node set for non-edge sampling
    train_nodes = set()
    for u, v in train_edges:
        train_nodes.add(u)
        train_nodes.add(v)

    # Compute features on training subgraph
    if feature_name == "kcore":
        feat = nx.core_number(G_train)
    elif feature_name == "degree":
        feat = dict(G_train.degree())
    else:
        return None

    # Filter test edges to those where both nodes have non-zero features
    valid_test_edges = [(u, v) for u, v in test_edges
                        if feat.get(u, 0) > 0 and feat.get(v, 0) > 0]

    if len(valid_test_edges) < 10:
        return None

    # Cap test edges
    if len(valid_test_edges) > MAX_EDGES:
        sel = rng.choice(len(valid_test_edges), MAX_EDGES, replace=False)
        valid_test_edges = [valid_test_edges[i] for i in sel]

    # Sample test non-edges (not in original graph)
    edge_set = set(G.edges()) | set((v, u) for u, v in G.edges())
    nodes_list = list(train_nodes)
    n_target = min(MAX_NONEDGES, len(valid_test_edges))
    test_non_edges = []
    attempts = 0
    while len(test_non_edges) < n_target and attempts < n_target * 30:
        u, v = rng.choice(nodes_list, 2, replace=False)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            if feat.get(u, 0) > 0 and feat.get(v, 0) > 0:
                test_non_edges.append((u, v))
        attempts += 1

    if len(test_non_edges) < 10:
        return None

    # Compute sym scores
    pos_f_u = np.array([float(feat.get(u, 0)) + 0.001 for u, _ in valid_test_edges])
    pos_f_v = np.array([float(feat.get(v, 0)) + 0.001 for _, v in valid_test_edges])
    pos_sym = sym_vec(pos_f_u, pos_f_v)

    neg_f_u = np.array([float(feat.get(u, 0)) + 0.001 for u, _ in test_non_edges])
    neg_f_v = np.array([float(feat.get(v, 0)) + 0.001 for _, v in test_non_edges])
    neg_sym = sym_vec(neg_f_u, neg_f_v)

    # Filter NaN
    pos_sym = pos_sym[~np.isnan(pos_sym)]
    neg_sym = neg_sym[~np.isnan(neg_sym)]

    if len(pos_sym) < 10 or len(neg_sym) < 10:
        return None

    auc = auc_from_scores(pos_sym, neg_sym)
    dprime = compute_dprime(pos_sym, neg_sym)

    return {"auc": auc, "dprime": dprime,
            "n_test_edges": len(pos_sym), "n_test_nonedges": len(neg_sym)}


def closed_world_evaluation(G, feature_name, rng):
    """Standard closed-world: AUC and d' on full graph."""
    if feature_name == "kcore":
        feat = nx.core_number(G)
    elif feature_name == "degree":
        feat = dict(G.degree())
    else:
        return None

    edges = list(G.edges())
    if len(edges) > MAX_EDGES:
        sel = rng.choice(len(edges), MAX_EDGES, replace=False)
        edges = [edges[i] for i in sel]

    # Sample non-edges
    nodes_list = list(G.nodes())
    edge_set = set(G.edges()) | set((v, u) for u, v in G.edges())
    n_target = min(MAX_NONEDGES, len(edges))
    non_edges = []
    attempts = 0
    while len(non_edges) < n_target and attempts < n_target * 30:
        u, v = rng.choice(nodes_list, 2, replace=False)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            non_edges.append((u, v))
        attempts += 1

    if len(non_edges) < 10:
        return None

    pos_f_u = np.array([float(feat.get(u, 0)) + 0.001 for u, _ in edges])
    pos_f_v = np.array([float(feat.get(v, 0)) + 0.001 for _, v in edges])
    pos_sym = sym_vec(pos_f_u, pos_f_v)

    neg_f_u = np.array([float(feat.get(u, 0)) + 0.001 for u, _ in non_edges])
    neg_f_v = np.array([float(feat.get(v, 0)) + 0.001 for _, v in non_edges])
    neg_sym = sym_vec(neg_f_u, neg_f_v)

    pos_sym = pos_sym[~np.isnan(pos_sym)]
    neg_sym = neg_sym[~np.isnan(neg_sym)]

    if len(pos_sym) < 10 or len(neg_sym) < 10:
        return None

    auc = auc_from_scores(pos_sym, neg_sym)
    dprime = compute_dprime(pos_sym, neg_sym)

    return {"auc": auc, "dprime": dprime,
            "n_edges": len(pos_sym), "n_nonedges": len(neg_sym)}


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    benchmark = json.load(open(BENCHMARK_PATH))
    all_names = [(key, benchmark[key]["name"]) for key in sorted(benchmark.keys())]

    print(f"{'='*70}")
    print(f"BENCHMARK AUDIT — {len(all_names)} networks")
    print(f"{'='*70}")

    # ── TEST 1: Directed vs undirected classification ─────────────
    print("\n[TEST 1] Classifying directed vs undirected sources...")
    audit = classify_directed()
    print(f"  Total: {audit['n_total']}")
    print(f"  Directed sources: {audit['n_directed']}")
    print(f"  Undirected sources: {audit['n_undirected']}")
    print(f"  Fraction directed: {audit['frac_directed']:.1%}")
    if audit["concern_above_30pct"]:
        print("  *** CONCERN: >30% of benchmark is symmetrized-from-directed ***")
    else:
        print("  OK: directed fraction is <= 30%")

    print("\n  Directed networks:")
    for net in audit["networks"]:
        if net["is_directed_source"]:
            print(f"    - {net['name']} ({net['domain']}, n={net['n_nodes']}, "
                  f"m={net['n_edges']}, <k>={net['mean_degree']})")
    print("\n  Undirected networks:")
    for net in audit["networks"]:
        if not net["is_directed_source"]:
            print(f"    - {net['name']} ({net['domain']}, n={net['n_nodes']}, "
                  f"m={net['n_edges']}, <k>={net['mean_degree']})")

    # ── TEST 2: Protocol sensitivity ──────────────────────────────
    print(f"\n{'='*70}")
    print("[TEST 2] Protocol sensitivity: d' under holdout vs closed-world")
    print(f"{'='*70}")

    per_network = {}
    for key, name in all_names:
        print(f"\n  Loading {name}...", end=" ", flush=True)
        try:
            G = load_network_by_name(name)
        except Exception as e:
            print(f"LOAD FAILED: {e}")
            per_network[key] = {"name": name, "error": str(e)}
            continue

        if G is None:
            print("SKIPPED (loader returned None)")
            per_network[key] = {"name": name, "error": "loader returned None"}
            continue

        n = G.number_of_nodes()
        m = G.number_of_edges()
        print(f"n={n}, m={m}", end=" ", flush=True)

        rng = np.random.RandomState(RNG_SEED)
        net_result = {"name": name, "n_nodes": n, "n_edges": m, "features": {}}

        for feat in FEATURES:
            # Closed-world
            cw = closed_world_evaluation(G, feat, np.random.RandomState(RNG_SEED))
            # Holdout
            ho = holdout_evaluation(G, feat, np.random.RandomState(RNG_SEED))

            net_result["features"][feat] = {
                "closed_world": cw,
                "holdout": ho,
            }
            if cw and ho:
                print(f"[{feat}: CW_AUC={cw['auc']:.3f}, HO_AUC={ho['auc']:.3f}]",
                      end=" ", flush=True)
            elif cw:
                print(f"[{feat}: CW_AUC={cw['auc']:.3f}, HO=skip]",
                      end=" ", flush=True)
            else:
                print(f"[{feat}: skip]", end=" ", flush=True)

        per_network[key] = net_result
        print()

    # ── Aggregate: Spearman rho(d', AUC) for each feature ─────────
    print(f"\n{'='*70}")
    print("AGGREGATE RESULTS")
    print(f"{'='*70}")

    aggregate = {}
    for feat in FEATURES:
        # Collect closed-world pairs
        cw_dprime = []
        cw_auc = []
        ho_dprime = []
        ho_auc = []

        for key, res in per_network.items():
            if "error" in res:
                continue
            fr = res["features"].get(feat, {})
            cw = fr.get("closed_world")
            ho = fr.get("holdout")

            if cw and not np.isnan(cw["auc"]) and not np.isnan(cw["dprime"]):
                cw_dprime.append(cw["dprime"])
                cw_auc.append(cw["auc"])
            if ho and not np.isnan(ho["auc"]) and not np.isnan(ho["dprime"]):
                ho_dprime.append(ho["dprime"])
                ho_auc.append(ho["auc"])

        feat_agg = {"n_closed_world": len(cw_dprime), "n_holdout": len(ho_dprime)}

        if len(cw_dprime) >= 5:
            rho_cw, p_cw = stats.spearmanr(cw_dprime, cw_auc)
            feat_agg["rho_dprime_auc_closed"] = round(float(rho_cw), 4)
            feat_agg["p_dprime_auc_closed"] = float(p_cw)
        else:
            feat_agg["rho_dprime_auc_closed"] = None
            feat_agg["p_dprime_auc_closed"] = None

        if len(ho_dprime) >= 5:
            rho_ho, p_ho = stats.spearmanr(ho_dprime, ho_auc)
            feat_agg["rho_dprime_auc_holdout"] = round(float(rho_ho), 4)
            feat_agg["p_dprime_auc_holdout"] = float(p_ho)
        else:
            feat_agg["rho_dprime_auc_holdout"] = None
            feat_agg["p_dprime_auc_holdout"] = None

        # Cross-protocol: rho between closed-world AUC and holdout AUC
        cross_cw_auc = []
        cross_ho_auc = []
        for key, res in per_network.items():
            if "error" in res:
                continue
            fr = res["features"].get(feat, {})
            cw = fr.get("closed_world")
            ho = fr.get("holdout")
            if (cw and ho and not np.isnan(cw["auc"]) and not np.isnan(ho["auc"])):
                cross_cw_auc.append(cw["auc"])
                cross_ho_auc.append(ho["auc"])

        if len(cross_cw_auc) >= 5:
            rho_cross, p_cross = stats.spearmanr(cross_cw_auc, cross_ho_auc)
            feat_agg["rho_auc_closed_vs_holdout"] = round(float(rho_cross), 4)
            feat_agg["p_auc_closed_vs_holdout"] = float(p_cross)
        else:
            feat_agg["rho_auc_closed_vs_holdout"] = None
            feat_agg["p_auc_closed_vs_holdout"] = None

        # Mean AUC drop
        if cross_cw_auc:
            mean_cw = np.mean(cross_cw_auc)
            mean_ho = np.mean(cross_ho_auc)
            feat_agg["mean_auc_closed"] = round(float(mean_cw), 4)
            feat_agg["mean_auc_holdout"] = round(float(mean_ho), 4)
            feat_agg["mean_auc_drop"] = round(float(mean_cw - mean_ho), 4)

        aggregate[feat] = feat_agg

        print(f"\n  Feature: {feat}")
        print(f"    Closed-world: n={feat_agg['n_closed_world']}, "
              f"rho(d',AUC)={feat_agg['rho_dprime_auc_closed']}")
        print(f"    Holdout:      n={feat_agg['n_holdout']}, "
              f"rho(d',AUC)={feat_agg['rho_dprime_auc_holdout']}")
        if feat_agg.get("rho_auc_closed_vs_holdout") is not None:
            print(f"    Cross-protocol rho(AUC_cw, AUC_ho)="
                  f"{feat_agg['rho_auc_closed_vs_holdout']}")
        if feat_agg.get("mean_auc_drop") is not None:
            print(f"    Mean AUC: closed={feat_agg['mean_auc_closed']:.4f}, "
                  f"holdout={feat_agg['mean_auc_holdout']:.4f}, "
                  f"drop={feat_agg['mean_auc_drop']:.4f}")

    # ── Verdicts ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("VERDICTS")
    print(f"{'='*70}")

    # Test 1 verdict
    d_frac = audit["frac_directed"]
    if d_frac > 0.30:
        t1_verdict = f"CONCERN — {d_frac:.0%} directed (>{0.30:.0%} threshold)"
    else:
        t1_verdict = f"OK — {d_frac:.0%} directed (<= 30% threshold)"
    print(f"\n  TEST 1 (directed fraction): {t1_verdict}")

    # Test 2 verdict per feature
    for feat in FEATURES:
        fa = aggregate[feat]
        rho_ho = fa.get("rho_dprime_auc_holdout")
        rho_cw = fa.get("rho_dprime_auc_closed")
        if rho_ho is not None:
            if rho_ho > 0.7:
                v = f"ROBUST (rho={rho_ho:.3f} > 0.7)"
            elif rho_ho > 0.5:
                v = f"MODERATE (0.5 < rho={rho_ho:.3f} <= 0.7)"
            else:
                v = f"PROTOCOL-SPECIFIC (rho={rho_ho:.3f} <= 0.5) — PROBLEM"
            print(f"  TEST 2 ({feat}): d' predicts AUC under holdout: {v}")
            if rho_cw is not None:
                print(f"    (closed-world rho={rho_cw:.3f} for comparison)")
        else:
            print(f"  TEST 2 ({feat}): INSUFFICIENT DATA")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.1f}s")

    # ── Save ──────────────────────────────────────────────────────
    output = {
        "test1_benchmark_audit": audit,
        "test2_protocol_sensitivity": {
            "per_network": per_network,
            "aggregate": aggregate,
        },
        "verdicts": {
            "test1": t1_verdict,
            "test2": {feat: {
                "rho_dprime_auc_holdout": aggregate[feat].get("rho_dprime_auc_holdout"),
                "rho_dprime_auc_closed": aggregate[feat].get("rho_dprime_auc_closed"),
                "verdict": (
                    "ROBUST" if (aggregate[feat].get("rho_dprime_auc_holdout") or 0) > 0.7
                    else "MODERATE" if (aggregate[feat].get("rho_dprime_auc_holdout") or 0) > 0.5
                    else "PROTOCOL-SPECIFIC" if aggregate[feat].get("rho_dprime_auc_holdout") is not None
                    else "INSUFFICIENT DATA"
                ),
            } for feat in FEATURES},
        },
        "runtime_seconds": round(elapsed, 1),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "benchmark_audit.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
