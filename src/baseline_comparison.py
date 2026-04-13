#!/usr/bin/env python3
"""Baseline comparison: k-core similarity vs standard link prediction methods.

Compares k-core similarity (AUC 0.703) against established baselines:
  - Common Neighbors (CN)
  - Adamic-Adar (AA)
  - Jaccard Coefficient (neighborhood-based)
  - Preferential Attachment (PA)

All methods use the SAME pipeline: same 67 networks, same edge/non-edge
sampling, same seeds. This determines whether the paper is a performance
claim or a mechanism study.

NOTE: CN/AA/Jaccard use neighborhood overlap — fundamentally different
from single-feature normalized similarity. The comparison is for context,
not to claim k-core similarity beats multi-hop methods.
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
RNG_SEED = 42


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


def sample_pairs(G, rng):
    """Sample edges and non-edges (same protocol as paper)."""
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


def compute_auc(scores_edge, scores_nonedge):
    """AUC via Mann-Whitney U statistic."""
    se = np.array(scores_edge, dtype=float)
    sn = np.array(scores_nonedge, dtype=float)
    n_pos, n_neg = len(se), len(sn)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Subsample for speed if needed
    if n_pos * n_neg > 1e8:
        rng = np.random.default_rng(42)
        se = se[rng.choice(n_pos, min(n_pos, 5000), replace=False)]
        sn = sn[rng.choice(n_neg, min(n_neg, 5000), replace=False)]
    u = np.sum(se[:, None] > sn[None, :]) + 0.5 * np.sum(se[:, None] == sn[None, :])
    return float(u / (len(se) * len(sn)))


def sym(a, b):
    """Normalized similarity."""
    s = a + b
    return (2 * min(a, b) / s) if s > 0 else 0.0


def common_neighbors_score(G, u, v):
    """Number of common neighbors."""
    return len(set(G.neighbors(u)) & set(G.neighbors(v)))


def adamic_adar_score(G, u, v):
    """Adamic-Adar index."""
    cn = set(G.neighbors(u)) & set(G.neighbors(v))
    return sum(1.0 / np.log(G.degree(w)) for w in cn if G.degree(w) > 1)


def jaccard_neighbors_score(G, u, v):
    """Jaccard coefficient of neighborhoods."""
    nu = set(G.neighbors(u))
    nv = set(G.neighbors(v))
    union = nu | nv
    if len(union) == 0:
        return 0.0
    return len(nu & nv) / len(union)


def preferential_attachment_score(G, u, v):
    """Preferential attachment: degree(u) * degree(v)."""
    return G.degree(u) * G.degree(v)


def run():
    print("=" * 80)
    print("BASELINE COMPARISON: K-core similarity vs standard link prediction methods")
    print("=" * 80)
    print()

    benchmark = json.load(open(BENCHMARK_PATH))
    rng = np.random.default_rng(RNG_SEED)

    methods = ["kcore_sym", "degree_sym", "CN", "AA", "Jaccard_nbr", "PA"]
    header = f"{'Network':35s}" + "".join(f" {m:>10s}" for m in methods) + "  winner"
    print(header)
    print("-" * len(header))

    results = []
    loaded = 0

    # Dev set (32 networks)
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
        if G.number_of_nodes() < 30:
            continue

        loaded += 1
        edges, non_edges = sample_pairs(G, rng)
        if len(edges) == 0 or len(non_edges) == 0:
            continue

        kcore = nx.core_number(G)
        degree = dict(G.degree())

        # Compute scores for each method
        aucs = {}

        # K-core similarity
        se = [sym(kcore[u], kcore[v]) for u, v in edges]
        sn = [sym(kcore[u], kcore[v]) for u, v in non_edges]
        aucs["kcore_sym"] = compute_auc(se, sn)

        # Degree similarity
        se = [sym(degree[u], degree[v]) for u, v in edges]
        sn = [sym(degree[u], degree[v]) for u, v in non_edges]
        aucs["degree_sym"] = compute_auc(se, sn)

        # Common Neighbors
        se = [common_neighbors_score(G, u, v) for u, v in edges]
        sn = [common_neighbors_score(G, u, v) for u, v in non_edges]
        aucs["CN"] = compute_auc(se, sn)

        # Adamic-Adar
        se = [adamic_adar_score(G, u, v) for u, v in edges]
        sn = [adamic_adar_score(G, u, v) for u, v in non_edges]
        aucs["AA"] = compute_auc(se, sn)

        # Jaccard (neighborhood)
        se = [jaccard_neighbors_score(G, u, v) for u, v in edges]
        sn = [jaccard_neighbors_score(G, u, v) for u, v in non_edges]
        aucs["Jaccard_nbr"] = compute_auc(se, sn)

        # Preferential Attachment
        se = [preferential_attachment_score(G, u, v) for u, v in edges]
        sn = [preferential_attachment_score(G, u, v) for u, v in non_edges]
        aucs["PA"] = compute_auc(se, sn)

        winner = max(aucs, key=lambda k: aucs[k])
        row = f"{name:35s}" + "".join(f" {aucs[m]:10.3f}" for m in methods) + f"  {winner}"
        print(row)

        results.append({
            "network": name,
            "corpus": "dev",
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "aucs": {m: round(aucs[m], 4) for m in methods},
            "winner": winner,
        })

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY (dev set)")
    print("=" * 80)
    n = len(results)
    for m in methods:
        vals = [r["aucs"][m] for r in results]
        wins = sum(1 for r in results if r["winner"] == m)
        print(f"  {m:12s}: mean={np.mean(vals):.3f}, median={np.median(vals):.3f}, wins={wins}/{n}")

    # Paired comparison: k-core vs each baseline
    print()
    print("PAIRED COMPARISONS (Wilcoxon signed-rank):")
    kcore_aucs = np.array([r["aucs"]["kcore_sym"] for r in results])
    for m in methods:
        if m == "kcore_sym":
            continue
        other_aucs = np.array([r["aucs"][m] for r in results])
        diff = kcore_aucs - other_aucs
        wins_kcore = np.sum(diff > 0)
        wins_other = np.sum(diff < 0)
        ties = np.sum(diff == 0)
        try:
            stat, p = stats.wilcoxon(kcore_aucs, other_aucs, alternative="two-sided")
        except ValueError:
            stat, p = 0, 1.0
        mean_diff = np.mean(diff)
        print(f"  kcore vs {m:12s}: kcore wins {wins_kcore}/{n}, other wins {wins_other}/{n}, "
              f"ties {ties}, mean_diff={mean_diff:+.3f}, p={p:.4f}")

    # Save
    out = RESULTS_DIR / "baseline_comparison.json"
    with open(out, "w") as f:
        json.dump({
            "methods": methods,
            "n_networks": n,
            "per_network": results,
            "summary": {
                m: {
                    "mean_auc": round(float(np.mean([r["aucs"][m] for r in results])), 4),
                    "median_auc": round(float(np.median([r["aucs"][m] for r in results])), 4),
                    "wins": sum(1 for r in results if r["winner"] == m),
                }
                for m in methods
            },
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
