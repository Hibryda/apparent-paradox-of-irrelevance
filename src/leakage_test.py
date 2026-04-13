#!/usr/bin/env python3
"""K-core target leakage test.

Tests whether computing k-core on the full graph (including test edges)
inflates AUC compared to computing on train-only graph.

Protocol:
  1. Hold out 20% of edges as test set
  2. Compute k-core on remaining 80% (train graph)
  3. Compute k-core on full graph
  4. Compare AUC for both
  5. Stratify by density (sparse <5 mean degree vs dense >=5)

If AUC degrades >0.02 for sparse networks, flag for paper revision.
"""
import sys
import json
import time
import warnings
import numpy as np
import networkx as nx
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

SM_ROOT = Path(__file__).resolve().parents[1] / ".." / "symmetry-mechanism"
sys.path.insert(0, str(SM_ROOT / "src"))

from phase6_benchmark import (
    load_netzschleuder_csv, load_airport, PHASE6_CONFIGS,
)
from multi_network_analysis import (
    NETWORK_CONFIGS, load_edge_list as mna_load_edge_list,
    load_bitcoin, load_string_network,
    load_epinions, load_slashdot, load_wiki_rfa,
)

BENCHMARK_PATH = SM_ROOT / "data" / "phase6_benchmark.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RNG_SEED = 42
HOLDOUT_FRAC = 0.2
MAX_NONEDGES = 4000


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


def sym(a, b):
    s = a + b
    return (2 * min(a, b) / s) if s > 0 else 0.0


def compute_auc(scores_pos, scores_neg):
    se = np.array(scores_pos, dtype=float)
    sn = np.array(scores_neg, dtype=float)
    if len(se) == 0 or len(sn) == 0:
        return 0.5
    u = np.sum(se[:, None] > sn[None, :]) + 0.5 * np.sum(se[:, None] == sn[None, :])
    return float(u / (len(se) * len(sn)))


def run():
    print("=" * 80)
    print("K-CORE TARGET LEAKAGE TEST")
    print("=" * 80)
    print(f"Protocol: {HOLDOUT_FRAC*100:.0f}% edges held out. Compare full-graph vs train-only k-core.\n")

    benchmark = json.load(open(BENCHMARK_PATH))
    rng = np.random.default_rng(RNG_SEED)

    header = f"{'Network':35s} {'<k>':>5s} {'AUC_full':>9s} {'AUC_train':>9s} {'diff':>7s} {'flag':>5s}"
    print(header)
    print("-" * len(header))

    results = []

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

        edges = list(G.edges())
        n_holdout = max(10, int(len(edges) * HOLDOUT_FRAC))
        idx = rng.choice(len(edges), n_holdout, replace=False)
        test_edges = [edges[i] for i in idx]
        train_edges = [edges[i] for i in range(len(edges)) if i not in set(idx)]

        # Train graph
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)

        # K-core on full graph
        kcore_full = nx.core_number(G)
        # K-core on train graph
        kcore_train = nx.core_number(G_train)

        # Sample non-edges from full graph
        nodes = list(G.nodes())
        edge_set = set(G.edges())
        non_edges = []
        target = min(MAX_NONEDGES, len(test_edges))
        attempts = 0
        while len(non_edges) < target and attempts < target * 20:
            u, v = rng.choice(nodes, 2, replace=False)
            if (u, v) not in edge_set and (v, u) not in edge_set:
                non_edges.append((u, v))
            attempts += 1

        if len(test_edges) < 10 or len(non_edges) < 10:
            continue

        # AUC with full-graph k-core
        se_full = [sym(kcore_full[u], kcore_full[v]) for u, v in test_edges]
        sn_full = [sym(kcore_full[u], kcore_full[v]) for u, v in non_edges]
        auc_full = compute_auc(se_full, sn_full)

        # AUC with train-only k-core
        se_train = [sym(kcore_train[u], kcore_train[v]) for u, v in test_edges]
        sn_train = [sym(kcore_train[u], kcore_train[v]) for u, v in non_edges]
        auc_train = compute_auc(se_train, sn_train)

        diff = auc_full - auc_train
        mean_deg = 2 * G.number_of_edges() / G.number_of_nodes()
        flag = "***" if abs(diff) > 0.02 else ""

        print(f"{name:35s} {mean_deg:5.1f} {auc_full:9.3f} {auc_train:9.3f} {diff:+7.3f} {flag:>5s}")

        results.append({
            "network": name,
            "mean_degree": round(mean_deg, 2),
            "auc_full": round(auc_full, 4),
            "auc_train": round(auc_train, 4),
            "diff": round(diff, 4),
            "sparse": mean_deg < 5,
        })

    # Summary
    print()
    diffs = [r["diff"] for r in results]
    sparse = [r for r in results if r["sparse"]]
    dense = [r for r in results if not r["sparse"]]
    print(f"Overall: mean diff = {np.mean(diffs):+.4f}, max = {np.max(np.abs(diffs)):.4f}")
    if sparse:
        sp_diffs = [r["diff"] for r in sparse]
        print(f"Sparse (<5): mean diff = {np.mean(sp_diffs):+.4f}, n={len(sparse)}")
    if dense:
        dn_diffs = [r["diff"] for r in dense]
        print(f"Dense (>=5): mean diff = {np.mean(dn_diffs):+.4f}, n={len(dense)}")

    flagged = [r for r in results if abs(r["diff"]) > 0.02]
    print(f"\nFlagged (|diff| > 0.02): {len(flagged)}/{len(results)}")
    for r in flagged:
        print(f"  {r['network']}: diff={r['diff']:+.4f}, <k>={r['mean_degree']}")

    # Save
    out = RESULTS_DIR / "leakage_test.json"
    with open(out, "w") as f:
        json.dump({
            "holdout_frac": HOLDOUT_FRAC,
            "n_networks": len(results),
            "mean_diff": round(float(np.mean(diffs)), 4),
            "max_abs_diff": round(float(np.max(np.abs(diffs))), 4),
            "n_flagged": len(flagged),
            "per_network": results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    t0 = time.time()
    run()
    print(f"\nTotal time: {time.time() - t0:.1f}s")
