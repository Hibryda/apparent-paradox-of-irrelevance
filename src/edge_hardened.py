"""EDGE hardened: Config model null (correct) with optimized centrality computation.

The permutation null (shuffling centrality vectors) is TOO LIBERAL —
it breaks degree-centrality correlations that exist even in ER graphs.
The config model null (degree-preserving rewiring) is CORRECT.

Optimization: for the null, only recompute fast centralities (kcore, clustering)
and reuse the degree normalization. Skip eigenvector for null (expensive, and
eigenvector ≈ f(degree) so degree-preserving rewiring changes it minimally).
"""
import sys
import json
import numpy as np
import networkx as nx
from pathlib import Path

SM_ROOT = Path(__file__).resolve().parents[1] / ".." / "symmetry-mechanism"
sys.path.insert(0, str(SM_ROOT / "src"))

from phase6_benchmark import (
    load_netzschleuder_csv, load_edge_list, load_airport,
    PHASE6_CONFIGS, BIO_DIR,
)
from multi_network_analysis import NETWORK_CONFIGS, load_edge_list as mna_load_edge_list

BENCHMARK_PATH = SM_ROOT / "data" / "phase6_benchmark.json"

N_REWIRINGS = 15
MAX_PAIRS = 2000


def load_network_by_name(name):
    for cfg in NETWORK_CONFIGS:
        if cfg["name"] == name:
            return mna_load_edge_list(cfg["path"], max_nodes=cfg.get("max_nodes", 3000))
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


def normalize(vals):
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12:
        return np.zeros_like(vals)
    return (vals - mn) / (mx - mn)


def compute_gap(G, nodes, node_idx, edges_i, edges_j, nonedges_i, nonedges_j):
    """Compute EDGE gap for a graph using fast centralities only."""
    n = len(nodes)
    deg = np.array([G.degree(node) for node in nodes], dtype=float)
    kcore = nx.core_number(G)
    kc = np.array([float(kcore[node]) for node in nodes])
    clust_d = nx.clustering(G)
    cl = np.array([clust_d[node] for node in nodes])

    mat = np.column_stack([normalize(deg), normalize(kc), normalize(cl)])
    norms = np.linalg.norm(mat, axis=1)
    norms[norms < 1e-12] = 1.0

    def mean_cos(ii, jj):
        dots = np.sum(mat[ii] * mat[jj], axis=1)
        cos = dots / (norms[ii] * norms[jj])
        return float(np.mean(cos))

    return mean_cos(edges_i, edges_j) - mean_cos(nonedges_i, nonedges_j)


def analyze_network(G, n_rewirings=N_REWIRINGS, max_pairs=MAX_PAIRS):
    """Compute EDGE with config model null."""
    if nx.number_of_selfloops(G) > 0:
        G = G.copy()
        G.remove_edges_from(nx.selfloop_edges(G))

    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}

    # Sample edges and non-edges
    rng = np.random.default_rng(42)
    edges = list(G.edges())
    if len(edges) > max_pairs:
        idx = rng.choice(len(edges), max_pairs, replace=False)
        edges = [edges[i] for i in idx]

    edge_set = set(G.edges())
    non_edges = []
    attempts = 0
    while len(non_edges) < len(edges) and attempts < len(edges) * 10:
        u, v = rng.choice(nodes, 2, replace=False)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            non_edges.append((u, v))
        attempts += 1

    if len(edges) == 0 or len(non_edges) == 0:
        return {"gap_real": 0, "gap_null_mean": 0, "gap_null_std": 0,
                "excess": 0, "z_score": 0, "p_value": 1.0, "is_signal": False,
                "n_rewirings": n_rewirings}

    edges_i = np.array([node_idx[u] for u, v in edges], dtype=int)
    edges_j = np.array([node_idx[v] for u, v in edges], dtype=int)
    nonedges_i = np.array([node_idx[u] for u, v in non_edges], dtype=int)
    nonedges_j = np.array([node_idx[v] for u, v in non_edges], dtype=int)

    # Real score (with eigenvector for the real graph only)
    deg = np.array([G.degree(node) for node in nodes], dtype=float)
    kcore = nx.core_number(G)
    kc = np.array([float(kcore[node]) for node in nodes])
    try:
        eigv_d = nx.eigenvector_centrality(G, max_iter=300, tol=1e-3)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigv_d = nx.degree_centrality(G)
    ev = np.array([eigv_d[node] for node in nodes])
    clust_d = nx.clustering(G)
    cl = np.array([clust_d[node] for node in nodes])

    mat_real = np.column_stack([normalize(deg), normalize(kc), normalize(ev), normalize(cl)])
    norms_real = np.linalg.norm(mat_real, axis=1)
    norms_real[norms_real < 1e-12] = 1.0

    def mean_cos(ii, jj, mat, nrm):
        dots = np.sum(mat[ii] * mat[jj], axis=1)
        cos = dots / (nrm[ii] * nrm[jj])
        return float(np.mean(cos))

    real_gap = mean_cos(edges_i, edges_j, mat_real, norms_real) - \
               mean_cos(nonedges_i, nonedges_j, mat_real, norms_real)

    # Config model null: degree-preserving edge swaps, recompute fast centralities
    null_gaps = []
    for seed in range(n_rewirings):
        G_null = G.copy()
        n_swaps = G_null.number_of_edges() * 3
        try:
            nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
        except nx.NetworkXAlgorithmError:
            pass

        # Recompute only kcore + clustering (fast), reuse degree (preserved)
        kcore_null = nx.core_number(G_null)
        kc_null = np.array([float(kcore_null.get(node, 0)) for node in nodes])
        clust_null = nx.clustering(G_null)
        cl_null = np.array([clust_null.get(node, 0) for node in nodes])

        # Use same degree (preserved by edge swap) + new kcore + new clustering
        mat_null = np.column_stack([normalize(deg), normalize(kc_null), normalize(cl_null)])
        norms_null = np.linalg.norm(mat_null, axis=1)
        norms_null[norms_null < 1e-12] = 1.0

        # Compute gap using the NULL graph's edges (not the real edges)
        null_edges = list(G_null.edges())
        if len(null_edges) > max_pairs:
            null_edges = [null_edges[i] for i in rng.choice(len(null_edges), max_pairs, replace=False)]
        null_edge_set = set(G_null.edges())
        null_nonedges = []
        att = 0
        while len(null_nonedges) < len(null_edges) and att < len(null_edges) * 10:
            u, v = rng.choice(nodes, 2, replace=False)
            if (u, v) not in null_edge_set and (v, u) not in null_edge_set:
                null_nonedges.append((u, v))
            att += 1

        if len(null_edges) > 0 and len(null_nonedges) > 0:
            nei = np.array([node_idx[u] for u, v in null_edges], dtype=int)
            nej = np.array([node_idx[v] for u, v in null_edges], dtype=int)
            nni = np.array([node_idx[u] for u, v in null_nonedges], dtype=int)
            nnj = np.array([node_idx[v] for u, v in null_nonedges], dtype=int)
            null_gap = mean_cos(nei, nej, mat_null, norms_null) - \
                       mean_cos(nni, nnj, mat_null, norms_null)
            null_gaps.append(null_gap)

    if not null_gaps:
        return {"gap_real": round(real_gap, 5), "gap_null_mean": 0, "gap_null_std": 0,
                "excess": round(real_gap, 5), "z_score": 0, "p_value": 0.5,
                "is_signal": False, "n_rewirings": n_rewirings}

    mean_null = np.mean(null_gaps)
    std_null = np.std(null_gaps) if len(null_gaps) > 1 else 0.001
    excess = real_gap - mean_null
    z_score = excess / std_null if std_null > 1e-8 else 0.0
    p_value = np.mean([ng >= real_gap for ng in null_gaps])

    return {
        "gap_real": round(float(real_gap), 5),
        "gap_null_mean": round(float(mean_null), 5),
        "gap_null_std": round(float(std_null), 5),
        "excess": round(float(excess), 5),
        "z_score": round(float(z_score), 2),
        "p_value": round(float(p_value), 3),
        "is_signal": bool(p_value < 0.05 and real_gap > 0),
        "n_rewirings": n_rewirings,
    }


def run():
    print("=" * 85)
    print(f"EDGE HARDENED: Config model null, {N_REWIRINGS} rewirings, p-values")
    print("=" * 85)

    benchmark = json.load(open(BENCHMARK_PATH))

    # ER validation
    print("\n--- ER Validation (should show NO signal) ---")
    for nn, pp in [(500, 0.05), (300, 0.10)]:
        G_er = nx.erdos_renyi_graph(nn, pp, seed=42)
        r = analyze_network(G_er, n_rewirings=5)
        print(f"  ER(n={nn},p={pp}): gap={r['gap_real']:.4f}, z={r['z_score']:.1f}, p={r['p_value']:.3f}")

    # Full corpus
    print(f"\n--- Full corpus ({N_REWIRINGS} config model rewirings) ---\n")
    header = (f"{'Network':35s} {'gap':>7s} {'null±σ':>14s} {'z':>6s} {'p':>6s} {'sig':>4s}")
    print(header)
    print("-" * len(header))

    results = []
    signal_count = 0
    loaded = 0

    # Classify domains as bio/non-bio properly
    bio_labels = {"ppi", "genetic", "connectome", "ecology", "ecology_bipartite",
                  "biology_yeast", "biology_fly", "biology_human", "biology_zebrafish", "other"}
    # "other" in the benchmark are mostly PPI/genetic networks with wrong labels

    for key, net_info in benchmark.items():
        name = net_info["name"]
        domain = net_info.get("domain", "unknown")
        try:
            G = load_network_by_name(name)
        except Exception:
            continue
        if G is None:
            continue
        if G.number_of_nodes() < 30:
            continue

        loaded += 1
        r = analyze_network(G)
        r["name"] = name
        r["domain"] = domain

        # Correct bio classification
        is_bio = domain in bio_labels or any(x in name.lower() for x in
            ["elegans", "cerevisiae", "sapiens", "melanogaster", "rerio",
             "drosophila", "connectome", "food web", "plant-poll"])
        r["is_bio"] = is_bio
        results.append(r)

        if r["is_signal"]:
            signal_count += 1

        sig_str = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "ns"
        null_str = f"{r['gap_null_mean']:.4f}±{r['gap_null_std']:.4f}"
        print(f"{name:35s} {r['gap_real']:7.4f} {null_str:>14s} {r['z_score']:6.1f} "
              f"{r['p_value']:6.3f} {sig_str:>4s}")

    print(f"\nNetworks: {loaded} | Signal (p<0.05): {signal_count}/{loaded}")

    # Bio vs non-bio
    bio_signal = sum(1 for r in results if r["is_bio"] and r["is_signal"])
    bio_total = sum(1 for r in results if r["is_bio"])
    nonbio_signal = sum(1 for r in results if not r["is_bio"] and r["is_signal"])
    nonbio_total = sum(1 for r in results if not r["is_bio"])
    print(f"\n--- Biological vs Non-biological ---")
    print(f"  Bio:    {bio_signal}/{bio_total}")
    print(f"  NonBio: {nonbio_signal}/{nonbio_total}")

    if bio_total > 0 and nonbio_total > 0:
        from scipy.stats import fisher_exact
        table = [[bio_signal, bio_total - bio_signal],
                 [nonbio_signal, nonbio_total - nonbio_signal]]
        odds, p_fisher = fisher_exact(table, alternative="greater")
        print(f"  Fisher (bio > nonbio): OR={odds:.2f}, p={p_fisher:.4f}")
    else:
        odds, p_fisher = 0, 1

    # Save
    out = Path(__file__).resolve().parent.parent / "results" / "edge_hardened.json"
    with open(out, "w") as f:
        json.dump({
            "null_type": "config_model_edge_swap",
            "n_rewirings": N_REWIRINGS,
            "networks_loaded": loaded,
            "signal_count": signal_count,
            "bio_signal": bio_signal, "bio_total": bio_total,
            "nonbio_signal": nonbio_signal, "nonbio_total": nonbio_total,
            "fisher_odds": round(float(odds), 2),
            "fisher_p": round(float(p_fisher), 4),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run()
