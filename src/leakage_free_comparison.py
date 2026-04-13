#!/usr/bin/env python3
"""Leakage-free comparison: train-only kcore vs train-only degree on all 65 networks.

Computes a strict leakage-free comparison: both features recomputed on train
graph only, evaluated on held-out edges.

Protocol:
  1. Hold out 20% of edges as test set
  2. Compute kcore AND degree on the remaining 80% (train graph)
  3. Score test edges and sampled non-edges using train-only features
  4. Compute AUC and signed d' for both features
  5. Report win rate, pooled rho, and per-feature rho
"""
import sys
import json
import time
import warnings
import gzip
import numpy as np
import networkx as nx
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

SM_ROOT = Path(__file__).resolve().parents[1] / ".." / "symmetry-mechanism"
sys.path.insert(0, str(SM_ROOT / "src"))

from phase6_benchmark import (
    load_netzschleuder_csv, load_airport, PHASE6_CONFIGS, BIO_DIR,
)
from multi_network_analysis import (
    NETWORK_CONFIGS, load_edge_list as mna_load_edge_list,
    load_bitcoin, load_string_network,
    load_epinions, load_slashdot, load_wiki_rfa,
)

BENCHMARK_PATH = SM_ROOT / "data" / "phase6_benchmark.json"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
HELDOUT_DIR = Path(__file__).resolve().parents[1] / "data" / "heldout"
HELDOUT2_DIR = Path(__file__).resolve().parents[1] / "data" / "heldout2"

RNG_SEED = 42
HOLDOUT_FRAC = 0.2
MAX_PAIRS = 4000


def load_dev_network(name):
    for cfg in NETWORK_CONFIGS:
        if cfg["name"] == name:
            loader = cfg.get("loader", "edge_list")
            max_n = cfg.get("max_nodes", 3000)
            if loader == "edge_list": return mna_load_edge_list(cfg["path"], max_nodes=max_n)
            elif loader == "bitcoin": return load_bitcoin(cfg["path"], max_nodes=max_n)
            elif loader == "string": return load_string_network(cfg["path"], max_nodes=max_n)
            elif loader == "epinions": return load_epinions(cfg["path"], max_nodes=max_n)
            elif loader == "slashdot": return load_slashdot(cfg["path"], max_nodes=max_n)
            elif loader == "wiki_rfa": return load_wiki_rfa(cfg["path"], max_nodes=max_n)
            else: return mna_load_edge_list(cfg["path"], max_nodes=max_n)
    for cfg in PHASE6_CONFIGS:
        if cfg["name"] == name:
            if cfg["loader"] == "netzschleuder":
                return load_netzschleuder_csv(cfg["path"], weight_col=cfg.get("weight_col", "weight"),
                    max_nodes=cfg.get("max_nodes", 3000), preserve_sign=cfg.get("preserve_sign", False))
            elif cfg["loader"] == "airport":
                return load_airport(cfg["path"], max_nodes=cfg.get("max_nodes", 3000))
    return None


def load_heldout_edge_list(path, max_nodes=2000):
    G = nx.Graph()
    opener = gzip.open if str(path).endswith('.gz') else open
    mode = 'rt' if str(path).endswith('.gz') else 'r'
    with opener(path, mode) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('%') or line.startswith('<'):
                continue
            parts = line.split(',') if ',' in line else (line.split('\t') if '\t' in line else line.split())
            if len(parts) >= 2:
                u, v = parts[0].strip(), parts[1].strip()
                if u != v and u and v:
                    G.add_edge(u, v)
    if nx.number_of_selfloops(G) > 0:
        G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        return G
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    if G.number_of_nodes() > max_nodes:
        rng = np.random.default_rng(42)
        sampled = set(rng.choice(list(G.nodes()), max_nodes, replace=False))
        G = G.subgraph(sampled).copy()
        if not nx.is_connected(G):
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def generate_synthetic(name):
    rng_seed = hash(name) % (2**31)
    if name.startswith("LFR"):
        mu = float(name.split("_mu")[1]) if "_mu" in name else 0.3
        try:
            from networkx.generators.community import LFR_benchmark_graph
            G = LFR_benchmark_graph(500, 3.0, 1.5, mu, min_degree=5, max_degree=50,
                                     min_community=20, max_community=100, seed=rng_seed)
            return nx.Graph(G)
        except: return None
    elif name.startswith("BA"):
        m = int(name.split("_m")[1]) if "_m" in name else 3
        return nx.barabasi_albert_graph(500, m, seed=rng_seed)
    elif name.startswith("WS"):
        p = float(name.split("_p")[1]) if "_p" in name else 0.1
        return nx.watts_strogatz_graph(500, 6, p, seed=rng_seed)
    elif name.startswith("SBM"):
        sizes = [200, 200, 200]
        p_matrix = [[0.05, 0.005, 0.005], [0.005, 0.05, 0.005], [0.005, 0.005, 0.05]]
        return nx.stochastic_block_model(sizes, p_matrix, seed=rng_seed)
    return None


def sym(a, b):
    s = a + b
    return (2 * min(a, b) / s) if s > 0 else 0.0


def compute_auc(se, sn):
    se, sn = np.array(se, dtype=float), np.array(sn, dtype=float)
    if len(se) == 0 or len(sn) == 0:
        return 0.5
    u = np.sum(se[:, None] > sn[None, :]) + 0.5 * np.sum(se[:, None] == sn[None, :])
    return float(u / (len(se) * len(sn)))


def signed_dprime(se, sn):
    mu_e, mu_ne = np.mean(se), np.mean(sn)
    sigma_e, sigma_ne = np.std(se, ddof=1), np.std(sn, ddof=1)
    pooled_sigma = np.sqrt((sigma_e**2 + sigma_ne**2) / 2)
    if pooled_sigma < 1e-12:
        return 0.0
    return (mu_e - mu_ne) / pooled_sigma  # sign from data, not external


def run():
    print("=" * 85)
    print("LEAKAGE-FREE COMPARISON: train-only kcore vs train-only degree")
    print(f"Protocol: {HOLDOUT_FRAC*100:.0f}% edges held out, features computed on train graph only")
    print("=" * 85)
    t0 = time.time()
    rng = np.random.default_rng(RNG_SEED)

    # Build registry (same as unified_analysis.py)
    benchmark = json.load(open(BENCHMARK_PATH))
    ho1 = json.load(open(RESULTS_DIR / "heldout_validation.json"))
    ho2 = json.load(open(RESULTS_DIR / "expanded_heldout.json"))

    HO1_FILES = {
        "E. coli Transcription": "ecoli_transcription_v1.0_edges.csv",
        "Fresh Webs Akatore": "fresh_webs_AkatoreA_edges.csv",
        "RouteViews AS": "route_views_19971108_edges.csv",
        "WebKB Wisconsin": "webkb_webkb_wisconsin_link1_edges.csv",
        "EU Procurements": "eu_procurements_alt_AT_2008_edges.csv",
        "Faculty Hiring CS": "faculty_hiring_computer_science_edges.csv",
        "Urban Streets Ahmedabad": "urban_streets_ahmedabad_edges.csv",
        "Ugandan Village Friendship": "ugandan_village_friendship-1_edges.csv",
        "Add Health Community": "ego_social_facebook_0_edges.csv",
        "Fullerene C1500": "openstreetmap_01-AL-cities-street_networks_0100124_Abbeville_edges.csv",
    }

    registry = []
    for key, info in benchmark.items():
        name = info["name"]
        domain = info.get("domain", "unknown")
        registry.append({"name": name, "corpus": "dev", "domain": domain, "loader": "dev"})

    for net in ho1["per_network"]:
        name = net["network"]
        registry.append({"name": name, "corpus": "held-out-1", "domain": net.get("domain", "unknown"),
                         "loader": "heldout1", "file": HO1_FILES.get(name)})

    for net in ho2["per_network"]:
        is_synth = net.get("source") == "synthetic" or net.get("domain", "").startswith("synthetic")
        registry.append({"name": net["network"], "corpus": "expanded", "domain": net.get("domain", "unknown"),
                         "is_synthetic": is_synth, "loader": "heldout2"})

    header = f"{'#':>3s} {'Network':35s} {'kc_train':>8s} {'dg_train':>8s} {'kc_full':>8s} {'dg_full':>8s} {'winner':>6s} {'leak':>5s}"
    print(header)
    print("-" * len(header))

    results = []

    for reg in registry:
        name = reg["name"]
        try:
            if reg["loader"] == "dev":
                G = load_dev_network(name)
            elif reg["loader"] in ("heldout1", "heldout2"):
                search_dir = HELDOUT_DIR if reg["loader"] == "heldout1" else HELDOUT2_DIR
                G = None
                name_norm = name.replace('/', '_').replace(':', '_').replace(' ', '_')
                candidate = search_dir / f"{name_norm}_edges.csv"
                if candidate.exists():
                    G = load_heldout_edge_list(candidate)
                if G is None:
                    candidate2 = search_dir / f"{name}_edges.csv"
                    if candidate2.exists():
                        G = load_heldout_edge_list(candidate2)
                if G is None:
                    name_lower = name_norm.lower()
                    for f in sorted(search_dir.glob("*_edges.csv")):
                        stem = f.stem.replace('_edges', '').lower()
                        if stem == name_lower:
                            G = load_heldout_edge_list(f)
                            break
                if G is None and reg.get("is_synthetic"):
                    G = generate_synthetic(name)
                if G is None:
                    continue
        except Exception:
            continue

        if G is None or G.number_of_nodes() < 30:
            continue
        if nx.number_of_selfloops(G) > 0:
            G.remove_edges_from(nx.selfloop_edges(G))

        edges = list(G.edges())
        n_edges = len(edges)
        if n_edges < 50:
            continue

        # Hold out 20% of edges
        n_holdout = max(10, int(n_edges * HOLDOUT_FRAC))
        idx = rng.choice(n_edges, n_holdout, replace=False)
        holdout_set = set(idx)
        test_edges = [edges[i] for i in idx]
        train_edges = [edges[i] for i in range(n_edges) if i not in holdout_set]

        # Build train graph (keep all nodes)
        G_train = nx.Graph()
        G_train.add_nodes_from(G.nodes())
        G_train.add_edges_from(train_edges)

        # Features on train graph
        kcore_train = nx.core_number(G_train)
        degree_train = dict(G_train.degree())

        # Features on full graph (for comparison)
        kcore_full = nx.core_number(G)
        degree_full = dict(G.degree())

        # Sample non-edges (from full graph complement)
        nodes = list(G.nodes())
        edge_set = set(G.edges()) | {(v, u) for u, v in G.edges()}
        non_edges = []
        target = min(MAX_PAIRS, len(test_edges))
        attempts = 0
        while len(non_edges) < target and attempts < target * 20:
            u, v = tuple(rng.choice(nodes, 2, replace=False))
            if (u, v) not in edge_set:
                non_edges.append((u, v))
            attempts += 1

        if len(test_edges) < 10 or len(non_edges) < 10:
            continue

        # Compute sym scores and AUC for train-only features
        kc_se_train = [sym(kcore_train[u], kcore_train[v]) for u, v in test_edges]
        kc_sn_train = [sym(kcore_train[u], kcore_train[v]) for u, v in non_edges]
        dg_se_train = [sym(degree_train[u], degree_train[v]) for u, v in test_edges]
        dg_sn_train = [sym(degree_train[u], degree_train[v]) for u, v in non_edges]

        auc_kc_train = compute_auc(kc_se_train, kc_sn_train)
        auc_dg_train = compute_auc(dg_se_train, dg_sn_train)

        # Full-graph features for reference
        kc_se_full = [sym(kcore_full[u], kcore_full[v]) for u, v in test_edges]
        kc_sn_full = [sym(kcore_full[u], kcore_full[v]) for u, v in non_edges]
        dg_se_full = [sym(degree_full[u], degree_full[v]) for u, v in test_edges]
        dg_sn_full = [sym(degree_full[u], degree_full[v]) for u, v in non_edges]

        auc_kc_full = compute_auc(kc_se_full, kc_sn_full)
        auc_dg_full = compute_auc(dg_se_full, dg_sn_full)

        # Newman r on train graph (for d' sign)
        nx.set_node_attributes(G_train, kcore_train, "kcore_val")
        nx.set_node_attributes(G_train, degree_train, "degree_val")
        try:
            r_kc = nx.numeric_assortativity_coefficient(G_train, "kcore_val")
        except:
            r_kc = 0.0
        try:
            r_dg = nx.numeric_assortativity_coefficient(G_train, "degree_val")
        except:
            r_dg = 0.0

        # Signed d' (sign from mu_e - mu_ne, not from newman_r)
        dp_kc = signed_dprime(kc_se_train, kc_sn_train)
        dp_dg = signed_dprime(dg_se_train, dg_sn_train)

        winner = "kcore" if auc_kc_train > auc_dg_train else "degree" if auc_dg_train > auc_kc_train else "tie"
        leak = f"{auc_kc_full - auc_kc_train:+.3f}"

        n = len(results) + 1
        print(f"{n:3d} {name:35s} {auc_kc_train:8.3f} {auc_dg_train:8.3f} {auc_kc_full:8.3f} {auc_dg_full:8.3f} {winner:>6s} {leak:>5s}")

        results.append({
            "network": name,
            "corpus": reg["corpus"],
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "mean_degree": round(2 * G.number_of_edges() / G.number_of_nodes(), 2),
            "auc_kcore_train": round(auc_kc_train, 4),
            "auc_degree_train": round(auc_dg_train, 4),
            "auc_kcore_full": round(auc_kc_full, 4),
            "auc_degree_full": round(auc_dg_full, 4),
            "dprime_kcore_train": round(dp_kc, 4),
            "dprime_degree_train": round(dp_dg, 4),
            "newman_r_kcore_train": round(r_kc, 4),
            "newman_r_degree_train": round(r_dg, 4),
            "kcore_leakage": round(auc_kc_full - auc_kc_train, 4),
            "degree_leakage": round(auc_dg_full - auc_dg_train, 4),
        })

    # Summary
    print(f"\n{'='*85}")
    print(f"RESULTS: {len(results)} networks analyzed")

    kc_wins = sum(1 for r in results if r["auc_kcore_train"] > r["auc_degree_train"])
    dg_wins = sum(1 for r in results if r["auc_degree_train"] > r["auc_kcore_train"])
    ties = len(results) - kc_wins - dg_wins
    print(f"\nLeakage-free win rate: kcore {kc_wins}/{len(results)}, degree {dg_wins}/{len(results)}, ties {ties}")
    print(f"Leakage-free kcore AUC mean: {np.mean([r['auc_kcore_train'] for r in results]):.4f}")
    print(f"Leakage-free degree AUC mean: {np.mean([r['auc_degree_train'] for r in results]):.4f}")

    # Leakage statistics
    kc_leaks = [r["kcore_leakage"] for r in results]
    dg_leaks = [r["degree_leakage"] for r in results]
    print(f"\nKcore leakage: mean={np.mean(kc_leaks):+.4f}, max={np.max(np.abs(kc_leaks)):.4f}")
    print(f"Degree leakage: mean={np.mean(dg_leaks):+.4f}, max={np.max(np.abs(dg_leaks)):.4f}")

    # Per-feature d' vs AUC rho (train-only)
    dp_kc = [r["dprime_kcore_train"] for r in results]
    auc_kc = [r["auc_kcore_train"] for r in results]
    dp_dg = [r["dprime_degree_train"] for r in results]
    auc_dg = [r["auc_degree_train"] for r in results]

    rho_kc, p_kc = stats.spearmanr(dp_kc, auc_kc)
    rho_dg, p_dg = stats.spearmanr(dp_dg, auc_dg)
    print(f"\nLeakage-free d' vs AUC:")
    print(f"  kcore-only: rho={rho_kc:.4f} (p={p_kc:.2e}, n={len(dp_kc)})")
    print(f"  degree-only: rho={rho_dg:.4f} (p={p_dg:.2e}, n={len(dp_dg)})")

    # Pooled
    all_dp = dp_kc + dp_dg
    all_auc = auc_kc + auc_dg
    rho_pooled, p_pooled = stats.spearmanr(all_dp, all_auc)
    print(f"  pooled: rho={rho_pooled:.4f} (p={p_pooled:.2e}, n={len(all_dp)})")

    # Magnitude-only: |d'| vs |AUC - 0.5|
    abs_dp = [abs(x) for x in all_dp]
    abs_auc = [abs(x - 0.5) for x in all_auc]
    rho_mag, p_mag = stats.spearmanr(abs_dp, abs_auc)
    print(f"  magnitude-only |d'| vs |AUC-0.5|: rho={rho_mag:.4f} (p={p_mag:.2e})")

    # Per-feature magnitude
    abs_kc_dp = [abs(x) for x in dp_kc]
    abs_kc_auc = [abs(x - 0.5) for x in auc_kc]
    abs_dg_dp = [abs(x) for x in dp_dg]
    abs_dg_auc = [abs(x - 0.5) for x in auc_dg]
    rho_kc_mag, _ = stats.spearmanr(abs_kc_dp, abs_kc_auc)
    rho_dg_mag, _ = stats.spearmanr(abs_dg_dp, abs_dg_auc)
    print(f"  kcore magnitude: rho={rho_kc_mag:.4f}")
    print(f"  degree magnitude: rho={rho_dg_mag:.4f}")

    # Networks where kcore loses under train-only
    losses = [r for r in results if r["auc_degree_train"] >= r["auc_kcore_train"]]
    if losses:
        print(f"\nNetworks where degree beats/ties kcore (train-only):")
        for r in losses:
            print(f"  {r['network']}: kc={r['auc_kcore_train']:.4f}, dg={r['auc_degree_train']:.4f}, gap={r['auc_kcore_train']-r['auc_degree_train']:+.4f}")

    # Save
    out = RESULTS_DIR / "leakage_free_comparison.json"
    with open(out, "w") as f:
        json.dump({
            "protocol": "20% edge holdout, features on train graph only",
            "n_networks": len(results),
            "kcore_wins": kc_wins,
            "degree_wins": dg_wins,
            "ties": ties,
            "kcore_auc_mean_train": round(float(np.mean([r["auc_kcore_train"] for r in results])), 4),
            "degree_auc_mean_train": round(float(np.mean([r["auc_degree_train"] for r in results])), 4),
            "dprime_rho_kcore_only": round(float(rho_kc), 4),
            "dprime_rho_degree_only": round(float(rho_dg), 4),
            "dprime_rho_pooled": round(float(rho_pooled), 4),
            "dprime_rho_magnitude_only": round(float(rho_mag), 4),
            "mean_kcore_leakage": round(float(np.mean(kc_leaks)), 4),
            "mean_degree_leakage": round(float(np.mean(dg_leaks)), 4),
            "per_network": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
