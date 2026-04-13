#!/usr/bin/env python3
"""Unified analysis across ALL 67 networks.

Runs the complete analysis pipeline on the full corpus:
  - 32 development networks (phase6_benchmark)
  - 10 held-out Netzschleuder-1
  - 25 expanded (17 Netzschleuder-2 + 8 synthetic)

Computes per network × feature:
  - AUC (sym similarity)
  - d' (signed by assortativity)
  - TER (tie enrichment ratio)
  - Mixture AUC decomposition (tie + continuous)
  - Metric independence (5 formulas)
  - BL2 variance prediction
  - EDGE diagnostic (config model null)
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

MAX_EDGES = 4000
MAX_NONEDGES = 4000
RNG_SEED = 42
FEATURES = ["degree", "kcore", "eigenvector", "clustering", "random"]
METRICS = ["sym", "jaccard", "cosine", "exp_diff", "binary_match"]
EDGE_N_REWIRINGS = 15


# ══════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════

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


def generate_synthetic(name):
    """Generate synthetic networks by name."""
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


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def sym(a, b):
    s = a + b
    return (2 * min(a, b) / s) if s > 0 else 0.0

def jaccard(a, b):
    return min(a, b) / max(a, b) if max(a, b) > 0 else 0.0

def cosine_scalar(a, b):
    return (a * b) / (np.sqrt(a*a) * np.sqrt(b*b)) if a > 0 and b > 0 else 0.0

def exp_diff(a, b):
    s = a + b
    return np.exp(-abs(a - b) / s) if s > 0 else 0.0

def binary_match(a, b):
    return 1.0 if a == b else 0.0

METRIC_FUNCS = {"sym": sym, "jaccard": jaccard, "cosine": cosine_scalar, "exp_diff": exp_diff, "binary_match": binary_match}


def compute_auc(se, sn):
    se, sn = np.array(se, dtype=float), np.array(sn, dtype=float)
    if len(se) == 0 or len(sn) == 0: return 0.5
    if len(se) * len(sn) > 1e8:
        rng = np.random.default_rng(42)
        se = se[rng.choice(len(se), min(len(se), 5000), replace=False)]
        sn = sn[rng.choice(len(sn), min(len(sn), 5000), replace=False)]
    u = np.sum(se[:, None] > sn[None, :]) + 0.5 * np.sum(se[:, None] == sn[None, :])
    return float(u / (len(se) * len(sn)))


def compute_features(G, rng):
    nodes = list(G.nodes())
    deg = dict(G.degree())
    kcore = nx.core_number(G)
    try:
        eigv = nx.eigenvector_centrality(G, max_iter=300, tol=1e-3)
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        eigv = nx.degree_centrality(G)
    clust = nx.clustering(G)
    rand = {n: abs(rng.normal(0, 1)) + 0.01 for n in nodes}
    return {"degree": deg, "kcore": kcore, "eigenvector": eigv, "clustering": clust, "random": rand}


def sample_pairs(G, rng):
    edges = list(G.edges())
    if len(edges) > MAX_EDGES:
        idx = rng.choice(len(edges), MAX_EDGES, replace=False)
        edges = [edges[i] for i in idx]
    nodes = list(G.nodes())
    edge_set = set(G.edges())
    non_edges = []
    target = min(MAX_NONEDGES, len(edges))
    att = 0
    while len(non_edges) < target and att < target * 20:
        u, v = rng.choice(nodes, 2, replace=False)
        if (u, v) not in edge_set and (v, u) not in edge_set:
            non_edges.append((u, v))
        att += 1
    return edges, non_edges


def normalize(vals):
    mn, mx = vals.min(), vals.max()
    if mx - mn < 1e-12: return np.zeros_like(vals)
    return (vals - mn) / (mx - mn)


def compute_edge_diagnostic(G, n_rewirings=EDGE_N_REWIRINGS):
    """EDGE diagnostic: cosine similarity of multi-centrality vectors vs config model null."""
    if G.number_of_nodes() < 30: return None
    nodes = list(G.nodes())
    n = len(nodes)
    node_idx = {nd: i for i, nd in enumerate(nodes)}

    # Use 3 centralities (degree, kcore, clustering) for BOTH real and null
    # Matches edge_hardened.py methodology — eigenvector skipped for consistency
    deg = np.array([G.degree(nd) for nd in nodes], dtype=float)
    kcore = nx.core_number(G)
    kc = np.array([float(kcore[nd]) for nd in nodes])
    clust = nx.clustering(G)
    cl = np.array([clust[nd] for nd in nodes])

    mat = np.column_stack([normalize(deg), normalize(kc), normalize(cl)])
    norms = np.linalg.norm(mat, axis=1)
    norms[norms < 1e-12] = 1.0

    rng = np.random.default_rng(42)
    edges = list(G.edges())
    if len(edges) > 2000:
        idx = rng.choice(len(edges), 2000, replace=False)
        edges = [edges[i] for i in idx]

    ei = np.array([node_idx[u] for u, v in edges])
    ej = np.array([node_idx[v] for u, v in edges])

    def mean_cos(ii, jj, m, nm):
        dots = np.sum(m[ii] * m[jj], axis=1)
        return float(np.mean(dots / (nm[ii] * nm[jj])))

    real_cos = mean_cos(ei, ej, mat, norms)

    null_gaps = []
    for seed in range(n_rewirings):
        G_null = G.copy()
        n_swaps = G_null.number_of_edges() * 3
        try: nx.double_edge_swap(G_null, nswap=n_swaps, max_tries=n_swaps * 10, seed=seed)
        except: pass
        kc_null = nx.core_number(G_null)
        cl_null = nx.clustering(G_null)
        mat_null = np.column_stack([normalize(deg), normalize(np.array([float(kc_null.get(nd, 0)) for nd in nodes])),
                                     normalize(np.array([cl_null.get(nd, 0) for nd in nodes]))])
        norms_null = np.linalg.norm(mat_null, axis=1)
        norms_null[norms_null < 1e-12] = 1.0
        null_edges = list(G_null.edges())
        if len(null_edges) > 2000:
            null_edges = [null_edges[i] for i in rng.choice(len(null_edges), 2000, replace=False)]
        nei = np.array([node_idx[u] for u, v in null_edges])
        nej = np.array([node_idx[v] for u, v in null_edges])
        null_gaps.append(mean_cos(nei, nej, mat_null, norms_null))

    if not null_gaps: return None
    mean_null = np.mean(null_gaps)
    std_null = np.std(null_gaps) if len(null_gaps) > 1 else 0.001
    z = (real_cos - mean_null) / std_null if std_null > 1e-8 else 0
    p = float(np.mean([ng >= real_cos for ng in null_gaps]))
    return {"gap_real": round(real_cos, 5), "gap_null_mean": round(float(mean_null), 5),
            "z_score": round(float(z), 2), "p_value": round(float(p), 3),
            "is_signal": bool(p < 0.05 and real_cos > mean_null)}


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def run():
    print("=" * 85)
    print("UNIFIED ANALYSIS: ALL 67 NETWORKS")
    print("=" * 85)
    t0 = time.time()

    rng = np.random.default_rng(RNG_SEED)
    benchmark = json.load(open(BENCHMARK_PATH))

    # Build network registry: name → (loader_func, corpus, domain, is_bio)
    registry = []

    # Dev set
    for key, info in benchmark.items():
        name = info["name"]
        domain = info.get("domain", "unknown")
        is_bio = domain in {"ppi", "genetic", "connectome", "ecology", "other"} or \
                 any(x in name.lower() for x in ["elegans", "cerevisiae", "sapiens", "melanogaster", "rerio", "drosophila", "food web", "plant-poll"])
        registry.append({"name": name, "corpus": "dev", "domain": domain, "is_bio": is_bio, "loader": "dev"})

    # Held-out 1 — explicit name-to-file mapping
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
    ho1 = json.load(open(RESULTS_DIR / "heldout_validation.json"))
    for net in ho1["per_network"]:
        name = net["network"]
        registry.append({"name": name, "corpus": "held-out-1", "domain": net.get("domain", "unknown"),
                         "is_bio": net.get("domain") in {"biology", "ecology"} or "bio" in net.get("domain", "").lower(),
                         "loader": "heldout1", "source": net.get("source", ""),
                         "file": HO1_FILES.get(name)})

    # Held-out 2 + synthetic
    ho2 = json.load(open(RESULTS_DIR / "expanded_heldout.json"))
    for net in ho2["per_network"]:
        is_synth = net.get("source") == "synthetic" or net.get("domain", "").startswith("synthetic")
        registry.append({"name": net["network"], "corpus": "expanded", "domain": net.get("domain", "unknown"),
                         "is_bio": net.get("domain") in {"biology", "ecology"},
                         "is_synthetic": is_synth, "loader": "heldout2"})

    print(f"Registry: {len(registry)} networks")
    print(f"  Dev: {sum(1 for r in registry if r['corpus']=='dev')}")
    print(f"  Held-out 1: {sum(1 for r in registry if r['corpus']=='held-out-1')}")
    print(f"  Expanded: {sum(1 for r in registry if r['corpus']=='expanded')}")
    print()

    results = []
    loaded = 0
    skipped = 0

    for reg in registry:
        name = reg["name"]
        try:
            if reg["loader"] == "dev":
                G = load_dev_network(name)
            elif reg["loader"] in ("heldout1", "heldout2"):
                search_dir = HELDOUT_DIR if reg["loader"] == "heldout1" else HELDOUT2_DIR
                G = None
                # Normalize name to filename: replace / : with _, append _edges.csv
                name_norm = name.replace('/', '_').replace(':', '_').replace(' ', '_')
                candidate = search_dir / f"{name_norm}_edges.csv"
                if candidate.exists():
                    G = load_heldout_edge_list(candidate)
                # Try without normalization
                if G is None:
                    candidate2 = search_dir / f"{name}_edges.csv"
                    if candidate2.exists():
                        G = load_heldout_edge_list(candidate2)
                # Try fuzzy match as fallback
                if G is None:
                    name_lower = name_norm.lower()
                    for f in sorted(search_dir.glob("*_edges.csv")):
                        stem = f.stem.replace('_edges', '').lower()
                        if stem == name_lower or name_lower == stem:
                            G = load_heldout_edge_list(f)
                            break
                # Synthetic networks: generate them
                if G is None and reg.get("is_synthetic"):
                    G = generate_synthetic(name)
                if G is None:
                    print(f"  SKIP {name}: file not found in {search_dir}")
                    skipped += 1
                    continue
        except Exception as e:
            print(f"  SKIP {name}: {e}")
            skipped += 1
            continue

        if G is None or G.number_of_nodes() < 30:
            skipped += 1
            continue

        if nx.number_of_selfloops(G) > 0:
            G.remove_edges_from(nx.selfloop_edges(G))

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        mean_deg = 2 * n_edges / n_nodes

        # Compute features
        feats = compute_features(G, rng)
        edges, non_edges = sample_pairs(G, rng)

        if len(edges) < 10 or len(non_edges) < 10:
            print(f"  SKIP {name}: insufficient edge/non-edge pairs ({len(edges)}/{len(non_edges)})")
            skipped += 1
            continue

        loaded += 1

        net_result = {
            "network": name, "corpus": reg["corpus"], "domain": reg["domain"],
            "is_bio": reg.get("is_bio", False), "is_synthetic": reg.get("is_synthetic", False),
            "n_nodes": n_nodes, "n_edges": n_edges, "mean_degree": round(mean_deg, 2),
            "features": {},
        }

        # NMI: kcore partition vs Louvain community
        try:
            kcore_part = nx.core_number(G)
            louvain_part = nx.community.louvain_communities(G, seed=42)
            # Convert louvain to node→community dict
            louvain_dict = {}
            for ci, comm in enumerate(louvain_part):
                for nd in comm:
                    louvain_dict[nd] = ci
            # Compute NMI via sklearn if available, else manual
            from sklearn.metrics import normalized_mutual_info_score
            nodes_list = list(G.nodes())
            kc_labels = [kcore_part[nd] for nd in nodes_list]
            lv_labels = [louvain_dict[nd] for nd in nodes_list]
            nmi = normalized_mutual_info_score(kc_labels, lv_labels)
        except Exception:
            nmi = None
        net_result["nmi_kcore_louvain"] = round(float(nmi), 3) if nmi is not None else None

        # Per feature analysis
        for feat_name in FEATURES:
            feat_vals = feats[feat_name]

            # AUC for all 5 metrics
            metric_aucs = {}
            for metric_name, metric_func in METRIC_FUNCS.items():
                se = [metric_func(feat_vals[u], feat_vals[v]) for u, v in edges]
                sn = [metric_func(feat_vals[u], feat_vals[v]) for u, v in non_edges]
                metric_aucs[metric_name] = round(compute_auc(se, sn), 4)

            auc = metric_aucs["sym"]

            # d' (signed)
            se_sym = np.array([sym(feat_vals[u], feat_vals[v]) for u, v in edges])
            sn_sym = np.array([sym(feat_vals[u], feat_vals[v]) for u, v in non_edges])
            mu_e, mu_ne = np.mean(se_sym), np.mean(sn_sym)
            sig_e, sig_ne = np.std(se_sym), np.std(sn_sym)
            pooled_sig = np.sqrt((sig_e**2 + sig_ne**2) / 2)
            dprime = (mu_e - mu_ne) / pooled_sig if pooled_sig > 1e-12 else 0.0
            # Sign: positive if assortative (edges have higher sym)
            signed_dprime = dprime  # already signed by mu_e - mu_ne

            # TER
            tie_tol = 1e-6
            p_e = np.mean(np.abs(se_sym - 1.0) < tie_tol)  # fraction of edge pairs that tie (sym=1)
            # More general: ties where edge sym equals any non-edge sym
            # Simple: fraction where sym = 1.0 (same feature value)
            edge_tie = float(np.mean([1 if feat_vals[u] == feat_vals[v] else 0 for u, v in edges]))
            nonedge_tie = float(np.mean([1 if feat_vals[u] == feat_vals[v] else 0 for u, v in non_edges]))
            ter = edge_tie / nonedge_tie if nonedge_tie > 1e-8 else float('inf')

            # Mixture AUC decomposition
            p_e_mix = edge_tie
            p_ne_mix = nonedge_tie
            auc_mix = p_e_mix * (1 - p_ne_mix) + 0.5 * p_e_mix * p_ne_mix
            # Continuous component
            non_tie_edges = [(u, v) for u, v in edges if feat_vals[u] != feat_vals[v]]
            non_tie_nonedges = [(u, v) for u, v in non_edges if feat_vals[u] != feat_vals[v]]
            if non_tie_edges and non_tie_nonedges:
                se_cont = [sym(feat_vals[u], feat_vals[v]) for u, v in non_tie_edges]
                sn_cont = [sym(feat_vals[u], feat_vals[v]) for u, v in non_tie_nonedges]
                auc_cont = compute_auc(se_cont, sn_cont)
            else:
                auc_cont = 0.5
            auc_mix_total = auc_mix + (1 - p_e_mix) * (1 - p_ne_mix) * auc_cont

            # BL2 variance prediction
            r_vals = np.array([min(feat_vals[u], feat_vals[v]) / max(feat_vals[u], feat_vals[v])
                              if max(feat_vals[u], feat_vals[v]) > 0 else 0
                              for u, v in edges + non_edges])
            var_r = np.var(r_vals) if len(r_vals) > 1 else 0
            e_r = np.mean(r_vals) if len(r_vals) > 0 else 0
            var_bl2 = 4 * var_r / (1 + e_r)**4 if (1 + e_r) > 0 else 0
            all_sym = np.concatenate([se_sym, sn_sym])
            var_emp = float(np.var(all_sym))

            # Newman assortativity for this feature
            try:
                # Set node attribute, then compute assortativity by attribute name
                attr_name = f"_feat_{feat_name}"
                nx.set_node_attributes(G, feat_vals, attr_name)
                r_newman = nx.numeric_assortativity_coefficient(G, attr_name)
            except Exception:
                r_newman = 0.0

            net_result["features"][feat_name] = {
                "auc": auc,
                "metric_aucs": metric_aucs,
                "dprime": round(float(dprime), 4),
                "signed_dprime": round(float(signed_dprime), 4),
                "assortative": bool(mu_e > mu_ne),
                "newman_r": round(float(r_newman), 4),
                "ter": round(float(ter), 4) if ter != float('inf') else "Inf",
                "tie_rate_edge": round(edge_tie, 4),
                "tie_rate_nonedge": round(nonedge_tie, 4),
                "mixture": {
                    "p_e": round(p_e_mix, 4), "p_ne": round(p_ne_mix, 4),
                    "auc_tie": round(float(auc_mix), 4),
                    "auc_continuous": round(float(auc_cont), 4),
                    "auc_predicted": round(float(auc_mix_total), 4),
                    "auc_observed": auc,
                },
                "bl2": {"var_predicted": round(float(var_bl2), 6), "var_observed": round(var_emp, 6)},
            }

        # EDGE diagnostic
        edge_result = compute_edge_diagnostic(G)
        net_result["edge_diagnostic"] = edge_result

        # Summary line
        kc_auc = net_result["features"]["kcore"]["auc"]
        dg_auc = net_result["features"]["degree"]["auc"]
        winner = "kcore" if kc_auc > dg_auc else "degree"
        edge_sig = "EDGE+" if edge_result and edge_result["is_signal"] else "edge-"
        print(f"  {loaded:2d} {name:35s} kc={kc_auc:.3f} dg={dg_auc:.3f} {winner:6s} {edge_sig} [{reg['corpus']}]")

        results.append(net_result)

    # ── Summary ──
    print(f"\n{'='*85}")
    print(f"LOADED: {loaded}, SKIPPED: {skipped}")

    kcore_aucs = [r["features"]["kcore"]["auc"] for r in results]
    degree_aucs = [r["features"]["degree"]["auc"] for r in results]
    kcore_wins = sum(1 for k, d in zip(kcore_aucs, degree_aucs) if k > d)

    print(f"\nK-core vs degree: {kcore_wins}/{loaded} ({100*kcore_wins/loaded:.0f}%)")
    print(f"K-core AUC: mean={np.mean(kcore_aucs):.3f}, median={np.median(kcore_aucs):.3f}")
    print(f"Degree AUC: mean={np.mean(degree_aucs):.3f}, median={np.median(degree_aucs):.3f}")

    # d' pooled
    all_dp = [(r["features"][f]["signed_dprime"], r["features"][f]["auc"])
              for r in results for f in ["kcore", "degree"]]
    dp_vals = [x[0] for x in all_dp]
    auc_vals = [x[1] for x in all_dp]
    rho_dp, p_dp = stats.spearmanr(dp_vals, auc_vals)
    print(f"\nSigned d' vs AUC (pooled, n={len(all_dp)}): rho={rho_dp:.3f}, p={p_dp:.2e}")

    # Assortativity
    kc_r = [r["features"]["kcore"]["newman_r"] for r in results]
    dg_r = [r["features"]["degree"]["newman_r"] for r in results]
    kc_assort = sum(1 for r in kc_r if r > 0)
    dg_disassort = sum(1 for r in dg_r if r < 0)
    print(f"\nAssortativity: kcore mean r={np.mean(kc_r):.3f}, {kc_assort}/{loaded} assortative (r>0)")
    print(f"  degree mean r={np.mean(dg_r):.3f}, {dg_disassort}/{loaded} disassortative (r<0)")

    # NMI
    nmis = [r.get("nmi_kcore_louvain") for r in results if r.get("nmi_kcore_louvain") is not None]
    if nmis:
        print(f"\nNMI(kcore, louvain): mean={np.mean(nmis):.3f}, n={len(nmis)}")

    # TER
    kc_ters = [r["features"]["kcore"]["ter"] for r in results if r["features"]["kcore"]["ter"] != "Inf"]
    dg_ters = [r["features"]["degree"]["ter"] for r in results if r["features"]["degree"]["ter"] != "Inf"]
    print(f"\nTER: kcore mean={np.mean(kc_ters):.2f}, degree mean={np.mean(dg_ters):.2f}")

    # EDGE
    edge_signal = [r for r in results if r["edge_diagnostic"] and r["edge_diagnostic"]["is_signal"]]
    edge_bio = [r for r in edge_signal if r["is_bio"]]
    edge_nonbio = [r for r in edge_signal if not r["is_bio"]]
    total_bio = sum(1 for r in results if r["is_bio"] and r["edge_diagnostic"])
    total_nonbio = sum(1 for r in results if not r["is_bio"] and r["edge_diagnostic"])
    print(f"\nEDGE: {len(edge_signal)}/{loaded} show signal")
    print(f"  Bio: {len(edge_bio)}/{total_bio}")
    print(f"  Non-bio: {len(edge_nonbio)}/{total_nonbio}")
    if total_bio > 0 and total_nonbio > 0:
        table = [[len(edge_bio), total_bio - len(edge_bio)],
                 [len(edge_nonbio), total_nonbio - len(edge_nonbio)]]
        odds, p_fisher = stats.fisher_exact(table, alternative="greater")
        print(f"  Fisher (one-sided): OR={odds:.2f}, p={p_fisher:.4f}")

    # BL2
    bl2_pred = [r["features"][f]["bl2"]["var_predicted"] for r in results for f in FEATURES if r["features"][f]["bl2"]["var_observed"] > 0]
    bl2_obs = [r["features"][f]["bl2"]["var_observed"] for r in results for f in FEATURES if r["features"][f]["bl2"]["var_observed"] > 0]
    if bl2_pred and bl2_obs:
        bl2_r2 = np.corrcoef(bl2_pred, bl2_obs)[0, 1] ** 2
        print(f"\nBL2: R²={bl2_r2:.3f} on {len(bl2_pred)} feature-network pairs")

    # Metric independence
    metric_rhos = []
    for m1 in ["sym", "jaccard", "cosine", "exp_diff"]:
        for m2 in ["sym", "jaccard", "cosine", "exp_diff"]:
            if m1 >= m2: continue
            a1 = [r["features"]["kcore"]["metric_aucs"][m1] for r in results]
            a2 = [r["features"]["kcore"]["metric_aucs"][m2] for r in results]
            rho_m, _ = stats.spearmanr(a1, a2)
            metric_rhos.append(rho_m)
    print(f"\nMetric independence (kcore, 4 continuous): mean cross-rho={np.mean(metric_rhos):.4f}")

    # Save
    out = RESULTS_DIR / "unified_analysis.json"
    # Clean for JSON
    for r in results:
        for f in FEATURES:
            for k, v in r["features"][f].items():
                if isinstance(v, (np.bool_, np.integer)):
                    r["features"][f][k] = bool(v) if isinstance(v, np.bool_) else int(v)
                elif isinstance(v, np.floating):
                    r["features"][f][k] = float(v)

    with open(out, "w") as fp:
        json.dump({
            "n_loaded": loaded,
            "n_skipped": skipped,
            "kcore_wins": kcore_wins,
            "kcore_auc_mean": round(float(np.mean(kcore_aucs)), 4),
            "degree_auc_mean": round(float(np.mean(degree_aucs)), 4),
            "dprime_rho_pooled": round(float(rho_dp), 4),
            "dprime_n_pooled": len(all_dp),
            "per_network": results,
        }, fp, indent=2, default=str)
    print(f"\nResults saved to {out}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
