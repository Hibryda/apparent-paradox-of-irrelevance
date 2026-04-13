#!/usr/bin/env python3
"""Expanded held-out validation: 18 real Netzschleuder networks + 8 synthetic.

Downloads networks from Netzschleuder (CSV edge lists), generates synthetic
benchmarks (LFR, BA, WS, SBM), and runs d' validation on all.

Saves results to results/expanded_heldout.json.
"""
import csv
import io
import json
import os
import sys
import time
import warnings
import zipfile
import urllib.request
import urllib.error
import numpy as np
import networkx as nx
from scipy import stats
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ══════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_DIR = ROOT / "data" / "heldout2"
RESULTS_DIR = ROOT / "results"
HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

MAX_EDGES = 3000
MAX_NONEDGES = 3000
MAX_NODES_SUBSAMPLE = 5000
MIN_NODES = 50
RNG_SEED = 42
FEATURES = ["degree", "kcore"]
TIE_TOL = 1e-6
DOWNLOAD_TIMEOUT = 15

# Network slug -> list of alternate slugs to try
REAL_NETWORKS = [
    # Tier 1: missing domains
    {"name": "uni_email", "domain": "communication",
     "slugs": ["uni_email"]},
    {"name": "dnc", "domain": "communication",
     "slugs": ["dnc"]},
    {"name": "cora", "domain": "citation",
     "slugs": ["cora"]},
    {"name": "dblp_cite", "domain": "citation",
     "slugs": ["dblp_cite", "dblp-cite"]},
    {"name": "jung", "domain": "software",
     "slugs": ["jung", "jung-j"]},
    {"name": "linux", "domain": "software",
     "slugs": ["linux"]},
    {"name": "eu_airlines", "domain": "transportation",
     "slugs": ["eu_airlines"]},
    {"name": "euroroad", "domain": "transportation",
     "slugs": ["euroroad"]},
    # Tier 2: strengthen thin
    {"name": "jazz_collab", "domain": "collaboration",
     "slugs": ["jazz_musicians", "jazz_collab", "jazz"]},
    {"name": "new_zealand_collab", "domain": "collaboration",
     "slugs": ["new_zealand_collab", "nz_collab"]},
    {"name": "internet_as", "domain": "technological",
     "slugs": ["internet_as", "as_relationships", "as-relationships"]},
    {"name": "polblogs", "domain": "political",
     "slugs": ["polblogs"]},
    {"name": "foodweb_little_rock", "domain": "ecological",
     "slugs": ["foodweb_littlerock", "foodweb_little_rock", "little_rock"]},
    {"name": "celegans_metabolic", "domain": "ecological",
     "slugs": ["celegans_metabolic", "c_elegans_metabolic"]},
    # Tier 3: size/density
    {"name": "chicago_road", "domain": "transportation",
     "slugs": ["chicago", "road_chicago", "chicago_road"]},
    {"name": "fao_trade", "domain": "economic",
     "slugs": ["fao_trade", "fao-trade"]},
    # Extra to fill gaps if some fail
    {"name": "dolphins", "domain": "social",
     "slugs": ["dolphins"]},
    {"name": "karate", "domain": "social",
     "slugs": ["karate"]},
]


# ══════════════════════════════════════════════════════════════════════
# NETZSCHLEUDER DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

def fetch_bytes(url, timeout=DOWNLOAD_TIMEOUT):
    """Fetch URL content as bytes, return None on failure."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "symmetry-balance-theory/2.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        return None


def parse_csv_to_graph(content):
    """Parse CSV/TSV content into an undirected networkx Graph.

    Auto-detects delimiter (comma or tab). First two columns are source, target.
    Skips comment lines (# or %).
    """
    G = nx.Graph()
    lines = content.strip().split("\n")
    if not lines:
        return G

    # Filter comment lines
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not stripped.startswith("%"):
            clean_lines.append(stripped)
    if not clean_lines:
        return G

    # Detect separator from first clean line
    first = clean_lines[0]
    sep = "\t" if "\t" in first else ","

    reader = csv.reader(io.StringIO("\n".join(clean_lines)), delimiter=sep)
    header = next(reader, None)
    if header is None:
        return G

    # Detect header: if first two fields look like column names, skip; else treat as data
    header_lower = [h.strip().lower().strip("#").strip('"') for h in header]
    is_header = any(
        h in ("source", "target", "src", "tgt", "node1", "node2", "from", "to", "i", "j")
        for h in header_lower
    )

    src_col, tgt_col = 0, 1
    if is_header:
        for idx, h in enumerate(header_lower):
            if h in ("source", "src", "node1", "from", "i"):
                src_col = idx
            elif h in ("target", "tgt", "node2", "to", "j"):
                tgt_col = idx
    else:
        # Not a header — treat first line as data
        if len(header) > 1:
            try:
                u, v = header[0].strip(), header[1].strip()
                try:
                    u = int(u)
                except ValueError:
                    pass
                try:
                    v = int(v)
                except ValueError:
                    pass
                if u != v:
                    G.add_edge(u, v)
            except (ValueError, IndexError):
                pass

    for row in reader:
        if len(row) <= max(src_col, tgt_col):
            continue
        try:
            u = row[src_col].strip().strip('"')
            v = row[tgt_col].strip().strip('"')
            try:
                u = int(u)
            except ValueError:
                pass
            try:
                v = int(v)
            except ValueError:
                pass
            if u != v:
                G.add_edge(u, v)
        except (ValueError, IndexError):
            continue

    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def download_netzschleuder_csv(slug):
    """Try to download a network from Netzschleuder as CSV.

    Tries multiple URL patterns:
    1. https://networks.skewed.de/net/{slug}/files/{slug}.csv  (plain CSV)
    2. https://networks.skewed.de/net/{slug}/files/{slug}.csv.zip  (zipped)

    Returns graph content string or None.
    """
    # Try plain CSV first
    url_csv = f"https://networks.skewed.de/net/{slug}/files/{slug}.csv"
    data = fetch_bytes(url_csv)
    if data is not None:
        try:
            content = data.decode("utf-8")
            # Quick sanity check: should have multiple lines
            if content.count("\n") >= 2:
                return content
        except UnicodeDecodeError:
            pass

    # Try zip
    url_zip = f"https://networks.skewed.de/net/{slug}/files/{slug}.csv.zip"
    data = fetch_bytes(url_zip)
    if data is not None:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                # Look for edges.csv or any csv file
                edges_file = None
                for name in names:
                    if "edges" in name.lower() and name.endswith(".csv"):
                        edges_file = name
                        break
                if edges_file is None:
                    # Just take the first CSV
                    for name in names:
                        if name.endswith(".csv"):
                            edges_file = name
                            break
                if edges_file is not None:
                    return zf.read(edges_file).decode("utf-8")
        except Exception:
            pass

    return None


def try_download_network(net_info):
    """Try all slug variants for a network. Returns (Graph, slug_used) or (None, None)."""
    for slug in net_info["slugs"]:
        cache_path = HELDOUT_DIR / f"{slug}_edges.csv"

        if cache_path.exists():
            content = cache_path.read_text()
        else:
            content = download_netzschleuder_csv(slug)
            if content is not None:
                cache_path.write_text(content)

        if content is None:
            continue

        G = parse_csv_to_graph(content)
        if G.number_of_nodes() == 0:
            continue

        # Take largest connected component
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()

        n = G.number_of_nodes()
        if n < MIN_NODES:
            continue

        # Subsample if too large
        if n > MAX_NODES_SUBSAMPLE:
            nodes = sorted(G.nodes())
            rng = np.random.RandomState(RNG_SEED)
            keep = set(rng.choice(nodes, MAX_NODES_SUBSAMPLE, replace=False))
            G = G.subgraph(keep).copy()
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            if G.number_of_nodes() < MIN_NODES:
                continue

        return G, slug

    return None, None


def download_all_networks():
    """Download all 18 real networks from Netzschleuder."""
    print("=" * 70)
    print("PART 1: Downloading real networks from Netzschleuder")
    print("=" * 70)

    loaded = []
    failed = []

    for i, net_info in enumerate(REAL_NETWORKS):
        name = net_info["name"]
        domain = net_info["domain"]
        print(f"\n  [{i+1}/{len(REAL_NETWORKS)}] {name} ({domain})...", end=" ", flush=True)

        G, slug_used = try_download_network(net_info)

        if G is not None:
            n, m = G.number_of_nodes(), G.number_of_edges()
            print(f"OK (slug={slug_used}, n={n}, m={m})")
            loaded.append({
                "name": name,
                "slug": slug_used,
                "domain": domain,
                "graph": G,
                "n_nodes": n,
                "n_edges": m,
                "source": "netzschleuder",
            })
        else:
            print(f"FAILED (tried: {net_info['slugs']})")
            failed.append(name)

    print(f"\n  Downloaded: {len(loaded)}/{len(REAL_NETWORKS)}")
    if failed:
        print(f"  Failed: {', '.join(failed)}")

    return loaded, failed


# ══════════════════════════════════════════════════════════════════════
# SYNTHETIC NETWORK GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_synthetic_networks():
    """Generate 8 synthetic networks: 3 LFR, 2 BA, 2 WS, 1 SBM."""
    print("\n" + "=" * 70)
    print("PART 2: Generating synthetic networks")
    print("=" * 70)

    synth = []

    # LFR benchmarks: mu = {0.1, 0.3, 0.5}
    for mu in [0.1, 0.3, 0.5]:
        print(f"\n  LFR mu={mu}...", end=" ", flush=True)
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=500,
                tau1=3,
                tau2=1.5,
                mu=mu,
                average_degree=15,
                min_community=30,
                max_degree=50,
                seed=RNG_SEED,
            )
            # Remove community attribute
            for node in G.nodes():
                if "community" in G.nodes[node]:
                    del G.nodes[node]["community"]
            G.remove_edges_from(nx.selfloop_edges(G))
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            n, m = G.number_of_nodes(), G.number_of_edges()
            print(f"OK (n={n}, m={m})")
            synth.append({
                "name": f"LFR_mu{mu}",
                "domain": "synthetic_LFR",
                "graph": G,
                "n_nodes": n,
                "n_edges": m,
                "source": "synthetic",
            })
        except Exception as e:
            print(f"FAILED: {e}")

    # Barabasi-Albert: m = {2, 5}
    for m_ba in [2, 5]:
        print(f"\n  BA m={m_ba}...", end=" ", flush=True)
        G = nx.barabasi_albert_graph(1000, m_ba, seed=RNG_SEED)
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"OK (n={n}, m={m})")
        synth.append({
            "name": f"BA_m{m_ba}",
            "domain": "synthetic_BA",
            "graph": G,
            "n_nodes": n,
            "n_edges": m,
            "source": "synthetic",
        })

    # Watts-Strogatz: p = {0.01, 0.5}
    for p_ws in [0.01, 0.5]:
        print(f"\n  WS p={p_ws}...", end=" ", flush=True)
        G = nx.watts_strogatz_graph(1000, 8, p_ws, seed=RNG_SEED)
        n, m = G.number_of_nodes(), G.number_of_edges()
        print(f"OK (n={n}, m={m})")
        synth.append({
            "name": f"WS_p{p_ws}",
            "domain": "synthetic_WS",
            "graph": G,
            "n_nodes": n,
            "n_edges": m,
            "source": "synthetic",
        })

    # Stochastic Block Model: 3 blocks of 200, p_in=0.1, p_out=0.005
    print(f"\n  SBM 3x200...", end=" ", flush=True)
    sizes = [200, 200, 200]
    p_matrix = [
        [0.1, 0.005, 0.005],
        [0.005, 0.1, 0.005],
        [0.005, 0.005, 0.1],
    ]
    G = nx.stochastic_block_model(sizes, p_matrix, seed=RNG_SEED)
    G.remove_edges_from(nx.selfloop_edges(G))
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    # Remove block attribute
    for node in G.nodes():
        if "block" in G.nodes[node]:
            del G.nodes[node]["block"]
    n, m = G.number_of_nodes(), G.number_of_edges()
    print(f"OK (n={n}, m={m})")
    synth.append({
        "name": "SBM_3x200",
        "domain": "synthetic_SBM",
        "graph": G,
        "n_nodes": n,
        "n_edges": m,
        "source": "synthetic",
    })

    return synth


# ══════════════════════════════════════════════════════════════════════
# CORE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def sym_vec(a, b):
    """Vectorised sym(a, b) = 1 - |a-b|/(a+b). Returns NaN where a+b==0."""
    s = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, 1.0 - np.abs(a - b) / s, np.nan)


def sample_pairs(G, rng):
    """Sample edges and non-edges, capped at MAX_EDGES/MAX_NONEDGES."""
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

    Adds epsilon 0.001 to avoid zero-sum pairs for integer features.
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


def dprime_signed(sym_e, sym_ne):
    """Signed d': positive when edges have higher sym (assortative).

    d' = (mu_e - mu_ne) / sqrt(0.5*(var_e + var_ne))
    Sign follows mean_edge - mean_nonedge.
    """
    if len(sym_e) < 5 or len(sym_ne) < 5:
        return np.nan
    mu_e, mu_ne = np.mean(sym_e), np.mean(sym_ne)
    var_e = np.var(sym_e, ddof=1)
    var_ne = np.var(sym_ne, ddof=1)
    denom = np.sqrt(0.5 * (var_e + var_ne))
    if denom < 1e-15:
        return np.nan
    sign = 1.0 if mu_e >= mu_ne else -1.0
    return float(sign * abs(mu_e - mu_ne) / denom)


def safe_spearman(x, y):
    """Spearman on finite pairs only. Returns (rho, p, n)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < 5:
        return np.nan, np.nan, n
    rho, p = stats.spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def bootstrap_spearman(x, y, B=500, seed=42):
    """Bootstrap CI for Spearman rho. Returns (lo, hi, median)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.RandomState(seed)
    rhos = []
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(idx)) < 4:
            continue
        r, _ = stats.spearmanr(x[idx], y[idx])
        if np.isfinite(r):
            rhos.append(r)
    if len(rhos) < 10:
        return np.nan, np.nan, np.nan
    rhos = np.sort(rhos)
    lo = float(np.percentile(rhos, 2.5))
    hi = float(np.percentile(rhos, 97.5))
    med = float(np.median(rhos))
    return lo, hi, med


# ══════════════════════════════════════════════════════════════════════
# NETWORK-LEVEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_network(net_info, rng):
    """Analyse one network: compute AUC and signed d' for degree and kcore."""
    G = net_info["graph"]
    name = net_info["name"]

    deg = dict(G.degree())
    kcore = nx.core_number(G)
    features = {"degree": deg, "kcore": kcore}

    edges, non_edges = sample_pairs(G, rng)

    if len(edges) < 20 or len(non_edges) < 20:
        print(f"    {name}: too few pairs — skip")
        return None

    result = {
        "network": name,
        "domain": net_info["domain"],
        "source": net_info["source"],
        "n_nodes": net_info["n_nodes"],
        "n_edges": net_info["n_edges"],
        "n_sampled_edges": len(edges),
        "n_sampled_nonedges": len(non_edges),
        "features": {},
    }

    # Degree assortativity
    try:
        assort = nx.degree_assortativity_coefficient(G)
        result["degree_assortativity"] = float(assort) if np.isfinite(assort) else None
    except Exception:
        result["degree_assortativity"] = None

    for feat_name in FEATURES:
        feat_dict = features[feat_name]
        sym_e, sym_ne = compute_sym_arrays(edges, non_edges, feat_dict)

        if len(sym_e) < 10 or len(sym_ne) < 10:
            result["features"][feat_name] = {
                "auc": None, "dprime_signed": None
            }
            continue

        auc = auc_mann_whitney(sym_e, sym_ne)
        d_signed = dprime_signed(sym_e, sym_ne)

        mu_e, mu_ne = float(np.mean(sym_e)), float(np.mean(sym_ne))
        tie_e = float(np.mean(np.abs(sym_e - 1.0) < TIE_TOL))
        tie_ne = float(np.mean(np.abs(sym_ne - 1.0) < TIE_TOL))

        result["features"][feat_name] = {
            "auc": float(auc) if np.isfinite(auc) else None,
            "dprime_signed": float(d_signed) if np.isfinite(d_signed) else None,
            "sym_mean_edge": mu_e,
            "sym_mean_nonedge": mu_ne,
            "assortative": bool(mu_e > mu_ne),
            "tie_rate_edge": tie_e,
            "tie_rate_nonedge": tie_ne,
        }

    return result


# ══════════════════════════════════════════════════════════════════════
# AGGREGATE ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def aggregate_results(per_network):
    """Compute aggregate statistics across networks."""
    # Collect per-feature vectors
    kcore_aucs, degree_aucs = [], []
    all_aucs, all_dprimes = [], []

    kcore_wins = 0
    degree_wins = 0
    ties = 0

    for r in per_network:
        kf = r["features"].get("kcore", {})
        df = r["features"].get("degree", {})

        k_auc = kf.get("auc")
        d_auc = df.get("auc")
        k_dp = kf.get("dprime_signed")
        d_dp = df.get("dprime_signed")

        if k_auc is not None:
            kcore_aucs.append(k_auc)
        if d_auc is not None:
            degree_aucs.append(d_auc)

        # Only add paired (auc, dprime) to correlation vectors
        if k_auc is not None and k_dp is not None:
            all_aucs.append(k_auc)
            all_dprimes.append(k_dp)
        if d_auc is not None and d_dp is not None:
            all_aucs.append(d_auc)
            all_dprimes.append(d_dp)

        # Win rate: kcore vs degree
        if k_auc is not None and d_auc is not None:
            if k_auc > d_auc + 0.001:
                kcore_wins += 1
            elif d_auc > k_auc + 0.001:
                degree_wins += 1
            else:
                ties += 1

    total_compared = kcore_wins + degree_wins + ties

    # Spearman d' vs AUC (across all feature-network pairs)
    rho, p, n_pairs = safe_spearman(all_dprimes, all_aucs)
    lo, hi, med = bootstrap_spearman(all_dprimes, all_aucs)

    # Separate for real vs synthetic
    real_aucs, real_dprimes = [], []
    synth_aucs, synth_dprimes = [], []
    for r in per_network:
        for feat_name in FEATURES:
            feat = r["features"].get(feat_name, {})
            auc = feat.get("auc")
            dp = feat.get("dprime_signed")
            if auc is not None and dp is not None:
                if r["source"] == "netzschleuder":
                    real_aucs.append(auc)
                    real_dprimes.append(dp)
                else:
                    synth_aucs.append(auc)
                    synth_dprimes.append(dp)

    rho_real, p_real, n_real = safe_spearman(real_dprimes, real_aucs)
    rho_synth, p_synth, n_synth = safe_spearman(synth_dprimes, synth_aucs)

    return {
        "n_networks_analysed": len(per_network),
        "kcore_vs_degree": {
            "kcore_wins": kcore_wins,
            "degree_wins": degree_wins,
            "ties": ties,
            "total": total_compared,
            "kcore_win_rate": round(kcore_wins / total_compared, 3) if total_compared > 0 else None,
        },
        "auc_means": {
            "kcore_mean": round(float(np.mean(kcore_aucs)), 3) if kcore_aucs else None,
            "degree_mean": round(float(np.mean(degree_aucs)), 3) if degree_aucs else None,
            "kcore_std": round(float(np.std(kcore_aucs)), 3) if kcore_aucs else None,
            "degree_std": round(float(np.std(degree_aucs)), 3) if degree_aucs else None,
        },
        "dprime_vs_auc_all": {
            "spearman_rho": round(rho, 3) if np.isfinite(rho) else None,
            "p_value": round(p, 4) if np.isfinite(p) else None,
            "n_pairs": n_pairs,
            "bootstrap_95ci": [round(lo, 3) if np.isfinite(lo) else None,
                               round(hi, 3) if np.isfinite(hi) else None],
            "bootstrap_median": round(med, 3) if np.isfinite(med) else None,
        },
        "dprime_vs_auc_real": {
            "spearman_rho": round(rho_real, 3) if np.isfinite(rho_real) else None,
            "p_value": round(p_real, 4) if np.isfinite(p_real) else None,
            "n_pairs": n_real,
        },
        "dprime_vs_auc_synthetic": {
            "spearman_rho": round(rho_synth, 3) if np.isfinite(rho_synth) else None,
            "p_value": round(p_synth, 4) if np.isfinite(p_synth) else None,
            "n_pairs": n_synth,
        },
    }


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED)

    # Part 1: Download real networks
    real_networks, failed = download_all_networks()

    # Part 2: Generate synthetic networks
    synthetic_networks = generate_synthetic_networks()

    all_networks = real_networks + synthetic_networks
    n_real = len(real_networks)
    n_synth = len(synthetic_networks)
    print(f"\n  Total networks: {len(all_networks)} (real: {n_real}, synthetic: {n_synth})")

    if len(all_networks) == 0:
        print("ERROR: No networks available. Aborting.")
        sys.exit(1)

    # Part 3: Validate
    print("\n" + "=" * 70)
    print("PART 3: Validation — d' and AUC computation")
    print("=" * 70)

    per_network = []
    for i, net in enumerate(all_networks):
        print(f"\n  [{i+1}/{len(all_networks)}] {net['name']} (n={net['n_nodes']}, m={net['n_edges']})...")
        result = analyse_network(net, rng)
        if result is not None:
            per_network.append(result)

            # Print quick summary
            for feat in FEATURES:
                fd = result["features"].get(feat, {})
                auc = fd.get("auc")
                dp = fd.get("dprime_signed")
                auc_str = f"{auc:.3f}" if auc is not None else "N/A"
                dp_str = f"{dp:+.3f}" if dp is not None else "N/A"
                print(f"    {feat:8s}: AUC={auc_str}  d'={dp_str}")

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    agg = aggregate_results(per_network)

    # Print summary table
    print(f"\n  Networks analysed: {agg['n_networks_analysed']}")
    kd = agg["kcore_vs_degree"]
    print(f"\n  Kcore vs Degree win rate:")
    print(f"    Kcore wins:  {kd['kcore_wins']}/{kd['total']}")
    print(f"    Degree wins: {kd['degree_wins']}/{kd['total']}")
    print(f"    Ties:        {kd['ties']}/{kd['total']}")
    print(f"    Kcore win rate: {kd['kcore_win_rate']}")

    am = agg["auc_means"]
    print(f"\n  Mean AUC:")
    print(f"    Kcore:  {am['kcore_mean']} +/- {am['kcore_std']}")
    print(f"    Degree: {am['degree_mean']} +/- {am['degree_std']}")

    da = agg["dprime_vs_auc_all"]
    print(f"\n  d' vs AUC (all):")
    print(f"    Spearman rho: {da['spearman_rho']} (p={da['p_value']})")
    print(f"    95% CI: [{da['bootstrap_95ci'][0]}, {da['bootstrap_95ci'][1]}]")
    print(f"    n pairs: {da['n_pairs']}")

    dar = agg["dprime_vs_auc_real"]
    print(f"\n  d' vs AUC (real only):")
    print(f"    Spearman rho: {dar['spearman_rho']} (p={dar['p_value']})")
    print(f"    n pairs: {dar['n_pairs']}")

    das = agg["dprime_vs_auc_synthetic"]
    print(f"\n  d' vs AUC (synthetic only):")
    print(f"    Spearman rho: {das['spearman_rho']} (p={das['p_value']})")
    print(f"    n pairs: {das['n_pairs']}")

    # Per-network detail table
    print("\n" + "-" * 90)
    print(f"  {'Network':<28s} {'Domain':<16s} {'n':>5s} {'m':>6s} "
          f"{'AUC_k':>6s} {'AUC_d':>6s} {'d_k':>7s} {'d_d':>7s} {'win':>5s}")
    print("-" * 90)
    for r in per_network:
        kf = r["features"].get("kcore", {})
        df = r["features"].get("degree", {})
        k_auc = kf.get("auc")
        d_auc = df.get("auc")
        k_dp = kf.get("dprime_signed")
        d_dp = df.get("dprime_signed")

        ka = f"{k_auc:.3f}" if k_auc is not None else "  N/A"
        da_ = f"{d_auc:.3f}" if d_auc is not None else "  N/A"
        kd_ = f"{k_dp:+.3f}" if k_dp is not None else "   N/A"
        dd_ = f"{d_dp:+.3f}" if d_dp is not None else "   N/A"

        if k_auc is not None and d_auc is not None:
            win = "kcore" if k_auc > d_auc + 0.001 else ("deg" if d_auc > k_auc + 0.001 else "tie")
        else:
            win = "?"

        print(f"  {r['network']:<28s} {r['domain']:<16s} {r['n_nodes']:5d} {r['n_edges']:6d} "
              f"{ka:>6s} {da_:>6s} {kd_:>7s} {dd_:>7s} {win:>5s}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    output = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_real": n_real,
            "n_synthetic": n_synth,
            "n_real_failed": len(failed),
            "failed_networks": failed,
            "elapsed_seconds": round(elapsed, 1),
        },
        "aggregate": agg,
        "per_network": per_network,
    }

    out_path = RESULTS_DIR / "expanded_heldout.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {out_path}")


if __name__ == "__main__":
    main()
