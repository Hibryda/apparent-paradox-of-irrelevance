#!/usr/bin/env python3
"""Held-out validation of the d' -> AUC relationship.

Downloads fresh networks from Netzschleuder (or falls back to LFR benchmarks),
computes sym-based d' and AUC for degree and kcore features, and tests whether
the development-set finding (d' rho=0.956) generalises.

Two parts:
  Part 1: Download 10 diverse networks from Netzschleuder API
  Part 2: Validate d' vs AUC on held-out networks

Saves results to results/heldout_validation.json.
"""
import json
import os
import sys
import time
import warnings
import csv
import io
import zipfile
import tempfile
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
HELDOUT_DIR = ROOT / "data" / "heldout"
RESULTS_DIR = ROOT / "results"
HELDOUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════

MAX_EDGES = 3000
MAX_NONEDGES = 3000
RNG_SEED = 42
FEATURES = ["degree", "kcore"]
TIE_TOL = 1e-6

# Netzschleuder slugs already used in the 32-network dev benchmark.
# These are approximate — we also string-match on common substrings.
DEV_BENCHMARK_SLUGS = {
    "bible_nouns", "netscience", "openflights", "us_airports", "ieee_300",
    "power", "contiguous_usa", "euroroad", "product_space",
    "celegans_2019",  # c_elegans_synapse_(2019)
    "budapest_connectome",  # budapest_human_connectome
}

# Hand-picked diverse candidates: (slug, net_id, domain)
# Chosen from the API listing, ensuring domain diversity and 50 < n < 10000.
# net_id is the sub-network ID from the API's "nets" field.
CANDIDATE_NETWORKS = [
    # Social (2)
    ("add_health", "comm1", "social"),            # n=71, m=305, Social/Offline
    ("ego_social", "facebook_0", "social"),       # n=333, m=2519, Social/Online
    ("ugandan_village", "friendship-1", "social"),# n=203, m=600, Social/Offline
    ("spanish_highschools", "1", "social"),        # n=409, m=8557, Social/Offline
    ("facebook_organizations", "S1", "social"),    # n=320, m=2369, Social/Online
    # Technological (2)
    ("route_views", "19971108", "tech"),           # n=3015, m=5539, Internet AS
    ("webkb", "webkb_wisconsin_link1", "tech"),   # n=300, m=1155, Web graph
    ("edit_wikiquote", "af", "tech"),              # n=1438, m=3450, Wiki edit
    # Biological (2)
    ("ecoli_transcription", "v1.0", "bio"),       # n=424, m=577, Gene regulation
    ("fresh_webs", "AkatoreA", "bio"),            # n=85, m=227, Food web
    ("kegg_metabolic", "aae", "bio"),             # n=926, m=2417, Metabolic
    ("celegans_interactomes", "wi2007", "bio"),   # n=1496, m=1816, Protein interactions
    ("malaria_genes", "HVR_1", "bio"),            # n=307, m=2812, Genetic
    # Economic (2)
    ("eu_procurements_alt", "AT_2008", "econ"),   # n=2271, m=2324, EU procurement
    ("faculty_hiring", "computer_science", "econ"), # n=206, m=4988, Employment
    ("board_directors", "net2m_2002-05-01", "econ"), # n=1217, m=1130, Affiliation
    # Infrastructure (2)
    ("urban_streets", "ahmedabad", "infra"),      # n=2870, m=4387, Roads
    ("openstreetmap", "01-AL-cities-street_networks:0100124_Abbeville", "infra"),  # n=351, m=866
    ("fullerene_structures", "C1500", "infra"),   # n=1500, m=2250, Chemical structure
    ("tree-of-life", "394", "infra"),             # n=1096, m=4863, Phylogenetic
]

DOMAIN_QUOTAS = {"social": 2, "tech": 2, "bio": 2, "econ": 2, "infra": 2}


# ══════════════════════════════════════════════════════════════════════
# NETZSCHLEUDER DOWNLOAD
# ══════════════════════════════════════════════════════════════════════

def fetch_bytes(url, timeout=30):
    """Fetch URL content as bytes, return None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "symmetry-balance-theory/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except Exception as e:
        print(f"    FAIL fetching {url}: {e}")
        return None


def download_netzschleuder_graph(slug, net_id):
    """Download csv.zip for a Netzschleuder network and return a networkx Graph.

    URL pattern: https://networks.skewed.de/net/{slug}/files/{net_id}.csv.zip
    Inside the zip: edges.csv (and possibly nodes.csv, gprops.csv)

    Returns (G, n_nodes, n_edges) or (None, 0, 0) on failure.
    """
    cache_path = HELDOUT_DIR / f"{slug}_{net_id.replace('/', '_').replace(':', '_')}_edges.csv"

    # Check if already cached (extracted edges.csv)
    if cache_path.exists():
        print(f"    Using cached {cache_path.name}")
        content = cache_path.read_text()
    else:
        # Download the zip
        url = f"https://networks.skewed.de/net/{slug}/files/{net_id}.csv.zip"
        data = fetch_bytes(url)
        if data is None:
            return None, 0, 0

        # Extract edges.csv from zip
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
                edges_file = None
                for name in names:
                    if name.endswith("edges.csv") or name == "edges.csv":
                        edges_file = name
                        break
                if edges_file is None:
                    print(f"    No edges.csv in zip (files: {names})")
                    return None, 0, 0
                content = zf.read(edges_file).decode("utf-8")
                cache_path.write_text(content)
        except Exception as e:
            print(f"    Failed to extract zip: {e}")
            return None, 0, 0

    # Parse CSV into graph
    G = nx.Graph()
    lines = content.strip().split("\n")
    if not lines:
        return None, 0, 0

    # Detect separator
    first_line = lines[0]
    sep = "," if "," in first_line else "\t"

    reader = csv.reader(io.StringIO(content), delimiter=sep)
    header = next(reader, None)
    if header is None:
        return None, 0, 0

    # Find source and target columns
    header_lower = [h.strip().lower().strip("#") for h in header]
    src_col, tgt_col = None, None

    for i, h in enumerate(header_lower):
        if h in ("source", "src", "node1", "from", "i"):
            src_col = i
        elif h in ("target", "tgt", "node2", "to", "j"):
            tgt_col = i

    # Fallback: first two columns
    if src_col is None or tgt_col is None:
        src_col, tgt_col = 0, 1

    for row in reader:
        if len(row) <= max(src_col, tgt_col):
            continue
        try:
            u = row[src_col].strip()
            v = row[tgt_col].strip()
            try:
                u = int(u)
            except ValueError:
                pass
            try:
                v = int(v)
            except ValueError:
                pass
            if u != v:  # Skip self-loops
                G.add_edge(u, v)
        except (ValueError, IndexError):
            continue

    # Remove self-loops explicitly (belt-and-suspenders)
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0:
        return None, 0, 0

    # Take largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    # Check size constraints
    n = G.number_of_nodes()
    if n < 50 or n > 10000:
        print(f"    n={n} outside [50, 10000] — skipping")
        return None, 0, 0

    return G, G.number_of_nodes(), G.number_of_edges()


def download_diverse_networks():
    """Download 10 diverse networks from Netzschleuder, respecting domain quotas."""
    print("=" * 70)
    print("PART 1: Downloading held-out networks from Netzschleuder")
    print("=" * 70)

    loaded = []
    domain_counts = {d: 0 for d in DOMAIN_QUOTAS}

    for slug, net_id, domain in CANDIDATE_NETWORKS:
        # Check domain quota
        if domain_counts[domain] >= DOMAIN_QUOTAS[domain]:
            continue

        # Check not in dev benchmark
        if slug in DEV_BENCHMARK_SLUGS:
            print(f"  Skipping {slug} — in dev benchmark")
            continue

        # Check if we already have 10
        if len(loaded) >= 10:
            break

        print(f"\n  [{len(loaded)+1}/10] {slug}/{net_id} ({domain})...")
        G, n, m = download_netzschleuder_graph(slug, net_id)

        if G is not None:
            print(f"    OK: n={n}, m={m}")
            loaded.append({
                "slug": f"{slug}/{net_id}",
                "domain": domain,
                "description": f"Netzschleuder: {slug}/{net_id}",
                "graph": G,
                "n_nodes": n,
                "n_edges": m,
                "source": "netzschleuder",
            })
            domain_counts[domain] += 1
        else:
            print(f"    FAILED or wrong size — skipping")

    return loaded, domain_counts


def generate_lfr_fallbacks(n_existing, rng):
    """Generate LFR benchmark graphs to fill gaps if Netzschleuder didn't yield enough."""
    need = 10 - n_existing
    if need <= 0:
        return []

    print(f"\n  Generating {need} LFR benchmark graphs as fallback...")
    lfr_graphs = []
    mus = [0.1, 0.2, 0.3, 0.4, 0.5]

    for i in range(min(need, len(mus))):
        mu = mus[i]
        try:
            G = nx.generators.community.LFR_benchmark_graph(
                n=500,
                tau1=3,
                tau2=1.5,
                mu=mu,
                average_degree=15,
                min_community=20,
                max_community=100,
                seed=RNG_SEED + i,
            )
            # Remove community attribute and self-loops
            for node in G.nodes():
                if "community" in G.nodes[node]:
                    del G.nodes[node]["community"]
            G.remove_edges_from(nx.selfloop_edges(G))

            # Take LCC
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()

            n, m = G.number_of_nodes(), G.number_of_edges()
            print(f"    LFR mu={mu}: n={n}, m={m}")
            lfr_graphs.append({
                "slug": f"lfr_mu{mu}",
                "domain": "synthetic",
                "description": f"LFR benchmark mu={mu}",
                "graph": G,
                "n_nodes": n,
                "n_edges": m,
                "source": "lfr_benchmark",
            })
        except Exception as e:
            print(f"    LFR mu={mu} failed: {e}")

    return lfr_graphs


# ══════════════════════════════════════════════════════════════════════
# CORE ANALYSIS (mirrors tie_effect_corrected.py logic)
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


def dprime(sym_e, sym_ne):
    """Fisher discriminant d' = |mu_e - mu_ne| / sqrt(0.5*(var_e + var_ne))."""
    if len(sym_e) < 5 or len(sym_ne) < 5:
        return np.nan
    mu_e, mu_ne = np.mean(sym_e), np.mean(sym_ne)
    var_e = np.var(sym_e, ddof=1)
    var_ne = np.var(sym_ne, ddof=1)
    denom = np.sqrt(0.5 * (var_e + var_ne))
    if denom < 1e-15:
        return np.nan
    return float(abs(mu_e - mu_ne) / denom)


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
# MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════════

def analyse_network(net_info, rng):
    """Analyse one network: compute AUC and d' for degree and kcore features."""
    G = net_info["graph"]
    slug = net_info["slug"]

    # Compute features
    deg = dict(G.degree())
    kcore = nx.core_number(G)

    features = {"degree": deg, "kcore": kcore}

    # Sample pairs
    edges, non_edges = sample_pairs(G, rng)

    if len(edges) < 20 or len(non_edges) < 20:
        print(f"    {slug}: too few pairs (edges={len(edges)}, non_edges={len(non_edges)}) — skip")
        return None

    results = {
        "network": slug,
        "domain": net_info["domain"],
        "source": net_info["source"],
        "n_nodes": net_info["n_nodes"],
        "n_edges": net_info["n_edges"],
        "n_sampled_edges": len(edges),
        "n_sampled_nonedges": len(non_edges),
        "features": {},
    }

    # Network-level assortativity
    try:
        assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        assort = None
    results["degree_assortativity"] = float(assort) if assort is not None and np.isfinite(assort) else None

    for feat_name in FEATURES:
        feat_dict = features[feat_name]
        sym_e, sym_ne = compute_sym_arrays(edges, non_edges, feat_dict)

        if len(sym_e) < 10 or len(sym_ne) < 10:
            results["features"][feat_name] = {"auc": None, "dprime": None}
            continue

        auc = auc_mann_whitney(sym_e, sym_ne)
        d = dprime(sym_e, sym_ne)

        # Signed d': positive when edges have higher sym (assortative),
        # negative when non-edges have higher sym (disassortative)
        mu_e, mu_ne = np.mean(sym_e), np.mean(sym_ne)
        sign = 1.0 if mu_e >= mu_ne else -1.0
        d_signed = sign * d if np.isfinite(d) else np.nan

        # Absolute AUC deviation from chance: |AUC - 0.5|
        auc_abs = abs(auc - 0.5) + 0.5 if np.isfinite(auc) else np.nan

        # Tie analysis
        tie_e = float(np.mean(np.abs(sym_e - 1.0) < TIE_TOL))
        tie_ne = float(np.mean(np.abs(sym_ne - 1.0) < TIE_TOL))

        results["features"][feat_name] = {
            "auc": float(auc) if np.isfinite(auc) else None,
            "auc_abs": float(auc_abs) if np.isfinite(auc_abs) else None,
            "dprime": float(d) if np.isfinite(d) else None,
            "dprime_signed": float(d_signed) if np.isfinite(d_signed) else None,
            "auc_gaussian": float(stats.norm.cdf(d / np.sqrt(2))) if np.isfinite(d) else None,
            "sym_mean_edge": float(mu_e),
            "sym_mean_nonedge": float(mu_ne),
            "assortative": bool(mu_e > mu_ne),
            "tie_rate_edge": tie_e,
            "tie_rate_nonedge": tie_ne,
        }

    return results


def main():
    t0 = time.time()
    rng = np.random.RandomState(RNG_SEED)

    # ──────────────────────────────────────────────────────────────────
    # PART 1: Download networks
    # ──────────────────────────────────────────────────────────────────

    networks, domain_counts = download_diverse_networks()

    # Fall back to LFR if not enough real networks
    if len(networks) < 10:
        lfr = generate_lfr_fallbacks(len(networks), rng)
        networks.extend(lfr)

    n_real = sum(1 for n in networks if n["source"] == "netzschleuder")
    n_lfr = sum(1 for n in networks if n["source"] == "lfr_benchmark")
    print(f"\n  Total networks: {len(networks)} (real: {n_real}, LFR: {n_lfr})")

    if len(networks) == 0:
        print("ERROR: No networks loaded. Cannot validate.")
        sys.exit(1)

    # ──────────────────────────────────────────────────────────────────
    # PART 2: Validate d' on held-out networks
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("PART 2: Validate d' -> AUC on held-out networks")
    print("=" * 70)

    per_network = []
    for net_info in networks:
        print(f"\n  Analysing {net_info['slug']}...")
        result = analyse_network(net_info, rng)
        if result is not None:
            per_network.append(result)
            for feat in FEATURES:
                fd = result["features"].get(feat, {})
                if fd.get("auc") is not None:
                    assort_flag = "+" if fd.get("assortative") else "-"
                    print(f"    {feat}: AUC={fd['auc']:.3f}, d'={fd['dprime']:.3f} ({assort_flag}assort)")

    print(f"\n  Networks analysed: {len(per_network)}")

    # Count assortative vs disassortative per feature
    for feat in FEATURES:
        n_assort = sum(1 for r in per_network
                       if r["features"].get(feat, {}).get("assortative") is True)
        n_disassort = sum(1 for r in per_network
                         if r["features"].get(feat, {}).get("assortative") is False)
        print(f"    {feat}: {n_assort} assortative, {n_disassort} disassortative")

    # ──────────────────────────────────────────────────────────────────
    # Aggregate: kcore vs degree win rate
    # ──────────────────────────────────────────────────────────────────

    kcore_wins = 0
    degree_wins = 0
    ties = 0
    for r in per_network:
        auc_deg = r["features"].get("degree", {}).get("auc")
        auc_kc = r["features"].get("kcore", {}).get("auc")
        if auc_deg is not None and auc_kc is not None:
            if auc_kc > auc_deg + 0.001:
                kcore_wins += 1
            elif auc_deg > auc_kc + 0.001:
                degree_wins += 1
            else:
                ties += 1

    total_compared = kcore_wins + degree_wins + ties
    print(f"\n  Kcore vs Degree:")
    print(f"    Kcore wins: {kcore_wins}/{total_compared}")
    print(f"    Degree wins: {degree_wins}/{total_compared}")
    print(f"    Ties (<0.001): {ties}/{total_compared}")
    if total_compared > 0:
        print(f"    Kcore win rate: {kcore_wins / total_compared:.1%}")

    # ──────────────────────────────────────────────────────────────────
    # d' vs AUC correlation (pooled across features) — ORIGINAL
    # ──────────────────────────────────────────────────────────────────

    all_dprime = []
    all_auc = []
    all_dprime_signed = []
    all_auc_abs = []
    all_labels = []

    for r in per_network:
        for feat in FEATURES:
            fd = r["features"].get(feat, {})
            d_val = fd.get("dprime")
            ds_val = fd.get("dprime_signed")
            auc_val = fd.get("auc")
            auc_abs_val = fd.get("auc_abs")
            if d_val is not None and auc_val is not None:
                all_dprime.append(d_val)
                all_auc.append(auc_val)
                all_labels.append(f"{r['network']}/{feat}")
            if ds_val is not None and auc_val is not None:
                all_dprime_signed.append(ds_val)
            if d_val is not None and auc_abs_val is not None:
                all_auc_abs.append(auc_abs_val)

    rho, p, n = safe_spearman(all_dprime, all_auc)
    ci_lo, ci_hi, ci_med = bootstrap_spearman(
        np.array(all_dprime), np.array(all_auc), B=500, seed=RNG_SEED
    )

    print(f"\n  d' vs AUC — ORIGINAL (pooled, n={n}):")
    print(f"    Spearman rho = {rho:.3f}  (p = {p:.2e})")
    print(f"    Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"    Development set:  rho=0.956 [0.865, 0.986]")

    # ──────────────────────────────────────────────────────────────────
    # DIAGNOSTIC: d' vs |AUC - 0.5| + 0.5 (direction-invariant)
    # ──────────────────────────────────────────────────────────────────

    rho_abs, p_abs, n_abs = safe_spearman(all_dprime, all_auc_abs)
    ci_lo_abs, ci_hi_abs, ci_med_abs = bootstrap_spearman(
        np.array(all_dprime), np.array(all_auc_abs), B=500, seed=RNG_SEED
    )

    print(f"\n  d' vs |AUC| — DIRECTION-INVARIANT (pooled, n={n_abs}):")
    print(f"    Spearman rho = {rho_abs:.3f}  (p = {p_abs:.2e})")
    print(f"    Bootstrap 95% CI: [{ci_lo_abs:.3f}, {ci_hi_abs:.3f}]")
    print(f"    (This tests whether d' predicts separation magnitude regardless of direction)")

    # ──────────────────────────────────────────────────────────────────
    # DIAGNOSTIC: signed d' vs AUC
    # ──────────────────────────────────────────────────────────────────

    rho_signed, p_signed, n_signed = safe_spearman(all_dprime_signed, all_auc)
    ci_lo_s, ci_hi_s, ci_med_s = bootstrap_spearman(
        np.array(all_dprime_signed), np.array(all_auc), B=500, seed=RNG_SEED
    )

    print(f"\n  signed d' vs AUC (pooled, n={n_signed}):")
    print(f"    Spearman rho = {rho_signed:.3f}  (p = {p_signed:.2e})")
    print(f"    Bootstrap 95% CI: [{ci_lo_s:.3f}, {ci_hi_s:.3f}]")
    print(f"    (Sign of d' encodes direction: + = assortative, - = disassortative)")

    # ──────────────────────────────────────────────────────────────────
    # d' vs AUC correlation (per feature)
    # ──────────────────────────────────────────────────────────────────

    per_feature_corr = {}
    for feat in FEATURES:
        feat_d = []
        feat_a = []
        feat_d_abs = []
        feat_a_abs = []
        for r in per_network:
            fd = r["features"].get(feat, {})
            d_val = fd.get("dprime")
            auc_val = fd.get("auc")
            auc_abs_val = fd.get("auc_abs")
            if d_val is not None and auc_val is not None:
                feat_d.append(d_val)
                feat_a.append(auc_val)
            if d_val is not None and auc_abs_val is not None:
                feat_d_abs.append(d_val)
                feat_a_abs.append(auc_abs_val)
        rho_f, p_f, n_f = safe_spearman(feat_d, feat_a)
        ci_lo_f, ci_hi_f, ci_med_f = bootstrap_spearman(
            np.array(feat_d), np.array(feat_a), B=500, seed=RNG_SEED
        )
        rho_f_abs, p_f_abs, n_f_abs = safe_spearman(feat_d_abs, feat_a_abs)
        per_feature_corr[feat] = {
            "rho_original": rho_f,
            "p_original": p_f,
            "n": n_f,
            "ci_95_lo": ci_lo_f,
            "ci_95_hi": ci_hi_f,
            "rho_direction_invariant": rho_f_abs,
            "p_direction_invariant": p_f_abs,
        }
        if np.isfinite(rho_f):
            print(f"\n  d' vs AUC ({feat}, n={n_f}):")
            print(f"    Original:            rho = {rho_f:.3f}  (p = {p_f:.2e})")
            if np.isfinite(rho_f_abs):
                print(f"    Direction-invariant:  rho = {rho_f_abs:.3f}  (p = {p_f_abs:.2e})")
        else:
            print(f"\n  d' vs AUC ({feat}): Insufficient data")

    # ──────────────────────────────────────────────────────────────────
    # Gaussian prediction quality: AUC_gaussian vs AUC_observed
    # ──────────────────────────────────────────────────────────────────

    gauss_pred = []
    gauss_obs = []
    for r in per_network:
        for feat in FEATURES:
            fd = r["features"].get(feat, {})
            ag = fd.get("auc_gaussian")
            ao = fd.get("auc")
            if ag is not None and ao is not None:
                gauss_pred.append(ag)
                gauss_obs.append(ao)

    if len(gauss_pred) >= 3:
        gauss_rmse = float(np.sqrt(np.mean((np.array(gauss_pred) - np.array(gauss_obs)) ** 2)))
        gauss_mae = float(np.mean(np.abs(np.array(gauss_pred) - np.array(gauss_obs))))
        print(f"\n  Gaussian prediction quality (n={len(gauss_pred)}):")
        print(f"    RMSE = {gauss_rmse:.4f}")
        print(f"    MAE  = {gauss_mae:.4f}")
    else:
        gauss_rmse = None
        gauss_mae = None

    # ──────────────────────────────────────────────────────────────────
    # Per-network detail table
    # ──────────────────────────────────────────────────────────────────

    dp_label = "d'"
    hdr = f"  {'Network':<40} {'Domain':<8} {'deg AUC':>8} {'deg '+dp_label:>8} {'kc AUC':>8} {'kc '+dp_label:>8} {'Winner':>8}"
    print(f"\n{hdr}")
    print("  " + "-" * 100)
    for r in per_network:
        deg_auc = r["features"].get("degree", {}).get("auc")
        deg_dp = r["features"].get("degree", {}).get("dprime")
        kc_auc = r["features"].get("kcore", {}).get("auc")
        kc_dp = r["features"].get("kcore", {}).get("dprime")
        winner = "---"
        if deg_auc is not None and kc_auc is not None:
            if kc_auc > deg_auc + 0.001:
                winner = "kcore"
            elif deg_auc > kc_auc + 0.001:
                winner = "degree"
            else:
                winner = "tie"
        if all(v is not None for v in [deg_auc, deg_dp, kc_auc, kc_dp]):
            net = r["network"]
            dom = r["domain"]
            print(f"  {net:<40} {dom:<8} {deg_auc:>8.3f} {deg_dp:>8.3f} {kc_auc:>8.3f} {kc_dp:>8.3f} {winner:>8}")
        else:
            print(f"  {r['network']:<40} {r['domain']:<8} (incomplete data)")

    # ──────────────────────────────────────────────────────────────────
    # Summary comparison with development set
    # ──────────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY: Held-out vs Development set")
    print("=" * 70)
    print(f"  Held-out networks:    {len(per_network)}")
    print(f"    Real (Netzschleuder): {sum(1 for r in per_network if r['source'] == 'netzschleuder')}")
    print(f"    Synthetic (LFR):      {sum(1 for r in per_network if r['source'] == 'lfr_benchmark')}")
    print(f"  Domains: {domain_counts}")
    print(f"\n  d' vs AUC (ORIGINAL — unsigned d', raw AUC):")
    print(f"    Dev set:     rho=0.956 [0.865, 0.986] (n=31 networks)")
    print(f"    Held-out:    rho={rho:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (n={n} points)")

    # Overlap test: does held-out CI overlap dev CI?
    dev_lo, dev_hi = 0.865, 0.986
    overlap = ci_lo <= dev_hi and ci_hi >= dev_lo
    print(f"    CIs overlap: {'YES' if overlap else 'NO'}")

    print(f"\n  d' vs |AUC| (DIRECTION-INVARIANT — unsigned d', |AUC-0.5|+0.5):")
    print(f"    Held-out:    rho={rho_abs:.3f} [{ci_lo_abs:.3f}, {ci_hi_abs:.3f}] (n={n_abs} points)")

    overlap_abs = ci_lo_abs <= dev_hi and ci_hi_abs >= dev_lo
    print(f"    CIs overlap with dev: {'YES' if overlap_abs else 'NO'}")

    print(f"\n  signed d' vs AUC:")
    print(f"    Held-out:    rho={rho_signed:.3f} [{ci_lo_s:.3f}, {ci_hi_s:.3f}] (n={n_signed} points)")

    # Disassortative network count
    n_disassort_net = sum(1 for r in per_network
                         if r.get("degree_assortativity") is not None
                         and r["degree_assortativity"] < 0)
    n_assort_net = sum(1 for r in per_network
                       if r.get("degree_assortativity") is not None
                       and r["degree_assortativity"] >= 0)
    print(f"\n  Network assortativity: {n_assort_net} assortative, {n_disassort_net} disassortative")

    if total_compared > 0:
        print(f"\n  Kcore vs Degree:")
        print(f"    Kcore win rate: {kcore_wins}/{total_compared} = {kcore_wins / total_compared:.1%}")

    # ──────────────────────────────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────────────────────────────

    elapsed = time.time() - t0

    # Custom serializer for NaN/inf
    def sanitize(obj):
        if isinstance(obj, float):
            if not np.isfinite(obj):
                return None
        return obj

    def sanitize_dict(d):
        if isinstance(d, dict):
            return {k: sanitize_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [sanitize_dict(v) for v in d]
        elif isinstance(d, float):
            return None if not np.isfinite(d) else d
        elif isinstance(d, np.integer):
            return int(d)
        elif isinstance(d, np.floating):
            return None if not np.isfinite(d) else float(d)
        return d

    output = sanitize_dict({
        "metadata": {
            "description": "Held-out validation of d' -> AUC relationship",
            "n_networks_total": len(per_network),
            "n_real": sum(1 for r in per_network if r["source"] == "netzschleuder"),
            "n_lfr": sum(1 for r in per_network if r["source"] == "lfr_benchmark"),
            "domain_counts": domain_counts,
            "max_edges": MAX_EDGES,
            "max_nonedges": MAX_NONEDGES,
            "rng_seed": RNG_SEED,
            "dev_set_reference": {
                "rho": 0.956,
                "ci_95_lo": 0.865,
                "ci_95_hi": 0.986,
                "n_networks": 31,
            },
            "runtime_sec": round(elapsed, 1),
        },
        "per_network": [{k: v for k, v in r.items() if k != "graph"} for r in per_network],
        "dprime_vs_auc_pooled": {
            "observed_rho": rho,
            "observed_p": p,
            "n_points": n,
            "bootstrap_ci_95_lo": ci_lo,
            "bootstrap_ci_95_hi": ci_hi,
            "bootstrap_median_rho": ci_med,
            "dev_set_rho": 0.956,
            "dev_set_ci": [0.865, 0.986],
            "cis_overlap": overlap,
        },
        "dprime_vs_auc_abs_pooled": {
            "description": "d' vs |AUC-0.5|+0.5 — direction-invariant",
            "observed_rho": rho_abs,
            "observed_p": p_abs,
            "n_points": n_abs,
            "bootstrap_ci_95_lo": ci_lo_abs,
            "bootstrap_ci_95_hi": ci_hi_abs,
            "cis_overlap_with_dev": overlap_abs,
        },
        "signed_dprime_vs_auc_pooled": {
            "description": "signed d' vs AUC — positive d' = assortative",
            "observed_rho": rho_signed,
            "observed_p": p_signed,
            "n_points": n_signed,
            "bootstrap_ci_95_lo": ci_lo_s,
            "bootstrap_ci_95_hi": ci_hi_s,
        },
        "dprime_vs_auc_per_feature": per_feature_corr,
        "kcore_vs_degree": {
            "kcore_wins": kcore_wins,
            "degree_wins": degree_wins,
            "ties": ties,
            "total": total_compared,
            "kcore_win_rate": kcore_wins / total_compared if total_compared > 0 else None,
        },
        "gaussian_prediction": {
            "rmse": gauss_rmse,
            "mae": gauss_mae,
            "n_points": len(gauss_pred),
        },
    })

    out_path = RESULTS_DIR / "heldout_validation.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")
    print(f"  Runtime: {elapsed:.1f}s")

    return output


if __name__ == "__main__":
    main()
