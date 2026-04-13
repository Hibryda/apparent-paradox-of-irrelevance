# Reproducibility Guide

## Environment

- **Python:** 3.14.0
- **OS:** Linux (Debian 13, kernel 6.12)
- **Key dependencies:** networkx 3.6, numpy 2.3, scipy 1.16, z3-solver 4.16, sympy 1.14
- **Full pinned versions:** see `requirements.txt`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Sources

### Development corpus (32 networks)

Loaded via `~/code/ai/symmetry-mechanism/data/phase6_benchmark.json`, which references:
- **Netzschleuder** (networks.skewed.de): 12 networks (downloaded as CSV edge lists)
- **STRING** (string-db.org): 3 PPI networks (high-confidence, score ≥ 700)
- **BioGRID** (thebiogrid.org): 6 biological networks
- **SNAP** (snap.stanford.edu): 4 social trust networks (Bitcoin, Epinions, Slashdot)
- **KONECT** (konect.cc): 3 networks (Wikipedia RfA, Word Association, Bible)
- **Other:** US Airport (BTS), Product Space (Harvard Atlas), IEEE 300-bus, Contiguous USA

### Held-out corpus 1 (10 networks)

`data/heldout/*.csv` — downloaded from Netzschleuder API. Networks NOT seen during development.

### Held-out corpus 2 (25 networks)

`data/heldout2/*.csv` — 17 Netzschleuder + 8 synthetic (LFR, BA, WS, SBM).

## Random Seeds

All experiments use `seed = 42` (numpy.random.default_rng). Edge/non-edge sampling: up to 4000 each per network.

## Key Experiments and Expected Output

### Main result: k-core vs degree

```bash
python3 src/tie_effect_corrected.py
```
Expected: kcore AUC mean ~0.706, degree AUC mean ~0.544 across 31 dev networks. d' rho ~0.956.

### Formal proofs

```bash
python3 src/proof_ceiling_effect.py
```
Expected: All 5 lemmas (L1-L5) PROVED, all 4 propositions (P1-P4) CONFIRMED, all 3 paradox theorems (T9-T11) PROVED.

### Baseline comparison

```bash
python3 src/baseline_comparison.py
```
Expected: CN ~0.848, AA ~0.851, PA ~0.854 mean AUC (all higher than kcore 0.706 — different method category).

### EDGE diagnostic

```bash
python3 src/edge_hardened.py
```
Expected: 9/25 biological networks show signal (Fisher p = 0.012). 0/8 non-biological.

### Leakage test

```bash
python3 src/leakage_test.py
```
Expected: Mean AUC inflation +0.047 from full-graph vs train-only k-core. 15/31 networks flagged.

## Reference AUCs (for validation)

These values should be reproduced exactly with the stated seeds:

| Network | n | m | kcore AUC | degree AUC |
|---------|---|---|-----------|------------|
| C. elegans Gene Network | 1995 | 43419 | 0.8172 | 0.7271 |
| S. cerevisiae (Yeast) | 1991 | 26987 | 0.7515 | 0.5554 |
| Bitcoin OTC Trust | 4762 | 15717 | 0.5068 | 0.3084 |

## File Manifest

### Source code (`src/`)

| File | Purpose |
|------|---------|
| `proof_ceiling_effect.py` | Z3+SymPy formal proofs (L1-L5, P1-P4, T9-T11) |
| `tie_effect_corrected.py` | Main analysis: d', TER, mixture AUC (6 corrections) |
| `tie_effect_tests.py` | Initial tie effect discovery |
| `ceiling_effect_tests.py` | Rigorous test suite (32 networks, 5 tests) |
| `bl1_bridge_lemma.py` | BL1 (Gaussian, inapplicable) |
| `bl2_bridge_lemma.py` | BL2 (distribution-free, R²=0.756) |
| `f6_metric_generalization.py` | 5-formula metric independence test |
| `blocker_tests.py` | Eigenvector anomaly, leakage control, CM control |
| `edge.py` | EDGE diagnostic (original) |
| `edge_hardened.py` | EDGE with config model null (hardened) |
| `edge_heldout.py` | EDGE held-out validation |
| `edge_sparsity.py` | EDGE sparsity investigation |
| `heldout_validation.py` | 10-network held-out (d' sign discovery) |
| `expanded_heldout.py` | 25-network expansion (d' rho=0.997) |
| `baseline_comparison.py` | CN/AA/Jaccard/PA baselines |
| `leakage_test.py` | Train-only k-core leakage test |
| `integration_tests.py` | Cross-component connection tests |
| `benchmark_audit.py` | Directed network audit |

### Results (`results/`)

All results are JSON files, one per experiment. Names match source files.

## Build Paper PDF

```bash
python3 build_paper.py
lualatex -output-directory=output output/document.tex
lualatex -output-directory=output output/document.tex
```

Requires: LuaLaTeX (TeX Live 2025), arxiv-classic template in `templates/`.
