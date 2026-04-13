# The Apparent Paradox of Irrelevance

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19560593.svg)](https://doi.org/10.5281/zenodo.19560593)

**Why does a coarser feature outperform a finer one for link prediction?**

Under normalized pairwise similarity, k-core consistently beats degree (mean AUC 0.659 vs 0.491) across 65 networks from 20 domains — despite degree being finer-grained. This repository contains the paper, analysis code, and data for the full resolution.

## Key Results

- **K-core wins 65/65** networks (100%, p < 10⁻¹⁹) under full-graph evaluation
- **Leakage-free: 64/65** (98.5%) under strict train-only protocol, d' ρ = 0.996
- **Mechanism:** assortativity-driven tie enrichment (TER 3.58× for k-core vs 0.54× for degree)
- **Exact mixture AUC decomposition** (RMSE = 0.000): ties account for 59% of AUC
- **Signed Fisher's d'** decomposes AUC into mean shift (assortativity) and variance (ceiling effect)
- **Metric independence:** three continuous similarity formulas give identical AUC on integer features (ρ = 1.000)
- **EDGE diagnostic:** detects positional signal in 10/20 biological vs 3/45 non-biological networks (Fisher p = 0.0002)
- **Cardinality control:** binning degree to match k-core cardinality eliminates the advantage (0.663 vs 0.667), confirming the tie mechanism

## Paper

- **Full paper:** [`docs/paper.md`](docs/paper.md)
- **Practitioner's guide:** [`docs/article.md`](docs/article.md)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note:** The development set (32 networks) requires the benchmark file from the parent project at `~/code/ai/symmetry-mechanism/data/phase6_benchmark.json`. Held-out and expanded networks are self-contained in `data/`.

## Core Scripts

| Script | Purpose |
|--------|---------|
| `src/unified_analysis.py` | Main pipeline: all 65 networks, AUC, d', TER, mixture, BL2, EDGE |
| `src/leakage_free_comparison.py` | Train-only k-core vs train-only degree (64/65, ρ = 0.996) |
| `src/proof_ceiling_effect.py` | Z3/SymPy formal proofs (L1–L5, P1–P4, T9–T11) |
| `src/edge_hardened.py` | EDGE diagnostic with config-model null |
| `generate_figures.py` | Regenerate all 5 paper figures from `results/unified_analysis.json` |
| `build_paper.py` | Convert `docs/paper.md` to LaTeX for arxiv-classic template |

## Reproducing Results

```bash
source .venv/bin/activate
python src/unified_analysis.py          # ~2 min, produces results/unified_analysis.json
python src/leakage_free_comparison.py   # ~40s, produces results/leakage_free_comparison.json
python src/proof_ceiling_effect.py      # Z3 + SymPy proofs
python generate_figures.py              # regenerate figures to output/figures/
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for full details.

## Repository Structure

```
docs/           Paper (paper.md) and companion guide (article.md)
src/            Analysis scripts (13 Python + 1 Lean)
results/        Pre-computed JSON results (12 files)
data/           Held-out network edge lists (27 CSVs)
build_paper.py  Markdown → LaTeX converter
generate_figures.py  Figure generation from unified results
```

## Citation

If you use this work, please cite:

> Hibryda. "The Apparent Paradox of Irrelevance: Fisher Discriminability Explains Feature Performance in Normalized Similarity Link Prediction." 2026.

## License

MIT
