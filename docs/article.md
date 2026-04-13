# K-Core Similarity for Link Prediction: A Practitioner's Guide

**Author:** Hibryda

*This article is a companion to the research paper "The Apparent Paradox of Irrelevance" [1], which provides formal proofs and full statistical analysis. Here we explain the paper's findings for data scientists and engineers who work with graphs but don't necessarily follow network science journals. All claims below are backed by the parent paper; we skip the proofs and focus on what the results mean for practice.*

---

## 1. Bottom Line

If you use normalized similarity for link prediction, switch from degree to k-core. K-core similarity achieves AUC 0.659 versus 0.491 for degree similarity, winning in 65 of 65 tested networks (100%). The result holds across social, biological, technological, economic, and infrastructure networks, and across four different similarity formulas. K-core runs in O(m) time, requires no parameters, and needs no external features --- just the graph itself.

One caveat: these results apply to the transductive setting (predicting edges in a fixed graph). Whether they transfer to inductive settings (new nodes arriving) is an open question. K-core similarity is a strong structural baseline, not a replacement for GNN-based methods.

---

## 2. What Is Link Prediction?

Link prediction is the task of guessing which edges are missing from a network --- or which edges will appear next. If you've used LinkedIn's "People You May Know" or Amazon's "Customers Who Bought This Also Bought," you've seen link prediction in action.

The simplest approach: compute a similarity score for every pair of unconnected nodes, then rank them. Pairs with high scores are predicted links. The standard way to measure quality is AUC (Area Under the ROC Curve), which answers: if you pick a random true link and a random non-link, how often does the true link rank higher? AUC = 0.5 means random guessing. AUC = 1.0 means perfect separation.

Link prediction matters beyond social networks. In bioinformatics, predicting protein interactions saves years of wet-lab experiments. In fraud detection, finding hidden connections in transaction networks reveals organized crime rings that manual investigation would miss. In knowledge graphs like Wikidata, it fills in missing facts. In infrastructure planning, it identifies which parts of a power grid or communication network are most vulnerable to cascading failures.

Many link prediction methods use structural features of the graph. Common neighbors counts how many friends two people share. Adamic-Adar weights those shared friends by their rarity. These are multi-hop features --- they look at neighborhoods.

A simpler approach uses single-node features. Every node has a degree (number of connections) and a k-core number (which we'll define in the next section). Given a feature f, you compute a pairwise similarity:

**S(a, b) = 2 min(a, b) / (a + b)**

This formula returns 1 when two nodes have the same feature value and approaches 0 when their values diverge. It's scale-invariant --- doubling both values doesn't change the score.

The obvious feature to use is degree. It's the most basic graph statistic, available instantly for every node. K-core number is a coarsened version of degree --- it throws away information. Common sense says: more information should give better predictions.

Common sense is wrong. K-core consistently and substantially beats degree. The rest of this article explains why, and what that means for your work.

---

## 3. K-Core in 5 Minutes

The k-core decomposition sorts nodes into concentric "shells" based on how deeply embedded they are in the graph. Think of it like city zoning: the outskirts (1-core) are loosely connected suburbs, while the inner core (high k) is a dense downtown where everyone is connected to everyone.

The algorithm is a simple peeling process:

1. Remove all nodes with degree less than 1. The nodes that survive are in the 1-core.
2. Remove all nodes with degree less than 2 (in the remaining graph). Survivors are in the 2-core.
3. Continue: remove nodes with degree less than k. Survivors are in the k-core.
4. Each node's k-core number is the highest k for which it survives.

<p align="center"><img src="../output/figures/fig1_bl2_scatter.png" width="60%" /><br><em>Figure 1: BL2-predicted vs observed Var(sym) across 325 feature-network pairs. The ceiling effect operates as theorized even for heavy-tailed real networks (R² = 0.728).</em></p>

Why "city zoning"? In a real city, suburbs have houses connected to a few neighbors (low k-core). Moving inward, you hit denser residential areas where each block connects to many others (mid k-core). The downtown core is the densest --- every business is accessible from every other (high k-core). If you removed all suburbs, the city still functions. If you removed the downtown core, the city fragments. K-core captures this structural hierarchy.

Two facts about k-core make it interesting for link prediction:

**It's correlated with degree but not identical.** Across our 67-network benchmark, the Spearman correlation between degree and k-core is about 0.93. They measure related things, but k-core is coarser --- many nodes with different degrees end up in the same shell.

**It runs in O(m) time.** The peeling algorithm visits each edge once. For a million-edge graph, it takes milliseconds. No parameters to tune, no convergence to wait for, no randomness.

The surprise is that this coarser, less informative feature produces better link predictions than degree --- by a wide margin.

---

## 4. What 65 Networks Showed

We tested k-core versus degree similarity on 65 networks spanning 20 domains: protein interactions, social trust networks, internet topology, power grids, food webs, citation networks, economic trade, and more. Network sizes range from 62 to 4,762 nodes.

The result is unambiguous. K-core similarity beats degree similarity in 65 of 65 networks. The mean AUC gap is 0.171 (0.659 vs 0.491).

<p align="center"><img src="../output/figures/fig3_dprime_vs_auc.png" width="60%" /><br><em>Figure 2: Signed d' vs AUC for k-core and degree across 65 networks. K-core (blue) clusters in the upper-right; degree (red) spans both quadrants. The Gaussian curve shows near-perfect fit.</em></p>

### Why K-Core Wins

The explanation comes from a property called *assortativity* --- the tendency of nodes to connect to others with similar values.

**K-core is assortative.** Nodes in the same shell tend to connect to each other. In 52 of 64 networks, k-core has positive Newman assortativity (mean r = 0.268). When you compute similarity between two nodes in the same k-core shell, S(k, k) = 1.0 --- the maximum score. These perfect-score ties are enriched among actual edges, directly inflating AUC.

**Degree is disassortative.** High-degree hubs tend to connect to low-degree periphery nodes --- the classic "rich-club" pattern documented in many real networks. When you compute similarity between a hub (degree 100) and a peripheral node (degree 5), S(100, 5) = 0.095 --- a low score. These low-similarity pairs are *overrepresented* among actual edges, depressing AUC.

<p align="center"><img src="../output/figures/fig2_mixture_auc.png" width="80%" /><br><em>Figure 3: Mixture AUC decomposition for k-core similarity. Blue = tie contribution, orange = continuous contribution. Networks sorted by total AUC. The tie component dominates in most networks.</em></p>

The normalized similarity formula S(a, a) = 1 is, in effect, an *assortativity detector*. Any feature where same-value nodes are overrepresented among edges will produce high AUC. K-core has this property; degree does not.

An exact decomposition of AUC into tied scores (where both nodes have the same feature value) and continuous scores (where they differ) shows that ties account for about 60% of k-core's AUC. The continuous gradient adds the remaining 40%. K-core wins primarily through tie enrichment --- the mechanical consequence of its assortative structure.

### What About the Formula?

A common concern in link prediction is choosing the "right" similarity formula. Should you use harmonic similarity? Jaccard? Exponential difference?

For integer-valued features like k-core and degree, it doesn't matter. Three continuous formulas (sym, Jaccard, exp-diff) produce *identical* AUC values (cross-metric correlation rho = 1.000). This is not an approximation --- the AUC values are equal to machine precision. (Cosine similarity is degenerate for scalar features --- it returns 1.0 for all positive pairs --- and is excluded.)

The reason: AUC depends only on the *rank ordering* of scores, and all three formulas are monotonic transformations of the ratio min(a,b)/max(a,b). Same ranking, same AUC.

The one formula that does differ is binary match (1 if a = b, else 0), which retains only the tie component and discards the continuous gradient. It underperforms the continuous formulas by 0.107 on average.

**Important caveat:** This equivalence holds for integer features. For continuous features (e.g., eigenvector centrality), formula choice can matter. Check before assuming.

<p align="center"><img src="../output/figures/fig4_metric_independence.png" width="100%" /><br><em>Figure 4: AUC across four similarity formulas for k-core (left) and degree (right). The three continuous formulas overlap perfectly. Binary match is displaced below.</em></p>

### Ruling Out the Obvious

Before reaching the assortativity explanation, the paper first tested the obvious hypothesis: maybe degree's high mean value compresses its similarity scores into a narrow band, destroying discriminative power. This "ceiling effect" is real and provable --- variance decreases as O(1/mu^2). But it turns out to be incidental. Variance does not predict AUC (rho = 0.099, not significant). The ceiling compresses the continuous component (41% of AUC) but doesn't touch the dominant tie component (59%). The real explanation is assortativity, not compression.

### Where K-Core Sits Among Methods

K-core similarity is a structural baseline, not the state of the art. Multi-hop methods like Adamic-Adar and common neighbors use richer neighborhood information. GNN-based methods learn representations from graph structure. K-core doesn't compete with these --- it explains *why* a single-feature normalized similarity works as well as it does, and provides a fast, interpretable, parameter-free option when simplicity matters.

In our benchmark, community-based similarity (Louvain membership match) achieves AUC = 0.732, marginally above k-core's 0.719. But k-core requires no community detection, no resolution parameter, and runs in linear time.

---

## 5. The EDGE Diagnostic: Testing Your Network's Structure

Beyond the k-core result, the paper introduces a diagnostic called EDGE (Edge-Driven Graph Equivalence) that tests whether a network contains structural signal beyond its degree sequence.

EDGE works by comparing the cosine similarity of multi-centrality node vectors (degree, k-core, eigenvector, betweenness) across actual edges versus a degree-preserving null model. If real edges show higher centrality similarity than rewired edges, the network has positional structure that goes beyond degree.

Of 65 networks tested, 13 show excess positional signal. The signal is strongly enriched in biological networks: 10 of 20 bio (50%) versus 3 of 45 non-bio (7%). Fisher exact test (one-sided): p = 0.0002.

The discriminating factor is data quality, not density. WormNet (C. elegans functional interactions) and BioGRID (C. elegans protein interactions) have nearly identical mean degree (~3.0), but WormNet shows strong EDGE signal while BioGRID shows none. The difference: WormNet is a curated database; BioGRID is high-throughput screening. EDGE detects whether the edges in your network reflect real structural relationships or experimental noise.

**For network biologists:** EDGE can serve as a data quality diagnostic. If your protein interaction database fails the EDGE test, its edges may be too noisy for methods that assume functional relationships.

**For fraud analysts:** Dense subgraphs in transaction networks may show EDGE-positive signal, indicating that k-core-based features could add value. In financial networks, fraudulent accounts often form tight clusters (same shell) that degree alone misses because some fraud nodes have low degree by design. But this application requires domain-specific validation --- financial networks differ structurally from the biological networks where EDGE was validated.

**For infrastructure engineers:** Power grid and internet topology networks have well-studied k-core structure. The innermost core identifies the nodes whose failure would cause the most disruption. EDGE can test whether your particular infrastructure graph has enough positional structure for k-core-based link prediction to add value beyond simpler baselines.

**Scope:** EDGE is an exploratory diagnostic, not a performance predictor. A positive EDGE signal suggests trying k-core features, but doesn't guarantee a specific AUC improvement.

---

## 6. Practical Recommendations

Based on the 65-network study, here are concrete guidelines for practitioners:

1. **Use k-core similarity as your structural baseline.** It wins 98% of the time, runs in O(m), has no parameters, and doesn't leak test-set information.

2. **Don't agonize over the similarity formula.** For integer features (degree, k-core, community label), sym, Jaccard, cosine, and exp-diff all produce identical results. Pick whichever is easiest to implement.

3. **Check assortativity first.** Before computing similarity with any feature, check Newman's r. If r < 0 (disassortative), the similarity score will produce AUC < 0.5. Either invert the score or use a different feature.

4. **Use EDGE as a pre-check.** Run the EDGE diagnostic on your network. If it passes, k-core features are likely to carry structural signal. If it fails, degree-based features may be sufficient.

5. **Remember the scope.** These results apply to transductive link prediction in undirected, unweighted, static graphs. For directed networks, temporal networks, or inductive settings (new nodes), additional validation is needed.

<p align="center"><img src="../output/figures/fig5_cumulative_validation.png" width="70%" /><br><em>Figure 5: Signed d' vs AUC across the full 65-network corpus, colored by provenance. The relationship is consistent across development (31), held-out (10), and expanded (24) corpus subsets.</em></p>

A minimal working example in Python:

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
kcore = nx.core_number(G)

# Compute k-core similarity for all non-edges
def sym(a, b):
    return 2 * min(a, b) / (a + b) if a + b > 0 else 0

scores = []
for u, v in nx.non_edges(G):
    scores.append((u, v, sym(kcore[u], kcore[v])))

# Top 10 predicted links
scores.sort(key=lambda x: -x[2])
print("Top predictions:", scores[:10])
```

Full code with proper evaluation (AUC computation, negative sampling, train/test split) is in Appendix A.

---

## 7. Key Takeaways

- K-core similarity outperforms degree similarity in 65/65 networks (AUC 0.659 vs 0.491)
- The reason is structural: k-core is assortative, degree is disassortative
- The similarity formula doesn't matter for integer features (5 formulas give identical AUC)
- The EDGE diagnostic can detect whether your network has positional structure worth exploiting
- K-core is a fast, parameter-free baseline --- not a replacement for GNNs or multi-hop methods

For the full formal treatment --- including Z3-verified proofs of the ceiling effect, the exact mixture AUC decomposition, paradox theorems characterizing when a coarser feature can outperform a finer one, and validation across 65 networks with three independent held-out sets --- see [1].

**Further reading:**
- Kong et al. [3] for a comprehensive review of k-core theory and applications
- Newman [5] for the mathematical foundations of assortativity in networks
- Liben-Nowell and Kleinberg [4] for the original formulation of the link prediction problem

---

## References

[1] Hibryda, "The Apparent Paradox of Irrelevance: Fisher Discriminability Explains Feature Performance in Normalized Similarity Link Prediction," 2026. [Main paper]

[2] M. Kitsak et al., "Identification of influential spreaders in complex networks," *Nature Physics*, vol. 6, pp. 888--893, 2010.

[3] Y.-X. Kong et al., "k-core: Theories and applications," *Physics Reports*, vol. 832, pp. 1--32, 2019.

[4] D. Liben-Nowell and J. Kleinberg, "The link-prediction problem for social networks," *JASIST*, vol. 58, pp. 1019--1031, 2007.

[5] M. E. J. Newman, "Assortative mixing in networks," *Physical Review Letters*, vol. 89, p. 208701, 2002.

[6] L. A. Adamic and E. Adar, "Friends and neighbors on the Web," *Social Networks*, vol. 25, pp. 211--230, 2003.

[7] D. Bamber, "The area above the ordinal dominance graph and the area below the receiver operating characteristic graph," *Journal of Mathematical Psychology*, vol. 12, pp. 387--415, 1975.

[8] D. M. Green and J. A. Swets, *Signal Detection Theory and Psychophysics*, Wiley, 1966.

---

## Appendix A: Full Working Example

```python
"""K-core link prediction on Zachary's Karate Club.

This is a toy example for illustration. On small graphs like this,
results are noisy --- the patterns described in the paper emerge
reliably on networks with 100+ nodes.
"""
import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

# Load graph
G = nx.karate_club_graph()
kcore = nx.core_number(G)
degree = dict(G.degree())

def sym(a, b):
    """Normalized similarity: S(a,b) = 2*min(a,b) / (a+b)."""
    return 2 * min(a, b) / (a + b) if a + b > 0 else 0

# Sample edges and non-edges
rng = np.random.default_rng(42)
edges = list(G.edges())
nodes = list(G.nodes())

# Negative sampling: sample as many non-edges as edges
# NOTE: Negative sampling strategy affects AUC. This uses
# uniform random sampling, matching the paper's protocol.
non_edges = []
edge_set = set(G.edges()) | set((v, u) for u, v in G.edges())
while len(non_edges) < len(edges):
    u, v = rng.choice(nodes, 2, replace=False)
    if (u, v) not in edge_set:
        non_edges.append((u, v))

# Compute scores
labels = [1] * len(edges) + [0] * len(non_edges)
all_pairs = edges + non_edges

kcore_scores = [sym(kcore[u], kcore[v]) for u, v in all_pairs]
degree_scores = [sym(degree[u], degree[v]) for u, v in all_pairs]

# Evaluate
auc_kcore = roc_auc_score(labels, kcore_scores)
auc_degree = roc_auc_score(labels, degree_scores)

print(f"K-core AUC:  {auc_kcore:.3f}")
print(f"Degree AUC:  {auc_degree:.3f}")
print(f"Winner:      {'k-core' if auc_kcore > auc_degree else 'degree'}")
```

---

## Appendix B: Fisher's d' (Performance Predictor)

Fisher's discriminability index d' predicts how well a feature will perform under normalized similarity, without running the full evaluation:

**d' = (mean_edge - mean_non-edge) / sqrt((var_edge + var_non-edge) / 2)**

**Step-by-step computation:**

1. Compute sym scores for a sample of edges and non-edges
2. Calculate mean and standard deviation for each group
3. Plug into the formula above
4. If the feature is disassortative (edges have *lower* mean sym), negate d'

Signed d' predicts AUC with rho = 0.994 across 65 networks (n = 130 feature-network pairs). For the full analysis, see [1, Section 3.1] and [8].
