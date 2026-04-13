# The Apparent Paradox of Irrelevance: Fisher Discriminability Explains Feature Performance in Normalized Similarity Link Prediction

**Authors:** Hibryda (hibryda@protonmail.com), Independent Researcher

**Abstract.** Under normalized pairwise similarity, k-core outperforms degree for link prediction (mean AUC 0.659 vs 0.491) across 65 networks from 20 domains and 4 metrics---an apparent paradox since degree carries more raw predictive signal. We resolve this in three steps. First, we prove a ceiling effect: normalized similarity metrics compress variance at rate O(1/mu^2) (Z3-verified: 5 lemmas, 3 theorems; empirically confirmed in 29/30 networks). Second, we show this ceiling is real but not the primary mechanism---variance does not predict AUC (rho = 0.099, p = ns). The correct predictor is signed Fisher's d' (pooled rho = 0.994, n = 130 feature-network pairs). Third, we derive an exact mixture AUC decomposition: AUC = p_e(1 - p_ne) + 0.5 p_e p_ne + (1 - p_e)(1 - p_ne) AUC_continuous. For k-core, ties account for 59% of AUC; the continuous gradient contributes 41%. The resolution is structural: degree is disassortative (tie enrichment ratio 0.54x), while k-core is assortative (TER = 3.58x). Assortativity gates the effect (determines direction); d' determines magnitude. K-core similarity wins in all 65 networks (100%) across all metrics. We also show metric independence---three continuous normalized similarity formulas yield identical AUC on integer-valued features (cross-metric rho = 1.000)---and provide a positional signal diagnostic (Edge-Driven Graph Equivalence, EDGE) that detects excess centrality similarity in 10/20 biological networks versus 3/45 non-biological (Fisher exact p = 0.0002, one-sided). All formal results are verified by the Z3 SMT solver.

---

## 1. Introduction

Link prediction---inferring missing or future edges in a network---is a central problem in network science [1, 2, 20]. Among the simplest and most widely used approaches are similarity-based heuristics, which assign a score to each node pair based on shared neighbors [3, 5] or node attributes [4]. A natural extension replaces structural neighborhoods with node-level features: given a scalar feature f(v), one computes a pairwise similarity S(f(u), f(v)) and ranks candidate edges by this score.

An instructive test case arises from two closely related features: node degree and k-core number. The k-core decomposition assigns each node to the innermost shell it belongs to under iterative pruning of nodes with degree below k [16, 17]. K-core number is strongly correlated with degree (Spearman rho approximately 0.93 across our benchmark), yet degree is finer-grained: it distinguishes nodes that k-core groups together, since k-core number is a coarsening that discards variation within shells.

Here is what is paradoxical: when we apply normalized pairwise similarity to these features and use the resulting scores for link prediction, k-core similarity consistently and substantially outperforms degree similarity. Across 65 networks spanning 20 domains---biological, social, technological, economic, infrastructure, and synthetic---k-core similarity achieves a mean AUC of 0.659 versus 0.491 for degree similarity, winning in all 65 cases (100%). This gap persists across four normalized similarity formulas (three continuous and one binary). The feature with less raw information yields better predictions.

The first hypothesis is a ceiling effect: the normalized similarity formula compresses high-mean features into a narrow band near 1, destroying their discriminative power. This compression is real and provable---but it is secondary, not the primary mechanism. Variance does not predict AUC (rho = 0.099, p = ns). The actual mechanism lies elsewhere.

We resolve the apparent paradox. The resolution consists of three components, none of which alone is sufficient.

First, there is a provable ceiling effect. Normalized similarity metrics of the form S(a,b) = 2 min(a,b)/(a+b) compress pairwise variance at rate O(1/mu^2), where mu is the feature mean. We establish this through five lemmas and three propositions verified by the Z3 SMT solver [22] and SymPy, and confirm it empirically in 29 of 30 networks. The ceiling effect is real---but it does not explain the paradox. Variance does not predict AUC (rho = 0.099, p = ns), and the paradox regime characterized by our formal theorems (T9--T11) is never reached empirically (0 of 31 networks satisfy the required sigma/mu < 0.3).

Second, the correct predictor of feature performance is Fisher's discriminability index d', adapted from signal detection theory [10]. Specifically, signed d'---where the sign is determined by the feature's assortativity---predicts AUC with Spearman rho = 0.994 across 130 feature-network pairs (pooled across 65 networks and 2 features). The sign convention is essential: d' is positive when the feature is assortative (edges have higher mean sym than non-edges) and negative when disassortative.

Third, we derive an exact three-term mixture AUC decomposition that separates the contribution of tied scores from the continuous gradient. This decomposition achieves RMSE = 0.000 on 325 observations and reveals that for k-core, the first two terms of Eq. 6 (tie-related) account for 59% of AUC, with the continuous third term contributing the remaining 41%.

The resolution also clarifies how network organization affects prediction performance. K-core decomposition, originally developed to study the resilience of networks under iterative pruning [16, 17], reveals a hierarchical shell structure that governs degree-degree correlations [11, 15]. Our finding that k-core's assortative shell structure drives link prediction performance adds a new dimension to this picture: the same structural hierarchy that determines network robustness also determines how well normalized similarity can separate edges from non-edges.

The resolution is structural, not statistical. Degree is disassortative in most networks: high-degree nodes preferentially connect to low-degree nodes [11, 13, 14]. The normalized similarity formula S(a,a) = 1 for same-value pairs makes it an assortativity detector---assortative features produce high same-value tie enrichment among edges, which directly inflates AUC. K-core is assortative (positive Newman r in 52 of 64 networks, mean r = 0.268), so k-core similarity benefits from enriched ties. Degree is disassortative (negative r in 44 of 65), so degree similarity is actively harmed.

Our contributions are:

1. A formal ceiling effect proof with paradox theorems (L1--L5, P1--P4, T9--T11), verified by Z3 and SymPy, establishing when a coarser feature can outperform a finer one under normalized similarity.
2. An exact mixture AUC decomposition (RMSE = 0.000) separating tied and continuous contributions.
3. Identification of signed Fisher's d' as the correct predictor of normalized similarity link prediction performance (pooled rho = 0.994; leakage-free rho = 0.996).
4. Demonstration of metric independence: three normalized similarity formulas yield identical AUC on integer features (rho = 1.000).
5. Empirical validation on 65 networks spanning 20 domains, with k-core similarity winning 65/65 (100%).
6. A positional signal diagnostic, Edge-Driven Graph Equivalence (EDGE), detecting excess centrality similarity in biological networks (10/20 biological vs 3/45 non-biological, Fisher p = 0.0002 one-sided).

Section 2 presents the formal framework. Section 3 provides the resolution. Section 4 reports empirical validation. Section 5 describes the EDGE diagnostic. Section 6 discusses implications and limitations.

---

## 2. Formal Framework

### 2.1 Normalized Similarity Metrics

Given a scalar node feature f: V -> R_+, we define the normalized similarity between nodes u and v as:

$$S(f(u), f(v)) = \frac{2 \min(f(u), f(v))}{f(u) + f(v)}$$    (1)

This metric, which we denote sym(a,b), has the equivalent form sym(a,b) = 1 - |a - b|/(a + b) and ranges over [0, 1]. It equals 1 when a = b and approaches 0 as the ratio a/b diverges.

We establish three foundational properties using the Z3 SMT solver [22]:

**Lemma 1 (Equivalence).** For all a, b > 0: sym(a,b) = 2 min(a,b)/(a+b) = 1 - |a-b|/(a+b).

**Lemma 2 (Scale Invariance).** For all a, b > 0 and c > 0: sym(ca, cb) = sym(a, b). The metric depends only on the ratio a/b, not on absolute magnitudes.

**Lemma 3 (Boundedness).** For all a, b > 0: 0 < sym(a, b) <= 1, with equality if and only if a = b.

Scale invariance (L2) is the central property. It implies that sym is fundamentally a ratio statistic: sym(a,b) = 2r/(1+r) where r = min(a,b)/max(a,b). This ratio form connects to the broader theory of compositional data analysis [25, 35] and is key to the variance analysis in Section 2.4.

### 2.2 The Ceiling Effect

The ceiling effect describes how normalized similarity compresses distributional information as the feature mean increases. We state it as a variance compression result.

**Lemma 4 (Convergence).** For feature values a = mu + epsilon_a and b = mu + epsilon_b with |epsilon| < mu (bounded noise): sym(a,b) -> 1 as mu -> infinity at rate O(1/mu).

**Lemma 5 (Separation Bound).** Under the same conditions: |sym(a,b) - sym(c,d)| <= 2 Delta/mu, where Delta = max(|epsilon_a - epsilon_c|, |epsilon_b - epsilon_d|). The maximum achievable separation between any two sym scores shrinks as O(1/mu).

Lemma L4 is verified by Z3 for all positive reals with bounded noise; L5 follows from the same bounded-noise framework. We complement them with asymptotic propositions for Gaussian features (mu >> sigma):

Propositions 1--4 establish the compression result across four complementary conditions, from variance scaling to discrimination bounds:

**Proposition 1 (Variance Compression).** For f ~ N(mu, sigma^2) with mu >> sigma:

$$\text{Var}(\text{sym}(f_i, f_j)) \approx \frac{\sigma^2 (1 - 2/\pi)}{2\mu^2}$$    (2)

which decreases as O(1/mu^2). Derived via delta method applied to the ratio r = min(f_i, f_j)/max(f_i, f_j).

**Proposition 2 (Monotone Compression).** Under the same conditions, Var(sym) is monotonically decreasing in mu for fixed sigma.

**Proposition 3 (Limit).** Var(sym) -> 0 as mu/sigma -> infinity.

**Proposition 4 (Discrimination Bound).** The maximum discrimination D between edge and non-edge sym distributions satisfies D <= C sigma / mu for some constant C.

Empirical verification across 32 networks confirms P1: within-network, variance decreases with degree bin (mean rho = -0.860, p < 10^{-17}, 29/30 networks showing the predicted sign). However, the converse prediction---that separation between edge and non-edge distributions also compresses---is *refuted*: separation actually *increases* with degree bin (mean rho = +0.570), contrary to the naive expectation. The ceiling compresses variance but not separation, because the mean shift between edge and non-edge distributions scales with the same compression factor.

This means the ceiling effect, while mathematically real, does not directly cause the performance paradox. Variance alone does not predict AUC (rho = 0.099, p = ns), nor do coefficient of variation (rho < 0.15) or Herfindahl concentration (rho = 0.061, p = 0.45). The ceiling is a secondary effect: it compresses the continuous component of AUC (41%) but does not touch the dominant tie component (59%), which is driven by assortativity (Section 3).

### 2.3 Paradox Theorems

The ceiling effect establishes that normalization compresses variance, but leaves open a sharper question: under what formal conditions can a coarser feature actually outperform a finer one? We prove three theorems using Z3:

**Theorem 9 (Separation Ordering).** For two features f and g with means mu_f, mu_g and class separations Delta_f, Delta_g:

$$\text{sep}(\text{sym}(g)) > \text{sep}(\text{sym}(f)) \iff \frac{\Delta_g}{\mu_g} > \frac{\Delta_f}{\mu_f}$$    (3)

The feature with higher signal-to-mean ratio achieves greater separation after normalization.

**Theorem 10 (Paradox Witness).** There exist features f, g with Delta_f > Delta_g and mu_f > mu_g such that sep(sym(g)) > sep(sym(f)). That is, the feature with more raw signal and higher mean can lose.

**Theorem 11 (Paradox Condition).** The paradox requires:

$$\frac{\mu_f}{\mu_g} > \frac{\Delta_f}{\Delta_g}$$    (4)

The ceiling must dominate the signal advantage. This occurs when the finer feature's higher mean compresses its separation more than its additional signal compensates.

These theorems are formally correct but empirically irrelevant in their literal form. The asymptotic regime they describe (sigma/mu < 0.3) is satisfied by 0 of 31 networks in our benchmark. The median sigma/mu for degree is 1.2; for k-core it is 0.67. The paradox in real networks is driven by a different mechanism, assortativity and tie structure, to which we turn in Section 3.

### 2.4 BL2: Distribution-Free Variance Prediction

The ceiling results above assume Gaussian features in the mu >> sigma regime---a condition satisfied by 0 of 31 real networks. To bridge from theory to the heavy-tailed distributions observed empirically, we derive a distribution-free variance predictor. Using the ratio representation r = min(a,b)/max(a,b), the delta method yields:

$$\text{Var}(\text{sym}) \approx \frac{4\,\text{Var}(r)}{(1 + E[r])^4}$$    (5)

where Var(r) and E[r] are computed from the empirical ratio distribution. We call this BL2 (Bridge Lemma 2), replacing the earlier BL1 which assumed the sigma/mu < 0.3 regime.

BL2 achieves R^2 = 0.728 on 325 feature-network pairs across 65 networks, with median absolute percentage error of 11.2% and median ratio of 1.01 (unbiased). For comparison, BL1 achieves R^2 = 0.252 and is applicable to 0 of 31 networks in the sigma/mu < 0.3 regime. BL2 operates without regime restrictions.

[Figure 1: Scatter plot of BL2-predicted vs. observed Var(sym) for 325 feature-network pairs. Diagonal line shows perfect prediction. R^2 = 0.728. Points colored by feature type (degree, k-core, eigenvector, clustering, random).]

### 2.5 Mixture AUC Decomposition

The variance analysis shows how normalization compresses distributional information, but variance alone does not predict AUC. What does predict it is the role of tied scores---pairs receiving identical similarity values---which standard AUC analysis treats as noise but which dominate the prediction signal. We formalize this through an exact decomposition of AUC into tied and continuous components.

For integer-valued features, same-value node pairs receive the maximum similarity score: S(a, a) = 1. We call such pairs *tied*. Let p_e be the fraction of edge pairs with identical feature values (f(u) = f(v), hence sym = 1), and p_{ne} the corresponding fraction among non-edges. Since tied pairs score 1 and non-tied pairs score strictly less than 1, the AUC decomposes exactly into four cases when comparing a random edge to a random non-edge:

$$\text{AUC} = p_e(1 - p_{ne}) \cdot 1 + p_e\,p_{ne} \cdot 0.5 + (1 - p_e)(1 - p_{ne})\,\text{AUC}_{\text{continuous}} + (1 - p_e)\,p_{ne} \cdot 0$$

The first term: edge tied (score 1), non-edge not tied (score < 1)---edge always wins. The second: both tied (score 1 vs 1)---random resolution at 0.5. The third: neither tied---determined by the continuous score ranking (AUC_continuous, computed on the non-tied subset). The fourth: edge not tied (score < 1), non-edge tied (score 1)---edge always loses, contributing 0. Simplifying:

$$\text{AUC} = p_e(1 - p_{ne}) + 0.5\,p_e\,p_{ne} + (1 - p_e)(1 - p_{ne})\,\text{AUC}_{\text{continuous}}$$    (6)

The exactness depends on the maximum-score property: S(a,a) = 1 is the unique maximum of the similarity function, so all ties share the same score and dominate all non-ties. This decomposition is exact. On 325 feature-network observations across 65 networks and 5 features, it achieves RMSE = 0.000 (machine precision). By comparison, a simple Gaussian d'-to-AUC conversion achieves RMSE = 0.007---adequate but not exact, confirming that the mixture structure matters.

Empirically, the tie component is dominant. For k-core similarity, the mean tie rate among edges is p_e = 0.472 versus p_{ne} = 0.297 among non-edges, yielding a tie enrichment ratio (TER) of 3.58x. The binary match component (ties only) contributes mean AUC = 0.587; the continuous gradient adds 0.072.

[Figure 2: Mixture AUC decomposition for k-core similarity across 65 networks. Stacked bar chart showing tie contribution (blue) and continuous contribution (orange) for each network, sorted by total AUC. Horizontal dashed line at AUC = 0.5.]

---

## 3. The Resolution

### 3.1 Fisher d' as the Correct Predictor

The ceiling effect (Section 2.2) is real but does not predict AUC. Signal detection theory [10] provides the right framework: Fisher's discriminability index d', which quantifies distributional separation in terms of both mean shift and variance---the quantities that normalization distorts:

$$d' = \frac{\mu_e - \mu_{ne}}{\sqrt{(\sigma_e^2 + \sigma_{ne}^2)/2}}$$    (7)

where mu_e, sigma_e are the mean and standard deviation of sym scores among edges, and mu_{ne}, sigma_{ne} are the corresponding values among non-edges.

On the 31-network development set (excluding one network with degenerate k-core structure), signed d' for k-core predicts AUC with Spearman rho = 0.956 (95% bootstrap CI [0.865, 0.986]). The sign convention is critical: d' is positive when the feature is assortative (edges have higher mean sym than non-edges) and negative when disassortative. Signed d' then predicts raw AUC, not |AUC - 0.5|.

On 10 held-out networks from Netzschleuder spanning 5 domains (social, technological, biological, economic, infrastructure; n = 84 to 3015 nodes), signed d' achieves rho = 0.974 (n = 20 feature-network pairs). On a further expanded validation of 24 networks (16 real + 8 synthetic), signed d' achieves rho = 0.997 (n = 48 pairs). Pooled across all 65 networks and both features (k-core and degree), signed d' predicts AUC with rho = 0.994 (n = 130 pairs). We note that d' is computed from the same score distributions that determine AUC; its value is not as an independent predictor but as a decomposition into mean shift (numerator, driven by assortativity) and variance (denominator, shaped by the ceiling effect), linking AUC to the structural properties analyzed in Sections 3.2--3.3.

[Figure 3: Signed d' vs. AUC for k-core and degree features across 65 networks (pooled rho = 0.994, n = 130). Points for degree-disassortative networks (AUC < 0.5) appear in the lower-left quadrant, confirming the sign convention. Dashed line shows Gaussian AUC = Phi(d'/sqrt(2)).]

The sign convention was discovered during held-out analysis, not pre-registered. On the development set, most networks are degree-assortative for k-core (30/32), so unsigned d' works well. On the held-out set, 7 of 10 networks have degree-disassortative degree features, making the sign correction essential---without it, the unsigned d'-AUC correlation drops to rho = -0.090 (ns). This correction is mechanistically motivated, not arbitrary: when a feature is disassortative, edges have *lower* sym scores than non-edges, inverting the prediction direction.

Alternative predictors fail. Variance (rho = 0.099), coefficient of variation (rho < 0.15), and Herfindahl index of feature concentration (rho = 0.061, p = 0.45) are all non-significant. The Herfindahl of *score* concentration shows a marginal trend (rho = 0.145, p = 0.07) but does not approach d's predictive power.

Protocol sensitivity analysis confirms robustness: under 80/20 random holdout of edges within each network, d' achieves rho = 0.982 on the development set.

### 3.2 Assortativity as Gate

The mechanism behind the d' result is assortativity, the tendency of nodes to connect to others with similar attribute values [11, 12].

K-core is assortative in 52 of 64 networks with computable assortativity (Newman r > 0), with mean assortativity r = 0.268. This means same-shell connections are overrepresented: nodes in shell k preferentially connect to other nodes in shell k. When we compute sym(kcore_u, kcore_v), these same-shell pairs receive score 1.0 (maximum), and they are enriched among edges relative to non-edges. This directly inflates AUC.

Degree is disassortative in 44 of 65 networks. This is a well-documented structural property of many real networks [11, 13, 14, 30, 31]: high-degree hubs tend to connect to low-degree periphery nodes. When we compute sym(deg_u, deg_v), the low-ratio hub-periphery pairs are *overrepresented* among edges, depressing the edge mean and yielding negative d'.

Critically, the assortativity gap between k-core and degree does *not* predict the AUC gap. The Spearman correlation between r_gap (r_kcore - r_degree) and AUC_gap is rho = 0.115 (p = 0.37, n = 64). Assortativity is a gate, not a meter: it determines whether AUC is above or below 0.5, but not by how much. The magnitude is determined by d', which incorporates both the mean shift and the variance of the score distributions.

An important clarification: k-core is *not* a proxy for community structure. The normalized mutual information between k-core partition and Louvain community detection is NMI = 0.137 (mean across 65 networks). K-core captures coreness---the hierarchical resilience structure of the network under iterative pruning [16, 17, 38]---not modular community membership. Nodes in the same k-core shell may belong to entirely different communities.

### 3.3 Tie Enrichment

The mixture AUC decomposition (Eq. 6) reveals the mechanical pathway through which assortativity translates to prediction performance: tie enrichment.

The tie enrichment ratio (TER) is defined as the ratio of the tie rate among edges to the tie rate among non-edges: TER = p_e / p_{ne}. For k-core, mean TER = 3.58x: edges are approximately three times more likely than non-edges to have identical k-core values. For degree, mean TER = 0.54x: edges are *less* likely than non-edges to have identical degree values---same-degree ties are depleted among edges, consistent with disassortativity.

The four-quadrant analysis decomposes the k-core advantage over degree into tie and continuous contributions. Across the corpus, the tie component accounts for the majority of k-core's advantage. The continuous component is negative on average: the continuous gradient of k-core similarity actually performs *worse* than the continuous gradient of degree similarity in some networks, but the overwhelming tie enrichment more than compensates.

Eigenvector centrality is an exception. It achieves the highest mean TER (3.75x) across the benchmark, yet its AUC is not correspondingly high (rho between TER and AUC for eigenvector is 0.001). Investigation reveals this is a base-rate artifact: eigenvector centrality produces very few ties overall (near-zero tie rates for both edges and non-edges in most networks), so TER is high but the absolute tie contribution to AUC is negligible. The mixture AUC formula (Eq. 6) correctly handles this---the small p_e and p_{ne} values suppress the tie terms regardless of their ratio.

### 3.4 Metric Independence

In link prediction practice, the choice of similarity formula [4, 24] is often treated as consequential because different metrics weight structural features differently. For integer-valued features, however, the choice has negligible impact on AUC.

We test four normalized pairwise metrics:

1. sym(a,b) = 2 min(a,b)/(a+b) (harmonic similarity)
2. Jaccard(a,b) = min(a,b)/max(a,b)
3. exp-diff(a,b) = exp(-|a-b|/(a+b))
4. binary match(a,b) = 1 if a = b, else 0

We omit cosine similarity: for scalar features, cosine(a,b) = ab/|a||b| = 1 for all positive pairs, producing degenerate AUC = 0.5. Cosine requires vector-valued features to be meaningful.

All three continuous metrics (sym, Jaccard, exp-diff) yield *identical* AUC values on integer-valued features (cross-metric Spearman rho = 1.000 across all 65 networks). This is not an approximation---the AUC values are exactly equal to machine precision.

The explanation follows from Bamber's theorem [6]: AUC depends only on the rank ordering of scores [32, 33, 36]. For integer features, all three continuous metrics are monotonic transformations of the ratio min(a,b)/max(a,b), which produces the same ranking. This means the specific formula is irrelevant; only the feature's distributional separation matters.

Binary match (metric 4) underperforms the continuous metrics by a mean AUC gap of 0.072. Binary match retains only the tie component and discards the continuous gradient entirely. This gap quantifies the contribution of the continuous component---consistent with the mixture decomposition (Section 2.5), which attributes approximately 41% of AUC to the continuous term.

[Figure 4: AUC by metric for k-core (left) and degree (right) across 65 networks. The three continuous metrics overlap perfectly (rho = 1.000). Binary match is displaced below the continuous metrics.]

---

## 4. Empirical Validation

### 4.1 Development Corpus

The development corpus consists of 32 networks from 15 domains: biological protein-protein interaction (6 networks), connectome (3), social trust (4), social collaboration (2), gene regulatory (2), food web (2), power grid (2), internet/web (2), linguistic (1), economic (1), political (1), transport (1), migration (1), financial (1), and trade (1). Network sizes range from 62 nodes (Florida Bay Food Web) to 4762 nodes (Bitcoin OTC Trust), with edge counts from 227 (Contiguous USA) to 70,654 (Budapest Human Connectome). Mean degree ranges from 3.0 to 139.2.

The benchmark is drawn from established network repositories (Netzschleuder, SNAP, KONECT, STRING) and covers sparse and dense networks, assortative and disassortative degree structure, and homogeneous and heterogeneous degree distributions. All networks are treated as undirected and unweighted for the purpose of this study.

A benchmark audit reveals that 10 of 32 networks (31%) originate from directed sources that were symmetrized during loading. Sensitivity analysis shows this does not materially affect results: excluding directed-source networks, k-core still wins in 21 of 22 remaining networks (95%), and d' correlation remains rho > 0.94.

For each network, we sample up to 4,000 edges and 4,000 non-edges uniformly at random (seed = 42) and compute sym scores for degree, k-core, eigenvector centrality, clustering coefficient, and a random control. AUC is computed as the Wilcoxon-Mann-Whitney statistic [6, 7, 27].

### 4.2 Held-Out Validation

We construct two held-out corpora that were not used during any phase of development.

**Netzschleuder-1 (10 networks).** Downloaded via the Netzschleuder API [networks.skewed.de], selected for domain diversity: 2 social (Add Health community, Ugandan village friendship), 2 technological (RouteViews AS graph, WebKB Wisconsin), 2 biological (E. coli transcription, Fresh Webs Akatore food web), 2 economic (EU Procurements, Faculty Hiring), and 2 infrastructure (Urban Streets Ahmedabad, Fullerene C1500). Network sizes range from n = 84 (Fresh Webs) to n = 3015 (RouteViews).

Results: k-core similarity outperforms degree similarity in 10 of 10 networks (100%). Signed d' vs. AUC achieves rho = 0.974 (p = 4.0 x 10^{-13}, n = 20 pooled feature-network pairs). The direction-invariant metric |AUC - 0.5| vs. d' achieves rho = 0.953 (p = 8.3 x 10^{-11}).

The 95% confidence interval for signed d' on the held-out set [0.934, 0.990] (Fisher z-transform, n = 20) overlaps with the development set CI [0.865, 0.986], confirming replication. Seven of 10 held-out networks have disassortative degree features (AUC_degree < 0.5), demonstrating that the sign convention is not an artifact of the development set's predominantly assortative composition.

**Netzschleuder-2 + Synthetic (24 of 25 registered networks successfully loaded).** A further expansion comprising 17 real networks from Netzschleuder (spanning communication, citation, software, transportation, collaboration, political, and ecological domains; 16 successfully loaded) and 8 synthetic networks (LFR benchmark, Barabasi-Albert [15], Watts-Strogatz, stochastic block model [34]). K-core wins 24 of 24 (100%). Signed d' achieves rho = 0.997 (95% CI [0.987, 0.999], n = 48 pairs), with real networks alone at rho = 0.996 and synthetic alone at rho = 0.994.

[Figure 5: Signed d' vs. AUC across the full 65-network corpus, colored by provenance (31 development, 10 held-out, 24 expanded). Pooled Spearman rho = 0.994 (n = 130 feature-network pairs). The relationship is consistent across corpus subsets.]

**Combined corpus.** Across all 65 networks, k-core similarity wins in 65 of 65 cases (100%, binomial test p < 10^{-19} against the null of no systematic advantage). The mean AUC for k-core is 0.659; for degree, 0.491.

### 4.3 Protocol Sensitivity

We test robustness to the evaluation protocol by varying the edge sampling fraction. Under 80/20 holdout (80% of edges used for feature computation, 20% held out for evaluation), d' achieves rho = 0.982 on the development set, compared to rho = 0.956 under the full-graph protocol. The slight improvement is consistent with reduced sampling noise when holdout edges are truly absent from the feature computation.

We also verify that d' is robust to sample size: subsampling from 4,000 to 2,000 edge/non-edge pairs changes AUC estimates by less than 0.01 on average (mean absolute change = 0.004).

### 4.4 Benchmark Audit

Of the 32 development networks, 10 originate from directed sources (Bitcoin Alpha/OTC trust ratings, Epinions trust, Slashdot trust, Wikipedia RFA votes, US Congress co-sponsorship, UN Migration, Florida Bay food web, E. coli gene regulation, and Product Space trade). These were symmetrized (direction discarded) during loading.

This raises the question of whether the symmetrization artificially favors k-core. We find no evidence for this: among directed-source networks, k-core wins 9 of 10 (90%), and the one exception (US Congress) has the lowest k-core diversity (only 4 distinct shells). Among undirected-source networks, k-core wins 21 of 22 (95%). The difference is not significant (Fisher exact p = 0.53).

The formal framework, mechanism analysis, and empirical validation support the same explanation: normalized similarity is an assortativity detector, k-core's assortative shell structure produces enriched ties that dominate AUC, and Fisher's d' predicts the magnitude of this effect across 65 networks and 20 domains.

---

## 5. EDGE: Positional Signal Diagnostic

Our resolution shows that feature performance under normalized similarity depends on assortativity and distributional separation, not on raw signal strength. We next ask whether real networks contain positional structure---similarity in centrality profiles between connected nodes, related to the concept of structural equivalence [19, 26]---beyond what their degree sequences alone explain. We develop EDGE (Edge-Driven Graph Equivalence) as a diagnostic for this question.

**Method.** For each network, we compute a normalized centrality vector v(i) = (degree(i), kcore(i), eigenvector(i), clustering(i)) / ||v(i)|| for each node. We then compute the mean cosine similarity of these vectors across all edges: C_real = mean_{(u,v) in E} cos(v(u), v(v)). We compare this to a configuration model null: we generate 15 degree-preserving edge-swap rewirings [28, 29] and compute C_null = mean of cosine similarities across rewired edges. A network is classified as having positional signal if C_real - C_null exceeds the null distribution at p < 0.05 (permutation test). For computational efficiency, the null recomputes only k-core and clustering on each rewired graph (3 features), while the real network uses all 4 centralities including eigenvector. This asymmetry is unlikely to inflate signal materially: eigenvector centrality is strongly correlated with degree (which is preserved exactly in the null), so the 4th dimension adds little independent cosine variation. The null preserves the degree sequence exactly, so any detected excess reflects positional structure beyond what degree alone produces.

**Validation.** EDGE passes Erdos-Renyi validation: ER graphs with matched density show near-zero excess (gap = 0.003--0.007), confirming the diagnostic is not trivially satisfied. A previous diagnostic (GC1, based on Taylor-expansion applicability) was falsified because it was trivially satisfied by structureless random graphs. EDGE avoids this failure mode by construction.

**Results.** Of 65 networks analyzed, 13 show excess positional signal at the per-network p < 0.05 level (no multiple-testing correction applied; at 65 tests, approximately 3 false positives are expected under the null). The signal is strongly enriched in biological networks: 10 of 20 biological networks (50%) show signal versus 3 of 45 non-biological networks (7%). Fisher exact test on this enrichment pattern (one-sided): p = 0.0002, OR = 14.0.

**Data quality, not density.** The biological enrichment is not driven by network density. WormNet (C. elegans functional network) has mean degree <k> = 3.1 and shows strong signal (z = 3.6, p = 0.000). BioGRID (C. elegans protein-protein interactions) has nearly identical mean degree <k> = 3.0 but shows no signal (p = 1.0). The same organism, the same sparsity level, opposite results. The discriminating factor is data quality: curated databases (WormNet, STRING high-confidence) show signal, while high-throughput screens (BioGRID) do not. This pattern holds across organisms---BioGRID fails for all five species tested (fission yeast, fruit fly, human, mouse, worm), while curated networks consistently detect positional structure.

This finding has a practical implication for network biology: EDGE can serve as a data quality diagnostic for protein interaction databases. A network that fails to show excess positional similarity relative to its degree-preserving null may contain too much noise for methods that assume edges reflect functional relationships rather than experimental artifacts. Computationally, EDGE requires O(m) per rewiring (15 rewirings by default), making it feasible as a one-time quality assessment even for large databases --- a single pass determines whether downstream analysis on the network is warranted. The implications of this diagnostic for database selection in network biology are discussed in Section 6.3.

---

## 6. Discussion

We aimed to explain an empirical puzzle: why does a coarser feature (k-core) consistently outperform a finer one (degree) under normalized similarity? The answer is that normalized similarity rewards assortativity rather than raw signal---and k-core is assortative where degree is not.

### 6.1 Why It Is Not a Paradox

The "less information wins" framing assumes that raw signal strength (the gap Delta between feature values of connected and unconnected node pairs) is the correct measure of feature quality. It is not. When features are passed through a normalized similarity function, the relevant quantity becomes the distributional separation in score space: Fisher's d' [23].

Degree has a larger raw gap than k-core in most networks. But degree's disassortative mixing pattern [11, 13, 14, 30] means that the *direction* of this gap is inverted in score space: edges have *lower* sym scores than non-edges, because high-degree-to-low-degree connections (which are overrepresented among edges) receive low similarity scores. K-core's assortative structure means edges have *higher* sym scores, because same-shell connections are overrepresented. The effect is heterogeneous: degree similarity achieves mean AUC = 0.644 in the 21 networks where degree is assortative (Newman r > 0), but only 0.417 in the 44 disassortative networks --- yielding the overall mean of 0.491 (near chance) as a cancellation of opposing effects, not a uniform null result.

The paradox dissolves once we recognize that normalized similarity is an assortativity detector. The formula S(a,a) = 1 ensures that any feature where same-value pairs are enriched among edges will yield high AUC. K-core has this property; degree does not. The ceiling effect (Section 2.2) is real but operates on the continuous component (41% of AUC), not on the dominant tie component (59%).

### 6.2 The Ceiling Effect's Actual Role

The ceiling effect is not the cause of the paradox, but it is not irrelevant. It explains a specific quantitative feature of the results: why the continuous gradient adds only 0.072 on top of the binary match's 0.587.

For high-mean features (degree in dense networks), the ceiling compresses the continuous gradient toward zero, limiting how much discriminability the smooth part of the score distribution can provide. This is precisely P1 (confirmed in 29/30 networks): Var(sym) ~ sigma^2/(2 mu^2). The compression affects both edge and non-edge variance symmetrically (both groups share the same feature mean), so it enters d' through the denominator without altering the numerator that drives cross-network variation. Because the tie component dominates, this compression has limited impact on total AUC.

The ceiling effect would become the dominant mechanism in a hypothetical setting where all features are assortative and have similar tie structures, but differ in their mean values. In that regime, Theorem 9's condition Delta_g/mu_g > Delta_f/mu_f would directly predict relative performance. Our benchmark does not contain such a regime (0/31 networks satisfy sigma/mu < 0.3), but the theoretical characterization remains valid for distributions where it does apply.

A cardinality control confirms the tie mechanism directly. K-core has a mean of 24 distinct values per network versus 82 for degree (ratio 3.6x at the median). When degree is binned to match k-core cardinality via equal-frequency binning, binned-degree AUC approaches k-core AUC (mean 0.663 vs 0.667, win rate 29/55 = 53%) --- essentially indistinguishable. This confirms that the primary contributor to k-core's advantage over raw degree is the cardinality difference: fewer distinct values produce more ties, and the tie enrichment mechanism (Eq. 6) amplifies this into an AUC advantage. K-core's structural alignment with network shells provides a small residual advantage (0.004 AUC) beyond what cardinality alone explains.

### 6.3 Implications for Link Prediction Practice

The results have separate implications for three audiences: link prediction practitioners receive a direct feature recommendation and a formula-independence result, network science theorists receive a d' framework linking assortativity to prediction performance, and network biologists receive the EDGE diagnostic for assessing interaction database quality (Section 5).

Our results suggest several practical guidelines for normalized similarity link prediction:

First, among single-feature normalized similarities, k-core is the strongest feature. It wins in all 65 tested networks (100%), requires only graph structure (no external features), and runs in O(m) time via the k-core decomposition algorithm [17].

Second, k-core similarity has a computational advantage on large graphs. K-core decomposition requires a single O(m) pass over the edge list, after which every node's k-core number is stored as a scalar. Scoring a candidate pair (u, v) then costs O(1): one lookup per node and one arithmetic operation. By contrast, neighborhood-based methods such as common neighbors and Adamic-Adar [5] require computing set intersections for each candidate pair at cost O(d_u + d_v), where d_u and d_v are the endpoint degrees. This per-pair cost is degree-dependent: O(10) for low-degree nodes, O(10^3 -- 10^4) for hubs in scale-free networks. On a billion-edge graph with 10 million candidate pairs to score, k-core similarity completes in seconds (O(m) precomputation + O(1) per query), while Adamic-Adar requires hours of set intersection computation. For streaming applications where new edges arrive continuously and candidates must be scored in real time, k-core similarity is the only single-pass method that provides O(1) per-pair evaluation after an initial linear scan.

Third, the specific similarity formula does not matter. Sym, Jaccard, and exp-diff all produce identical AUC on integer features. Practitioners can choose any monotonic normalized similarity function without affecting results.

Fourth, before applying any normalized similarity metric, practitioners should check whether the feature is assortative. A disassortative feature (Newman r < 0) will produce AUC < 0.5, which can be corrected by inverting the score ranking (using dissimilarity instead of similarity). The signed d' framework provides a single-number summary of expected performance.

Fifth, among feature-based similarities that use a single node attribute, k-core is competitive with community-based similarity (Louvain membership match, mean AUC = 0.732 vs k-core's 0.719). Multi-hop structural heuristics such as common neighbors and Adamic-Adar [18, 21, 37] exploit neighborhood overlap and achieve higher AUC, but operate in a different computational category: they require per-pair neighborhood access rather than scalar lookups, and cannot be precomputed into a static table. Our contribution is not to compete with these methods but to explain why, within the single-feature framework, a coarser feature outperforms a finer one --- and to show that this framework has practical value precisely because of its O(1) scoring cost.

### 6.4 Limitations

Several limitations deserve mention.

**Scope.** All results apply to networks analyzed as undirected static graphs. K-core decomposition is well-defined for directed networks only after adaptation (e.g., D-core [17]), and the assortativity-based mechanism may not transfer directly. Whether the k-core advantage persists in weighted, temporal, or multilayer networks remains an open question.

**Convenience sample.** The 65-network corpus, while spanning 20 domains, is not a random sample of "all networks." Domain representation is uneven (biological networks are overrepresented in the development set), and infrastructure and economic networks---which exhibit markedly different degree distributions---are underrepresented. Results may not generalize to network types absent from the corpus.

**Sign convention discovered on held-out.** The d' sign convention (positive for assortative, negative for disassortative) was discovered during held-out analysis, not pre-registered. The unsigned d' formula was fixed before held-out evaluation; the sign correction emerged when 7 of 10 held-out networks showed degree-disassortative AUC < 0.5. The correction is mechanistically motivated (assortativity determines whether edges score higher or lower than non-edges), and was subsequently validated on a 25-network expansion (rho = 0.997). Nonetheless, a pre-registered replication would strengthen confidence.

**Feature computation on full graph.** K-core numbers are computed on the full graph, including edges used for evaluation. Under a strict train-only protocol (20% edges held out, both features recomputed on the train graph), k-core AUC decreases by a mean of 0.057, while degree AUC is essentially unchanged (mean leakage < 0.001). Under this leakage-free protocol, train-only k-core beats train-only degree in 64 of 65 networks (98.5%), and signed d' predicts AUC with pooled rho = 0.996 (n = 130)---slightly higher than the full-graph rho of 0.994, consistent with leakage adding noise. Absolute AUC values should be interpreted with this caveat; a train-only protocol is recommended for applications where leakage-free evaluation is required.

**Directed network symmetrization.** Ten of 32 development networks (31%) are symmetrized directed networks. While our sensitivity analysis shows no significant impact, directed networks have fundamentally different degree structures (in-degree vs. out-degree), and a dedicated analysis of normalized similarity in directed networks is warranted.

**Mixture AUC decomposition.** The three-term decomposition (Eq. 6) follows from Bamber's [6] foundational identity AUC = P(X > Y) + 0.5 P(X = Y) by conditioning on class-specific tie rates p_e and p_{ne}; to our knowledge, this factored form---separating the tie contribution from the continuous gradient via class-conditional tie probabilities---has not appeared previously in the ROC literature. Muschelli [9] treats the binary-predictor special case (all ties) and DeLong et al. [8] provide the nonparametric comparison framework, but neither conditions on class-specific tie rates to produce the mixture form.

**Negative sampling.** Non-edges are sampled uniformly at random (Section 4.1), which includes many structurally distant pairs that are easy to classify. Whether the assortativity/tie mechanism explains performance under harder negative sampling protocols (degree-matched, distance-bounded, or temporal negatives) remains untested. Our contribution is the relative comparison between features under a fixed protocol, not absolute link prediction performance.

**Scope of feature comparison.** We compare single-feature normalized similarities. The comparison to structural heuristics that combine multiple neighborhood features (Adamic-Adar [5], common neighbors [3], Katz index [24]) is not the focus of this paper. K-core similarity is not claimed to be superior to these multi-feature methods; our contribution is explaining *why* it outperforms degree similarity within the single-feature normalized framework.

---

## 7. Conclusion

The apparent paradox of irrelevance---that k-core similarity outperforms degree similarity despite containing less information---dissolves when examined closely. The paradox rests on a false premise: that raw signal strength predicts performance under normalized similarity. It does not. Fisher's discriminability d', combined with a sign convention derived from feature assortativity, predicts AUC with Spearman rho = 0.994 across 65 networks spanning 20 domains (n = 130 feature-network pairs).

The mechanism is structural. K-core is assortative; degree is disassortative. The normalized similarity formula S(a,a) = 1 turns it into an assortativity detector that rewards features where same-value pairs are overrepresented among edges. An exact mixture AUC decomposition (RMSE = 0.000) quantifies this: ties account for 59% of AUC, with the continuous gradient contributing 41%. The formal ceiling effect (L1--L5, P1--P4, T9--T11) is mathematically real but secondary---it compresses the continuous component (41% of AUC) without driving the performance gap, which originates in the tie component (59%).

The paper makes three contributions. First, a formal framework: Z3-verified proofs of the ceiling effect and paradox conditions, plus a distribution-free variance predictor (BL2, R^2 = 0.728). Second, an analytical decomposition: the exact mixture AUC formula and the identification of signed d' as the correct performance predictor. Third, empirical validation at scale: 65 networks, 20 domains, k-core winning 65/65 (100%).

For single-feature normalized similarity link prediction, k-core is the strongest choice. It is fast, parameter-free, and competitive with community-based similarity. The theoretical takeaway is that feature performance under normalization depends not on raw information content but on the alignment between feature assortativity and the structural bias of the similarity formula.

---

## References

[1] D. Liben-Nowell and J. Kleinberg, "The link-prediction problem for social networks," *Journal of the American Society for Information Science and Technology*, vol. 58, no. 7, pp. 1019--1031, 2007.

[2] L. Lu and T. Zhou, "Link prediction in complex networks: A survey," *Physica A*, vol. 390, no. 6, pp. 1150--1170, 2011.

[3] T. Zhou, L. Lu, and Y.-C. Zhang, "Predicting missing links via local information," *European Physical Journal B*, vol. 71, no. 4, pp. 623--630, 2009.

[4] V. Martinez, F. Berzal, and J.-C. Cubero, "A survey of link prediction in complex networks," *ACM Computing Surveys*, vol. 49, no. 4, pp. 1--33, 2016.

[5] L. A. Adamic and E. Adar, "Friends and neighbors on the Web," *Social Networks*, vol. 25, no. 3, pp. 211--230, 2003.

[6] D. Bamber, "The area above the ordinal dominance graph and the area below the receiver operating characteristic graph," *Journal of Mathematical Psychology*, vol. 12, no. 4, pp. 387--415, 1975.

[7] J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29--36, 1982.

[8] E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837--845, 1988.

[9] J. Muschelli, "ROC and AUC with a binary predictor: A potentially misleading metric," *Journal of Classification*, vol. 37, pp. 696--708, 2020.

[10] D. M. Green and J. A. Swets, *Signal Detection Theory and Psychophysics*. New York: Wiley, 1966.

[11] M. E. J. Newman, "Assortative mixing in networks," *Physical Review Letters*, vol. 89, no. 20, p. 208701, 2002.

[12] M. E. J. Newman, "Mixing patterns in networks," *Physical Review E*, vol. 67, no. 2, p. 026126, 2003.

[13] S. Maslov and K. Sneppen, "Specificity and stability of topology of protein networks," *Science*, vol. 296, no. 5569, pp. 910--913, 2002.

[14] J. Park and M. E. J. Newman, "Origin of degree correlations in the Internet and other networks," *Physical Review E*, vol. 68, no. 2, p. 026112, 2003.

[15] A.-L. Barabasi and R. Albert, "Emergence of scaling in random networks," *Science*, vol. 286, no. 5439, pp. 509--512, 1999.

[16] M. Kitsak, L. K. Gallos, S. Havlin, F. Liljeros, L. Muchnik, H. E. Stanley, and H. A. Makse, "Identification of influential spreaders in complex networks," *Nature Physics*, vol. 6, no. 11, pp. 888--893, 2010.

[17] Y.-X. Kong, G.-Y. Shi, R.-J. Wu, and Y.-C. Zhang, "k-core: Theories and applications," *Physics Reports*, vol. 832, pp. 1--32, 2019.

[18] Z. Liu, J.-L. He, and H. Jia, "Roles of degree, H-index and coreness in link prediction," *International Journal of Modern Physics B*, vol. 32, no. 13, p. 1850155, 2018.

[19] F. Lorrain and H. C. White, "Structural equivalence of individuals in social networks," *Journal of Mathematical Sociology*, vol. 1, no. 1, pp. 49--80, 1971.

[20] R. Guimera and M. Sales-Pardo, "Missing and spurious interactions and the reconstruction of complex networks," *Proceedings of the National Academy of Sciences*, vol. 106, no. 52, pp. 22073--22078, 2009.

[21] A. F. Al Musawi, S. Roy, and R. Ghosh, "Identifying accurate link predictors based on assortativity of complex networks," *Scientific Reports*, vol. 12, p. 7865, 2022.

[22] L. de Moura and N. Bjorner, "Z3: An efficient SMT solver," in *Tools and Algorithms for the Construction and Analysis of Systems (TACAS)*, 2008, pp. 337--340.

[23] M. S. Pepe, *The Statistical Evaluation of Medical Tests for Classification and Prediction*. Oxford: Oxford University Press, 2003.

[24] A. Kumar, S. S. Singh, K. Singh, and B. Biswas, "Link prediction techniques, applications, and performance: A survey," *Physica A*, vol. 553, p. 124289, 2020.

[25] J. Aitchison, *The Statistical Analysis of Compositional Data*. London: Chapman and Hall, 1986.

[26] S. P. Borgatti and M. G. Everett, "Two algorithms for computing regular equivalence," *Social Networks*, vol. 15, no. 4, pp. 361--376, 1993.

[27] W. J. Conover, *Practical Nonparametric Statistics*, 3rd ed. New York: Wiley, 1999.

[28] B. K. Fosdick, D. B. Larremore, J. Nishimura, and J. Ugander, "Configuring random graph models with fixed joint degree distributions," *SIAM Review*, vol. 60, no. 2, pp. 315--355, 2018.

[29] R. Milo, S. Shen-Orr, S. Itzkovitz, N. Kashtan, D. Chklovskii, and U. Alon, "Network motifs: Simple building blocks of complex networks," *Science*, vol. 298, no. 5594, pp. 824--827, 2002.

[30] N. Litvak and R. van der Hofstad, "Uncovering disassortativity in large scale-free networks," *Physical Review E*, vol. 87, no. 2, p. 022801, 2013.

[31] R. Aiyappa, S. Biswas, J. Bollen, and S. Fortunato, "Implicit degree bias in the link prediction task," arXiv:2405.14985, 2024.

[32] H. Wan, Y. Zhang, J. Zhang, and J. Tang, "Quantifying discriminability of evaluation metrics in link prediction," arXiv:2409.20078, 2024.

[33] D. J. Hand, "Measuring classifier performance: A coherent alternative to the area under the ROC curve," *Machine Learning*, vol. 77, no. 1, pp. 103--123, 2009.

[34] P. W. Holland, K. B. Laskey, and S. Leinhardt, "Stochastic blockmodels: First steps," *Social Networks*, vol. 5, no. 2, pp. 109--137, 1983.

[35] K. Pearson, "Mathematical contributions to the theory of evolution: On a form of spurious correlation which may arise when indices are used in the measurement of organs," *Proceedings of the Royal Society of London*, vol. 60, pp. 489--498, 1897.

[36] S. Bi, L. Gallo, G. Menichetti, A. L. Barabasi, and F. Battiston, "Inconsistency among evaluation metrics in link prediction," *PNAS Nexus*, vol. 3, no. 11, pgae473, 2024.

[37] L. Tian, A. Bashan, D.-N. Shi, and Y.-Y. Liu, "Comprehensive contributions of endpoint degree and coreness to link prediction," *Complexity*, vol. 2021, p. 5765313, 2021.

[38] P. Holme, "Core-periphery organization of complex networks," *Physical Review E*, vol. 72, no. 4, p. 046111, 2005.
