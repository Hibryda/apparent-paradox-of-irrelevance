# Reference List for "The Apparent Paradox of Irrelevance"

## MUST-CITE (22 papers)

### Link prediction foundations
1. Liben-Nowell & Kleinberg (2007) "The link-prediction problem for social networks" JASIST 58(7)
2. Lü & Zhou (2011) "Link prediction in complex networks: A survey" Physica A 390(6)
3. Zhou, Lü & Zhang (2009) "Predicting missing links via local information" EPJ B 71(4)
4. Martínez, Berzal & Cubero (2016) "A survey of link prediction in complex networks" ACM CSUR 49
5. Adamic & Adar (2003) "Friends and neighbors on the Web" Social Networks 25(3)

### AUC and ROC theory
6. Bamber (1975) "Area above ordinal dominance graph" J Math Psychology 12(4)
7. Hanley & McNeil (1982) "Meaning and use of AUC" Radiology 143(1)
8. DeLong, DeLong & Clarke-Pearson (1988) "Comparing AUCs" Biometrics 44(3)
9. Muschelli (2020) "ROC and AUC with a binary predictor" J Classification 37

### Signal detection
10. Green & Swets (1966) Signal Detection Theory and Psychophysics. Wiley.

### Network structure
11. Newman (2002) "Assortative mixing in networks" PRL 89, 208701
12. Newman (2003) "Mixing patterns in networks" PRE 67, 026126
13. Maslov & Sneppen (2002) "Specificity and stability in topology of protein networks" Science 296
14. Park & Newman (2003) "Origin of degree correlations" PRE 68, 026112
15. Barabási & Albert (1999) "Emergence of scaling in random networks" Science 286

### K-core
16. Kitsak et al. (2010) "Identification of influential spreaders" Nature Physics 6
17. Kong et al. (2019) "k-core: Theories and applications" Physics Reports 832
18. Liu, He & Jia (2018) "Roles of degree, H-index and coreness in link prediction" IJMPB 32(13)

### Structural equivalence
19. Lorrain & White (1971) "Structural equivalence of individuals in social networks" J Math Sociology 1(1)
20. Guimerà & Sales-Pardo (2009) "Missing and spurious interactions" PNAS 106(52)

### Assortativity and link prediction
21. Al Musawi, Roy & Ghosh (2022) "Identifying accurate link predictors based on assortativity" Scientific Reports

### Methodology
22. De Moura & Bjørner (2008) "Z3: An efficient SMT solver" TACAS 2008

## SHOULD-CITE (10 papers)

23. Pepe (2003) Statistical Evaluation of Medical Tests. Oxford.
24. Kumar et al. (2020) "Link prediction techniques, applications, performance" Physica A 553
25. Aitchison (1986) The Statistical Analysis of Compositional Data. Chapman & Hall.
26. Borgatti & Everett (1993) "Two algorithms for computing regular equivalence" Social Networks 15(4)
27. Conover (1999) Practical Nonparametric Statistics, 3rd ed. Wiley.
28. Fosdick et al. (2018) "Configuring random graph models" SIAM Review 60(2)
29. Milo et al. (2002) "Network motifs" Science 298
30. Litvak & van der Hofstad (2013) "Uncovering disassortativity in large scale-free networks" PRE 87
31. Aiyappa et al. (2024) "Implicit degree bias in link prediction" arXiv:2405.14985
32. Wan et al. (2024) "Quantifying discriminability of evaluation metrics" arXiv:2409.20078

## NICE-TO-CITE (5 papers)

33. Hand (2009) "Measuring classifier performance" Machine Learning 77(1)
34. Holland, Laskey & Leinhardt (1983) "Stochastic blockmodels: First steps" Social Networks 5(2)
35. Pearson (1897) "Spurious correlation" Proc Royal Society 60
36. Bi et al. (2024) "Inconsistency among evaluation metrics in link prediction" PNAS Nexus 3(11)
37. Tian et al. (2021) "Comprehensive contributions of endpoint degree and coreness" Complexity

## NOVELTY ASSESSMENT

| Contribution | Novel? | Closest prior art |
|-------------|--------|------------------|
| Mixture AUC 3-term decomposition | YES (application) | Bamber 1975 (foundation), Muschelli 2020 (binary case) |
| Fisher d' for LP feature comparison | YES | Green & Swets 1966 (SDT origin, never applied to LP) |
| Kcore as pairwise similarity score | YES | Liu 2018 (uses kcore as endpoint weight, not pairwise) |
| Assortativity gates prediction mechanism | YES (mechanism) | Al Musawi 2022 (correlational, no AUC decomposition) |
| Z3 for network metric proofs | YES (methodology) | No prior art in network science |
| Metric independence for integer features | YES (observation) | Bamber 1975 (AUC = rank statistic, corollary) |
