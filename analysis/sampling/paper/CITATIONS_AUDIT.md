# Citation Audit

Every citation used in `main.tex` was independently web-verified against authoritative
sources (DOI resolvers, publisher pages, arXiv/repository copies, DBLP, Crossref,
Semantic Scholar). For each: the canonical URL, the verified metadata, the claim it
supports in our paper, verbatim quote(s) where retrievable, a short synopsis, and an
in-context check. Where a verbatim quote could not be fetched (paywall/403), this is
stated explicitly rather than paraphrase being passed off as a quote.

## Corrections applied as a result of this audit

| Key | Problem found | Fix |
| --- | --- | --- |
| `Brandes2007` | venue/DOI wrong; cited DOI resolved to an **unrelated** paper | → IJBC 17(7):2303–2318, doi 10.1142/S0218127407018403 |
| `Borassi2019` | cited DOI resolved to an **unrelated** ESA paper; "ESA 2019" does not exist | → ACM JEA 24(1), 2019, doi 10.1145/3284359 (KADABRA journal version) |
| `Bergamini2019` | **author list fabricated** (real authors Matta, Ercal, Sinha); art. no. wrong | authors fixed; key renamed `Matta2019`; art. 5→2 |
| `Cooper2018` | DOI did not resolve; venue/year/vol/pages all wrong | → SoftwareX 12:100525, 2020, doi 10.1016/j.softx.2020.100525 |
| `Pellegrina2023` | page span wrong | 1–40 → 1–55 |
| `Freeman1979` | year inconsistent with DOI | 1979 → 1978 (DOI encodes `(78)`) |
| `Simons2022` | vol/issue/pages missing | added 50(5):1328–1344 |
| `Turner2007` | issue missing | added no. 3 |
| `Strano2013` (in-text) | "block length 80–180 m" cited to Boeing2017 (OSMnx tool, not a block-length source) and loose | dropped Boeing2017 from the claim; range corrected to city-mean ≈95–122 m |
| `Eppstein2004` | pages — one verifier claimed 39–45, another 27–38 | JGAA publisher page confirms **39–45** (no change) |

No citation was found to be wholly fabricated (every cited work exists); the serious
issues were incorrect bibliographic pointers (wrong DOIs/venues) and one fabricated
author list, all now corrected. All in-text usages were judged faithful to their
sources (see per-citation "Usage" below); two carry a precision note (`Bader2007`,
`Turner2007`).

---

## Foundational centrality

### Freeman1977
- **URL:** https://doi.org/10.2307/3033543 (JSTOR 3033543)
- **Metadata:** Linton C. Freeman, "A Set of Measures of Centrality Based on Betweenness", *Sociometry* 40(1):35–41, 1977. CONFIRMED.
- **Cited for:** origin of betweenness centrality (a node as intermediary on shortest paths).
- **Quote:** verbatim not retrieved (JSTOR 403); indexed abstract describes centrality "in terms of the degree to which a point falls on the shortest path between others."
- **Synopsis:** Introduces betweenness as the extent to which a node lies on shortest paths between other pairs, giving control over communication.
- **Usage:** IN-CONTEXT — canonical originating reference.

### Freeman1979 (key) — published 1978
- **URL:** https://doi.org/10.1016/0378-8733(78)90021-7
- **Metadata:** Linton C. Freeman, "Centrality in social networks: conceptual clarification", *Social Networks* 1(3):215–239, **1978** (DOI encodes 1978; widely also cited 1979).
- **Cited for:** conceptual basis for closeness centrality.
- **Quote:** verbatim not retrieved (ScienceDirect 403).
- **Synopsis:** Clarifies the degree/betweenness/closeness families; closeness framed as a node's proximity in reaching all others.
- **Usage:** IN-CONTEXT.

### Rochat2009
- **URL:** http://infoscience.epfl.ch/record/200525 (EPFL Infoscience; no DOI — ASNA 2009 conference paper)
- **Metadata:** Yannick Rochat, "Closeness Centrality Extended to Unconnected Graphs: The Harmonic Centrality Index", Applications of Social Network Analysis (ASNA 2009), Zürich. CONFIRMED.
- **Cited for:** source of harmonic closeness (sum of reciprocal distances).
- **Quote:** verbatim not retrieved (repository fetch blocked).
- **Synopsis:** Defines harmonic centrality as the sum of reciprocals of shortest-path distances; well-defined on disconnected graphs (infinite distances contribute 0).
- **Usage:** IN-CONTEXT.

## Sampling / approximation algorithms

### Eppstein2004
- **URL:** https://doi.org/10.7155/jgaa.00081 · preprint https://arxiv.org/abs/cs/0009005
- **Metadata:** David Eppstein & Joseph Wang, "Fast Approximation of Centrality", *J. Graph Algorithms and Applications* 8(1):39–45, 2004. CONFIRMED (pages 39–45 per JGAA publisher page).
- **Cited for:** bounding the number of sampled sources for closeness via Hoeffding + union bound over n nodes — the foundational source-sampling method we localise.
- **Quote (verbatim, from arXiv full text):** "RAND randomly chooses k sample vertices and computes single-source shortest-paths (SSSP) from each sample vertex to all other vertices." · "using Θ(log n / ε²) samples will cause the probability of error at any vertex to be bounded above by … 1/n², giving at most 1/n probability of having greater than εΔ error anywhere in the graph."
- **Synopsis:** Samples k sources uniformly, runs SSSP from each, averages distances; Hoeffding bounds per-vertex error and a union bound over all n vertices sets k = Θ(log n/ε²).
- **Usage:** IN-CONTEXT — the source explicitly derives the source-sample count via Hoeffding + union bound over all vertices, exactly as we cite.

### Brandes2001
- **URL:** https://doi.org/10.1080/0022250X.2001.9990249 (open PDF: snap.stanford.edu/class/cs224w-readings/brandes01centrality.pdf)
- **Metadata:** Ulrik Brandes, "A Faster Algorithm for Betweenness Centrality", *J. Mathematical Sociology* 25(2):163–177, 2001. CONFIRMED.
- **Cited for:** Brandes' back-propagation computes betweenness from each source traversal.
- **Quote (abstract, via indexed copies):** "new algorithms for betweenness are introduced … They require O(n + m) space and run in O(nm) and O(nm + n² log n) time on unweighted and weighted networks." (Recursive dependency relation was only reproducible from secondary sources, not the original PDF — flagged.)
- **Synopsis:** Per source, runs SSSP computing path counts/predecessors, then accumulates one-sided dependencies in a back-propagation phase; summing over sources gives betweenness without explicit pairwise summation.
- **Usage:** IN-CONTEXT.

### Brandes2007 (corrected)
- **URL:** https://doi.org/10.1142/S0218127407018403 (open PDF: uni-konstanz.de/mmsp/pubsys/publishedFiles/BrPi07.pdf)
- **Metadata:** Ulrik Brandes & Christian Pich, "Centrality Estimation in Large Networks", *Int. J. Bifurcation and Chaos* 17(7):2303–2318, 2007. CORRECTED from a wrong LNCS/DOI that resolved to an unrelated paper.
- **Cited for:** established source/pivot-sampling estimation empirically for both closeness and betweenness.
- **Quote (abstract):** "Centrality scores can be estimated … from a limited number of SSSP computations. … we present results from an experimental study of the quality of such estimates under various selection strategies for the source vertices."
- **Synopsis:** Empirical study of closeness and betweenness estimation from SSSP runs rooted at sampled "pivots", comparing pivot-selection strategies vs. number of pivots.
- **Usage:** IN-CONTEXT.

### Bader2007
- **URL:** https://doi.org/10.1007/978-3-540-77004-6_10 (abstract: pure.psu.edu/en/publications/approximating-betweenness-centrality)
- **Metadata:** Bader, Kintali, Madduri, Mihail, "Approximating Betweenness Centrality", WAW 2007, LNCS 4863:124–137. CONFIRMED.
- **Cited for:** adaptive sampling for betweenness with a per-vertex guarantee.
- **Quote (abstract, verbatim):** "Our approximation algorithm is based on an adaptive sampling technique that significantly reduces the number of single-source shortest path computations for vertices with high centrality."
- **Synopsis:** Adaptive-sampling estimate of the betweenness of a *given* vertex; sample count scales inversely with that vertex's betweenness (∝ n²/BC(v)), so high-centrality vertices need few samples.
- **Usage:** IN-CONTEXT, with **precision note** — the per-vertex framing is accurate, but our table's "O(1/(ε² b(v)))" simplifies the paper's multiplicative (factor-1/ε) bound; "samples inversely proportional to the vertex's betweenness" is the precise statement.

### Geisberger2008
- **URL:** https://doi.org/10.1137/1.9781611972887.9
- **Metadata:** Geisberger, Sanders, Schultes, "Better Approximation of Betweenness Centrality", ALENEX 2008, pp. 90–100. CONFIRMED.
- **Cited for:** refined source sampling with distance-weighted (linear-scaling) corrections.
- **Quote:** paper PDF blocked; Semantic Scholar TLDR (not the paper's own words): "a framework for unbiased approximation of betweenness … generalizes a previous approach by Brandes."
- **Synopsis:** Generalises Brandes–Pich into a framework of unbiased estimators with rescaling (linear/bisection) so nodes near a pivot are not overestimated; better accuracy on real networks.
- **Usage:** IN-CONTEXT (linear scaling is the distance-aware correction).

### Riondato2016
- **URL:** https://doi.org/10.1007/s10618-015-0423-0
- **Metadata:** Riondato & Kornaropoulos, "Fast approximation of betweenness centrality through sampling", *Data Mining and Knowledge Discovery* 30(2):438–475, 2016 (extends WSDM 2014). CONFIRMED.
- **Cited for:** path-sampling (random source–target pairs; one shortest path per sample); additive global guarantee via VC-dimension / vertex-diameter.
- **Quote:** Semantic Scholar TLDR (summary, not verbatim source text): bounds "the sample size needed" via "the VC-dimension of a range set associated with the problem."
- **Synopsis:** Samples shortest paths between random vertex pairs; bounds sample size via VC-dimension in terms of the vertex-diameter for a uniform additive (ε,δ) guarantee.
- **Usage:** IN-CONTEXT.

### Borassi2019 (corrected)
- **URL:** https://doi.org/10.1145/3284359 · preprint https://arxiv.org/abs/1604.08553
- **Metadata:** Borassi & Natale, "KADABRA is an ADaptive Algorithm for Betweenness via Random Approximation", *ACM J. Experimental Algorithmics* 24(1), art. 1.2, 2019 (conference precursor: ESA 2016). CORRECTED from a wrong DOI that resolved to an unrelated ESA paper.
- **Cited for:** adaptive algorithm with network-dependent sample counts; path-sampling with adaptive stopping.
- **Quote (abstract, verbatim):** "a new rigorous application of the adaptive sampling technique. This approach decreases the total number of shortest paths that need to be sampled to compute all betweenness centralities with a given absolute error."
- **Synopsis:** Balanced bidirectional shortest-path sampling with a data-dependent stopping rule; sub-linear per-sample cost on realistic models; extends to top-k.
- **Usage:** IN-CONTEXT.

### Pellegrina2023
- **URL:** https://doi.org/10.1145/3628601 · preprint https://arxiv.org/abs/2106.03462
- **Metadata:** Pellegrina & Vandin, "SILVAN: Estimating Betweenness Centralities with Progressive Sampling and Non-uniform Rademacher Bounds", *ACM TKDD* 18(3):1–55, 2023 (online; print 2024). CONFIRMED.
- **Cited for:** variance-aware / non-uniform progressive sampling bounds.
- **Quote (abstract, verbatim):** "SILVAN follows a progressive sampling approach, and builds on novel bounds based on Monte-Carlo Empirical Rademacher Averages …" · "non-uniform bounds on the deviation of the estimates of the betweenness centrality of all the nodes."
- **Synopsis:** Progressive sampling with per-node, data-dependent (Rademacher) bounds, requiring fewer samples than prior methods at comparable guarantees.
- **Usage:** IN-CONTEXT.

### Matta2019 (was mislabelled `Bergamini2019`)
- **URL:** https://doi.org/10.1186/s40649-019-0062-5
- **Metadata:** **John Matta, Gunes Ercal, Koushik Sinha**, "Comparing the speed and accuracy of approaches to betweenness centrality approximation", *Computational Social Networks* 6(1), art. 2, 2019. CORRECTED — the prior author list (Bergamini et al.) was fabricated.
- **Cited for:** a comprehensive empirical comparison of betweenness approximation approaches.
- **Quote (abstract, verbatim):** "Overall, the speed of betweenness centrality can be reduced several orders of magnitude by using approximation algorithms." · "we run two tests, clustering and immunization, on identical hardware."
- **Synopsis:** Benchmarks multiple betweenness-approximation methods for speed and accuracy via clustering and immunization tasks on identical hardware.
- **Usage:** IN-CONTEXT (only the citation key was wrong, now fixed).

## Concentration & sampling theory

### Hoeffding1963
- **URL:** https://doi.org/10.1080/01621459.1963.10500830
- **Metadata:** Wassily Hoeffding, "Probability Inequalities for Sums of Bounded Random Variables", *JASA* 58(301):13–30, 1963. CONFIRMED.
- **Cited for:** Hoeffding's inequality bounds deviation of a sample mean from its expectation.
- **Quote (abstract):** "Upper bounds are derived for the probability that the sum S of n independent random variables exceeds its mean ES by a positive number nt … depend only on the endpoints of the ranges of the summands."
- **Synopsis:** Seminal exponential concentration bound for sums of independent bounded variables.
- **Usage:** IN-CONTEXT.

### Horvitz1952
- **URL:** https://doi.org/10.1080/01621459.1952.10483446 (open PDF: stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf)
- **Metadata:** Horvitz & Thompson, "A Generalization of Sampling Without Replacement from a Finite Universe", *JASA* 47(260):663–685, 1952. CONFIRMED.
- **Cited for:** the inverse-probability-weighted (Horvitz–Thompson) estimator, unbiased when inclusion prob > 0.
- **Quote (abstract):** "a general technique for the treatment of samples drawn without replacement from finite universes when unequal selection probabilities are used."
- **Synopsis:** Defines the HT estimator (weight by inverse inclusion probability), establishing unbiasedness and unbiased variance estimation.
- **Usage:** IN-CONTEXT.

## Urban / space-syntax

### Cooper2015
- **URL:** https://doi.org/10.1080/13658816.2015.1018834
- **Metadata:** Crispin H. V. Cooper, "Spatial localization of closeness and betweenness measures …", *Int. J. Geographical Information Science* 29(8):1293–1309, 2015. CONFIRMED.
- **Cited for:** localised / distance-thresholded closeness and betweenness in urban network analysis.
- **Quote:** verbatim not retrieved (T&F/ORCA blocked).
- **Synopsis:** Examines distance-thresholded closeness/betweenness where locality and shortest-path metrics differ; formally self-contradictory yet empirically useful for movement prediction.
- **Usage:** IN-CONTEXT.

### Cooper2018 (key) — actually SoftwareX 2020
- **URL:** https://doi.org/10.1016/j.softx.2020.100525 (ADS 2020SoftX..1200525C)
- **Metadata:** Cooper & Chiaradia, "sDNA: 3-d spatial network analysis for GIS, CAD, Command Line & Python", *SoftwareX* 12:100525, **2020**. CORRECTED from a non-resolving EPB 2018 record.
- **Cited for:** sDNA as a tool implementing distance-thresholded centrality.
- **Quote:** verbatim not retrieved (ScienceDirect blocked); abstract describes accessibility, betweenness/flow, and efficiency measures within radial distance bands.
- **Synopsis:** Toolbox for 3-D spatial network analysis (QGIS/ArcGIS/CAD/CLI/Python) computing accessibility, flow and efficiency localised within radial neighbourhoods.
- **Usage:** IN-CONTEXT (venue corrected).

### Turner2007
- **URL:** https://doi.org/10.1068/b32067 (post-print: discovery.ucl.ac.uk/id/eprint/2092)
- **Metadata:** Alasdair Turner, "From axial to road-centre lines …", *Environment and Planning B: Planning and Design* 34(3):539–555, 2007. CONFIRMED.
- **Cited for:** angular (simplest-path) distance, which does not satisfy subpath optimality.
- **Quote (verbatim, p.3):** "This angular sum is treated as the 'cost' of a putative journey through the graph, and from it a shortest (that is, least cost) path from one segment to another across the system can be calculated."
- **Synopsis:** Introduces angular segment analysis on road-centre lines; cumulative angular turn is the path cost, over which betweenness ("choice") is computed.
- **Usage:** IN-CONTEXT, with **precision note** — Turner is cited for *angular distance*; the subpath-optimality-violation point is our analytical inference, not a verbatim claim in the source.

### Strano2013
- **URL:** https://doi.org/10.1068/b38216 · preprint https://arxiv.org/abs/1211.0259
- **Metadata:** Strano, Viana, da Fontoura Costa, Cardillo, Porta, Latora, "Urban street networks, a comparative analysis of ten European cities", *Environment and Planning B: Planning and Design* 40(6):1071–1086, 2013. CONFIRMED.
- **Cited for:** (a) cross-city comparison of street networks; (b) typical mean street-segment length (to justify the conservative s = 175 m).
- **Quote (verbatim):** Table 1 ⟨ℓ⟩ (m): Barcelona 110.7, Bologna 119.1, Catania 55.9, Edinburgh 110.0, Geneva 122.4, Lancaster 96.8, Leicester 98.5, Oxford 103.0, Sheffield 111.0, Worcester 94.8. Text (p.1076): "the average street length lies between 94.8 m (Worcester) and 122.4 m (Geneva)" (Catania ~56 m the lone outlier).
- **Synopsis:** Comparative analysis of ten European cities' primal street networks; reports geometric indices (incl. mean segment length) and centralities; classifies cities by PCA.
- **Usage:** IN-CONTEXT — text corrected to the verified city-mean range ≈95–122 m (the previous "80–180 m" with a Boeing2017 co-cite was loose; Boeing2017 dropped from this claim).

## Data sources / tooling

### Gil2017
- **URL:** https://doi.org/10.1177/0265813516650678
- **Metadata:** Jorge Gil, "Street network analysis 'edge effects': Examining the sensitivity of centrality measures to boundary conditions", *Environment and Planning B: Urban Analytics and City Science* 44(5):819–836, 2017. CONFIRMED.
- **Cited for:** the standard remedy of buffering a study area to mitigate edge effects.
- **Quote:** verbatim not retrieved (paywall); verified abstract: an empirical study of "the impact of different network model boundaries on the results of closeness and betweenness centrality analysis."
- **Synopsis:** Canonical reference for boundary/edge effects on street-network centrality and their mitigation.
- **Usage:** IN-CONTEXT (specific buffer-node sentence not verbatim-verified).

### Boeing2017
- **URL:** https://doi.org/10.1016/j.compenvurbsys.2017.05.004 · preprint https://arxiv.org/abs/1611.01890
- **Metadata:** Geoff Boeing, "OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks", *Computers, Environment and Urban Systems* 65:126–139, 2017. CONFIRMED.
- **Cited for:** OSMnx used to geocode administrative boundaries from OpenStreetMap.
- **Quote (abstract, verbatim):** "OSMnx contributes five significant capabilities … the automated downloading of political boundaries and building footprints."
- **Synopsis:** Python tool to download/construct/analyse/visualise OpenStreetMap street networks; includes automated download of political boundaries.
- **Usage:** IN-CONTEXT.

### Hagberg2008
- **URL:** https://proceedings.scipy.org/articles/TCWV9851 (doi 10.25080/TCWV9851)
- **Metadata:** Hagberg, Schult, Swart, "Exploring Network Structure, Dynamics, and Function using NetworkX", Proc. 7th Python in Science Conf (SciPy 2008), pp. 11–15. CONFIRMED.
- **Cited for:** NetworkX, used to cross-verify the Rust implementation.
- **Quote (abstract, verbatim):** "NetworkX is a Python language package for exploration and analysis of networks and network algorithms."
- **Synopsis:** The standard NetworkX reference — graph data structures and a large algorithm/metric library.
- **Usage:** IN-CONTEXT.

### Simons2022
- **URL:** https://doi.org/10.1177/23998083221133827 · preprint https://arxiv.org/abs/2106.15314
- **Metadata:** Gareth Simons, "The cityseer Python package for pedestrian-scale network-based urban analysis", *Environment and Planning B: Urban Analytics and City Science* 50(5):1328–1344 (online 2022; print 2023). vol/issue/pages CONFIRMED (RePEc `v50y2023i5p1328-1344`).
- **Cited for:** the package implementing the schedule.
- **Quote (preprint abstract):** "cityseer-api is a Python package consisting of computational tools for fine-grained street-network and land-use analysis."
- **Synopsis:** The cityseer toolkit — node/segment centralities, land-use accessibility, mixed-use, distance-weighted network-constrained measures.
- **Usage:** IN-CONTEXT (online-2022 / print-2023 year nuance only).
