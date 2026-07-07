---
layout: '@src/layouts/PageLayout.astro'
---

# Interpretation

This page maps practical planning questions to the metrics that answer them, then explains how to read the resulting numbers. It assumes familiarity with the concepts on the [Fundamentals](/guide/fundamentals) page; the [Centrality](/guide/centrality) and [Land-Use](/guide/land-use) pages document each metric in full.

## Which metric answers which question

| Question | Metric | Typical thresholds | Guide | Recipe |
| --- | --- | --- | --- | --- |
| Which streets will carry pedestrian through-movement? | Betweenness (`cc_betweenness_{d}`), or angular betweenness (`cc_betweenness_{d}_ang`) on the dual graph, which better reflects how pedestrians actually choose routes | 800–1600m | [Centrality](/guide/centrality#shortest-path-centrality) | [Metric Centrality](/examples/centrality/gpd-metric-centrality) |
| Where are the preconditions for local vibrancy and retail? | Gravity-weighted closeness (the default `cc_decay_{d}` metric, or a custom gravity expression) together with mixed-use diversity (`cc_hill_q0_{d}`) | 400–800m | [Land-Use](/guide/land-use#mixed-use-diversity) | [Mixed Uses](/examples/accessibility/gpd-mixed-uses) |
| How well served is each location by amenities? | Accessibility counts (`cc_{category}_{d}`) and nearest distances (`cc_{category}_nearest_max_{d}`) per land-use category | 400–800m for daily amenities; longer for occasional destinations | [Land-Use](/guide/land-use#accessibility) | [OSM Accessibility](/examples/accessibility/osm-accessibility) |
| What is the movement hierarchy of the network? | Angular betweenness (`cc_betweenness_{d}_ang`) compared across several thresholds: streets prominent at all scales are the primary structure, streets prominent only locally are neighbourhood streets | 400m through 5000m or more, computed together | [Centrality](/guide/centrality#simplest-path-centrality) | [Angular Centrality](/examples/centrality/gpd-angular-centrality) |
| How intense is local street connectivity? | Density (`cc_density_{d}`, the count of reachable nodes) and harmonic closeness (`cc_harmonic_{d}`) | 400–800m | [Centrality](/guide/centrality#shortest-path-centrality) | [Metric Centrality](/examples/centrality/gpd-metric-centrality) |

Two notes on the table. First, the `CityNetwork` centrality methods compute a lean default set: harmonic closeness and unweighted betweenness. The other columns above are part of the fuller default set of the lower-level functions; with `CityNetwork`, request them explicitly via expression dicts, for example `closeness={"density": "1", "decay": "exp(-4 * p)"}` (see [Custom metrics](/guide/centrality#custom-metrics)). Second, several questions are best answered by combinations: a street that scores highly for both through-movement and local closeness, and that has diverse reachable land uses, is a stronger candidate for a high street than one that scores highly on any single metric.

A worked pattern for the movement-hierarchy question: compute angular betweenness at, say, 800m, 1600m, and 5000m in one call, then compare. A street in a high quantile at 5000m but not at 800m is a through-route serving city-scale movement; the reverse pattern indicates a locally important street with little wider significance. The same multi-scale reading applies to closeness: locally central but globally peripheral locations are typical of self-contained neighbourhood centres.

## Reading the numbers

Centrality values are relative, not absolute. Closeness metrics are sums over all nodes reachable within the threshold, and betweenness sums contributions over all origin–destination pairs, so raw magnitudes scale with how many nodes the network representation contains. Two networks with identical street layouts but different node densities, for example because one was decomposed and the other was not, or one is primal and the other dual, will produce systematically different raw values for the same places. Compare values within a single run, not across runs on differently prepared networks.

For mapping, classify by quantiles rather than by raw value breaks. Quantile classification shows the spatial pattern, which is the meaningful output, and is robust to the arbitrary scaling of the raw sums. This is the convention used throughout the [examples](/examples); the [From Results to Maps](/examples/networks/results-to-maps) recipe shows the full workflow, from exporting a GeoPackage to a quantile-classed figure.

When comparisons across networks or study areas are genuinely needed, two mechanisms make magnitudes less dependent on network representation (see [Node weights](/guide/centrality#node-weights) and [Segment-weighted centrality](/guide/centrality#segment-weighted-centrality)):

- `segment_weighted=True` weights each node by its street segment length, so closeness reflects total reachable street length in metres, a physical quantity, rather than a node count that depends on how the network was drawn.
- Node weights let you weight centrality by a real quantity such as population or floorspace, so results reflect reachable opportunities rather than reachable nodes.

Even so, treat cross-network comparisons of raw magnitudes with caution: differences in data source, cleaning, and boundary treatment all leave traces in the numbers. Rankings and spatial patterns travel better than magnitudes.

## Choosing distance thresholds

The [distance thresholds table](/guide/fundamentals#distance-thresholds) on the Fundamentals page lists common choices and their walking-time equivalents; this section adds only the judgement. Match the threshold to the behaviour being studied: everyday walking trips to shops and services concentrate below 800m, so vibrancy and amenity questions belong at 400–800m; through-movement and route choice operate at larger scales, so betweenness questions belong at 800m and upward. Because `cityseer` computes all requested thresholds in a single pass, the practical answer is usually to compute several, for example `distances=[400, 800, 1600]`, and compare: how the pattern changes across scales is itself informative, and it protects conclusions from resting on one arbitrary cutoff.
