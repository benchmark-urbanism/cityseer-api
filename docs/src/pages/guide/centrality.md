---
layout: '@src/layouts/PageLayout.astro'
---

# Centrality

Centrality metrics describe each location's position in the street network: how easily it reaches its surroundings, and how much movement passes through it. `cityseer` computes multiple centrality measures simultaneously for any combination of distance thresholds in a single pass.

## Closeness and betweenness

Most centrality measures answer one of two questions.

**Closeness** asks how near a location is to everywhere it can reach within the threshold. A street within a short walk of many destinations scores high; one at the end of a long cul-de-sac scores low. High closeness marks places that are well-placed to reach their surroundings, typically local centres.

**Betweenness** asks how many of the shortest routes between other places pass through a location. A street that lies on the way between many origins and destinations scores high and carries through-movement; a quiet back street that no route uses scores low. High betweenness marks the through-routes and high streets.

The two often diverge. A local centre can be easy to reach from its neighbourhood (high closeness) without lying on any through-route (low betweenness); a bypass can carry heavy through-movement (high betweenness) while being cut off from its immediate surroundings (low closeness). Computing both, at several distance thresholds, is usually more informative than either alone.

The rest of this page covers how these measures are defined and read: the expression engine, the default metrics, and the shortest-path and angular variants.

The examples on this page use the [`CityNetwork`](/api/network) methods, which are the recommended interface: network construction, cleaning, and the dual graph are handled automatically. The same computations exist as lower-level functions in [`metrics.networks`](/metrics/networks) for direct control over the network structures. The older `node_centrality_shortest`, `node_centrality_simplest`, and `segment_centrality` functions are deprecated: they exist only for backwards compatibility with pre-5.0 code (see the [migration guide](/guide/migration)).

## Expression-based metrics

Metrics are defined as `{name: expression}` dictionaries using two variables:

- **`c`** (cost): the raw routing cost to each reached node. For shortest paths, `c` is the metric distance in metres. For simplest paths, `c` is the cumulative angular change in degrees.
- **`p`** (progress): normalised progress from 0 at the source to 1 at the threshold. For shortest paths this is `p = c / threshold`. For simplest paths the routing cost `c` is angular, so `p` instead tracks the metric budget consumed (`elapsed_time / max_time`), not the angular change.

Metrics fall into four categories:

| Category | Role | Example |
| --- | --- | --- |
| **Closeness** | Expression is evaluated once per reached node and summed across all reachable nodes. | `{"harmonic": "1/c"}` accumulates $\sum_j 1/c_j$ |
| **Betweenness** | Expression weights the contribution of each destination during Brandes backpropagation. | `{"betweenness": "1"}` counts paths equally |
| **Cycles** | Boolean flag; computes the circuit rank of the locally reachable subgraph (shortest-path only). | `cycles=True` |
| **Postprocess** | Derives new columns from previously computed metrics using simple arithmetic (`+`, `-`, `*`, `/`, `**`). | `{"hillier": "density**2 / farness"}` |

Pass `None` to use the defaults for a category, or `{}` to skip it entirely. The default sets described below are those of the lower-level `metrics.networks` functions; the `CityNetwork` methods default to a leaner set, a single harmonic closeness and a single unweighted betweenness, with cycles and postprocess off. Pass the expression dicts explicitly to compute any of the fuller set.

## Shortest-path centrality

[`centrality_shortest`](/api/network#centrality_shortest) (or [`centrality_shortest`](/metrics/networks#centrality_shortest) in the lower-level API) computes the following default metrics for each distance threshold `d`. In the formulas below, the sum is over all nodes $j$ reachable within $d$, $c_j$ is the shortest-path distance in metres to node $j$, and $p_j = c_j / d$.

**Default closeness** (pass `closeness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_density_{d}` | `"1"` | $\sum_j 1$ | Count of nodes reachable within distance $d$. A simple measure of local connectivity. |
| `cc_farness_{d}` | `"c"` | $\sum_j c_j$ | Sum of metric distances to all reachable nodes. Lower values indicate better average proximity. |
| `cc_harmonic_{d}` | `"1/c"` | $\sum_j 1 / c_j$ | Harmonic closeness: sum of inverse distances. Higher values indicate better proximity. Unlike standard closeness, harmonic closeness handles distance-bounded analysis correctly because unreachable nodes contribute 0 rather than distorting the average. Nodes within a few metres of one another, especially below 1 m, contribute very large $1/c$ values and inflate scores severely; `CityNetwork` construction removes such degenerate edges automatically, but manually built networks must be consolidated first. |
| `cc_decay_{d}` | `"exp(-4 * p)"` | $\sum_j e^{-4 p_j}$ | Exponential decay-weighted closeness. Nearby nodes contribute most; at the distance threshold ($p = 1$), weight drops to $e^{-4} \approx 1.8\%$. This is the continuous equivalent of the historical $\beta$-weighted metric. |

**Default betweenness** (pass `betweenness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_betweenness_{d}` | `"1"` | $\sum_{s,t} \sigma_{st}(v) / \sigma_{st}$ | Unweighted betweenness: for each origin–destination pair $(s, t)$, counts the fraction of shortest paths that pass through node $v$. High values indicate through-movement potential. |
| `cc_betweenness_decay_{d}` | `"exp(-4 * p)"` | $\sum_{s,t} e^{-4 p_t} \cdot \sigma_{st}(v) / \sigma_{st}$ | Decay-weighted betweenness: each pair's contribution is downweighted by the exponential decay applied to the destination's normalised distance $p_t$. Paths to distant destinations count less. |

**Default postprocess** (pass `postprocess=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_hillier_{d}` | `"density**2 / farness"` | $n^2 / \sum_j c_j$ | Hillier normalisation: rewards locations that are both well-connected (high density $n$) and proximate (low farness). This is the standard normalisation used in space syntax research. |

**Cycles** (`cycles=True` by default):

| Column | Description |
| --- | --- |
| `cc_cycles_{d}` | Circuit rank of the locally reachable subgraph: the number of independent loops ($e - n + 1$ where $e$ is edges and $n$ is nodes in the subgraph). In grid-like networks, each loop roughly corresponds to a city block; in less regular networks, it measures the redundancy of route choices. |

## Simplest-path centrality

[`centrality_simplest`](/api/network#centrality_simplest) (or [`centrality_simplest`](/metrics/networks#centrality_simplest) in the lower-level API) computes angular centrality metrics. For simplest paths, `c` is the cumulative angular change in degrees (the total turning at each junction along a route) rather than metric distance. The variable `p` is normalised elapsed time (not angular cost), so the distance reachability budget is still metric. Note the `_ang` suffix on all output columns.

**Default closeness** (pass `closeness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_density_{d}_ang` | `"1"` | $\sum_j 1$ | Count of nodes reachable within distance $d$ via angular routing. |
| `cc_farness_{d}_ang` | `"1 + c / 90"` | $\sum_j (1 + c_j / 90)$ | Angular farness. The $1 + c/90$ transform maps a straight-ahead path ($0°$) to 1 and a single $90°$ turn to 2, giving a meaningful scale that avoids division-by-zero at the source. |
| `cc_harmonic_{d}_ang` | `"1 / (1 + c / 90)"` | $\sum_j 1 / (1 + c_j / 90)$ | Angular harmonic closeness: the inverse of the farness expression. Higher values indicate better angular proximity (straighter routes to more destinations). |

**Default betweenness** (pass `betweenness=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_betweenness_{d}_ang` | `"1"` | $\sum_{s,t} \sigma_{st}(v) / \sigma_{st}$ | Angular betweenness: fraction of simplest (minimum angular change) paths through each node. Equivalent to angular choice in space syntax. |

**Default postprocess** (pass `postprocess=None` or omit):

| Column | Expression | Formula | Description |
| --- | --- | --- | --- |
| `cc_hillier_{d}_ang` | `"density**2 / farness"` | $n^2 / \sum_j (1 + c_j / 90)$ | A node-count normalisation of angular closeness ($n^2$ over angular farness). This is one classical form of what space syntax calls _integration_. |

Simplest-path centrality does not apply decay-weighted forms by default. Angular cost is cumulative turning, which does not decrease continuously with distance, so it is not a meaningful basis for distance decay. If you need a decay-weighted or otherwise custom angular metric, define it with a [custom expression](#custom-metrics).

## Choosing shortest or angular

Shortest-path centrality uses metric distance, which is less sensitive to how the network is drawn.

Simplest-path (angular) centrality weights routes by cumulative turning, following the space-syntax observation that people tend to prefer straighter routes. On a clean, well-consolidated network, angular measures can be preferable and can correspond more closely with observed movement. Angular cost is computed from turn angles, so spurious nodes, unconsolidated parallel edges, and roundabouts drawn as rings distort it (see [Network Cleaning](/guide/cleaning)). When the network has not been cleaned to a high standard, the shortest path may be the safer choice.

## Custom metrics

To define custom metrics, pass a dictionary of `{name: expression}` pairs. Closeness and betweenness expressions (and `decay_fn` expressions elsewhere) are evaluated by the runtime expression engine: they can use the variables `c` and `p`, the operators `+`, `-`, `*`, `/`, and `^` (power), and the functions `exp`, `ln`, `log10`, `sqrt`, `abs`, `sin`, `cos`, `tan`, `floor`, `ceil`, and `round`, plus the constants `PI` and `E`. Note that `**` is not accepted (write `^`), `min` and `max` are not available, and unary minus binds tighter than the power operator, so write `-(p^2)` rather than `-p^2` when the square should be negated.

Postprocess expressions follow different rules: they are evaluated in Python over the previously computed metric columns and support only the arithmetic operators `+`, `-`, `*`, `/`, and `**` (power), with no functions. In postprocess, power is written `**`, not `^`.

```python
# Custom gravity model and linear-decay betweenness
cn.centrality_shortest(
    distances=[800],
    closeness={"gravity": "1 / c^2", "reach": "1"},
    betweenness={"bt_linear": "1 - p"},
    postprocess={},  # skip hillier
)

# Angular centrality with decay-weighted betweenness
cn.centrality_simplest(
    distances=[800],
    betweenness={"betweenness": "1", "bt_decay": "exp(-4 * p)"},
)
```

## Node weights

Every node carries a `weight` (default `1.0`). Set it on the nodes `GeoDataFrame`, or add a `weight` attribute to your NetworkX nodes before ingestion, to apply gravity-style weighting to centrality:

- **Closeness** weights each reachable node by its destination weight, so `density` becomes $\sum_j w_j$ (the sum of reachable node weights) rather than a plain count, and other closeness expressions are scaled accordingly. A node's own weight does **not** rescale its own score; weighting reflects the *opportunities it can reach*.
- **Betweenness** weights each origin–destination pair by the **product** of its endpoint weights $w_s \cdot w_t$, the standard gravity-flow form.

The same weighting is applied identically whether or not [adaptive sampling](#adaptive-sampling) is used. With the default weights of `1.0` the results are unchanged from an unweighted analysis.

:::note
Node weights affect **centrality only**. Land-use accessibility, mixed-use diversity, and statistical aggregations are intentionally *not* node-weighted; they weight reachable land-use data features (optionally by [distance decay](/guide/land-use#decay-functions)), not network nodes.
:::

## Segment-weighted centrality

`segment_weighted=True` is a convenience preset over the node `weight` mechanism above: it temporarily sets each dual-graph node's weight to its corresponding street segment length, then restores the original weights afterwards. This means:

- **Closeness** metrics reflect total reachable street length rather than node counts (e.g. density becomes total metres of reachable street within the threshold).
- **Betweenness** weights each origin–destination pair by both endpoint segment lengths, so longer streets contribute more to betweenness flows.

This requires a dual graph representation (which `CityNetwork` builds automatically), and works for every `CityNetwork` construction path. Pass the flag per centrality call, or set it once at construction as the default for all subsequent centrality calls:

```python
cn.centrality_shortest(distances=[800], segment_weighted=True)

# or as a construction-time default
cn = CityNetwork.from_geopandas(edges_gdf, segment_weighted=True)
cn.centrality_shortest(distances=[800])
```

## Convenience wrappers

For cases where only closeness or only betweenness is needed, convenience functions skip the unused category:

- [`closeness_shortest`](/metrics/networks#closeness_shortest) / [`closeness_simplest`](/metrics/networks#closeness_simplest): closeness only (betweenness disabled)
- [`betweenness_shortest`](/metrics/networks#betweenness_shortest) / [`betweenness_simplest`](/metrics/networks#betweenness_simplest): betweenness only (closeness and cycles disabled)

## Origin–destination and demand betweenness

Standard betweenness treats every node pair equally. When you have real or modelled travel flows, `cityseer` can route those flows instead, with [`betweenness_od`](/metrics/networks#betweenness_od) for an explicit origin–destination matrix and [`betweenness_demand`](/metrics/networks#betweenness_demand) for a modelled, singly-constrained spatial interaction model. This has its own dedicated section: see the [Origin-Destination Flows guide](/guide/flows) and the [flow recipes](/examples/flows).

## Centrality recipes

- [Metric Centrality from GeoDataFrame](/examples/centrality/gpd-metric-centrality) -- shortest-path centrality workflow
- [Angular Centrality from GeoDataFrame](/examples/centrality/gpd-angular-centrality) -- simplest-path centrality workflow
- [OSM Centrality](/examples/centrality/osm-centrality) -- end-to-end from OpenStreetMap
- [Custom Expressions](/examples/centrality/custom-expressions) -- defining custom metrics, selecting only what you need, postprocess, and statistic selection
- [Sampled Centrality](/examples/centrality/sampled-centrality) -- adaptive sampling on a large network, validated against exact results
- [Demand Betweenness](/examples/flows/demand-flows) -- demand-weighted flows from a singly constrained spatial interaction model

## Performance and scale

The underlying algorithms are parallelised in Rust and scale to large networks. Computation scales with the number of edges, the number of distance thresholds, and the reachability at each threshold. Simplest-path (angular) centrality is typically faster than shortest-path because angular routing explores fewer paths. For large networks at long distance thresholds, consider [adaptive sampling](#adaptive-sampling).

## Adaptive sampling

For large networks at long distance thresholds, `cityseer` offers an experimental adaptive sampling feature. Rather than using every node as a source, each node is included with its own probability derived from its measured local reach: a cheap pilot polls the network with bounded shortest-path traversals from a small sample of sources to estimate each node's reach, and the Hoeffding inequality converts that reach into the minimum sampling rate needed to keep the approximation error within a specified tolerance. Sparse areas are sampled more heavily and dense areas less, so precision is uniform across the network. Results are corrected using inverse-probability weighting (IPW): if a node had a 25% chance of being selected as a source, its contribution is multiplied by 4 (the reciprocal of 0.25), so that the sampled subset approximates the result of using all nodes without bias.

![Per-node reach-based sampling schematic.](/images/sampling_method_schematic.svg) *The pilot measures each node's catchment (A) and converts the measurements into per-node inclusion probabilities (B); a draw under those probabilities gives each catchment the samples it needs (C), whereas a fixed rate spending the same budget oversamples the dense core and misses most of the sparse catchment it should count in full (D).*

Sampling is applied per distance threshold. For each distance, the estimated cost of properly powered sampling is compared against exact computation, and the cheaper option is selected automatically; short distances and networks with small live areas therefore run exactly, while long distances on large networks are sampled. The [sampling page](/metrics/sampling) illustrates this decision.

The `epsilon` parameter controls the error tolerance. The default of 0.05 is calibrated on real networks spanning the density range, from dense metropolitan grids to a sparse suburb, so that node rankings are preserved (Spearman ρ ≥ 0.95); denser networks clear the target comfortably, while networks sparser than a typical suburb may require a tighter tolerance (see the [sampling page](/metrics/sampling)). Both `centrality_shortest` and `centrality_simplest` support sampling. Pass `sample=True` to enable:

```python
cn.centrality_shortest(
    distances=[800, 2000, 5000],
    sample=True,
)
```

:::warning
Adaptive sampling is experimental: API and behaviour may change in future releases.
:::

For technical details, see the [`metrics.sampling`](/metrics/sampling) documentation.
