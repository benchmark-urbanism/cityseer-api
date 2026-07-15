---
layout: '@src/layouts/PageLayout.astro'
---

# Demand Betweenness (OD Flow)

Accessible via **Processing > Cityseer > Demand Betweenness (OD Flow)**. Computes demand-weighted flow betweenness from a spatial interaction model. Trips are allocated from weighted origins (for example population) to weighted destinations (for example shops or amenities) with distance decay, then routed along shortest network paths so that intermediate streets accumulate the flow passing through them. Each origin's full weight is conserved and distributed across its reachable destinations.

## Input Parameters

| Parameter                     | Description                                                                                                                                                                                      | Default       |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------- |
| **Street network line layer** | A line layer in a projected metre-based CRS                                                                                                                                                      | _(required)_  |
| **Origins layer**             | Point or polygon layer of demand origins. Polygon centroids are used.                                                                                                                            | _(required)_  |
| **Origin weight field**       | Numeric column giving each origin's weight, for example population                                                                                                                               | _(required)_  |
| **Destinations layer**        | Point or polygon layer of destinations. Polygon centroids are used.                                                                                                                              | _(required)_  |
| **Destination weight field**  | Numeric column giving each destination's attractiveness weight                                                                                                                                   | _(required)_  |
| **Distance thresholds**       | Comma-separated distances in metres                                                                                                                                                              | `800`         |
| **Max snap distance**         | Maximum distance (metres) to snap origins and destinations to the network. Points beyond this are dropped.                                                                                       | `100`         |
| **Closest destination only**  | Route each origin's full weight to its single nearest reachable destination instead of allocating across all of them                                                                             | `False`       |
| **Boundary polygon**          | Optional polygon layer. Segments inside the boundary are written to the output.                                                                                                                  | _(none)_      |
| **Decay expression**          | Advanced. Distance-decay for the allocation, using `c` (metric distance) and `p` (progress = c / threshold). For a classic gravity model on absolute distance use for example `exp(-0.002 * c)`. | `exp(-4 * p)` |
| **Shortest-path tolerance %** | Advanced. Spreads flow across near-shortest routes. 0 = exact shortest paths only. Keep below 2%.                                                                                                | `0.0`         |
| **Time thresholds**           | Advanced. Comma-separated minutes; overrides distances when set. Converted to metres using the walking speed.                                                                                    | _(none)_      |
| **Walking speed**             | Advanced. Metres per second, used to convert minutes to distances.                                                                                                                               | `1.33`        |

## Output

The output is a line layer with the original street segments and a flow column per distance threshold:

```text
cc_demand_<distance>
```

A segment's value is the total origin weight routed through it, so values are comparable across runs that use the same origin weights.

## Model

For each origin, the allocated flow to each reachable destination is proportional to the destination's weight multiplied by the decay function of the network distance, normalised so that the origin's full weight is conserved. This is a singly (origin-)constrained spatial interaction model; the classic gravity model is recovered with an exponential decay on absolute distance. See the [Origin-Destination Flows guide](/guide/flows) for background and worked examples in Python.
