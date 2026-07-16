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
| **Max snap distance**         | Maximum distance (metres) for assigning origins and destinations to the network (nearest street, with the assignment offset included in routed distances). Points beyond this are dropped.       | `100`         |
| **Closest destination only**  | Route each origin's full weight to its single nearest reachable destination instead of allocating across all of them                                                                             | `False`       |
| **Boundary polygon**          | Optional polygon layer. Segments inside the boundary are written to the output.                                                                                                                  | _(none)_      |
| **Decay expression**          | Advanced. Distance-decay for the allocation, using `c` (metric distance) and `p` (progress = c / threshold). For a classic gravity model on absolute distance use for example `exp(-0.002 * c)`. | `exp(-4 * p)` |
| **Flow expressions**          | Advanced. Semicolon-separated `name: expression` pairs; each weights the allocated flow by trip distance and emits its own column. Empty uses the paired `demand` + `demand_decay` default.      | _(paired)_    |
| **Shortest-path tolerance %** | Advanced. Spreads flow across near-shortest routes. 0 = exact shortest paths only. Keep below 2%.                                                                                                | `0.0`         |
| **Time thresholds**           | Advanced. Comma-separated minutes; overrides distances when set. Converted to metres using the walking speed.                                                                                    | _(none)_      |
| **Walking speed**             | Advanced. Metres per second, used to convert minutes to distances.                                                                                                                               | `1.33`        |

## Output

The output is a line layer with the original street segments and one flow column per expression per distance threshold. The default emits two channels from a single traversal:

```text
cc_demand_<distance>
cc_demand_decay_<distance>
```

`cc_demand` holds conserved flow: each trip contributes its full allocated weight to every street it crosses, so a segment's value is the total origin weight routed through it. `cc_demand_decay` attenuates each trip's contribution by its network distance, reflecting trip frequency falling with trip length; this channel is usually the better predictor of observed pedestrian volumes.

## Model

For each origin, the allocated flow to each reachable destination is proportional to the destination's weight multiplied by the decay function of the network distance, normalised so that the origin's full weight is conserved. This is a singly (origin-)constrained spatial interaction model; the classic gravity model is recovered with an exponential decay on absolute distance. The decay expression and the flow-weighting expressions play distinct roles: the decay shapes destination choice within the allocation and, being normalised, never changes how much travel occurs; the flow-weighting expressions scale each trip's contribution by its own distance. See the [Origin-Destination Flows guide](/guide/flows) for background and worked examples in Python.
