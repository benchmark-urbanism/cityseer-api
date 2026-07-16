---
layout: '@src/layouts/PageLayout.astro'
---

# Origin-Destination Flows

Betweenness centrality counts how often each street lies on the shortest path between other streets. In its standard form it treats every origin-destination pair as equally important: each pair contributes one unit of flow to the streets between them. That gives a baseline measure of a street's structural position, but it does not describe movement, because real travel concentrates on the routes between where people live and the places they travel to.

Origin-destination (OD) flow analysis weights each pair of locations by how much travel occurs between them, then routes that flow along the network so every street it passes through accumulates it. The output estimates through-movement rather than structural position.

## The problem uniform betweenness leaves open

Consider a district with a dense residential quarter on one side and a cluster of shops and stations on the other. Standard betweenness weights a path between two quiet residential streets the same as a path between a housing block and the main market. The streets connecting home to amenity carry most of the real movement, but uniform betweenness cannot see this, because it has no information about where trips begin or end.

Weighting the flows by demand corrects this. Streets on the routes people actually take receive higher scores, and streets that are structurally central but lightly used recede.

## How cityseer models it

`cityseer` provides two routes to weighted flows. Both accumulate weighted movement along shortest network paths; they differ in where the pair weights come from.

### Explicit OD matrix

When you have observed trip data, a travel survey, ticketing records, or mobile-phone traces, you already know the weight for each origin-destination pair. Build an OD matrix from that table and the zones it refers to, then route it:

- [`build_od_matrix`](/api/network#build_od_matrix) takes a flow table (origin zone, destination zone, trip weight) and a `GeoDataFrame` of zones, snaps each zone centroid to the nearest network node, and returns a sparse matrix.
- [`betweenness_od`](/api/network#betweenness_od) routes that matrix: only origins with outbound trips are traversed, and each shortest-path contribution is scaled by its pair weight.

### Modelled demand

Often you do not have observed trips, only where people live and where they might go. A spatial interaction model fills the gap by predicting the flows from the two sets of weights and the network distances between them.

[`betweenness_demand`](/metrics/networks#betweenness_demand) uses a **singly (origin-)constrained** model. Each origin distributes its full weight $W_o$ across the destinations it can reach within the distance threshold, giving each destination a share proportional to its attractiveness $W_d$ discounted by the cost of getting there:

$$W_{od} = W_o \cdot \frac{W_d \, f(c_{od})}{\sum_{d'} W_{d'} \, f(c_{od'})}$$

Here $f$ is a distance-decay (deterrence) function, supplied as a [`decay_fn`](/api/decay) expression, and $c_{od}$ is the network distance from origin to destination. With an exponential decay this is the classic gravity model. "Singly constrained" means only the origin totals are held fixed: each origin sends out exactly its own weight, while destination totals are left free (constraining both ends would require a doubly-constrained, or Furness, model). The allocation and the routing happen in one traversal per origin, so no explicit matrix is materialised. Origins and destinations are assigned to the network with the same workflow as the data layers, and the assignment offsets are included in the allocation distances, the decay, and the radius cutoffs.

### Two levers, two questions

Because the allocation is normalised, `decay_fn` decides only where trips go, never how much travel occurs: every origin emits exactly its weight regardless of how far its destinations sit. How much travel occurs is a separate question, answered by the `participation` share: a stay-home option joins the destination choice set, so each origin participates at rate $A_o / (K + A_o)$, where $A_o$ is its accessibility and $K$ is derived from `participation` against the run's own median accessibility. Locations with a rich reachable offer send out nearly their full weight; poorly connected locations send out less. This is the standard discrete-choice treatment of trip generation (a logit with a no-travel alternative), and it makes modelled volumes respond to accessibility rather than being fixed by construction.

`participation = 1` (the default) is full participation, the classic conserved model. For predicting pedestrian volumes, walking mode shares suggest starting around `0.2`, one in five people travelling at a typical location; use a local travel survey's share when available. The fit is not knife-edge in this setting.

## Which method to use

| You have | Use | Notes |
| --- | --- | --- |
| Observed trips between zones | `build_od_matrix` + `betweenness_od` | The weights are your data; no model assumptions |
| Weighted origins and destinations, no trips | `betweenness_demand` | The model predicts the pair weights from distance and attractiveness |
| Only the network | standard `betweenness` (see [Centrality](/guide/centrality)) | Uniform demand; the structural baseline |

The choice of `decay_fn` matters for the modelled route: a steeper decay concentrates flow onto short local trips, a gentler decay spreads it onto longer journeys. The [decay module](/api/decay) provides ready expressions.

## Worked examples

The [Origin-Destination Flows recipes](/examples/flows) work through this on a central-Madrid network: [modelling home-to-amenity demand](/examples/flows/demand-flows), [routing an explicit OD matrix](/examples/flows/explicit-od) under two contrasting demand patterns, and [comparing weighted flow against the uniform betweenness baseline](/examples/flows/flows-vs-betweenness).
