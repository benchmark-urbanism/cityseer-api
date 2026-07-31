---
layout: '@src/layouts/PageLayout.astro'
---

# Origin-Destination Flows

Standard betweenness treats every origin-destination pair as equally important. Origin-destination flow analysis weights the movement instead, so streets carrying more actual travel receive higher scores. The [Origin-Destination Flows guide](/guide/flows) explains the challenge and the two methods `cityseer` provides: routing an explicit OD matrix, or modelling demand from weighted origins and destinations with a singly-constrained spatial interaction model.

| Notebook | Description |
| -------- | ----------- |
| [demand_flows](/examples/flows/demand-flows) | Modelled demand flows: a singly-constrained spatial interaction model from population origins to hospitality destinations, with a distance-decay deterrence function. |
| [explicit_od](/examples/flows/explicit-od) | Routing an explicit origin-destination matrix with `build_od_matrix` and `betweenness_od`, and how two different demand patterns reshape the flows on the same network. |
| [flows_vs_betweenness](/examples/flows/flows-vs-betweenness) | Comparing demand-weighted flow against the uniform betweenness baseline on the same network. |
| [london_cycling](/examples/flows/london-cycling) | Cycle-commute through-movement across central London, mapped two ways: observed trips routed with `betweenness_od`, and modelled trips from `betweenness_demand`, using open data. |
