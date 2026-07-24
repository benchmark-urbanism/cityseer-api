---
layout: '@src/layouts/PageLayout.astro'
---

# Glossary

Key terms used throughout the examples and guide.

### Accessibility

A measure of how easily land uses (shops, parks, restaurants, etc.) can be reached from each point in the street network. Computed using network distance, not straight-line distance. See [Accessibility recipes](/examples/accessibility) and [`compute_accessibilities`](/metrics/layers#compute_accessibilities).

### Betweenness Centrality

Counts how often a node (or street segment) lies on the shortest paths between all other pairs of nodes. High betweenness indicates streets that serve as major through-movement corridors. See [Centrality recipes](/examples/centrality).

### Closeness / Harmonic Centrality

Measures how close a node is to all other reachable nodes. The harmonic variant sums the inverse distances, making it robust to disconnected components. Higher values indicate more central, well-connected locations.

### Continuity

Assesses how consistently named routes, road classifications, or route numbers extend through the network. Useful for understanding legibility and identifying major through-routes. See [Continuity recipes](/examples/continuity).

### CRS / EPSG

A **Coordinate Reference System** defines how geographic coordinates map to locations on the Earth's surface. **EPSG** codes are numeric identifiers for specific CRS definitions (e.g., `EPSG:4326` for WGS 84 longitude/latitude, `EPSG:3035` for ETRS89-LAEA Europe). Always use a **projected** CRS (with metre units) for distance and area calculations.

### Density (Network)

Counts the number of other nodes reachable within a given network distance threshold. A simple measure of how well-connected a location is at a particular scale.

### Distance Decay / Beta

A weighting that reduces the influence of features as distance from the analysis point increases. Earlier versions controlled this through a beta parameter derived automatically from the distance thresholds; the current API takes explicit decay expressions instead, either within metric expressions (e.g. `"exp(-4 * p)"`) or via the `decay_fn` parameter, with helpers in the [`cityseer.decay`](/api/decay) module. Not to be confused with the per-edge `imp_factor` traversal impedance described in [Edge Impedance](/guide/networks#edge-impedance). See [Decay Functions](/guide/land-use#decay-functions).

### Dual Graph

A graph representation where **streets become nodes** and **intersections become edges**. This is the inverse of a primal graph. Working with the dual graph is recommended for centrality analysis because it expresses metrics relative to street segments rather than intersections, and it is required for angular (simplest-path) analysis. `CityNetwork` builds the dual graph automatically; when using the lower-level API, convert manually with [`nx_to_dual`](/tools/graphs#nx_to_dual).

### Edge Rolloff

A boundary distortion effect that occurs when the network does not extend far enough beyond the study area. Nodes near the boundary have fewer reachable destinations, producing artificially low metric values. Prevent this by buffering the network and setting boundary nodes to `live=False`. See the [Edge Rolloff](/guide/fundamentals#edge-rolloff) section.

### Live Nodes

Nodes with `live=True` are within the study area and will have metrics computed. Nodes with `live=False` exist in the buffer zone: they are used for routing (preventing edge rolloff) but their results are excluded from output. See the [Edge Rolloff](/guide/fundamentals#edge-rolloff) section.

### Mixed-Use

A measure of the diversity of land uses reachable from each node within a given distance threshold. Higher mixed-use values indicate areas where many different types of amenities are accessible. See [`compute_mixed_uses`](/metrics/layers#compute_mixed_uses).

### Network Distance

The shortest-path distance along the street network between two points, as opposed to Euclidean (straight-line) distance. All `cityseer` metrics use network distance by default.

### NetworkStructure

A `cityseer` data structure (backed by Rust) that holds the network topology for efficient metric computation. `CityNetwork` creates and manages this structure automatically; in the lower-level API it is created from a `networkx` graph using [`network_structure_from_nx`](/tools/io#network_structure_from_nx). Can be reused across multiple metric calculations as long as the network doesn't change.

### Primal Graph

The conventional graph representation where **intersections are nodes** and **streets are edges**. This is the default output from most network creation functions. Convert to a dual graph using `nx_to_dual` for street-level centrality analysis.

### Visibility

Measures how enclosed or open the street environment is at each point, based on surrounding built form. Computed by casting rays from network nodes and measuring how far they travel before hitting a building or obstruction. See [Visibility recipes](/examples/visibility).
