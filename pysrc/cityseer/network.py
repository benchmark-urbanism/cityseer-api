"""High-level CityNetwork API for urban network analysis.

The [`CityNetwork`](#citynetwork) class wraps network construction, centrality computation, land-use analysis, and
export into a single interface. It builds dual graphs (where street segments become nodes rather than intersections)
directly from LineString geometries, enabling both shortest-path (metric distance) and simplest-path (angular change)
analysis. See the [Guide](/guide/fundamentals) for concepts, conventions, and worked examples.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from pyproj import CRS
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry

from . import rustalgos
from .metrics import layers, networks
from .tools import dual, io, util

_BASE_NODE_COLUMNS = {"ns_node_idx", "x", "y", "z", "live", "weight"}


def _normalize_crs(crs: Any | None) -> CRS | None:
    if crs is None:
        return None
    return CRS.from_user_input(crs)


def _require_projected_crs(crs: Any | None) -> CRS:
    if crs is None:
        raise ValueError("A projected CRS is required.")
    normalized = _normalize_crs(crs)
    if normalized is None or not normalized.is_projected:
        raise ValueError("The network CRS must be projected, i.e. not geographic.")
    return normalized


def _load_boundary(boundary_wkt: str | None) -> BaseGeometry | None:
    if boundary_wkt is None:
        return None
    return shapely_wkt.loads(boundary_wkt)


def _merge_saved_columns(
    rebuilt_nodes_gdf: gpd.GeoDataFrame,
    saved_nodes_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    geometry_name = saved_nodes_gdf.geometry.name
    extra_columns = [col for col in saved_nodes_gdf.columns if col != geometry_name and col != "ns_node_idx"]
    if extra_columns:
        idx = saved_nodes_gdf.index
        rebuilt_nodes_gdf.loc[idx, extra_columns] = saved_nodes_gdf.loc[idx, extra_columns]
    return rebuilt_nodes_gdf


def _prepare_dual_node_key(start_nd_key: Any, end_nd_key: Any, edge_idx: int) -> str:
    start_end = sorted([str(start_nd_key), str(end_nd_key)])
    return f"{start_end[0]}_{start_end[1]}_k{edge_idx}"


def _rebuild_dual_network(
    state: dict[str, Any],
    nodes_gdf: gpd.GeoDataFrame,
) -> tuple[rustalgos.graph.NetworkStructure, gpd.GeoDataFrame, dict[str, Any]]:
    network_structure = rustalgos.graph.NetworkStructure()
    network_structure.set_is_dual(True)
    if state.get("directions") is not None:
        network_structure.set_is_directed(True)
    node_idx: dict[Any, int] = {}
    rebuilt_nodes_gdf = nodes_gdf.copy()
    geoms = state.get("geoms", {})
    for node_key, row in rebuilt_nodes_gdf.iterrows():
        z = None
        if "z" in rebuilt_nodes_gdf.columns and pd.notna(row.get("z")):
            z = float(row["z"])
        # the dual node's primal street geometry (representation-aware data assignment)
        street_geom = geoms.get(node_key)
        node_idx[node_key] = network_structure.add_street_node(
            node_key=node_key,
            x=float(row["x"]),
            y=float(row["y"]),
            live=bool(row["live"]),
            weight=float(row["weight"]),
            z=z,
            street_geom_wkt=street_geom.wkt if street_geom is not None else None,
        )
    for record in sorted(state["edge_records"].values(), key=lambda item: item["edge_idx"]):
        network_structure.add_street_edge(
            node_idx[record["start_key"]],
            node_idx[record["end_key"]],
            int(record["edge_idx"]),
            record["start_key"],
            record["end_key"],
            record["geom_wkt"],
            float(record.get("imp_factor", 1.0)),
            shared_primal_node_key=record.get("shared_primal_node_key"),
        )
    network_structure.validate()
    network_structure.build_edge_rtree()
    rebuilt_nodes_gdf["ns_node_idx"] = [node_idx[node_key] for node_key in rebuilt_nodes_gdf.index]
    state["node_idx"] = node_idx
    state["midpoints"] = {
        node_key: (
            float(rebuilt_nodes_gdf.at[node_key, "x"]),
            float(rebuilt_nodes_gdf.at[node_key, "y"]),
        )
        for node_key in rebuilt_nodes_gdf.index
    }
    state["ns"] = network_structure
    return network_structure, rebuilt_nodes_gdf, state


def _merge_input_columns(
    nodes_gdf: gpd.GeoDataFrame,
    input_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    geometry_name = input_gdf.geometry.name
    extra_columns = [col for col in input_gdf.columns if col != geometry_name]
    if extra_columns:
        shared_idx = nodes_gdf.index.intersection(input_gdf.index)
        if len(shared_idx) > 0:
            nodes_gdf.loc[shared_idx, extra_columns] = input_gdf.loc[shared_idx, extra_columns]
    return nodes_gdf


class CityNetwork:
    """High-level interface for urban network analysis.

    Wraps network construction, centrality computation, and land-use analysis into a single object that manages graph
    topology, node attributes, and coordinate reference systems. The network is built as a dual graph where street
    segments become nodes and intersections become edges, enabling both shortest-path (metric) and simplest-path
    (angular) centrality analysis.

    Construct instances via the class methods rather than calling ``__init__`` directly:

    - [`from_geopandas`](#from_geopandas) -- from a GeoDataFrame of LineString geometries
    - [`from_wkts`](#from_wkts) -- from a dictionary of WKT strings or Shapely geometries
    - [`from_nx`](#from_nx) -- from a cityseer-compatible NetworkX MultiGraph
    - [`from_osm`](#from_osm) -- from OpenStreetMap via a bounding polygon
    - [`load`](#load) -- from a previously saved parquet/pickle pair

    Most methods return ``self`` to support method chaining:

    ```python
    cn = (
        CityNetwork.from_geopandas(edges_gdf, crs=32632)
        .set_boundary(boundary_polygon)
        .centrality_shortest(distances=[500, 1000, 2000])
    )
    ```

    :::note
    The underlying graph construction automatically cleans input geometries: short self-loops are removed, chains
    of segments meeting at filler (degree-2) endpoints are welded into single segments, short danglers are removed,
    and near-duplicate parallel edges are merged. Each step is controlled by a constructor parameter
    (``remove_fillers``, ``remove_danglers``, ``merge_parallel_dist``) and can be disabled to pass a network
    through as-is. Use the ``feature_status`` property to inspect which input features were filtered or merged
    and why.
    :::

    ### Dual graph architecture

    ``CityNetwork`` always constructs a dual graph internally. In the dual representation, each street segment becomes
    a node (positioned at the segment midpoint) and edges connect segments that share a common intersection. This
    enables both shortest-path and simplest-path (angular) analysis from a single topology:

    - **Shortest-path** analysis uses metric distances along street segments.
    - **Simplest-path** analysis uses cumulative angular change along streets and at intersections as the routing cost.

    Because the dual is built automatically, there is no need to call ``nx_to_dual`` when using ``CityNetwork``.
    Although the topology is dual internally, results are visualised and exported as the original street segment
    geometries via [`to_geopandas`](#to_geopandas), so each row in the output corresponds to one input street.

    ### Working with results

    All computed metrics are written to the internal ``nodes_gdf`` GeoDataFrame. Since ``CityNetwork`` uses a dual
    graph, each row in ``nodes_gdf`` represents a street segment, with a Point geometry at the segment midpoint.

    To retrieve results with the original LineString geometries, use [`to_geopandas`](#to_geopandas):

    ```python
    cn = CityNetwork.from_geopandas(edges_gdf, crs=32632)
    cn.centrality_shortest(distances=[800])

    # Midpoint geometries (internal representation)
    cn.nodes_gdf["cc_harmonic_800"]

    # Original LineString geometries with the same computed columns
    result_gdf = cn.to_geopandas()
    result_gdf["cc_harmonic_800"]
    ```

    Column names follow the ``cc_{metric}_{distance}`` convention described in the
    [Column Naming Conventions](/guide/fundamentals#column-naming-conventions) section.

    ### Feature cleaning

    Input geometries are automatically cleaned during construction, on the primal edge set before dual
    conversion, in this order:

    1. **Short self-loops** (under 1 m) are removed as corrupt geometry.
    2. **Filler welding** (``remove_fillers=True``): chains of segments meeting at degree-2 endpoints are welded
       into single segments, so a street subdivided during digitisation is treated as one segment. The welded
       feature keeps the id of the longest constituent; absorbed features are marked ``"merged"``.
    3. **Danglers** (``remove_danglers=10.0``): dead-end stubs up to this length are removed iteratively, judged
       on the full welded length. ``0`` disables.
    4. **Parallel merging** (``merge_parallel_dist=2.0``): edges whose endpoints fall within this distance of
       another edge's endpoints, with near-identical lengths, are the same street drawn twice; the longest is
       kept. ``0`` disables.

    Directed networks skip filler welding and parallel merging, which do not preserve one-way semantics. To pass
    a network through with minimal intervention, disable the steps:

    ```python
    cn = CityNetwork.from_geopandas(edges_gdf, remove_fillers=False, remove_danglers=0, merge_parallel_dist=0)
    ```

    The ``feature_status`` property returns a Series indicating the status of each input feature:

    ```python
    cn = CityNetwork.from_geopandas(edges_gdf, crs=32632)
    print(cn.feature_status.value_counts())
    # active              142
    # merged                6
    # short_dangler         3
    # duplicate             1
    ```

    ### Saving and loading

    Networks can be serialised to disk and restored later, preserving all computed metrics:

    ```python
    cn.save("my_network")
    # Creates: my_network.nodes.parquet, my_network.state.pkl

    cn_restored = CityNetwork.load("my_network")
    ```

    ### Updating geometries

    To analyse modified geometries, construct a fresh network from the updated data; construction is fast and
    stateless, and cleaning runs exactly once per construction. (Controlled in-place editing exists for the QGIS
    plugin via ``tools.dual.incremental_update``, where the layer-edit workflow warrants it.)

    ```python
    cn = CityNetwork.from_geopandas(updated_edges_gdf, crs=32632)
    cn.centrality_shortest(distances=[800])
    ```

    ### Typical workflow

    ```python
    import geopandas as gpd
    from cityseer.network import CityNetwork
    from cityseer import decay

    # 1. Load street network edges
    edges_gdf = gpd.read_file("streets.gpkg")

    # 2. Build the network
    cn = CityNetwork.from_geopandas(edges_gdf, crs="EPSG:32632")

    # 3. Compute centrality at multiple scales
    cn.centrality_shortest(distances=[400, 800, 1600])
    cn.centrality_simplest(distances=[400, 800, 1600])

    # 4. Compute land-use accessibility
    landuses_gdf = gpd.read_file("landuses.gpkg")
    cn = cn.compute_accessibilities(
        data_gdf=landuses_gdf,
        landuse_column_label="category",
        accessibility_keys=["retail", "park"],
        distances=[400, 800],
        decay_fn=decay.exponential(),
    )

    # 5. Compute statistical aggregations
    prices_gdf = gpd.read_file("property_prices.gpkg")
    cn = cn.compute_stats(
        data_gdf=prices_gdf,
        stats_column_labels=["price"],
        distances=[800, 1600],
        decay_fn=decay.gaussian(peak=400, cutoff=1600),
    )

    # 6. Export results with original LineString geometries
    result_gdf = cn.to_geopandas()
    result_gdf.to_file("results.gpkg")
    ```

    For end-to-end worked examples with real-world data, see the
    [Cityseer Examples](https://cityseer.benchmarkurbanism.com/examples) site, including the
    [Quickstart](https://cityseer.benchmarkurbanism.com/examples/recipes/quickstart) notebook.
    """

    def __init__(
        self,
        network_structure: rustalgos.graph.NetworkStructure,
        nodes_gdf: gpd.GeoDataFrame,
        *,
        _state: dict[str, Any],
        _crs: Any | None = None,
    ):
        self._network_structure = network_structure
        self._nodes_gdf = nodes_gdf
        self._state = _state
        self._state["ns"] = network_structure
        self._crs = _normalize_crs(_crs if _crs is not None else nodes_gdf.crs)
        self._sync_feature_status_column()

    @property
    def network_structure(self) -> rustalgos.graph.NetworkStructure:
        """The underlying Rust-optimised [`NetworkStructure`](/rustalgos/graph#networkstructure)."""
        return self._network_structure

    @property
    def nodes_gdf(self) -> gpd.GeoDataFrame:
        """The internal nodes GeoDataFrame with point geometries at segment midpoints.

        Columns include ``ns_node_idx``, ``x``, ``y``, ``z`` (if elevation data is present), ``live``, ``weight``,
        and any centrality or layer results
        that have been computed. This is the *working* GeoDataFrame used by centrality and layer methods.
        Use [`to_geopandas`](#to_geopandas) to obtain a GeoDataFrame with the original LineString geometries.
        """
        return self._nodes_gdf

    def to_geopandas(self) -> gpd.GeoDataFrame:
        """Return a GeoDataFrame with the original input LineString geometries.

        The returned GeoDataFrame contains all computed columns (centrality metrics, layer results, etc.)
        joined to the original edge geometries rather than the midpoint representations used internally.

        Returns
        -------
        gdf: GeoDataFrame
            A new GeoDataFrame indexed by feature id with LineString geometries.

        Examples
        --------
        ```python
        cn.centrality_shortest(distances=[800])
        result_gdf = cn.to_geopandas()

        # result_gdf has LineString geometries (not midpoint Points)
        print(result_gdf.geometry.geom_type.unique())  # ['LineString']

        # All computed columns are present
        print(result_gdf["cc_harmonic_800"])

        # Export to file
        result_gdf.to_file("centrality_results.gpkg")
        ```
        """
        return gpd.GeoDataFrame(  # type: ignore
            {col: self._nodes_gdf[col] for col in self._nodes_gdf.columns if col != self._nodes_gdf.geometry.name},
            index=self._nodes_gdf.index,
            geometry=[self._state["geoms"][fid] for fid in self._nodes_gdf.index],
            crs=self._crs,
        )

    @property
    def is_dual(self) -> bool:
        """Whether the network is a dual graph (always ``True`` for CityNetwork)."""
        return bool(self._network_structure.is_dual)

    @property
    def is_directed(self) -> bool:
        """Whether the network is a directed graph respecting one-way streets."""
        return bool(self._network_structure.is_directed)

    @property
    def crs(self) -> CRS | None:
        """The projected coordinate reference system."""
        return self._crs

    @property
    def node_count(self) -> int:
        """Number of nodes (street segments) in the network."""
        return len(self._nodes_gdf)

    @property
    def feature_status(self) -> pd.Series:
        """Status of each input feature after geometry cleaning.

        Possible values: ``"active"``, ``"invalid_geometry"``, ``"short_self_loop"``,
        ``"merged"`` (welded into an adjacent feature by ``remove_fillers``),
        ``"duplicate"``, ``"short_dangler"``, ``"deleted"``.
        """
        return pd.Series(self._state.get("feature_status", {}), name="feature_status")

    @classmethod
    def _from_dual_result(
        cls,
        network_structure: rustalgos.graph.NetworkStructure,
        nodes_gdf: gpd.GeoDataFrame | None,
        state: dict[str, Any],
        crs: CRS,
    ) -> CityNetwork:
        if nodes_gdf is None:
            raise RuntimeError("Fast dual build did not produce nodes GeoDataFrame.")
        return cls(network_structure, nodes_gdf, _state=state, _crs=crs)

    @classmethod
    def from_wkts(
        cls,
        wkts: dict[Any, str] | dict[Any, BaseGeometry],
        *,
        crs: Any,
        boundary: BaseGeometry | None = None,
        directed: bool = False,
        oneway_fids: set[Any] | None = None,
        impedances: dict[Any, float] | None = None,
        remove_fillers: bool = True,
        remove_danglers: float = 10.0,
        merge_parallel_dist: float = 2.0,
        segment_weighted: bool = False,
    ) -> CityNetwork:
        """Construct a CityNetwork from a dictionary of WKT strings or Shapely geometries.

        Parameters
        ----------
        wkts: dict[Any, str] | dict[Any, BaseGeometry]
            A mapping from feature identifiers to WKT strings or Shapely LineString geometries.
            Input geometries may include z (elevation) coordinates, which are preserved and used
            for slope-based walking impedance calculations.
        crs: Any
            A projected coordinate reference system (EPSG code, CRS object, or proj string).
        boundary: BaseGeometry
            Optional polygon in the same projected CRS; nodes inside are marked as ``live``,
            nodes outside as ``dead``.
        directed: bool
            If ``True``, build a directed network. Requires ``oneway_fids``. Features in
            ``oneway_fids`` are one-way (in LineString coordinate order); all other features
            are bidirectional.
        oneway_fids: set[Any] | None
            Feature IDs that are one-way when ``directed=True``. Ignored if ``directed=False``.
        impedances: dict[Any, float] | None
            Optional mapping from primal feature ID to its impedance factor. Each dual edge's
            ``imp_factor`` becomes the length-weighted mean of its two adjacent primal segments'
            impedances; missing entries default to ``1.0``. See the [Edge Impedance](/guide/networks#edge-impedance)
            section of the guide.
        remove_fillers: bool
            Weld chains of segments meeting at filler (degree-2) endpoints into single segments,
            so a street subdivided during digitisation is treated as one segment. Welded features
            keep the id of the longest constituent; absorbed features are marked ``"merged"`` in
            [`feature_status`](#feature_status). Skipped for directed networks. Default ``True``.
        remove_danglers: float
            Remove dead-end stubs up to this length in metres (iteratively, after filler welding,
            so a welded chain is judged on its full length). ``0`` disables. Default ``10.0``.
        merge_parallel_dist: float
            Merge near-duplicate parallel edges: edges whose endpoints fall within this distance
            of another edge's endpoints and whose lengths are near-identical are the same street
            drawn twice; the longest is kept. ``0`` disables. Skipped for directed networks.
            Default ``2.0``.
        segment_weighted: bool
            Default for the ``segment_weighted`` option of the centrality methods: weight each
            node by its street segment length rather than a unit count. Can be overridden per
            centrality call. Default ``False``.

        Returns
        -------
        network: CityNetwork
            A new CityNetwork instance.

        Raises
        ------
        ValueError
            If ``directed=True`` but ``oneway_fids`` is not provided.

        Examples
        --------
        ```python
        from shapely.geometry import LineString
        from cityseer.network import CityNetwork

        wkts = {
            "street_a": LineString([(0, 0), (100, 0)]),
            "street_b": LineString([(100, 0), (100, 100)]),
            "street_c": LineString([(100, 0), (200, 0)]),
        }
        cn = CityNetwork.from_wkts(wkts, crs=32632)
        cn.centrality_shortest(distances=[200])
        ```
        """
        normalized_crs = _require_projected_crs(crs)
        directions = None
        if directed:
            if oneway_fids is None:
                raise ValueError("directed=True requires oneway_fids specifying which features are one-way.")
            directions = {fid: (True, False) if fid in oneway_fids else (True, True) for fid in wkts}
        ns, nodes_gdf, state = dual.build_dual(
            wkts,
            crs=normalized_crs,
            boundary=boundary,
            directions=directions,
            impedances=impedances,
            remove_fillers=remove_fillers,
            remove_danglers=remove_danglers,
            merge_parallel_dist=merge_parallel_dist,
        )
        state["segment_weighted_default"] = bool(segment_weighted)
        return cls._from_dual_result(ns, nodes_gdf, state, normalized_crs)

    @classmethod
    def from_geopandas(
        cls,
        gdf: gpd.GeoDataFrame,
        *,
        crs: Any | None = None,
        boundary: BaseGeometry | None = None,
        directed: bool = False,
        remove_fillers: bool = True,
        remove_danglers: float = 10.0,
        merge_parallel_dist: float = 2.0,
        segment_weighted: bool = False,
    ) -> CityNetwork:
        """Construct a CityNetwork from a GeoDataFrame of LineString geometries.

        Extra columns from the input GeoDataFrame are carried through to the internal nodes GeoDataFrame. The CRS
        is read from the GeoDataFrame unless explicitly overridden.

        Parameters
        ----------
        gdf: GeoDataFrame
            A GeoDataFrame with LineString or MultiLineString geometries. The index must be unique.
            Input geometries may include z (elevation) coordinates, which are preserved and used
            for slope-based walking impedance calculations.
        crs: Any
            Optional projected CRS override. If ``None``, uses the GeoDataFrame's CRS.
        boundary: BaseGeometry
            Optional polygon in the same projected CRS; nodes inside are marked as ``live``,
            nodes outside as ``dead``.
        directed: bool
            If ``True``, build a directed network. Requires a boolean ``oneway`` column in the
            GeoDataFrame. Features with ``oneway=True`` are one-way in LineString coordinate
            order; features with ``oneway=False`` are bidirectional.
        remove_fillers: bool
            Weld chains of segments meeting at filler (degree-2) endpoints into single segments,
            so a street subdivided during digitisation is treated as one segment. Welded features
            keep the id of the longest constituent; absorbed features are marked ``"merged"`` in
            [`feature_status`](#feature_status). Skipped for directed networks. Default ``True``.
        remove_danglers: float
            Remove dead-end stubs up to this length in metres (iteratively, after filler welding,
            so a welded chain is judged on its full length). ``0`` disables. Default ``10.0``.
        merge_parallel_dist: float
            Merge near-duplicate parallel edges: edges whose endpoints fall within this distance
            of another edge's endpoints and whose lengths are near-identical are the same street
            drawn twice; the longest is kept. ``0`` disables. Skipped for directed networks.
            Default ``2.0``.
        segment_weighted: bool
            Default for the ``segment_weighted`` option of the centrality methods: weight each
            node by its street segment length rather than a unit count. Can be overridden per
            centrality call. Default ``False``.

        Notes
        -----
        An optional ``imp_factor`` column on the input ``GeoDataFrame`` is propagated to each
        dual edge as the length-weighted mean of the two adjacent primal segments' impedances;
        omit it to leave every dual edge at the default ``1.0``. See the
        [Edge Impedance](/guide/networks#edge-impedance) section of the guide.

        Returns
        -------
        network: CityNetwork
            A new CityNetwork instance.

        Raises
        ------
        ValueError
            If ``directed=True`` but the GeoDataFrame has no ``oneway`` column.

        Examples
        --------
        ```python
        import geopandas as gpd
        from shapely.geometry import LineString
        from cityseer.network import CityNetwork

        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (100, 100)]),
                    LineString([(100, 0), (200, 0)]),
                ]
            },
            crs="EPSG:32632",
        )
        cn = CityNetwork.from_geopandas(gdf)
        cn.centrality_shortest(distances=[200])
        print(cn.nodes_gdf[["cc_harmonic_200", "cc_betweenness_200"]])
        ```

        With directed one-way streets:

        ```python
        gdf = gpd.GeoDataFrame(
            {
                "geometry": [
                    LineString([(0, 0), (100, 0)]),
                    LineString([(100, 0), (200, 0)]),
                ],
                "oneway": [True, False],
            },
            crs="EPSG:32632",
        )
        cn = CityNetwork.from_geopandas(gdf, directed=True)
        ```
        """
        normalized_crs = _require_projected_crs(crs if crs is not None else gdf.crs)
        directions = None
        if directed:
            if "oneway" not in gdf.columns:
                raise ValueError("directed=True requires a boolean 'oneway' column in the GeoDataFrame.")
            oneway_raw = gdf["oneway"]
            if oneway_raw.isna().any():
                raise ValueError("The 'oneway' column contains NaN values.")
            if not pd.api.types.is_bool_dtype(oneway_raw):
                raise TypeError(f"The 'oneway' column must contain boolean values, but found dtype {oneway_raw.dtype}.")
            oneway = oneway_raw.astype(bool)
            directions = {fid: (True, False) if ow else (True, True) for fid, ow in oneway.items()}
        # Forward an optional `imp_factor` column on the input GeoDataFrame as per-fid impedance.
        impedances: dict[Any, float] | None = None
        if "imp_factor" in gdf.columns:
            impedances = {fid: float(val) for fid, val in gdf["imp_factor"].items()}
        ns, nodes_gdf, state = dual.build_dual(
            gdf,
            crs=normalized_crs,
            boundary=boundary,
            directions=directions,
            impedances=impedances,
            remove_fillers=remove_fillers,
            remove_danglers=remove_danglers,
            merge_parallel_dist=merge_parallel_dist,
        )
        if nodes_gdf is not None:
            nodes_gdf = _merge_input_columns(nodes_gdf, gdf)
        state["segment_weighted_default"] = bool(segment_weighted)
        return cls._from_dual_result(ns, nodes_gdf, state, normalized_crs)

    @classmethod
    def from_nx(
        cls,
        graph: nx.MultiGraph,
        *,
        boundary: BaseGeometry | None = None,
        remove_fillers: bool = True,
        remove_danglers: float = 10.0,
        merge_parallel_dist: float = 2.0,
        segment_weighted: bool = False,
    ) -> CityNetwork:
        """Construct a CityNetwork from a cityseer-compatible NetworkX graph.

        The input graph must be a *primal* edge graph (not a dual graph) with ``geom`` attributes on edges and a
        ``crs`` attribute on the graph. Node ``live`` attributes are preserved.

        When a ``MultiDiGraph`` is passed, directed mode is enabled automatically: each directed edge becomes
        its own one-way dual node (in the coordinate order of the directed edge). Two-way streets should be
        represented as two reciprocal edges (A to B and B to A), which produce two separate dual nodes.

        Parameters
        ----------
        graph: nx.MultiGraph | nx.MultiDiGraph
            A cityseer-compatible primal NetworkX graph. ``MultiDiGraph`` enables directed routing.
            Any ``imp_factor`` edge attribute is propagated to each dual edge as the length-weighted
            mean of the two adjacent primal segments' impedances (default ``1.0`` if absent). See the
            [Edge Impedance](/guide/networks#edge-impedance) section of the guide.
        boundary: BaseGeometry
            Optional polygon in the same projected CRS; nodes inside are marked as ``live``,
            nodes outside as ``dead``.
        remove_fillers: bool
            Weld chains of segments meeting at filler (degree-2) endpoints into single segments,
            so a street subdivided during digitisation is treated as one segment. Welded features
            keep the id of the longest constituent; absorbed features are marked ``"merged"`` in
            [`feature_status`](#feature_status). Skipped for directed networks. Default ``True``.
        remove_danglers: float
            Remove dead-end stubs up to this length in metres (iteratively, after filler welding,
            so a welded chain is judged on its full length). ``0`` disables. Default ``10.0``.
        merge_parallel_dist: float
            Merge near-duplicate parallel edges: edges whose endpoints fall within this distance
            of another edge's endpoints and whose lengths are near-identical are the same street
            drawn twice; the longest is kept. ``0`` disables. Skipped for directed networks.
            Default ``2.0``.
        segment_weighted: bool
            Default for the ``segment_weighted`` option of the centrality methods: weight each
            node by its street segment length rather than a unit count. Can be overridden per
            centrality call. Default ``False``.

        Returns
        -------
        network: CityNetwork
            A new CityNetwork instance.

        Raises
        ------
        ValueError
            If the input graph is a dual graph.

        Examples
        --------
        From an undirected graph:

        ```python
        import networkx as nx
        from shapely.geometry import LineString
        from cityseer.network import CityNetwork

        G = nx.MultiGraph(crs="EPSG:32632")
        G.add_node("a", x=0.0, y=0.0)
        G.add_node("b", x=100.0, y=0.0)
        G.add_node("c", x=200.0, y=0.0)
        G.add_edge("a", "b", geom=LineString([(0, 0), (100, 0)]))
        G.add_edge("b", "c", geom=LineString([(100, 0), (200, 0)]))

        cn = CityNetwork.from_nx(G)
        ```

        From a directed MultiDiGraph (e.g. via OSMnx):

        ```python
        G = nx.MultiDiGraph(crs="EPSG:32632")
        G.add_node("a", x=0.0, y=0.0)
        G.add_node("b", x=100.0, y=0.0)
        # One-way: a -> b only
        G.add_edge("a", "b", key=0, geom=LineString([(0, 0), (100, 0)]))
        cn = CityNetwork.from_nx(G)
        assert cn.is_directed
        ```
        """
        primal_graph = util.validate_cityseer_networkx_graph(graph)
        if primal_graph.graph.get("is_dual", False):
            raise ValueError("CityNetwork.from_nx expects a primal edge graph, not a dual graph.")

        is_digraph = isinstance(primal_graph, nx.MultiDiGraph)
        wkts: dict[str, str] = {}
        live_overrides: dict[str, bool] = {}
        edge_attrs: dict[str, dict[str, Any]] = {}
        impedances: dict[str, float] = {}
        directions: dict[str, tuple[bool, bool]] | None = None

        def _collect_edge(dual_node_key: str, u: Any, v: Any, k: int, data: dict[str, Any]) -> None:
            wkts[dual_node_key] = data["geom"].wkt
            if "live" in primal_graph.nodes[u] and "live" in primal_graph.nodes[v]:
                live_overrides[dual_node_key] = bool(primal_graph.nodes[u]["live"] or primal_graph.nodes[v]["live"])
            edge_attrs[dual_node_key] = {key: value for key, value in data.items() if key != "geom"}
            edge_attrs[dual_node_key]["primal_edge_node_a"] = u
            edge_attrs[dual_node_key]["primal_edge_node_b"] = v
            edge_attrs[dual_node_key]["primal_edge_idx"] = int(k)
            impedances[dual_node_key] = float(data.get("imp_factor", 1.0))

        if is_digraph:
            # Each directed edge becomes its own dual node, marked one-way in its
            # coordinate direction.  Using directed (unsorted) keys avoids collisions
            # between distinct A->B and B->A edges that happen to share the same key.
            directions = {}
            for u, v, k, data in primal_graph.edges(keys=True, data=True):
                dual_node_key = f"{u}_{v}_k{k}"
                _collect_edge(dual_node_key, u, v, k, data)
                directions[dual_node_key] = (True, False)
        else:
            for u, v, k, data in primal_graph.edges(keys=True, data=True):
                dual_node_key = _prepare_dual_node_key(u, v, int(k))
                _collect_edge(dual_node_key, u, v, k, data)

        normalized_crs = _require_projected_crs(primal_graph.graph["crs"])
        ns, nodes_gdf, state = dual.build_dual(
            wkts,
            crs=normalized_crs,
            boundary=boundary,
            directions=directions,
            impedances=impedances,
            remove_fillers=remove_fillers,
            remove_danglers=remove_danglers,
            merge_parallel_dist=merge_parallel_dist,
        )
        if nodes_gdf is None:
            raise RuntimeError("Fast dual build did not produce nodes GeoDataFrame.")
        for node_key, live in live_overrides.items():
            if node_key in state["node_idx"]:
                ns.set_node_live(state["node_idx"][node_key], live)
                nodes_gdf.at[node_key, "live"] = live
        if edge_attrs:
            attr_df = pd.DataFrame.from_dict(edge_attrs, orient="index")
            shared_idx = attr_df.index.intersection(nodes_gdf.index)
            if len(shared_idx) > 0:
                nodes_gdf.loc[shared_idx, attr_df.columns] = attr_df.loc[shared_idx, attr_df.columns]
        state["nx_source_graph"] = graph
        state["segment_weighted_default"] = bool(segment_weighted)
        return cls._from_dual_result(ns, nodes_gdf, state, normalized_crs)

    @classmethod
    def from_osm(
        cls,
        poly_geom: BaseGeometry,
        *,
        poly_crs_code: int = 4326,
        to_crs_code: int | None = None,
        simplify: bool = True,
        boundary: BaseGeometry | None = None,
        remove_fillers: bool = True,
        remove_danglers: float = 10.0,
        merge_parallel_dist: float = 2.0,
        segment_weighted: bool = False,
        **kwargs: Any,
    ) -> CityNetwork:
        """Construct a CityNetwork from OpenStreetMap data within a bounding polygon.

        Downloads the road network and converts it to a dual CityNetwork.

        For directed (one-way) routing with OSM data, fetch a directed graph via
        `OSMnx <https://osmnx.readthedocs.io/>`_ and pass it to :meth:`from_nx`
        or convert it with :func:`io.nx_from_osm_nx(directed=True) <cityseer.tools.io.nx_from_osm_nx>`.

        Parameters
        ----------
        poly_geom: BaseGeometry
            A Shapely polygon defining the area of interest.
        poly_crs_code: int
            EPSG code for ``poly_geom``. Defaults to 4326 (WGS84).
        to_crs_code: int
            Target projected EPSG code. If ``None``, an appropriate UTM zone is inferred.
        simplify: bool
            Whether to simplify the OSM graph topology (the aggressive, OSM-tuned pipeline:
            junction consolidation, parallel carriageway merging by midline, ironing).
            Defaults to ``True``.
        boundary: BaseGeometry
            Optional polygon for live/dead node assignment (in the target projected CRS).
        remove_fillers: bool
            Weld chains of segments meeting at filler (degree-2) endpoints into single segments
            during dual construction. Default ``True``.
        remove_danglers: float
            Remove dead-end stubs up to this length in metres during dual construction.
            ``0`` disables. Default ``10.0``.
        merge_parallel_dist: float
            Merge near-duplicate parallel edges (endpoints within this distance, near-identical
            lengths) during dual construction. ``0`` disables. Default ``2.0``.
        segment_weighted: bool
            Default for the ``segment_weighted`` option of the centrality methods. Can be
            overridden per centrality call. Default ``False``.
        **kwargs
            Additional keyword arguments passed to
            [`io.osm_graph_from_poly`](/tools/io#osm_graph_from_poly).

        Returns
        -------
        network: CityNetwork
            A new CityNetwork instance.

        Examples
        --------
        ```python
        from shapely.geometry import box
        from cityseer.network import CityNetwork

        # Bounding box in WGS84 (lon/lat)
        polygon = box(-0.13, 51.51, -0.12, 51.52)
        cn = CityNetwork.from_osm(polygon, to_crs_code=32630)
        cn.centrality_shortest(distances=[400, 800])
        ```
        """
        graph = io.osm_graph_from_poly(
            poly_geom,
            poly_crs_code=poly_crs_code,
            to_crs_code=to_crs_code,
            simplify=simplify,
            **kwargs,
        )
        return cls.from_nx(
            graph,
            boundary=boundary,
            remove_fillers=remove_fillers,
            remove_danglers=remove_danglers,
            merge_parallel_dist=merge_parallel_dist,
            segment_weighted=segment_weighted,
        )

    def _clear_metric_columns(self) -> None:
        metric_columns = [col for col in self._nodes_gdf.columns if col.startswith("cc_")]
        if metric_columns:
            self._nodes_gdf.drop(columns=metric_columns, inplace=True)

    def _sync_feature_status_column(self) -> None:
        if "feature_status" in self._state:
            self._nodes_gdf["feature_status"] = [
                self._state["feature_status"].get(fid, "active") for fid in self._nodes_gdf.index
            ]

    def set_boundary(self, polygon: BaseGeometry) -> CityNetwork:
        """Set live/dead node status based on a boundary polygon.

        Nodes whose midpoints fall inside the polygon are marked ``live``; others are marked ``dead``.
        Dead nodes participate in traversal but their own values are not reported. Closeness skips dead
        sources in exact mode (a cost saving); betweenness sources from every node so that routes through
        the study area — including those between dead nodes — credit the live nodes they traverse.

        Parameters
        ----------
        polygon: BaseGeometry
            A Shapely polygon in the same projected CRS as the network.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining.
        """
        from shapely.geometry import Point
        from shapely.prepared import prep

        prepared = prep(polygon)
        for idx, row in self._nodes_gdf.iterrows():
            live = prepared.contains(Point(float(row["x"]), float(row["y"])))
            self._network_structure.set_node_live(int(row["ns_node_idx"]), live)
            self._nodes_gdf.at[idx, "live"] = live
        self._state["boundary_wkt"] = polygon.wkt
        return self

    def set_all_live(self) -> CityNetwork:
        """Mark all nodes as live, clearing any boundary restriction.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining.
        """
        for idx, row in self._nodes_gdf.iterrows():
            self._network_structure.set_node_live(int(row["ns_node_idx"]), True)
            self._nodes_gdf.at[idx, "live"] = True
        self._state["boundary_wkt"] = None
        return self

    def save(self, path: str | Path) -> None:
        """Save the network to disk as a parquet/pickle pair.

        Creates two files: ``<path>.nodes.parquet`` (the nodes GeoDataFrame with all computed columns) and
        ``<path>.state.pkl`` (source and cleaned WKTs, boundary, cleaning parameters, and feature
        status). Use [`load`](#load) to restore.

        Parameters
        ----------
        path: str | Path
            Base file path. File extensions are replaced automatically.

        Examples
        --------
        ```python
        cn.centrality_shortest(distances=[800])
        cn.save("my_network")
        # Creates: my_network.nodes.parquet, my_network.state.pkl

        # Later, restore the full network with all metrics
        cn_restored = CityNetwork.load("my_network")
        print(cn_restored.nodes_gdf["cc_harmonic_800"])
        ```
        """
        path = Path(path)
        self._nodes_gdf.to_parquet(path.with_suffix(".nodes.parquet"))
        payload = {
            "source_wkts": dict(self._state.get("source_wkts", self._state["wkts"])),
            "boundary_wkt": self._state.get("boundary_wkt"),
            "feature_status": dict(self._state.get("feature_status", {})),
            "directions": self._state.get("directions"),
            "impedances": dict(self._state.get("impedances", {})),
            "clean_params": dict(self._state.get("clean_params", dual.LEGACY_CLEAN_PARAMS)),
            "segment_weighted_default": bool(self._state.get("segment_weighted_default", False)),
            # the cleaned (post-weld) geometries are the canonical network: load() reconstructs
            # from these with cleaning disabled, so cleaning runs exactly once, at construction,
            # and a reloaded network is identical regardless of cleaning changes between versions
            "cleaned_wkts": {fid: geom.wkt for fid, geom in self._state.get("geoms", {}).items()},
            "effective_impedances": dict(self._state.get("effective_impedances", self._state.get("impedances", {}))),
            "merges": dict(self._state.get("merges", {})),
        }
        with path.with_suffix(".state.pkl").open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> CityNetwork:
        """Load a previously saved CityNetwork from disk.

        Reconstructs the graph topology from the saved cleaned geometries with cleaning disabled, so the
        loaded network is identical to the saved one (cleaning runs exactly once, at construction), and
        merges any previously computed
        columns (centrality metrics, layer results) from the saved nodes GeoDataFrame.

        Parameters
        ----------
        path: str | Path
            Base file path (same as was passed to [`save`](#save)).

        Returns
        -------
        network: CityNetwork
            The restored CityNetwork instance.
        """
        path = Path(path)
        saved_nodes_gdf = gpd.read_parquet(path.with_suffix(".nodes.parquet"))
        with path.with_suffix(".state.pkl").open("rb") as file:
            payload = pickle.load(file)
        boundary = _load_boundary(payload.get("boundary_wkt"))
        source_wkts = payload.get("source_wkts", payload.get("wkts"))
        directions = payload.get("directions")
        impedances = payload.get("impedances")
        clean_params = payload.get("clean_params", dict(dual.LEGACY_CLEAN_PARAMS))
        cleaned_wkts = payload.get("cleaned_wkts")
        # Build state (geoms, edge_records, endpoint_to_fids, etc.)
        # but skip building nodes_gdf — we'll use the saved one.
        if cleaned_wkts:
            # pure reconstruction: rebuild from the saved cleaned (post-weld) geometries with
            # cleaning disabled, so cleaning runs exactly once, at construction, and a reloaded
            # network is identical to the saved one regardless of cleaning changes between versions
            active_directions = {fid: d for fid, d in directions.items() if fid in cleaned_wkts} if directions else None
            _ns, _nodes_gdf, state = dual.build_dual(
                cleaned_wkts,
                crs=saved_nodes_gdf.crs,
                boundary=boundary,
                build_nodes_gdf=False,
                directions=active_directions,
                impedances=payload.get("effective_impedances", impedances),
                remove_fillers=False,
                remove_danglers=0,
                merge_parallel_dist=0,
            )
            # restore the construction-time bookkeeping: diffs and rebuilds run against the
            # original source with the original cleaning parameters and impedances
            state["source_wkts"] = dict(source_wkts)
            state["impedances"] = dict(impedances or {})
            state["clean_params"] = clean_params
            state["merges"] = dict(payload.get("merges", {}))
            state["directions"] = directions
        else:
            # states saved before cleaned geometries were persisted: rebuild from source with
            # the saved cleaning parameters (legacy behaviour if the state predates those too)
            _ns, _nodes_gdf, state = dual.build_dual(
                source_wkts,
                crs=saved_nodes_gdf.crs,
                boundary=boundary,
                build_nodes_gdf=False,
                directions=directions,
                impedances=impedances,
                **clean_params,
            )
        state["feature_status"] = payload.get("feature_status", state.get("feature_status", {}))
        state["segment_weighted_default"] = bool(payload.get("segment_weighted_default", False))
        # Merge saved columns (metrics, user attrs) onto fresh topology,
        # then rebuild NS once from the merged result.
        merged_gdf = dual._build_nodes_gdf(
            _ns,
            state["fid_list"],
            state["node_idx"],
            state["midpoints"],
            saved_nodes_gdf.crs,
            geoms=state["geoms"],
        )
        merged_gdf = _merge_saved_columns(merged_gdf, saved_nodes_gdf)
        ns, nodes_gdf, state = _rebuild_dual_network(state, merged_gdf)
        return cls(ns, nodes_gdf, _state=state, _crs=nodes_gdf.crs)

    def centrality_shortest(self, **kwargs: Any) -> CityNetwork:
        """Compute shortest-path (metric) centrality.

        Wraps [`centrality_shortest`](/metrics/networks#centrality_shortest). All keyword arguments
        are forwarded; see that function for the full parameter list including ``distances``,
        ``minutes``, ``closeness``, ``betweenness``, ``cycles``, ``postprocess``,
        ``segment_weighted``, ``sample``, and ``epsilon``.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining. Results are written to ``nodes_gdf``.

        Examples
        --------
        ```python
        # Multiple distance thresholds
        cn.centrality_shortest(distances=[400, 800, 1600])

        # With segment-length weighting
        cn.centrality_shortest(distances=[400, 800], segment_weighted=True)

        # With sampling for large networks
        cn.centrality_shortest(
            distances=[800, 2000, 5000],
            sample=True,
            epsilon=0.05,
        )

        # Custom closeness metric
        cn.centrality_shortest(
            distances=[800],
            closeness={"harmonic": "1/c", "gravity": "exp(-0.005 * c)"},
            betweenness={},
        )
        ```

        By default this emits just ``cc_harmonic_{d}`` (closeness) and ``cc_betweenness_{d}``, with cycles
        off. Pass ``closeness`` / ``betweenness`` expression dicts (and ``cycles=True``) to compute any of the
        metrics below (see [Column Naming Conventions](/guide/fundamentals#column-naming-conventions)):

        | Column | Description |
        | --- | --- |
        | ``cc_density_{d}`` | Count of reachable nodes (or total reachable street length if segment_weighted). |
        | ``cc_harmonic_{d}`` | Harmonic closeness: sum of inverse distances to reachable nodes (default). |
        | ``cc_farness_{d}`` | Sum of distances to reachable nodes. |
        | ``cc_hillier_{d}`` | Hillier normalisation (density² / farness). |
        | ``cc_cycles_{d}`` | Circuit rank: count of independent loops in the reachable subgraph. |
        | ``cc_decay_{d}`` | Decay-weighted closeness (e.g. ``exp(-4 * p)``). |
        | ``cc_betweenness_{d}`` | Betweenness: count of shortest paths passing through each node (default). |
        | ``cc_betweenness_decay_{d}`` | Decay-weighted betweenness (e.g. ``exp(-4 * p)``). |
        """
        # New-API defaults: a single closeness (harmonic) and a single betweenness, cycles off.
        # The functional `networks.centrality_shortest` keeps its fuller default; pass
        # closeness / betweenness / cycles here to override.
        kwargs.setdefault("closeness", {"harmonic": "1/c"})
        kwargs.setdefault("betweenness", {"betweenness": "1"})
        kwargs.setdefault("cycles", False)
        kwargs.setdefault("postprocess", {})
        kwargs.setdefault("segment_weighted", bool(self._state.get("segment_weighted_default", False)))
        self._nodes_gdf = networks.centrality_shortest(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            **kwargs,
        )
        return self

    def centrality_simplest(self, **kwargs: Any) -> CityNetwork:
        """Compute simplest-path (angular) centrality.

        Wraps [`centrality_simplest`](/metrics/networks#centrality_simplest). All keyword arguments
        are forwarded; see that function for the full parameter list.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining. Results are written to ``nodes_gdf``.

        Examples
        --------
        ```python
        cn.centrality_simplest(distances=[400, 800, 1600])
        ```

        By default this emits just ``cc_harmonic_{d}_ang`` (closeness) and ``cc_betweenness_{d}_ang``. Pass
        ``closeness`` / ``betweenness`` expression dicts to compute any of the metrics below (note the
        ``_ang`` suffix):

        | Column | Description |
        | --- | --- |
        | ``cc_density_{d}_ang`` | Count of reachable nodes (or total reachable street length if segment_weighted). |
        | ``cc_harmonic_{d}_ang`` | Harmonic closeness using angular cost as impedance (default). |
        | ``cc_farness_{d}_ang`` | Sum of angular costs to reachable nodes. |
        | ``cc_hillier_{d}_ang`` | Hillier normalisation (density² / farness). |
        | ``cc_betweenness_{d}_ang`` | Betweenness via simplest angular paths (angular choice; default). |
        """
        # New-API defaults: a single (angular) harmonic closeness and a single betweenness.
        kwargs.setdefault("closeness", {"harmonic": "1 / (1 + c / 90)"})
        kwargs.setdefault("betweenness", {"betweenness": "1"})
        kwargs.setdefault("postprocess", {})
        kwargs.setdefault("segment_weighted", bool(self._state.get("segment_weighted_default", False)))
        self._nodes_gdf = networks.centrality_simplest(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            **kwargs,
        )
        return self

    def build_od_matrix(
        self,
        od_df: pd.DataFrame,
        zones_gdf: gpd.GeoDataFrame,
        **kwargs: Any,
    ) -> rustalgos.centrality.OdMatrix:
        """Build an origin-destination (OD) matrix for OD-weighted betweenness.

        In standard betweenness, every pair of nodes contributes equally. OD-weighted betweenness
        instead weights each pair by observed trip counts between their respective zones (e.g.
        commuter cycling counts), so streets carrying more actual traffic receive higher scores.

        Wraps [`build_od_matrix`](/metrics/networks#build_od_matrix). See that function for
        the full parameter list.

        Parameters
        ----------
        od_df: pd.DataFrame
            Origin-destination flow data with columns for origin zone, destination zone, and trip weight.
        zones_gdf: GeoDataFrame
            Zone boundaries (polygons) or centroids (points) corresponding to the OD matrix.

        Returns
        -------
        od_matrix: OdMatrix
            An OD matrix for use with [`betweenness_od`](#betweenness_od).
        """
        return networks.build_od_matrix(
            od_df=od_df,
            zones_gdf=zones_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )

    def betweenness_od(
        self,
        od_matrix: rustalgos.centrality.OdMatrix,
        **kwargs: Any,
    ) -> CityNetwork:
        """Compute OD-weighted betweenness centrality.

        Weights betweenness by actual origin-destination trip counts rather than treating all paths
        equally. Only source nodes with outbound trips are traversed, and each shortest-path
        contribution is scaled by the corresponding OD weight.

        Wraps [`betweenness_od`](/metrics/networks#betweenness_od). See that function for
        the full parameter list.

        Parameters
        ----------
        od_matrix: OdMatrix
            An OD matrix from [`build_od_matrix`](#build_od_matrix).

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining. Results are written to ``nodes_gdf``.
        """
        self._nodes_gdf = networks.betweenness_od(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            od_matrix=od_matrix,
            **kwargs,
        )
        return self

    def betweenness_demand(
        self,
        origins_gdf: gpd.GeoDataFrame,
        destinations_gdf: gpd.GeoDataFrame,
        origin_weight_col: str,
        destination_weight_col: str,
        **kwargs: Any,
    ) -> CityNetwork:
        """Compute demand-weighted (flow) betweenness from a spatial interaction model.

        Trips are allocated between weighted origins (e.g. population) and weighted destinations
        (e.g. attractors) using a singly (origin-)constrained spatial interaction model, then routed
        along shortest network paths so intermediate streets accumulate the flow passing through them.
        This is the modelled-matrix counterpart to [`betweenness_od`](#betweenness_od): the per-pair
        weights are derived from the network distances revealed during routing, rather than supplied
        as an explicit OD matrix.

        Wraps [`betweenness_demand`](/metrics/networks#betweenness_demand). All additional keyword
        arguments are forwarded; see that function for the full parameter list including ``distances``,
        ``minutes``, ``decay_fn``, ``closest_destination``, ``participation``, ``metric_name``,
        ``max_netw_assign_dist``, and ``barriers_gdf``. Origins and destinations are assigned to the
        network with the same workflow as the data layers (representation-aware nearest-street
        assignment, offsets included in routed distances). The model has two distance levers, each
        with one job: ``decay_fn`` shapes destination choice within the allocation, and
        ``participation`` (the share of people at a typical location who make a trip; ``1.0`` by
        default, e.g. ``0.2`` for pedestrian flows) makes trip generation accessibility-elastic via
        a stay-home option in the choice set. The output is a single conserved
        ``cc_demand_{distance}`` column. For externally weighted or bespoke flow conventions use
        [`betweenness_od`](#betweenness_od), which retains full expression support.

        Parameters
        ----------
        origins_gdf: GeoDataFrame
            A GeoDataFrame of demand origins (points or centroids).
        destinations_gdf: GeoDataFrame
            A GeoDataFrame of demand destinations / attractors (points or centroids).
        origin_weight_col: str
            Column in ``origins_gdf`` giving each origin's weight (e.g. population).
        destination_weight_col: str
            Column in ``destinations_gdf`` giving each destination's attractiveness weight.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining. A ``cc_{metric_name}_{distance}`` column (default
            ``cc_demand_{distance}``) is written to ``nodes_gdf``.

        Examples
        --------
        ```python
        cn.betweenness_demand(
            origins_gdf=population_gdf,
            destinations_gdf=amenities_gdf,
            origin_weight_col="population",
            destination_weight_col="weight",
            distances=[800],
            decay_fn="exp(-0.002 * c)",
        )
        result_gdf = cn.to_geopandas()  # cc_demand_800 projected onto the street segments
        ```
        """
        self._nodes_gdf = networks.betweenness_demand(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            origins_gdf=origins_gdf,
            destinations_gdf=destinations_gdf,
            origin_weight_col=origin_weight_col,
            destination_weight_col=destination_weight_col,
            **kwargs,
        )
        return self

    def __iter__(self):
        # Migration guard (v5.8.0): compute_accessibilities / compute_mixed_uses / compute_stats
        # once returned a (CityNetwork, data_gdf) tuple, a carry-over from when assignment details were
        # written back onto the data frame. They now return the CityNetwork itself so calls can be
        # chained. Unpacking two values would iterate the CityNetwork, so raise a clear message
        # instead of the default "cannot unpack" error.
        raise TypeError(
            "CityNetwork is not iterable. As of v5.8.0, compute_accessibilities, compute_mixed_uses, "
            "and compute_stats return the CityNetwork itself (chainable) rather than a "
            "(CityNetwork, data_gdf) tuple; the second value was a legacy return that is no longer "
            "produced. Replace `cn, data = cn.compute_...(...)` with `cn = cn.compute_...(...)`."
        )

    def compute_accessibilities(self, data_gdf: gpd.GeoDataFrame, **kwargs: Any) -> CityNetwork:
        """Compute land-use accessibility metrics.

        Counts how many instances of each specified land-use category (e.g. retail, parks) are reachable
        within each distance threshold, and records the nearest distance to each category.

        Wraps [`compute_accessibilities`](/metrics/layers#compute_accessibilities). All additional keyword
        arguments are forwarded; see that function for the full parameter list including
        ``landuse_column_label``, ``accessibility_keys``, ``distances``, ``minutes``, ``decay_fn``,
        and ``angular``. ``decay_fn`` accepts a single expression or a ``{label: expression}`` dict that
        computes several decay variants in one network traversal, each label suffixing its output columns.

        Parameters
        ----------
        data_gdf: GeoDataFrame
            A GeoDataFrame of land-use points with categorical columns.

        Returns
        -------
        self: CityNetwork
            Returns self with accessibility columns added to ``nodes_gdf``.

        Examples
        --------
        ```python
        from cityseer import decay

        cn = cn.compute_accessibilities(
            data_gdf=landuses_gdf,
            landuse_column_label="category",
            accessibility_keys=["retail", "cafe", "park"],
            distances=[400, 800],
            decay_fn=decay.exponential(),
        )
        # Count of reachable "retail" within 800m
        print(cn.nodes_gdf["cc_retail_800"])
        # Nearest distance to "park" at the maximum threshold
        print(cn.nodes_gdf["cc_park_nearest_max_800"])
        ```
        """
        # New-API default: a single (unweighted) column. The functional layers.* calls keep the
        # legacy two-column (_nw + _wt) default; pass decay_fn here for weighting or both columns.
        kwargs.setdefault("decay_fn", "1")
        self._nodes_gdf = layers.compute_accessibilities(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self

    def compute_mixed_uses(self, data_gdf: gpd.GeoDataFrame, **kwargs: Any) -> CityNetwork:
        """Compute mixed-use diversity metrics.

        Measures the diversity of land-use categories reachable from each node using Hill numbers:
        q=0 counts how many distinct types are present, q=1 measures diversity accounting for how
        evenly types are represented, and q=2 gives greater weight to the most common types.

        Wraps [`compute_mixed_uses`](/metrics/layers#compute_mixed_uses). All additional keyword
        arguments are forwarded; see that function for the full parameter list including
        ``landuse_column_label``, ``distances``, ``minutes``, ``compute_hill``, ``compute_shannon``,
        ``compute_gini``, ``decay_fn``, and ``angular``. ``decay_fn`` accepts a single expression or a
        ``{label: expression}`` dict that computes several decay variants in one network traversal, each
        label suffixing its output columns (only the Hill measures vary with decay).

        Parameters
        ----------
        data_gdf: GeoDataFrame
            A GeoDataFrame of land-use points with categorical columns.

        Returns
        -------
        self: CityNetwork
            Returns self with mixed-use columns added to ``nodes_gdf``.

        Examples
        --------
        ```python
        cn = cn.compute_mixed_uses(
            data_gdf=landuses_gdf,
            landuse_column_label="category",
            distances=[400, 800],
        )
        # Hill q=0 (count of distinct land-use types) at 800m
        print(cn.nodes_gdf["cc_hill_q0_800"])
        ```
        """
        kwargs.setdefault("decay_fn", "1")  # new-API single-column default (functional keeps _nw + _wt)
        self._nodes_gdf = layers.compute_mixed_uses(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self

    def compute_stats(self, data_gdf: gpd.GeoDataFrame, **kwargs: Any) -> CityNetwork:
        """Compute statistical aggregations of numerical data over the network.

        Aggregates numerical attributes (e.g. property prices, floor areas) within network-distance
        thresholds, computing weighted statistics (mean, sum, min, max, variance, etc.) at each node.

        Wraps [`compute_stats`](/metrics/layers#compute_stats). All additional keyword arguments are forwarded;
        see that function for the full parameter list including ``stats_column_labels``, ``distances``,
        ``minutes``, ``decay_fn``, and ``angular``. ``decay_fn`` accepts a single expression or a
        ``{label: expression}`` dict that computes several decay variants in one network traversal, each
        label suffixing its output columns.

        Parameters
        ----------
        data_gdf: GeoDataFrame
            A GeoDataFrame of data points with numerical columns.

        Returns
        -------
        self: CityNetwork
            Returns self with statistical columns added to ``nodes_gdf``.

        Examples
        --------
        ```python
        from cityseer import decay

        cn = cn.compute_stats(
            data_gdf=prices_gdf,
            stats_column_labels=["price", "floor_area"],
            distances=[800, 1600],
            decay_fn=decay.exponential(),
        )
        # Weighted mean of "price" at 800m
        print(cn.nodes_gdf["cc_price_mean_800"])
        # Weighted sum of "floor_area" at 1600m
        print(cn.nodes_gdf["cc_floor_area_sum_1600"])
        ```
        """
        kwargs.setdefault("decay_fn", "1")  # new-API single-column default (functional keeps _nw + _wt)
        self._nodes_gdf = layers.compute_stats(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self

    def add_gtfs(
        self,
        gtfs_path: str,
        *,
        crs: Any | None = None,
        max_netw_assign_dist: int = 400,
    ) -> CityNetwork:
        """Add GTFS (General Transit Feed Specification) public transport data to the network.

        Integrates transit stops and routes so that centrality and accessibility analyses account
        for public transport connections.

        Wraps [`io.add_transport_gtfs`](/tools/io#add_transport_gtfs).

        Parameters
        ----------
        gtfs_path: str
            Path to a GTFS zip file or directory.
        crs: Any
            Optional CRS override for the GTFS data.
        max_netw_assign_dist: int
            Maximum distance (metres) for assigning transit stops to the network (nearest street,
            via the shared assignment workflow). Stops beyond this distance are excluded.
            Defaults to 400.

        Returns
        -------
        self: CityNetwork
            Returns self for method chaining.
        """
        network_crs = _require_projected_crs(crs if crs is not None else self._crs)
        self._network_structure, _stops, _pairs = io.add_transport_gtfs(
            gtfs_path,
            self._network_structure,
            network_crs.to_epsg() or int(network_crs.to_authority()[1]),
            max_netw_assign_dist=max_netw_assign_dist,
        )
        return self

    def to_nx(self) -> nx.MultiGraph | nx.MultiDiGraph:
        """Convert the network to a NetworkX MultiGraph (or MultiDiGraph if directed).

        If the network was built with [`from_nx`](#from_nx), returns a copy of the original graph with computed
        centrality and layer columns added to each edge's data dictionary. Otherwise builds a new
        cityseer-compatible undirected graph from the internal GeoDataFrame.

        Returns
        -------
        graph: nx.MultiGraph | nx.MultiDiGraph
            A primal edge graph with computed metrics added to edge data.

        Raises
        ------
        NotImplementedError
            If the network is directed but was not built via [`from_nx`](#from_nx) (no source graph to export).

        Examples
        --------
        ```python
        cn = CityNetwork.from_nx(G)
        cn.centrality_shortest(distances=[800])

        # Round-trip: get back a NetworkX graph with metrics on edges
        G_with_metrics = cn.to_nx()
        u, v, k, data = list(G_with_metrics.edges(keys=True, data=True))[0]
        print(data["cc_harmonic_800"])
        ```
        """
        source_graph: nx.MultiGraph | None = self._state.get("nx_source_graph")
        if source_graph is not None:
            g = source_graph.copy()
            result = self.to_geopandas()
            cc_cols = [c for c in result.columns if c.startswith("cc_")]
            if cc_cols:
                for _, row in result.iterrows():
                    u = row["primal_edge_node_a"]
                    v = row["primal_edge_node_b"]
                    k = int(row["primal_edge_idx"])
                    if g.has_edge(u, v, k):
                        for col in cc_cols:
                            g[u][v][k][col] = row[col]  # type: ignore
            return g
        if self.is_directed:
            raise NotImplementedError(
                "to_nx() is not supported for directed networks built via from_geopandas or from_wkts. "
                "Build from a MultiDiGraph via from_nx() to enable round-trip conversion."
            )
        return io.nx_from_generic_geopandas(self.to_geopandas())

    def __repr__(self) -> str:
        crs_repr = None if self._crs is None else self._crs.to_string()
        parts = f"node_count={self.node_count}, is_dual={self.is_dual}, is_directed={self.is_directed}"
        return f"CityNetwork({parts}, crs={crs_repr})"
