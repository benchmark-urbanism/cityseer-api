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
_NON_PRESERVED_UPDATE_COLUMNS = {"ns_node_idx", "x", "y", "live"}


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
    extra_columns = [
        col for col in saved_nodes_gdf.columns
        if col != geometry_name and col != "ns_node_idx"
    ]
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
    node_idx: dict[Any, int] = {}
    rebuilt_nodes_gdf = nodes_gdf.copy()
    for node_key, row in rebuilt_nodes_gdf.iterrows():
        z = None
        if "z" in rebuilt_nodes_gdf.columns and pd.notna(row.get("z")):
            z = float(row["z"])
        node_idx[node_key] = network_structure.add_street_node(
            node_key=node_key,
            x=float(row["x"]),
            y=float(row["y"]),
            live=bool(row["live"]),
            weight=float(row["weight"]),
            z=z,
        )
    for record in sorted(
        state["edge_records"].values(), key=lambda item: item["edge_idx"]
    ):
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
    rebuilt_nodes_gdf["ns_node_idx"] = [
        node_idx[node_key] for node_key in rebuilt_nodes_gdf.index
    ]
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
            nodes_gdf.loc[shared_idx, extra_columns] = (
                input_gdf.loc[shared_idx, extra_columns]
            )
    return nodes_gdf


class CityNetwork:
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
        return self._network_structure

    @property
    def nodes_gdf(self) -> gpd.GeoDataFrame:
        return self._nodes_gdf

    def to_geopandas(self) -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame(
            {
                col: self._nodes_gdf[col]
                for col in self._nodes_gdf.columns
                if col != self._nodes_gdf.geometry.name
            },
            index=self._nodes_gdf.index,
            geometry=[
                self._state["geoms"][fid]
                for fid in self._nodes_gdf.index
            ],
            crs=self._crs,
        )

    @property
    def is_dual(self) -> bool:
        return bool(self._network_structure.is_dual)

    @property
    def crs(self) -> CRS | None:
        return self._crs

    @property
    def node_count(self) -> int:
        return len(self._nodes_gdf)

    @property
    def feature_status(self) -> pd.Series:
        return pd.Series(
            self._state.get("feature_status", {}), name="feature_status"
        )

    @classmethod
    def _from_dual_result(
        cls,
        network_structure: rustalgos.graph.NetworkStructure,
        nodes_gdf: gpd.GeoDataFrame | None,
        state: dict[str, Any],
        crs: CRS,
    ) -> CityNetwork:
        if nodes_gdf is None:
            raise RuntimeError(
                "Fast dual build did not produce nodes GeoDataFrame."
            )
        return cls(network_structure, nodes_gdf, _state=state, _crs=crs)

    @classmethod
    def from_wkts(
        cls,
        wkts: dict[Any, str] | dict[Any, BaseGeometry],
        *,
        crs: Any,
        boundary: BaseGeometry | None = None,
    ) -> CityNetwork:
        normalized_crs = _require_projected_crs(crs)
        ns, nodes_gdf, state = dual.build_dual(
            wkts, crs=normalized_crs, boundary=boundary
        )
        return cls._from_dual_result(ns, nodes_gdf, state, normalized_crs)

    @classmethod
    def from_geopandas(
        cls,
        gdf: gpd.GeoDataFrame,
        *,
        crs: Any | None = None,
        boundary: BaseGeometry | None = None,
    ) -> CityNetwork:
        normalized_crs = _require_projected_crs(
            crs if crs is not None else gdf.crs
        )
        ns, nodes_gdf, state = dual.build_dual(
            gdf, crs=normalized_crs, boundary=boundary
        )
        if nodes_gdf is not None:
            nodes_gdf = _merge_input_columns(nodes_gdf, gdf)
        return cls._from_dual_result(ns, nodes_gdf, state, normalized_crs)

    @classmethod
    def from_nx(
        cls,
        graph: nx.MultiGraph,
        *,
        boundary: BaseGeometry | None = None,
    ) -> CityNetwork:
        primal_graph = util.validate_cityseer_networkx_graph(graph)
        if primal_graph.graph.get("is_dual", False):
            raise ValueError(
                "CityNetwork.from_nx expects a primal edge graph, "
                "not a dual graph."
            )

        wkts: dict[str, str] = {}
        live_overrides: dict[str, bool] = {}
        edge_attrs: dict[str, dict[str, Any]] = {}
        for start_nd_key, end_nd_key, edge_idx, edge_data in primal_graph.edges(
            keys=True, data=True
        ):
            dual_node_key = _prepare_dual_node_key(
                start_nd_key, end_nd_key, int(edge_idx)
            )
            wkts[dual_node_key] = edge_data["geom"].wkt
            if (
                "live" in primal_graph.nodes[start_nd_key]
                and "live" in primal_graph.nodes[end_nd_key]
            ):
                live_overrides[dual_node_key] = bool(
                    primal_graph.nodes[start_nd_key]["live"]
                    or primal_graph.nodes[end_nd_key]["live"]
                )
            edge_attrs[dual_node_key] = {
                key: value
                for key, value in edge_data.items()
                if key != "geom"
            }
            edge_attrs[dual_node_key]["primal_edge_node_a"] = start_nd_key
            edge_attrs[dual_node_key]["primal_edge_node_b"] = end_nd_key
            edge_attrs[dual_node_key]["primal_edge_idx"] = int(edge_idx)

        normalized_crs = _require_projected_crs(primal_graph.graph["crs"])
        ns, nodes_gdf, state = dual.build_dual(
            wkts, crs=normalized_crs, boundary=boundary
        )
        if nodes_gdf is None:
            raise RuntimeError(
                "Fast dual build did not produce nodes GeoDataFrame."
            )
        for node_key, live in live_overrides.items():
            if node_key in state["node_idx"]:
                ns.set_node_live(state["node_idx"][node_key], live)
                nodes_gdf.at[node_key, "live"] = live
        if edge_attrs:
            attr_df = pd.DataFrame.from_dict(edge_attrs, orient="index")
            nodes_gdf.loc[attr_df.index, attr_df.columns] = (
                attr_df.loc[attr_df.index, attr_df.columns]
            )
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
        **kwargs: Any,
    ) -> CityNetwork:
        graph = io.osm_graph_from_poly(
            poly_geom,
            poly_crs_code=poly_crs_code,
            to_crs_code=to_crs_code,
            simplify=simplify,
            **kwargs,
        )
        return cls.from_nx(graph, boundary=boundary)

    def _clear_metric_columns(self) -> None:
        metric_columns = [
            col for col in self._nodes_gdf.columns
            if col.startswith("cc_")
        ]
        if metric_columns:
            self._nodes_gdf.drop(columns=metric_columns, inplace=True)

    def _sync_feature_status_column(self) -> None:
        if "feature_status" in self._state:
            self._nodes_gdf["feature_status"] = [
                self._state["feature_status"].get(fid, "active")
                for fid in self._nodes_gdf.index
            ]

    def _restore_non_topology_columns(
        self, previous_nodes_gdf: gpd.GeoDataFrame
    ) -> None:
        shared_idx = self._nodes_gdf.index.intersection(
            previous_nodes_gdf.index
        )
        geometry_name = previous_nodes_gdf.geometry.name
        restore_columns = [
            col
            for col in previous_nodes_gdf.columns
            if col not in _NON_PRESERVED_UPDATE_COLUMNS
            and col != geometry_name
            and not col.startswith("cc_")
        ]
        if restore_columns:
            self._nodes_gdf.loc[shared_idx, restore_columns] = (
                previous_nodes_gdf.loc[shared_idx, restore_columns]
            )

    def update(
        self,
        data: dict[Any, str] | dict[Any, BaseGeometry] | gpd.GeoDataFrame,
    ) -> CityNetwork:
        new_wkts, _discovered_crs = dual.extract_wkts(data)
        if new_wkts == self._state.get("source_wkts", self._state["wkts"]):
            return self
        previous_nodes_gdf = self._nodes_gdf.copy()
        boundary = _load_boundary(self._state.get("boundary_wkt"))
        ns, nodes_gdf, state = dual.incremental_update(
            self._state,
            data,
            crs=self._crs,
            boundary=boundary,
        )
        if nodes_gdf is None:
            raise RuntimeError(
                "Fast dual update did not produce nodes GeoDataFrame."
            )
        self._network_structure = ns
        self._nodes_gdf = nodes_gdf
        self._state = state
        self._state["ns"] = ns
        self._clear_metric_columns()
        self._restore_non_topology_columns(previous_nodes_gdf)
        self._sync_feature_status_column()
        return self

    def set_boundary(self, polygon: BaseGeometry) -> CityNetwork:
        from shapely.geometry import Point
        from shapely.prepared import prep

        prepared = prep(polygon)
        for idx, row in self._nodes_gdf.iterrows():
            live = prepared.contains(
                Point(float(row["x"]), float(row["y"]))
            )
            self._network_structure.set_node_live(
                int(row["ns_node_idx"]), live
            )
            self._nodes_gdf.at[idx, "live"] = live
        self._state["boundary_wkt"] = polygon.wkt
        return self

    def set_all_live(self) -> CityNetwork:
        for idx, row in self._nodes_gdf.iterrows():
            self._network_structure.set_node_live(
                int(row["ns_node_idx"]), True
            )
            self._nodes_gdf.at[idx, "live"] = True
        self._state["boundary_wkt"] = None
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        self._nodes_gdf.to_parquet(path.with_suffix(".nodes.parquet"))
        payload = {
            "source_wkts": dict(
                self._state.get("source_wkts", self._state["wkts"])
            ),
            "boundary_wkt": self._state.get("boundary_wkt"),
            "feature_status": dict(
                self._state.get("feature_status", {})
            ),
        }
        with path.with_suffix(".state.pkl").open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path) -> CityNetwork:
        path = Path(path)
        saved_nodes_gdf = gpd.read_parquet(
            path.with_suffix(".nodes.parquet")
        )
        with path.with_suffix(".state.pkl").open("rb") as file:
            payload = pickle.load(file)
        boundary = _load_boundary(payload.get("boundary_wkt"))
        source_wkts = payload.get("source_wkts", payload.get("wkts"))
        # Build state (geoms, edge_records, endpoint_to_fids, etc.)
        # but skip building nodes_gdf — we'll use the saved one.
        _ns, _nodes_gdf, state = dual.build_dual(
            source_wkts,
            crs=saved_nodes_gdf.crs,
            boundary=boundary,
            build_nodes_gdf=False,
        )
        state["feature_status"] = payload.get(
            "feature_status", state.get("feature_status", {})
        )
        # Merge saved columns (metrics, user attrs) onto fresh topology,
        # then rebuild NS once from the merged result.
        merged_gdf = dual._build_nodes_gdf(
            _ns,
            state["fid_list"],
            state["node_idx"],
            state["midpoints"],
            saved_nodes_gdf.crs,
        )
        merged_gdf = _merge_saved_columns(merged_gdf, saved_nodes_gdf)
        ns, nodes_gdf, state = _rebuild_dual_network(state, merged_gdf)
        return cls(ns, nodes_gdf, _state=state, _crs=nodes_gdf.crs)

    def centrality_shortest(self, **kwargs: Any) -> CityNetwork:
        self._nodes_gdf = networks.node_centrality_shortest(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            **kwargs,
        )
        return self

    def centrality_simplest(self, **kwargs: Any) -> CityNetwork:
        self._nodes_gdf = networks.node_centrality_simplest(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            **kwargs,
        )
        return self

    def segment_centrality(self, **kwargs: Any) -> CityNetwork:
        self._nodes_gdf = networks.segment_centrality(
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
        self._nodes_gdf = networks.betweenness_od(
            network_structure=self._network_structure,
            nodes_gdf=self._nodes_gdf,
            od_matrix=od_matrix,
            **kwargs,
        )
        return self

    def compute_accessibilities(
        self, data_gdf: gpd.GeoDataFrame, **kwargs: Any
    ) -> tuple[CityNetwork, gpd.GeoDataFrame]:
        self._nodes_gdf, data_gdf = layers.compute_accessibilities(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self, data_gdf

    def compute_mixed_uses(
        self, data_gdf: gpd.GeoDataFrame, **kwargs: Any
    ) -> tuple[CityNetwork, gpd.GeoDataFrame]:
        self._nodes_gdf, data_gdf = layers.compute_mixed_uses(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self, data_gdf

    def compute_stats(
        self, data_gdf: gpd.GeoDataFrame, **kwargs: Any
    ) -> tuple[CityNetwork, gpd.GeoDataFrame]:
        self._nodes_gdf, data_gdf = layers.compute_stats(
            data_gdf=data_gdf,
            nodes_gdf=self._nodes_gdf,
            network_structure=self._network_structure,
            **kwargs,
        )
        return self, data_gdf

    def add_gtfs(
        self,
        gtfs_path: str,
        *,
        crs: Any | None = None,
        max_netw_assign_dist: int = 400,
    ) -> CityNetwork:
        network_crs = _require_projected_crs(
            crs if crs is not None else self._crs
        )
        self._network_structure, _stops, _pairs = io.add_transport_gtfs(
            gtfs_path,
            self._network_structure,
            network_crs.to_epsg() or int(network_crs.to_authority()[1]),
            max_netw_assign_dist=max_netw_assign_dist,
        )
        return self

    def to_nx(self) -> nx.MultiGraph:
        return io.nx_from_generic_geopandas(self.to_geopandas())

    def __repr__(self) -> str:
        crs_repr = None if self._crs is None else self._crs.to_string()
        return (
            f"CityNetwork(node_count={self.node_count}, "
            f"is_dual={self.is_dual}, crs={crs_repr})"
        )
