"""Typing information for metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import numpy as np
import numpy.typing as npt


class DataPoint(TypedDict):
    """DataPoint type for type-hinting."""

    x: float
    y: float


DataDictType = dict[str, list[DataPoint]]


@dataclass
class CentralityMetricsState:
    """Centrality metrics typing scaffold."""

    # node shortest
    node_density: dict[int, npt.NDArray[np.float_]] | None = None
    node_farness: dict[int, npt.NDArray[np.float_]] | None = None
    node_cycles: dict[int, npt.NDArray[np.float_]] | None = None
    node_harmonic: dict[int, npt.NDArray[np.float_]] | None = None
    node_beta: dict[int, npt.NDArray[np.float_]] | None = None
    node_betweenness: dict[int, npt.NDArray[np.float_]] | None = None
    node_betweenness_beta: dict[int, npt.NDArray[np.float_]] | None = None
    # node simplest
    node_harmonic_angular: dict[int, npt.NDArray[np.float_]] | None = None
    node_betweenness_angular: dict[int, npt.NDArray[np.float_]] | None = None
    # segment shortest
    segment_density: dict[int, npt.NDArray[np.float_]] | None = None
    segment_harmonic: dict[int, npt.NDArray[np.float_]] | None = None
    segment_beta: dict[int, npt.NDArray[np.float_]] | None = None
    segment_betweenness: dict[int, npt.NDArray[np.float_]] | None = None
    # segment simplest
    segment_harmonic_hybrid: dict[int, npt.NDArray[np.float_]] | None = None
    segment_betweeness_hybrid: dict[int, npt.NDArray[np.float_]] | None = None


@dataclass
class MixedUsesMetricsState:
    """Mixed-uses metrics typing scaffold."""

    # hill measures have q keys
    hill: dict[float | int, dict[int, npt.NDArray[np.float_]]] | None = None
    hill_branch_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]] | None = None
    hill_pairwise_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]] | None = None
    hill_pairwise_disparity: dict[float | int, dict[int, npt.NDArray[np.float_]]] | None = None
    # non-hill do not have q keys
    shannon: dict[int, npt.NDArray[np.float_]] | None = None
    gini_simpson: dict[int, npt.NDArray[np.float_]] | None = None
    raos_pairwise_disparity: dict[int, npt.NDArray[np.float_]] | None = None


@dataclass
class AccessibilityMetricsState:
    """Accessibility metrics typing scaffold."""

    weighted: dict[str, dict[int, npt.NDArray[np.float_]]]
    non_weighted: dict[str, dict[int, npt.NDArray[np.float_]]]


@dataclass
class StatsMetricsState(TypedDict):
    """Stats metrics typing scaffold."""

    max: dict[int, npt.NDArray[np.float_]]
    min: dict[int, npt.NDArray[np.float_]]
    sum: dict[int, npt.NDArray[np.float_]]
    sum_weighted: dict[int, npt.NDArray[np.float_]]
    mean: dict[int, npt.NDArray[np.float_]]
    mean_weighted: dict[int, npt.NDArray[np.float_]]
    variance: dict[int, npt.NDArray[np.float_]]
    variance_weighted: dict[int, npt.NDArray[np.float_]]


class MetricsState:
    """Metrics typing scaffold."""

    centrality: CentralityMetricsState | None = None
    mixed_uses: MixedUsesMetricsState | None = None
    accessibility: AccessibilityMetricsState | None = None
    stats: dict[str, StatsMetricsState] | None = None

    def extract_node_metrics(self, node_idx: int) -> dict:  # type: ignore
        """Extract metrics for a given node idx."""
        node_state = {}
        # centrality
        node_state["centrality"] = {}
        for k in [
            "node_density",
            "node_farness",
            "node_cycles",
            "node_harmonic",
            "node_beta",
            "node_betweenness",
            "node_betweenness_beta",
            "node_harmonic_angular",
            "node_betweenness_angular",
            "segment_density",
            "segment_harmonic",
            "segment_beta",
            "segment_betweenness",
            "segment_harmonic_hybrid",
            "segment_betweeness_hybrid",
        ]:
            if hasattr(self.centrality, k):
                node_state["centrality"][k] = {}
                for d_key, d_val in getattr(self.centrality, k):
                    node_state["centrality"][k][d_key] = d_val[node_idx]
        # mixed-uses - hill
        node_state["mixed_uses"] = {}
        for k in [
            "hill",
            "hill_branch_wt",
            "hill_pairwise_wt",
            "hill_pairwise_disparity",
        ]:
            if hasattr(self.mixed_uses, k):
                node_state["mixed_uses"][k] = {}
                for q_key, q_val in getattr(self.mixed_uses, k).items():
                    node_state["mixed_uses"][k][q_key] = {}
                    for d_key, d_val in q_val.items():
                        node_state["mixed_uses"][k][q_key][d_key] = d_val[node_idx]
        # mixed-uses non-hill
        for k in [
            "shannon",
            "gini_simpson",
            "raos_pairwise_disparity",
        ]:
            if hasattr(self.mixed_uses, k):
                node_state["mixed_uses"][k] = {}
                for d_key, d_val in getattr(self.mixed_uses, k).items():
                    node_state["mixed_uses"][k][d_key] = d_val[node_idx]
        # accessibility
        node_state["accessibility"] = {"non_weighted": {}, "weighted": {}}
        # non-weighted
        for cl_key, cl_val in self.accessibility.non_weighted.items():
            node_state["accessibility"]["non_weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():
                node_state["accessibility"]["non_weighted"][cl_key][d_key] = d_val[node_idx]
        # weighted
        for cl_key, cl_val in self.accessibility.weighted.items():
            node_state["accessibility"]["weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():
                node_state["accessibility"]["weighted"][cl_key][d_key] = d_val[node_idx]
        # stats
        node_state["stats"] = {}
        for th_key in self.stats.keys():
            node_state["stats"][th_key] = {}
            for stat_attr in [
                "max",
                "min",
                "sum",
                "sum_weighted",
                "mean",
                "mean_weighted",
                "variance",
                "variance_weighted",
            ]:
                node_state["stats"][th_key][stat_attr] = {}
                stat_val = getattr(self.stats[th_key], stat_attr)
                for d_key, d_val in stat_val.items():
                    node_state["stats"][th_key][stat_attr][d_key] = d_val[node_idx]

        return node_state
