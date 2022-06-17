from dataclasses import dataclass, field
from typing import Any, TypedDict, Union

import numpy as np
import numpy.typing as npt


class ColourMap:  # pylint: disable=too-few-public-methods
    """Specifies global colour presets."""

    primary: str = "#0091ea"
    accent: str = "#64c1ff"
    info: str = "#0064b7"
    secondary: str = "#d32f2f"
    warning: str = "#9a0007"
    error: str = "#ffffff"
    background: str = "#19181B"


qsType = Union[  # pylint: disable=invalid-name
    int,
    float,
    Union[list[int], list[float]],
    Union[tuple[int], tuple[float]],
    Union[npt.NDArray[np.int_], npt.NDArray[np.float32]],
    None,
]


class DataPoint(TypedDict):
    """DataPoint type for type-hinting."""

    x: float
    y: float


DataDictType = dict[Union[str, int], DataPoint]


class CentralityMetricsState(TypedDict, total=False):
    """Centrality metrics typing scaffold."""

    # node shortest
    node_density: dict[int, npt.NDArray[np.float_]]
    node_farness: dict[int, npt.NDArray[np.float_]]
    node_cycles: dict[int, npt.NDArray[np.float_]]
    node_harmonic: dict[int, npt.NDArray[np.float_]]
    node_beta: dict[int, npt.NDArray[np.float_]]
    node_betweenness: dict[int, npt.NDArray[np.float_]]
    node_betweenness_beta: dict[int, npt.NDArray[np.float_]]
    # node simplest
    node_harmonic_angular: dict[int, npt.NDArray[np.float_]]
    node_betweenness_angular: dict[int, npt.NDArray[np.float_]]
    # segment shortest
    segment_density: dict[int, npt.NDArray[np.float_]]
    segment_harmonic: dict[int, npt.NDArray[np.float_]]
    segment_beta: dict[int, npt.NDArray[np.float_]]
    segment_betweenness: dict[int, npt.NDArray[np.float_]]
    # segment simplest
    segment_harmonic_hybrid: dict[int, npt.NDArray[np.float_]]
    segment_betweeness_hybrid: dict[int, npt.NDArray[np.float_]]


class MixedUsesMetricsState(TypedDict, total=False):
    """Mixed-uses metrics typing scaffold."""

    # hill measures have q keys
    hill: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_branch_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_pairwise_wt: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    hill_pairwise_disparity: dict[float | int, dict[int, npt.NDArray[np.float_]]]
    # non-hill do not have q keys
    shannon: dict[int, npt.NDArray[np.float_]]
    gini_simpson: dict[int, npt.NDArray[np.float_]]
    raos_pairwise_disparity: dict[int, npt.NDArray[np.float_]]


@dataclass()
class AccessibilityMetricsState:
    """Accessibility metrics typing scaffold."""

    weighted: dict[str, dict[int, npt.NDArray[np.float_]]] = field(default_factory=dict)
    non_weighted: dict[str, dict[int, npt.NDArray[np.float_]]] = field(default_factory=dict)


class StatsMetricsState(TypedDict, total=False):
    """Stats metrics typing scaffold."""

    max: dict[int, npt.NDArray[np.float_]]
    min: dict[int, npt.NDArray[np.float_]]
    sum: dict[int, npt.NDArray[np.float_]]
    sum_weighted: dict[int, npt.NDArray[np.float_]]
    mean: dict[int, npt.NDArray[np.float_]]
    mean_weighted: dict[int, npt.NDArray[np.float_]]
    variance: dict[int, npt.NDArray[np.float_]]
    variance_weighted: dict[int, npt.NDArray[np.float_]]


NodeMetrics = dict[str, dict[str, Any]]
DictNodeMetrics = dict[Union[str, int], NodeMetrics]


class MetricsState:
    """Metrics typing scaffold."""

    centrality: CentralityMetricsState
    mixed_uses: MixedUsesMetricsState
    accessibility: AccessibilityMetricsState
    stats: dict[str, StatsMetricsState]

    def __init__(self) -> None:
        """Instance a MetricsState class."""
        self.centrality = CentralityMetricsState()
        self.mixed_uses = MixedUsesMetricsState()
        self.accessibility = AccessibilityMetricsState()
        self.stats = {}

    def extract_node_metrics(self, node_idx: int) -> NodeMetrics:
        """Extract metrics for a given node idx."""
        node_state: NodeMetrics = {}
        # centrality
        node_state["centrality"] = {}
        # pylint: disable=duplicate-code
        for key in [
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
            if key in self.centrality:
                node_state["centrality"][key] = {}
                for d_key, d_val in self.centrality[key].items():
                    node_state["centrality"][key][d_key] = d_val[node_idx]
        # mixed-uses - hill
        node_state["mixed_uses"] = {}
        for key in [
            "hill",
            "hill_branch_wt",
            "hill_pairwise_wt",
            "hill_pairwise_disparity",
        ]:
            if key in self.mixed_uses:
                node_state["mixed_uses"][key] = {}
                for q_key, q_val in self.mixed_uses[key].items():  # type: ignore
                    node_state["mixed_uses"][key][q_key] = {}
                    for d_key, d_val in q_val.items():
                        node_state["mixed_uses"][key][q_key][d_key] = d_val[node_idx]
        # mixed-uses non-hill
        for key in [
            "shannon",
            "gini_simpson",
            "raos_pairwise_disparity",
        ]:
            if key in self.mixed_uses:
                node_state["mixed_uses"][key] = {}
                for d_key, d_val in self.mixed_uses[key].items():
                    node_state["mixed_uses"][key][d_key] = d_val[node_idx]
        # accessibility
        node_state["accessibility"] = {"non_weighted": {}, "weighted": {}}
        # non-weighted
        for cl_key, cl_val in self.accessibility.non_weighted.items():
            node_state["accessibility"]["non_weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():  # type: ignore
                node_state["accessibility"]["non_weighted"][cl_key][d_key] = d_val[node_idx]
        # weighted
        for cl_key, cl_val in self.accessibility.weighted.items():
            node_state["accessibility"]["weighted"][cl_key] = {}
            for d_key, d_val in cl_val.items():  # type: ignore
                node_state["accessibility"]["weighted"][cl_key][d_key] = d_val[node_idx]
        # stats
        node_state["stats"] = {}
        for th_key in self.stats:  # pylint: disable=consider-using-dict-items
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
                d_key: str
                d_val: npt.NDArray[np.float32]
                for d_key, d_val in self.stats[th_key][stat_attr].items():
                    node_state["stats"][th_key][stat_attr][d_key] = d_val[node_idx]

        return node_state
