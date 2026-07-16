# pyright: basic
from __future__ import annotations

import numpy as np
import pytest
from cityseer import config, rustalgos
from cityseer.metrics import networks
from cityseer.tools import io


def test_centrality_shortest(primal_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    # test different combinations of closeness and betweenness
    for _closeness, _betweenness in [(False, True), (True, False), (True, True)]:
        closeness_dict = None if _closeness else {}
        betweenness_dict = None if _betweenness else {}
        cycles_flag = _closeness
        nodes_gdf = networks.centrality_shortest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            distances=distances,
            closeness=closeness_dict,
            betweenness=betweenness_dict,
            cycles=cycles_flag,
        )
        for dist_key in distances:
            if _closeness is True:
                # test closeness against underlying Rust method
                closeness_items = list(networks.DEFAULT_SHORTEST_CLOSENESS.items())
                node_result_short = network_structure.centrality_shortest(
                    closeness_exprs=closeness_items,
                    betweenness_exprs=[],
                    compute_cycles=True,
                    distances=distances,
                )
                metrics = node_result_short.metrics
                for measure_key in ["decay", "cycles", "density", "farness", "harmonic"]:
                    data_key = config.prep_gdf_key(measure_key, dist_key)
                    assert np.allclose(
                        nodes_gdf[data_key],
                        metrics[measure_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                with np.errstate(divide="ignore", invalid="ignore"):
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key("hillier", dist_key)],
                        metrics["density"][dist_key] ** 2 / metrics["farness"][dist_key],
                        equal_nan=True,
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
            if _betweenness is True:
                # test betweenness against underlying Rust method
                betweenness_items = list(networks.DEFAULT_SHORTEST_BETWEENNESS.items())
                betweenness_result = network_structure.centrality_shortest(
                    closeness_exprs=[],
                    betweenness_exprs=betweenness_items,
                    distances=[dist_key],
                )
                metrics = betweenness_result.metrics
                for measure_key in ["betweenness", "betweenness_decay"]:
                    data_key = config.prep_gdf_key(measure_key, dist_key)
                    assert np.allclose(
                        nodes_gdf[data_key],
                        metrics[measure_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )


def test_centrality_simplest(dual_graph):
    """
    Underlying methods also tested via test_networks.test_network_centralities
    """
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    # Test with different farness scaling expressions
    for _far_scale_off, _ang_scale_unit in [(0, 180), (0, 90), (1, 180)]:
        closeness_exprs = {
            "density": "1",
            "farness": f"{_far_scale_off} + c / {_ang_scale_unit}",
            "harmonic": f"1 / (1 + c / {_ang_scale_unit})",
        }
        betweenness_exprs = {"betweenness": "1"}
        nodes_gdf = networks.centrality_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf,
            distances=distances,
            closeness=closeness_exprs,
            betweenness=betweenness_exprs,
        )
        for dist_key in distances:
            # test closeness against underlying Rust method
            node_result_simplest = network_structure.centrality_simplest(
                closeness_exprs=list(closeness_exprs.items()),
                betweenness_exprs=[],
                distances=distances,
            )
            metrics = node_result_simplest.metrics
            for measure_key in ["density", "farness", "harmonic"]:
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key(measure_key, dist_key, angular=True)],
                    metrics[measure_key][dist_key],
                    equal_nan=True,
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                assert np.allclose(
                    nodes_gdf[config.prep_gdf_key("hillier", dist_key, angular=True)],
                    metrics["density"][dist_key] ** 2 / metrics["farness"][dist_key],
                    equal_nan=True,
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
            # test betweenness against underlying Rust method
            betw_result = network_structure.centrality_simplest(
                closeness_exprs=[],
                betweenness_exprs=list(betweenness_exprs.items()),
                distances=distances,
            )
            assert np.allclose(
                nodes_gdf[config.prep_gdf_key("betweenness", dist_key, angular=True)],
                betw_result.metrics["betweenness"][dist_key],
                equal_nan=True,
                atol=config.ATOL,
                rtol=config.RTOL,
            )


def test_segment_weighted(dual_graph):
    """Test segment_weighted=True on a dual graph."""
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    # Verify it requires a dual graph
    assert network_structure.is_dual
    assert "primal_edge" in nodes_gdf.columns
    # Compute with segment_weighted
    nodes_gdf_sw = networks.centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=True,
    )
    # Compute without segment_weighted
    nodes_gdf_plain = networks.centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=False,
    )
    for d in distances:
        density_key = config.prep_gdf_key("density", d)
        harmonic_key = config.prep_gdf_key("harmonic", d)
        betw_key = config.prep_gdf_key("betweenness", d)
        # Segment-weighted density should be total reachable street length (> node count)
        assert nodes_gdf_sw[density_key].sum() > nodes_gdf_plain[density_key].sum()
        # All expected columns should be present and non-null for live nodes
        assert not nodes_gdf_sw[density_key].isna().all()
        assert not nodes_gdf_sw[harmonic_key].isna().all()
        assert not nodes_gdf_sw[betw_key].isna().all()
    # Verify weights are restored after computation
    for nd_idx in network_structure.node_indices():
        assert network_structure.get_node_weight(nd_idx) == 1.0
    # Test simplest path variant too
    nodes_gdf_sw_ang = networks.centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        segment_weighted=True,
    )
    for d in distances:
        density_key = config.prep_gdf_key("density", d, angular=True)
        assert not nodes_gdf_sw_ang[density_key].isna().all()


def test_closeness_shortest(primal_graph):
    """Test standalone closeness_shortest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    nodes_gdf_result = networks.closeness_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("decay", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("cycles", dist) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist) in nodes_gdf_result.columns


def test_closeness_shortest_seeded_determinism(primal_graph):
    """Same seed produces identical adaptive closeness_shortest results."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    kwargs = dict(
        network_structure=network_structure,
        distances=distances,
        random_seed=42,
    )
    r1 = networks.closeness_shortest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    r2 = networks.closeness_shortest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    for dist in distances:
        key = config.prep_gdf_key("density", dist)
        assert np.allclose(r1[key].values, r2[key].values), f"Non-deterministic at {dist}m"


def test_closeness_simplest(dual_graph):
    """Test standalone closeness_simplest with adaptive sampling."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    nodes_gdf_result = networks.closeness_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
        random_seed=42,
    )
    for dist in distances:
        assert config.prep_gdf_key("harmonic", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("density", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("farness", dist, angular=True) in nodes_gdf_result.columns
        assert config.prep_gdf_key("hillier", dist, angular=True) in nodes_gdf_result.columns


def test_closeness_simplest_seeded_determinism(dual_graph):
    """Same seed produces identical adaptive closeness_simplest results."""
    distances = [200, 400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    kwargs = dict(
        network_structure=network_structure,
        distances=distances,
        random_seed=42,
    )
    r1 = networks.closeness_simplest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    r2 = networks.closeness_simplest(nodes_gdf=nodes_gdf.copy(), **kwargs)
    for dist in distances:
        key = config.prep_gdf_key("density", dist, angular=True)
        assert np.allclose(r1[key].values, r2[key].values), f"Non-deterministic at {dist}m"


def test_simplest_wrappers_require_dual_graph(primal_graph):
    distances = [200, 400]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    with pytest.raises(ValueError, match="dual graph"):
        networks.closeness_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )
    with pytest.raises(ValueError, match="dual graph"):
        networks.betweenness_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )


def _build_line(weights: list[float], spacing: float = 100.0) -> rustalgos.graph.NetworkStructure:
    """Build a simple path graph 0-1-2-... with the given per-node weights."""
    ns = rustalgos.graph.NetworkStructure()
    coords = [(i * spacing, 0.0) for i in range(len(weights))]
    for i, (x, y) in enumerate(coords):
        ns.add_street_node(node_key=str(i), x=x, y=y, live=True, weight=weights[i])
    for i in range(len(weights) - 1):
        wkt = f"LINESTRING ({coords[i][0]} 0, {coords[i + 1][0]} 0)"
        ns.add_street_edge(i, i + 1, i, str(i), str(i + 1), wkt)
        ns.add_street_edge(i + 1, i, i, str(i + 1), str(i), wkt)
    ns.validate()
    ns.build_edge_rtree()
    return ns


def test_node_weight_closeness_is_gravity():
    """Node weight should weight reachable destinations (gravity), not rescale the node's own score.

    For density (f="1") with all nodes within the threshold, closeness[N] must equal the sum of the
    OTHER nodes' weights, i.e. A(N) = sum_{j != N} w_j.
    """
    weights = [1.0, 1.0, 3.0, 1.0, 1.0]
    ns = _build_line(weights)
    res = ns.centrality_shortest(
        distances=[1000],
        closeness_exprs=[("density", "1")],
        betweenness_exprs=[],
        compute_cycles=False,
        pbar_disabled=True,
    )
    total = sum(weights)
    expected = [total - w for w in weights]  # [6, 6, 4, 6, 6]
    assert np.allclose(res.metrics["density"][1000], expected)


def test_node_weight_full_matches_sampled():
    """Under non-uniform weights, the full computation must match sampling at p=1.0.

    Regression guard: previously full closeness weighted by the source/self while sampling weighted
    by the destination, so the two diverged whenever weights were non-uniform.
    """
    weights = [1.0, 2.0, 3.0, 1.0, 4.0]
    closeness = [("density", "1"), ("harmonic", "1/c")]
    full = _build_line(weights).centrality_shortest(
        distances=[1000],
        closeness_exprs=closeness,
        betweenness_exprs=[],
        compute_cycles=False,
        pbar_disabled=True,
    )
    sampled = _build_line(weights).centrality_shortest(
        distances=[1000],
        closeness_exprs=closeness,
        betweenness_exprs=[],
        compute_cycles=False,
        sample_probability=1.0,
        random_seed=1,
        pbar_disabled=True,
    )
    for name in ("density", "harmonic"):
        assert np.allclose(full.metrics[name][1000], sampled.metrics[name][1000])


def test_node_weight_betweenness_is_product():
    """Betweenness should weight each O-D pair by the PRODUCT of endpoint weights (gravity flow).

    On a 0-1-2 path the only intermediate node is 1, carrying the single pair {0, 2}, so its
    betweenness must equal w_0 * w_2 (not the average 0.5 * (w_0 + w_2)).
    """
    for w0, w2 in [(1.0, 1.0), (3.0, 1.0), (1.0, 5.0), (3.0, 5.0)]:
        ns = _build_line([w0, 1.0, w2])
        res = ns.centrality_shortest(
            distances=[1000],
            closeness_exprs=[],
            betweenness_exprs=[("b", "1")],
            compute_cycles=False,
            pbar_disabled=True,
        )
        assert res.metrics["b"][1000][1] == pytest.approx(w0 * w2)


def test_zero_weight_node_no_nan_with_sampling_cycles():
    """A zero-weight source under sampling + cycles must not produce NaN/inf.

    Regression guard for cycles_wt = wt / weight(src), which was 0/0 = NaN for a zero-weight node.
    """
    ns = _build_line([1.0, 0.0, 1.0, 1.0, 1.0])
    res = ns.centrality_shortest(
        distances=[1000],
        closeness_exprs=[("density", "1")],
        betweenness_exprs=[("b", "1")],
        compute_cycles=True,
        sample_probability=0.5,
        random_seed=1,
        pbar_disabled=True,
    )
    for name in ("density", "b", "cycles"):
        vals = np.asarray(res.metrics[name][1000], dtype=float)
        assert np.isfinite(vals).all()


def test_betweenness_demand(primal_graph):
    """Demand-weighted flow betweenness: aggregation invariant, allocation, closest mode, empties."""
    import geopandas as gpd
    from shapely.geometry import Point

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    pts = nodes_gdf.geometry
    origins_gdf = gpd.GeoDataFrame({"geometry": pts, "pop": np.full(len(pts), 100.0)}, crs=nodes_gdf.crs)

    def run(dests_gdf, **kwargs):
        out = networks.betweenness_demand(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            origins_gdf=origins_gdf,
            destinations_gdf=dests_gdf,
            origin_weight_col="pop",
            destination_weight_col="w",
            distances=[800],
            decay_fn="exp(-0.002 * c)",
            **kwargs,
        )
        return out["cc_demand_800"].to_numpy()

    # The headline fix: two destinations snapped to the same node with weights (3, 7) must produce
    # exactly the same flow as a single destination of weight 10 at that node (weights aggregated,
    # not overwritten).
    coord0 = pts.iloc[0]
    two_dests = gpd.GeoDataFrame({"geometry": [coord0, coord0], "w": [3.0, 7.0]}, crs=nodes_gdf.crs)
    one_dest = gpd.GeoDataFrame({"geometry": [coord0], "w": [10.0]}, crs=nodes_gdf.crs)
    flow_two = run(two_dests)
    flow_one = run(one_dest)
    assert np.allclose(flow_two, flow_one, equal_nan=True, atol=config.ATOL, rtol=config.RTOL)

    # output column present and non-negative; some flow is generated
    assert np.nanmax(flow_one) > 0.0

    # closest_destination is a distinct, valid code path
    flow_closest = run(one_dest, closest_destination=True)
    assert np.nanmax(flow_closest) > 0.0

    # empty destinations: no crash, no flow
    empty = gpd.GeoDataFrame({"geometry": [], "w": []}, crs=nodes_gdf.crs, geometry="geometry")
    flow_empty = run(empty)
    assert np.nansum(flow_empty) == 0.0

    # a destination too far to snap is dropped -> no flow, no crash
    far = gpd.GeoDataFrame({"geometry": [Point(1e7, 1e7)], "w": [10.0]}, crs=nodes_gdf.crs)
    flow_far = run(far, max_netw_assign_dist=50.0)
    assert np.nansum(flow_far) == 0.0


def test_betweenness_demand_outside_option(primal_graph):
    """Outside option: s=1 identity, participation damping, monotonicity in s, closest-mode scaling."""
    import geopandas as gpd

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    pts = nodes_gdf.geometry
    origins_gdf = gpd.GeoDataFrame({"geometry": pts, "pop": np.full(len(pts), 100.0)}, crs=nodes_gdf.crs)
    dests_gdf = gpd.GeoDataFrame({"geometry": [pts.iloc[0], pts.iloc[-1]], "w": [10.0, 5.0]}, crs=nodes_gdf.crs)

    def run(**kwargs):
        out = networks.betweenness_demand(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            origins_gdf=origins_gdf,
            destinations_gdf=dests_gdf,
            origin_weight_col="pop",
            destination_weight_col="w",
            distances=[800],
            decay_fn="exp(-0.002 * c)",
            **kwargs,
        )
        return out["cc_demand_800"].to_numpy()

    base = run()
    # full participation (s = 1) reproduces the default exactly (derived K = 0)
    full = run(participation=1.0)
    assert np.allclose(base, full, equal_nan=True, atol=config.ATOL, rtol=config.RTOL)

    # s < 1 damps flows everywhere (participation < 1) and never increases them
    damped = run(participation=0.5)
    finite = np.isfinite(base) & np.isfinite(damped)
    assert (damped[finite] <= base[finite] + config.ATOL).all()
    positive = finite & (base > 0)
    assert positive.any()
    assert (damped[positive] < base[positive]).all()

    # monotone: lower participation, lower flows
    damped_more = run(participation=0.2)
    finite2 = np.isfinite(damped) & np.isfinite(damped_more)
    assert (damped_more[finite2] <= damped[finite2] + config.ATOL).all()

    # closest_destination scales by the same participation rate
    closest_base = run(closest_destination=True)
    closest_damped = run(closest_destination=True, participation=0.5)
    finite3 = np.isfinite(closest_base) & np.isfinite(closest_damped)
    assert (closest_damped[finite3] <= closest_base[finite3] + config.ATOL).all()
    assert np.nanmax(closest_damped) > 0.0

    # invalid shares rejected
    with pytest.raises(ValueError):
        run(participation=0.0)
    with pytest.raises(ValueError):
        run(participation=1.5)

    # metric_name renames the output column
    named = networks.betweenness_demand(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        origins_gdf=origins_gdf,
        destinations_gdf=dests_gdf,
        origin_weight_col="pop",
        destination_weight_col="w",
        distances=[800],
        decay_fn="exp(-0.002 * c)",
        metric_name="flows",
    )
    assert np.allclose(base, named["cc_flows_800"].to_numpy(), equal_nan=True, atol=config.ATOL, rtol=config.RTOL)


def test_betweenness_demand_offsets(primal_graph):
    """Assignment offsets enter routed distances: composite offset_o + graph + offset_d vs radius."""
    import geopandas as gpd
    from shapely.geometry import Point

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    pts = nodes_gdf.geometry
    # a single origin and destination, each held ~60 m off-network from well-separated nodes
    o_pt = Point(pts.iloc[0].x + 60.0, pts.iloc[0].y)
    d_pt = Point(pts.iloc[-1].x - 60.0, pts.iloc[-1].y)
    origins_gdf = gpd.GeoDataFrame({"geometry": [o_pt], "pop": [100.0]}, crs=nodes_gdf.crs)
    dests_gdf = gpd.GeoDataFrame({"geometry": [d_pt], "w": [10.0]}, crs=nodes_gdf.crs)

    def run(distance):
        out = networks.betweenness_demand(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            origins_gdf=origins_gdf,
            destinations_gdf=dests_gdf,
            origin_weight_col="pop",
            destination_weight_col="w",
            distances=[distance],
            decay_fn="1",
            max_netw_assign_dist=200.0,
        )
        return np.nansum(out[f"cc_demand_{distance}"].to_numpy())

    # generous radius: the pair connects and flow is routed
    assert run(5000) > 0.0
    # radius below the summed offsets alone (~120 m): the composite distance can never fit,
    # so no flow routes even though both points assign to the network successfully
    assert run(100) == 0.0
    """The deprecated 4.24 shim reproduces centrality_shortest output under legacy column names."""
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    # modern defaults emit cc_decay_* / cc_betweenness_decay_*
    modern = networks.centrality_shortest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
    )
    # the deprecated shim emits the legacy cc_beta_* / cc_betweenness_beta_* names
    with pytest.warns(DeprecationWarning):
        legacy = networks.node_centrality_shortest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )
    for d in distances:
        assert f"cc_beta_{d}" in legacy
        assert f"cc_betweenness_beta_{d}" in legacy
        # legacy beta-weighting == modern normalised-progress decay, numerically
        # (equal_nan: hillier is density**2 / farness, which is NaN at farness-zero nodes in both)
        assert np.allclose(legacy[f"cc_beta_{d}"], modern[f"cc_decay_{d}"], equal_nan=True)
        assert np.allclose(legacy[f"cc_betweenness_beta_{d}"], modern[f"cc_betweenness_decay_{d}"], equal_nan=True)
        # every other metric is unchanged
        for stem in ("cc_density", "cc_farness", "cc_harmonic", "cc_cycles", "cc_hillier", "cc_betweenness"):
            assert np.allclose(legacy[f"{stem}_{d}"], modern[f"{stem}_{d}"], equal_nan=True)


def test_node_centrality_simplest_compat(dual_graph):
    """The deprecated 4.24 simplest shim reproduces centrality_simplest output (angular columns)."""
    distances = [400, 800]
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(dual_graph)
    modern = networks.centrality_simplest(
        network_structure=network_structure,
        nodes_gdf=nodes_gdf.copy(),
        distances=distances,
    )
    with pytest.warns(DeprecationWarning):
        legacy = networks.node_centrality_simplest(
            network_structure=network_structure,
            nodes_gdf=nodes_gdf.copy(),
            distances=distances,
        )
    # the shim produces exactly the same cc_ columns and values as the modern default
    modern_cc = sorted(c for c in modern.columns if c.startswith("cc_"))
    legacy_cc = sorted(c for c in legacy.columns if c.startswith("cc_"))
    assert legacy_cc == modern_cc
    for col in modern_cc:
        assert np.allclose(legacy[col], modern[col], equal_nan=True)
