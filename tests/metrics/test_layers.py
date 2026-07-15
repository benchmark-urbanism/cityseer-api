# pyright: basic
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from cityseer import config
from cityseer.metrics import layers
from cityseer.tools import io, mock
from shapely import geometry


def test_build_data_map(primal_graph):
    _nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    for typ in [int, float, str]:
        for data_id_col in [None, "data_id"]:
            data_gdf = mock.mock_data_gdf(primal_graph)
            data_gdf.index = data_gdf.index.astype(typ)
            for to_poly in [False, True]:
                # handle both points and polys
                if to_poly is True:
                    data_gdf.geometry = data_gdf.geometry.buffer(10)
                #
                data_map = layers.build_data_map(
                    data_gdf, network_structure, max_netw_assign_dist=400, data_id_col=data_id_col
                )
                for _data_key, data_entry in data_map.entries.items():
                    data_gdf_row = data_gdf.loc[data_entry.data_key_py]  # type: ignore
                    assert data_entry.geom_wkt == data_gdf_row.geometry.wkt
                    if data_id_col is not None:
                        assert data_entry.dedupe_key_py == data_gdf_row[data_id_col]
                    else:
                        assert data_entry.dedupe_key_py == data_entry.data_key_py
    # check with different geometry column name
    data_gdf = mock.mock_data_gdf(primal_graph)
    data_gdf.rename(columns={"geometry": "geom"}, inplace=True)
    data_gdf.set_geometry("geom", inplace=True)
    data_map = layers.build_data_map(data_gdf, network_structure, max_netw_assign_dist=400, data_id_col="data_id")
    # catch non unique indices
    data_gdf = gpd.GeoDataFrame(
        {
            "data_idx": [1, 2, 2],
            "geometry": [
                geometry.Point(0, 0),
                geometry.Point(1, 1),
                geometry.Point(2, 2),
            ],
        },
        crs="EPSG:3857",
    )
    data_gdf.set_index("data_idx", inplace=True)
    with pytest.raises(ValueError):
        data_map = layers.build_data_map(
            data_gdf,
            network_structure,
            max_netw_assign_dist=400,
        )


def test_compute_accessibilities(primal_graph, dual_graph):
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    for angular in [False, True]:
        nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
        network_structure = network_structure_dual if angular else network_structure_primal
        for data_id_col in [None, "data_id"]:
            for key_set in (["a"], ["b"], ["a", "b"]):
                nodes_gdf, data_gdf = layers.compute_accessibilities(
                    data_gdf,  # type: ignore
                    "categorical_landuses",
                    key_set,
                    nodes_gdf,  # type: ignore
                    network_structure,
                    max_netw_assign_dist=max_assign_dist,
                    distances=distances,
                    data_id_col=data_id_col,
                    angular=angular,
                    decay_fn="1",
                )
                # test against manual implementation over underlying method
                landuses_map = dict(data_gdf["categorical_landuses"])  # type: ignore
                data_map = layers.build_data_map(
                    data_gdf,
                    network_structure,
                    max_netw_assign_dist=max_assign_dist,
                    data_id_col=data_id_col,
                )
                accessibility_data = data_map.accessibility(
                    network_structure,
                    landuses_map,  # type: ignore
                    key_set,
                    distances=distances,
                    angular=angular,
                )
                for acc_key in key_set:
                    for dist_key in distances:
                        acc_data_key = config.prep_gdf_key(acc_key, dist_key, angular)
                        assert np.allclose(
                            nodes_gdf[acc_data_key].values,  # type: ignore
                            accessibility_data.result[acc_key].count[dist_key],
                            atol=config.ATOL,
                            rtol=config.RTOL,
                            equal_nan=True,
                        )
                        acc_data_key_dist = config.prep_gdf_key(f"{acc_key}_nearest_max", dist_key, angular)
                        if dist_key == max(distances):
                            assert np.allclose(
                                nodes_gdf[acc_data_key_dist].values,  # type: ignore
                                accessibility_data.result[acc_key].distance[dist_key],
                                atol=config.ATOL,
                                rtol=config.RTOL,
                                equal_nan=True,
                            )
                        else:
                            assert acc_data_key_dist not in nodes_gdf.columns  # type: ignore
                # most integrity checks happen in underlying method
                with pytest.raises(ValueError):
                    nodes_gdf = layers.compute_accessibilities(
                        data_gdf,  # type: ignore
                        "categorical_landuses-TYPO",
                        ["c"],
                        nodes_gdf,  # type: ignore
                        network_structure,
                        max_netw_assign_dist=max_assign_dist,
                        distances=distances,
                    )


def test_compute_mixed_uses(primal_graph, dual_graph):
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    max_assign_dist = 400
    # test against manual implementation over underlying method
    for data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
            network_structure = network_structure_dual if angular else network_structure_primal
            nodes_gdf, data_gdf = layers.compute_mixed_uses(
                data_gdf,
                "categorical_landuses",
                nodes_gdf,
                network_structure,
                distances=distances,
                compute_hill=True,
                compute_shannon=True,
                compute_gini=True,
                data_id_col=data_id_col,
                angular=angular,
                max_netw_assign_dist=max_assign_dist,
                decay_fn="1",
            )
            # generate manually
            data_map = layers.build_data_map(
                data_gdf,
                network_structure,
                max_netw_assign_dist=max_assign_dist,
                data_id_col=data_id_col,
            )
            landuses_map = dict(data_gdf["categorical_landuses"])
            mu_data = data_map.mixed_uses(
                network_structure,
                landuses_map,
                compute_hill=True,
                compute_shannon=True,
                compute_gini=True,
                distances=distances,
                angular=angular,
            )
            for dist_key in distances:
                for q_key in [0, 1, 2]:
                    hill_data_key = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular=angular)
                    assert np.allclose(
                        nodes_gdf[hill_data_key].values,
                        mu_data.hill[q_key][dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                    )
                shannon_data_key = config.prep_gdf_key("shannon", dist_key, angular=angular)
                assert np.allclose(
                    nodes_gdf[shannon_data_key].values,
                    mu_data.shannon[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )
                gini_data_key = config.prep_gdf_key("gini", dist_key, angular=angular)
                assert np.allclose(
                    nodes_gdf[gini_data_key].values,
                    mu_data.gini[dist_key],
                    atol=config.ATOL,
                    rtol=config.RTOL,
                )


def test_layer_methods_skip_missing_landuse(primal_graph):
    """Points with a missing (NaN) land-use category are excluded rather than erroring (issue #146).

    A frame with NaN categories must produce the same result as the same frame with those rows
    already dropped: uncategorised points belong to no land use, so they contribute to neither
    accessibility counts nor mixed-use diversity.
    """
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph, random_seed=13)
    distances = [400, 800]
    # introduce missing categories on a subset, and a reference frame with those rows removed
    data_nan = data_gdf.copy()
    data_nan.loc[data_nan.index[::4], "categorical_landuses"] = np.nan
    data_dropped = data_nan.dropna(subset=["categorical_landuses"])
    assert data_nan["categorical_landuses"].isna().any()

    keys = ["a", "b"]
    acc_nan, ret_gdf = layers.compute_accessibilities(
        data_nan.copy(),
        "categorical_landuses",
        keys,
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn="1",
    )
    acc_dropped, _ = layers.compute_accessibilities(
        data_dropped.copy(),
        "categorical_landuses",
        keys,
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn="1",
    )
    # the returned data frame is the caller's, unchanged (NaN rows preserved)
    assert ret_gdf["categorical_landuses"].isna().any()
    for key in keys:
        for dist_key in distances:
            col = config.prep_gdf_key(key, dist_key, False)
            assert np.allclose(acc_nan[col].values, acc_dropped[col].values, atol=config.ATOL, rtol=config.RTOL)

    mu_nan, _ = layers.compute_mixed_uses(
        data_nan.copy(),
        "categorical_landuses",
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        compute_hill=True,
        compute_shannon=True,
        compute_gini=True,
        decay_fn="1",
    )
    mu_dropped, _ = layers.compute_mixed_uses(
        data_dropped.copy(),
        "categorical_landuses",
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        compute_hill=True,
        compute_shannon=True,
        compute_gini=True,
        decay_fn="1",
    )
    for dist_key in distances:
        for q_key in [0, 1, 2]:
            col = config.prep_gdf_key(f"hill_q{q_key}", dist_key, angular=False)
            assert np.allclose(mu_nan[col].values, mu_dropped[col].values, atol=config.ATOL, rtol=config.RTOL)


def test_compute_stats_skips_nan(primal_graph):
    """NaN values in a numeric column are skipped by the aggregation, not errored on or poisoning.

    With a single stats column, skipping NaN values is equivalent to removing those rows, so the
    result on a frame with NaN must match the result on the same frame with the NaN rows dropped.
    Guards the Rust-side ``if num.is_nan() { continue }`` behaviour against regression.
    """
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    distances = [400, 800]
    col = "mock_numerical_1"
    data_nan = data_gdf.copy()
    data_nan.loc[data_nan.index[::3], col] = np.nan
    data_dropped = data_nan.dropna(subset=[col])
    assert data_nan[col].isna().any()

    res_nan, _ = layers.compute_stats(
        data_nan.copy(), [col], nodes_gdf.copy(), network_structure, distances=distances, decay_fn="1"
    )
    res_dropped, _ = layers.compute_stats(
        data_dropped.copy(), [col], nodes_gdf.copy(), network_structure, distances=distances, decay_fn="1"
    )
    for measure in ["sum", "mean", "count"]:
        for dist_key in distances:
            key = config.prep_gdf_key(f"{col}_{measure}", dist_key, angular=False)
            assert np.allclose(
                res_nan[key].values, res_dropped[key].values, atol=config.ATOL, rtol=config.RTOL, equal_nan=True
            )


def test_compute_stats(primal_graph, dual_graph):
    """
    Test stats component
    """
    nodes_gdf_primal, _edges_gdf_primal, network_structure_primal = io.network_structure_from_nx(primal_graph)
    nodes_gdf_dual, _edges_gdf_dual, network_structure_dual = io.network_structure_from_nx(dual_graph)
    data_gdf = mock.mock_numerical_data(primal_graph, num_arrs=2)
    max_assign_dist = 400
    distances = [400, 800]
    for _data_id_col in [None, "data_id"]:
        for angular in [False, True]:
            nodes_gdf = nodes_gdf_dual.copy() if angular else nodes_gdf_primal.copy()
            network_structure = network_structure_dual if angular else network_structure_primal
            data_map = layers.build_data_map(
                data_gdf,
                network_structure,
                max_netw_assign_dist=max_assign_dist,
                data_id_col=None,
            )
            nodes_gdf, data_gdf = layers.compute_stats(
                data_gdf,
                ["mock_numerical_1", "mock_numerical_2"],
                nodes_gdf,
                network_structure,
                distances=distances,
                angular=angular,
                max_netw_assign_dist=max_assign_dist,
                decay_fn="1",
            )
            # compare to manual
            for stats_key in ["mock_numerical_1", "mock_numerical_2"]:
                stats_map = dict(data_gdf[stats_key])  # type: ignore
                # generate stats
                stats_results = data_map.stats(
                    network_structure,
                    numerical_maps=[stats_map],
                    distances=distances,
                    angular=angular,
                )
                stats_result = stats_results.result[0]
                for dist_key in distances:
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_sum", dist_key, angular=angular)],
                        stats_result.sum[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_mean", dist_key, angular=angular)],
                        stats_result.mean[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_count", dist_key, angular=angular)],
                        stats_result.count[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_var", dist_key, angular=angular)],
                        stats_result.variance[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_max", dist_key, angular=angular)],
                        stats_result.max[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
                    assert np.allclose(
                        nodes_gdf[config.prep_gdf_key(f"{stats_key}_min", dist_key, angular=angular)],
                        stats_result.min[dist_key],
                        atol=config.ATOL,
                        rtol=config.RTOL,
                        equal_nan=True,
                    )
    # check that problematic column labels are raised
    with pytest.raises(ValueError):
        layers.compute_stats(
            data_gdf,
            ["typo"],
            nodes_gdf,
            network_structure,
            distances=distances,
        )


def test_angular_layer_wrappers_require_dual_graph(primal_graph):
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    landuse_gdf = mock.mock_landuse_categorical_data(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_accessibilities(
            landuse_gdf,
            "categorical_landuses",
            ["a"],
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_mixed_uses(
            landuse_gdf,
            "categorical_landuses",
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )
    with pytest.raises(ValueError, match="dual graph"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=[400],
            angular=True,
        )


def test_custom_decay_fn(primal_graph):
    """Test that custom decay_fn expressions produce different results from default."""
    from cityseer import decay

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    landuse_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [800]
    col_mean = config.prep_gdf_key("mock_numerical_1_mean", 800)
    col_acc = config.prep_gdf_key("a", 800)
    # --- compute_stats ---
    n_default, _ = layers.compute_stats(
        numerical_gdf, ["mock_numerical_1"], nodes_gdf.copy(), network_structure, distances=distances, decay_fn="1"
    )
    n_exp, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.exponential(),
    )
    n_linear, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.linear(),
    )
    # default is flat ("1"), so default and exponential should differ
    assert not np.allclose(n_default[col_mean].dropna(), n_exp[col_mean].dropna(), atol=0.1)
    # default and linear should differ
    assert not np.allclose(n_default[col_mean].dropna(), n_linear[col_mean].dropna(), atol=0.1)
    # exponential and linear should also differ from each other
    assert not np.allclose(n_exp[col_mean].dropna(), n_linear[col_mean].dropna(), atol=0.1)
    # --- compute_accessibilities ---
    a_default, _ = layers.compute_accessibilities(
        landuse_gdf,
        "categorical_landuses",
        ["a"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn="1",
    )
    a_linear, _ = layers.compute_accessibilities(
        landuse_gdf,
        "categorical_landuses",
        ["a"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=decay.linear(),
    )
    assert not np.allclose(a_default[col_acc].dropna(), a_linear[col_acc].dropna(), atol=0.1)
    # --- helper-generated expressions ---
    n_gauss, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=[1200],
        decay_fn=decay.gaussian(peak=400, cutoff=1200, std=150),
    )
    col_gauss = config.prep_gdf_key("mock_numerical_1_mean", 1200)
    assert not n_gauss[col_gauss].dropna().empty
    # --- invalid expression ---
    with pytest.raises(ValueError, match="parse"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=distances,
            decay_fn="invalid !! expression",
        )


def test_per_label_decay_fns(primal_graph):
    """A dict of {label: decay} computes all decays in one traversal, matching separate calls.

    Also verifies the str/None forms remain byte-identical (no column suffix) for backwards
    compatibility, across compute_stats, compute_accessibilities, and compute_mixed_uses.
    """
    from cityseer import decay

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    landuse_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    gauss = decay.gaussian(peak=200, cutoff=800, std=100)
    flat = decay.flat()

    # --- compute_stats: dict form vs two separate single-decay calls ---
    combo, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn={"grav": gauss, "raw": flat},
    )
    sep_g, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=gauss,
    )
    sep_f, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=flat,
    )
    for measure in ["mean", "sum", "count", "max", "min", "median", "var", "mad"]:
        for d in distances:
            grav_col = config.prep_gdf_key(f"mock_numerical_1_{measure}_grav", d)
            raw_col = config.prep_gdf_key(f"mock_numerical_1_{measure}_raw", d)
            base_col = config.prep_gdf_key(f"mock_numerical_1_{measure}", d)
            # f32 summation order (HashMap iteration) differs between calls, so compare
            # with the library's standard tolerance rather than bit-exactly.
            assert np.allclose(combo[grav_col], sep_g[base_col], equal_nan=True, atol=config.ATOL, rtol=config.RTOL)
            assert np.allclose(combo[raw_col], sep_f[base_col], equal_nan=True, atol=config.ATOL, rtol=config.RTOL)
    # the two decays must actually produce different results
    g800 = config.prep_gdf_key("mock_numerical_1_mean_grav", 800)
    r800 = config.prep_gdf_key("mock_numerical_1_mean_raw", 800)
    assert not np.allclose(combo[g800].dropna(), combo[r800].dropna(), atol=0.1)

    # --- back-compat: str/None forms produce the original unsuffixed column names ---
    bc, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=gauss,
    )
    assert config.prep_gdf_key("mock_numerical_1_mean", 800) in bc.columns
    assert config.prep_gdf_key("mock_numerical_1_mean_grav", 800) not in bc.columns

    # --- compute_accessibilities: dict form vs separate ---
    a_combo, _ = layers.compute_accessibilities(
        landuse_gdf,
        "categorical_landuses",
        ["a"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn={"grav": gauss, "raw": flat},
    )
    a_sep_g, _ = layers.compute_accessibilities(
        landuse_gdf,
        "categorical_landuses",
        ["a"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=gauss,
    )
    for d in distances:
        assert np.allclose(
            a_combo[config.prep_gdf_key("a_grav", d)],
            a_sep_g[config.prep_gdf_key("a", d)],
            equal_nan=True,
            atol=config.ATOL,
            rtol=config.RTOL,
        )

    # --- compute_mixed_uses: dict form vs separate ---
    m_combo, _ = layers.compute_mixed_uses(
        landuse_gdf,
        "categorical_landuses",
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn={"grav": gauss, "raw": flat},
    )
    m_sep_g, _ = layers.compute_mixed_uses(
        landuse_gdf,
        "categorical_landuses",
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        decay_fn=gauss,
    )
    for d in distances:
        assert np.allclose(
            m_combo[config.prep_gdf_key("hill_q0_grav", d)],
            m_sep_g[config.prep_gdf_key("hill_q0", d)],
            equal_nan=True,
            atol=config.ATOL,
            rtol=config.RTOL,
        )

    # --- an empty decay dict is rejected ---
    with pytest.raises(ValueError, match="at least one"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=distances,
            decay_fn={},
        )


def test_stats_measures_selection(primal_graph):
    """`measures` selects which statistics are computed; the subset matches the full run."""
    from cityseer import decay

    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    numerical_gdf = mock.mock_numerical_data(primal_graph, num_arrs=1)
    distances = [400, 800]
    all_measures = ["sum", "mean", "count", "var", "median", "mad", "max", "min"]

    full, _ = layers.compute_stats(
        numerical_gdf, ["mock_numerical_1"], nodes_gdf.copy(), network_structure, distances=distances, decay_fn="1"
    )
    sub, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        measures=["mean", "count"],
        decay_fn="1",
    )
    # selected measures match the full computation exactly
    for measure in ["mean", "count"]:
        for d in distances:
            k = config.prep_gdf_key(f"mock_numerical_1_{measure}", d)
            assert np.allclose(sub[k], full[k], equal_nan=True)
    # only the requested measures are emitted
    sub_cc = [c for c in sub.columns if c.startswith("cc_")]
    for measure in all_measures:
        present = any(f"_{measure}_" in c for c in sub_cc)
        assert present == (measure in {"mean", "count"})
    # default (None) computes all eight
    full_cc = [c for c in full.columns if c.startswith("cc_")]
    for measure in all_measures:
        assert any(f"_{measure}_" in c for c in full_cc)
    # invalid measure rejected
    with pytest.raises(ValueError, match="Unknown stats measure"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=distances,
            measures=["nope"],
        )
    # empty list rejected (use None for all)
    with pytest.raises(ValueError, match="at least one measure"):
        layers.compute_stats(
            numerical_gdf,
            ["mock_numerical_1"],
            nodes_gdf.copy(),
            network_structure,
            distances=distances,
            measures=[],
        )
    # composes with the decay dict
    combo, _ = layers.compute_stats(
        numerical_gdf,
        ["mock_numerical_1"],
        nodes_gdf.copy(),
        network_structure,
        distances=distances,
        measures=["mean"],
        decay_fn={"grav": decay.gaussian(peak=200, cutoff=800, std=150), "raw": decay.flat()},
    )
    assert config.prep_gdf_key("mock_numerical_1_mean_grav", 800) in combo.columns
    assert config.prep_gdf_key("mock_numerical_1_mean_raw", 800) in combo.columns


def test_compute_accessibilities_default_back_compat(primal_graph):
    """A bare compute_accessibilities call restores the pre-4.25 _nw + _wt columns and values."""
    nodes_gdf, _edges_gdf, network_structure = io.network_structure_from_nx(primal_graph)
    data_gdf = mock.mock_landuse_categorical_data(primal_graph)
    distances = [400, 800]
    common = dict(
        data_gdf=data_gdf,
        landuse_column_label="categorical_landuses",
        accessibility_keys=["a"],
        network_structure=network_structure,
        distances=distances,
    )
    # bare default -> both legacy columns
    default_gdf, _ = layers.compute_accessibilities(nodes_gdf=nodes_gdf.copy(), **common)
    # explicit single-decay equivalents: "1" is the unweighted (_nw), exp(-4*p) is the weighted (_wt)
    nw_gdf, _ = layers.compute_accessibilities(nodes_gdf=nodes_gdf.copy(), decay_fn="1", **common)
    wt_gdf, _ = layers.compute_accessibilities(nodes_gdf=nodes_gdf.copy(), decay_fn="exp(-4 * p)", **common)
    for d in distances:
        assert f"cc_a_{d}_nw" in default_gdf
        assert f"cc_a_{d}_wt" in default_gdf
        assert np.allclose(default_gdf[f"cc_a_{d}_nw"], nw_gdf[f"cc_a_{d}"], equal_nan=True)
        assert np.allclose(default_gdf[f"cc_a_{d}_wt"], wt_gdf[f"cc_a_{d}"], equal_nan=True)
    # nearest distance is decay-independent: one unsuffixed column, not duplicated per variant
    assert f"cc_a_nearest_max_{max(distances)}" in default_gdf
    assert f"cc_a_nearest_max_{max(distances)}_nw" not in default_gdf
