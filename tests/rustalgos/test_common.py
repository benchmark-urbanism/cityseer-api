# pyright: basic
from __future__ import annotations

import numpy as np
import pytest

from cityseer import config, rustalgos
from cityseer.tools import mock


def test_coord():
    c1 = rustalgos.Coord(0, 1)
    c2 = rustalgos.Coord(1, 2)
    assert c1.x == 0 and c1.y == 1
    assert c2.x == 1 and c2.y == 2
    assert c1.xy() == (0, 1)
    assert c2.xy() == (1, 2)
    assert np.isclose(c1.hypot(c2), np.sqrt(2))
    assert np.isclose(c2.hypot(c1), np.sqrt(2))
    assert c1.difference(c2).xy() == (-1, -1)
    assert c2.difference(c1).xy() == (1, 1)


def test_calculate_rotation():
    c1 = rustalgos.Coord(0, 0)
    c2 = rustalgos.Coord(10, 10)
    assert rustalgos.calculate_rotation(c1, c2) == -45
    assert rustalgos.calculate_rotation(c2, c1) == 45
    c3 = rustalgos.Coord(-10, 0)
    c4 = rustalgos.Coord(10, 0)
    assert rustalgos.calculate_rotation(c1, c3) == -180
    assert rustalgos.calculate_rotation(c1, c4) == 0


def test_calculate_rotation_smallest():
    c1 = rustalgos.Coord(0, 0)
    c2 = rustalgos.Coord(10, 10)
    c3 = rustalgos.Coord(-10, 0)
    c4 = rustalgos.Coord(10, -10)
    # calculates anticlockwise
    assert rustalgos.calculate_rotation_smallest(c2.difference(c1), c3.difference(c1)) == 135
    assert rustalgos.calculate_rotation_smallest(c2.difference(c1), c4.difference(c1)) == 90
    assert rustalgos.calculate_rotation_smallest(c3.difference(c1), c4.difference(c1)) == 225


def test_check_numerical_data(primal_graph):
    # catch single dimensions
    with pytest.raises(TypeError):
        mock_numerical = mock.mock_numerical_data(primal_graph)
        corrupt_numerical = mock_numerical.mock_numerical_1.values
        assert corrupt_numerical.ndim == 1
        rustalgos.check_numerical_data(corrupt_numerical)
    # catch infinites
    mock_numerical = mock.mock_numerical_data(primal_graph, num_arrs=2)
    # should work
    ok_numerical = mock_numerical[["mock_numerical_1", "mock_numerical_2"]].values
    rustalgos.check_numerical_data(ok_numerical)
    with pytest.raises(ValueError):
        corrupt_numerical = mock_numerical[["mock_numerical_1", "mock_numerical_2"]].values
        corrupt_numerical[0, 0] = np.inf
        rustalgos.check_numerical_data(corrupt_numerical)


def test_check_categorical_data(primal_graph):
    pass


def test_distances_from_betas():
    # some basic checks using float form
    for b, d in zip([0.04, 0.0025], [100, 1600]):
        # simple straight check against corresponding distance
        assert rustalgos.distances_from_betas([b]) == [d]
        # circular check
        assert np.allclose(
            rustalgos.betas_from_distances(rustalgos.distances_from_betas([b])), [b], atol=config.ATOL, rtol=config.RTOL
        )
    # check that custom min_threshold_wt works
    d = rustalgos.distances_from_betas([0.04], min_threshold_wt=0.001)
    assert np.allclose(d, 173, atol=config.ATOL, rtol=config.RTOL)
    # check on array form
    arr = rustalgos.distances_from_betas([0.04, 0.0025])
    assert np.allclose(arr, [100, 1600], atol=config.ATOL, rtol=config.RTOL)
    # check that duplicates or decreasing ordering is caught
    with pytest.raises(ValueError):
        rustalgos.distances_from_betas([0.04, 0.04])
    with pytest.raises(ValueError):
        rustalgos.distances_from_betas([0.0025, 0.04])
    # check that invalid beta values raise an error
    with pytest.raises(TypeError):
        rustalgos.distances_from_betas("boo")
    for b in ([None], None):
        with pytest.raises(TypeError):
            rustalgos.distances_from_betas(b)
    for b in ([-0.04], [0], [-0], [-0.0], [0.0], []):
        with pytest.raises(ValueError):
            rustalgos.distances_from_betas(b)


def test_betas_from_distances():
    # some basic checks
    for dist, b in zip([100, 1600], [0.04, 0.0025]):
        # simple straight check against corresponding distance
        assert np.allclose(rustalgos.betas_from_distances([dist]), [b], atol=config.ATOL, rtol=config.RTOL)
        # circular check
        assert np.allclose(
            rustalgos.distances_from_betas(rustalgos.betas_from_distances([dist])),
            [dist],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
        # array form check
        assert np.allclose(
            rustalgos.betas_from_distances([dist]),
            [b],
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # check that custom min_threshold_wt works
    b = rustalgos.betas_from_distances([173], min_threshold_wt=0.001)
    assert np.allclose([b], [0.03992922231554985], atol=config.ATOL, rtol=config.RTOL)
    # check on array form
    arr = rustalgos.betas_from_distances([100, 1600])
    assert np.allclose(arr, [0.04, 0.0025], atol=config.ATOL, rtol=config.RTOL)
    # check that duplicates or decreasing ordering is caught
    with pytest.raises(ValueError):
        rustalgos.betas_from_distances([100, 100])
    with pytest.raises(ValueError):
        rustalgos.betas_from_distances([100, 50])
    # check that invalid distance values raise an error
    with pytest.raises(TypeError):
        rustalgos.betas_from_distances("boo")
    for d in ([None], None):
        with pytest.raises(TypeError):
            rustalgos.betas_from_distances(d)
    for d in ([0], []):  # [-1],  negative gives overflow error
        with pytest.raises(ValueError):
            rustalgos.betas_from_distances(d)


def test_pair_distances_and_betas():
    betas = [0.02, 0.01, 0.005, 0.0025]
    distances = rustalgos.distances_from_betas(betas)
    # should raise if both provided
    with pytest.raises(ValueError):
        rustalgos.pair_distances_and_betas(distances, betas)
    # should raise if neither provided
    with pytest.raises(ValueError):
        rustalgos.pair_distances_and_betas(None, None)
    distances, betas_1 = rustalgos.pair_distances_and_betas(distances, None)
    assert np.allclose(betas_1, betas, atol=config.ATOL, rtol=config.RTOL)
    distances_1, betas = rustalgos.pair_distances_and_betas(None, betas)
    assert np.allclose(distances_1, distances, atol=config.ATOL, rtol=config.RTOL)


def test_avg_distances_for_betas():
    betas = rustalgos.betas_from_distances([100, 200, 400, 800, 1600])
    assert np.allclose(
        rustalgos.avg_distances_for_betas(betas),
        [35.11949, 70.23898, 140.47797, 280.95593, 561.91187],
        atol=config.ATOL,
        rtol=config.RTOL,
    )


def test_clip_wts_curve():
    distances = [400, 800, 1600]
    betas = rustalgos.betas_from_distances(distances)
    # should be 1 if dist buffer is zero
    max_curve_wts = rustalgos.clip_wts_curve(distances, betas, 0)
    assert np.allclose([1, 1, 1], max_curve_wts, atol=config.ATOL, rtol=config.RTOL)
    # check for a random distance
    max_curve_wts = rustalgos.clip_wts_curve(distances, betas, 50)
    assert np.allclose([0.60653067, 0.7788008, 0.8824969], max_curve_wts, atol=config.ATOL, rtol=config.RTOL)
    # should raise if buffer_distance is less than zero
    # with pytest.raises(ValueError):
    #     max_curve_wts = rustalgos.clip_wts_curve(distances, betas, -1)  # negative raises overflow error
    # should raise if buffer_distance is greater than distances
    with pytest.raises(ValueError):
        max_curve_wts = rustalgos.clip_wts_curve(distances, betas, 401)


def test_clipped_beta_wt():
    distances = [400, 800, 1600]
    betas = rustalgos.betas_from_distances(distances)
    # try a range of spatial tolerances
    for spatial_tolerance in [0, 50, 400]:
        # calculate curve thresholds
        max_curve_wts = rustalgos.clip_wts_curve(distances, betas, spatial_tolerance)
        # iter
        for dist, beta, max_wt in zip(distances, betas, max_curve_wts):
            # try a range of datapoint distances
            for data_dist in [0, 25, 50, 100, 400, 800, 1600]:
                # continue if data distance exceeds current distance threshold - these are ignored in computations
                if data_dist > dist:
                    continue
                # calculate raw vs. clipped
                wt_raw = np.exp(-beta * data_dist)
                wt_clip = rustalgos.clipped_beta_wt(beta, max_wt, data_dist)
                # calculated clipped manually for cross checking
                wt_manual = min(np.exp(-beta * data_dist), max_wt) / max_wt
                assert np.isclose(wt_clip, wt_manual, atol=config.ATOL, rtol=config.RTOL)
                # if tolerance is zero - raw and clipped should match regardless
                if spatial_tolerance == 0:
                    assert np.isclose(wt_raw, wt_clip, atol=config.ATOL, rtol=config.RTOL)
                    continue
                # if distance is zero, then either case should be 1
                if data_dist == 0:
                    assert np.isclose(wt_raw, 1, atol=config.ATOL, rtol=config.RTOL)
                    assert np.isclose(wt_raw, wt_clip, atol=config.ATOL, rtol=config.RTOL)
                # if distance equals threshold, then either case should be min wt threshold
                elif data_dist == dist:
                    assert np.isclose(wt_raw, config.MIN_THRESH_WT, atol=config.ATOL, rtol=config.RTOL)
                    assert wt_clip > wt_raw
                # if distance is less than spatial tolerance, then clipped should be 1
                elif data_dist < spatial_tolerance:
                    assert np.isclose(wt_clip, 1, atol=config.ATOL, rtol=config.RTOL)
                    assert wt_raw < 1
                # if distance is greater than tolerance, then clipped should be greater than
                elif data_dist > spatial_tolerance:
                    assert wt_clip > wt_raw
