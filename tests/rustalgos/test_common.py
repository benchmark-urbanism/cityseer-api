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
    for b, d in zip([0.04, 0.0025], [100, 1600], strict=True):
        # simple straight check against corresponding distance
        assert rustalgos.distances_from_betas([b]) == [d]
        # circular check
        assert np.allclose(
            rustalgos.betas_from_distances(rustalgos.distances_from_betas([b])),
            [b],
            atol=config.ATOL,
            rtol=config.RTOL,
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
    for dist, b in zip([100, 1600], [0.04, 0.0025], strict=True):
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


def test_distances_from_seconds():
    distances = [400, 600, 800, 1600, 2000, 2400]
    minutes = [5, 7.5, 10, 20, 25, 30]
    seconds = [round(t * 60) for t in minutes]
    for dist, time in zip(distances, seconds, strict=True):
        assert np.allclose(
            rustalgos.distances_from_seconds([time]),
            dist,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # array form check
    ds = rustalgos.distances_from_seconds(seconds)
    assert np.allclose(distances, ds, atol=config.ATOL, rtol=config.RTOL)
    # round trip check
    ts = rustalgos.seconds_from_distances(ds)
    assert np.allclose(seconds, ts, atol=config.ATOL, rtol=config.RTOL)
    # check that duplicates or decreasing ordering is caught
    with pytest.raises(ValueError):
        rustalgos.distances_from_seconds([10, 10])
    with pytest.raises(ValueError):
        rustalgos.distances_from_seconds([10, 5])


def test_seconds_from_distances():
    distances = [400, 600, 800, 1600, 2000, 2400]
    minutes = [5, 7.5, 10, 20, 25, 30]
    seconds = [round(t * 60) for t in minutes]
    for dist, time in zip(distances, seconds, strict=True):
        assert np.allclose(
            rustalgos.seconds_from_distances([dist]),
            time,
            atol=config.ATOL,
            rtol=config.RTOL,
        )
    # array form check
    ts = rustalgos.seconds_from_distances(distances)
    assert np.allclose(seconds, ts, atol=config.ATOL, rtol=config.RTOL)
    # round trip check
    ds = rustalgos.distances_from_seconds(ts)
    assert np.allclose(distances, ds, atol=config.ATOL, rtol=config.RTOL)
    # check that duplicates or decreasing ordering is caught
    with pytest.raises(ValueError):
        rustalgos.seconds_from_distances([100, 100])
    with pytest.raises(ValueError):
        rustalgos.seconds_from_distances([100, 50])


def test_pair_distances_betas_time():
    distances = [400, 600, 800, 1600, 2000, 10000, 20000]
    minutes = [5.0, 7.5, 10.0, 20.0, 25.0, 125.0, 250.0]
    seconds = [round(t * 60) for t in minutes]
    betas = [0.01, 0.00667, 0.005, 0.0025, 0.002, 0.0004, 0.0002]
    # should raise
    with pytest.raises(ValueError):
        rustalgos.pair_distances_betas_time(config.SPEED_M_S, None, None, None)
    with pytest.raises(ValueError):
        rustalgos.pair_distances_betas_time(config.SPEED_M_S, distances, betas, None)
    with pytest.raises(ValueError):
        rustalgos.pair_distances_betas_time(config.SPEED_M_S, distances, None, minutes)
    with pytest.raises(ValueError):
        rustalgos.pair_distances_betas_time(config.SPEED_M_S, None, betas, minutes)
    with pytest.raises(ValueError):
        rustalgos.pair_distances_betas_time(config.SPEED_M_S, distances, betas, minutes)
    # should match
    ds, bs, ts = rustalgos.pair_distances_betas_time(config.SPEED_M_S, distances, None, None)
    assert np.allclose(ds, distances, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(bs, betas, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(ts, seconds, atol=config.ATOL, rtol=config.RTOL)
    #
    ds, bs, ts = rustalgos.pair_distances_betas_time(config.SPEED_M_S, None, betas, None)
    assert np.allclose(ds, distances, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(bs, betas, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(ts, seconds, atol=config.ATOL, rtol=config.RTOL)
    #
    ds, bs, ts = rustalgos.pair_distances_betas_time(config.SPEED_M_S, None, None, minutes)
    assert np.allclose(ds, distances, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(bs, betas, atol=config.ATOL, rtol=config.RTOL)
    assert np.allclose(ts, seconds, atol=config.ATOL, rtol=config.RTOL)


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
        for dist, beta, max_wt in zip(distances, betas, max_curve_wts, strict=True):
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
