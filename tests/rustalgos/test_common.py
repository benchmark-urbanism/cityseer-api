# pyright: basic
from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config, rustalgos
from cityseer.metrics import networks
from cityseer.tools import mock


def test_check_numerical_data(primal_graph):
    mock_numerical = mock.mock_numerical_data(primal_graph)
    # check for malformed data
    # single dimension
    with pytest.raises(ValueError):
        corrupt_numerical = mock_numerical["mock_numerical_1"].values
        assert corrupt_numerical.ndim == 1
        common.check_numerical_data(corrupt_numerical)
    # catch infinites
    with pytest.raises(ValueError):
        mock_numerical.at[0, "mock_numerical_1"] = np.inf
        common.check_numerical_data(mock_numerical["mock_numerical_1"].values)


def test_check_categorical_data(primal_graph):
    mock_categorical = mock.mock_landuse_categorical_data(primal_graph)
    # check for malformed data
    corrupt_categorical = mock_categorical.categorical_landuses.values.copy()
    # negatives
    with pytest.raises((ValueError, numba.TypingError)):
        corrupt_categorical[0] = -1
        common.check_categorical_data(corrupt_categorical)
    # NaN
    with pytest.raises((ValueError, numba.TypingError)):
        corrupt_categorical[0] = np.nan
        common.check_categorical_data(corrupt_categorical)
    # floats
    with pytest.raises((ValueError, numba.TypingError)):
        common.check_categorical_data(mock_categorical.categorical_landuses.values.astype(np.float_))


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
    assert np.allclose(d, 172.6938934326172, atol=config.ATOL, rtol=config.RTOL)
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
    b = rustalgos.betas_from_distances([172.69388197455342], min_threshold_wt=0.001)
    assert np.allclose([b], [0.04], atol=config.ATOL, rtol=config.RTOL)
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
    for d in ([0], [-100], []):
        with pytest.raises(ValueError):
            rustalgos.betas_from_distances(d)


def test_pair_distances_and_betas():
    betas = np.array([0.02, 0.01, 0.005, 0.0025], np.float32)
    distances = rustalgos.distances_from_betas(betas)
    # should raise if both provided
    with pytest.raises(ValueError):
        rustalgos.pair_distances_and_betas(distances, betas)
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
    with pytest.raises(ValueError):
        max_curve_wts = rustalgos.clip_wts_curve(distances, betas, -1)
    # should raise if buffer_distance is greater than distances
    with pytest.raises(ValueError):
        max_curve_wts = rustalgos.clip_wts_curve(distances, betas, 401)


def test_clipped_beta_wt():
    distances = np.array([400, 800, 1600], dtype=np.int_)
    betas = rustalgos.betas_from_distances(distances)
    # try a range of spatial tolerances
    for spatial_tolerance in [0, 50, 400]:
        # calculate curve thresholds
        max_curve_wts = rustalgos.clip_wts_curve(distances, betas, spatial_tolerance)
        # iter
        for dist, beta, max_wt in zip(distances, betas, max_curve_wts):
            # try a range of datapoint distances
            for data_dist in np.array([0, 25, 50, 100, 400, 800, 1600], dtype=np.int_):
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
