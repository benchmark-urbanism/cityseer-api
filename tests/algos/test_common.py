# pyright: basic
from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pytest

from cityseer import config
from cityseer.algos import common
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


def test_check_distances_and_betas():
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025], np.float32)
    distances = networks.distance_from_beta(betas)
    # zero length arrays
    with pytest.raises(ValueError):
        common.check_distances_and_betas(np.array([]), betas)
    with pytest.raises(ValueError):
        common.check_distances_and_betas(distances, np.array([]))
    # mismatching array lengths
    with pytest.raises(ValueError):
        common.check_distances_and_betas(np.array(distances[:-1]), betas)
    with pytest.raises(ValueError):
        common.check_distances_and_betas(distances, betas[:-1])
    # check that duplicates are caught
    dup_betas = np.array([0.02, 0.02])
    dup_distances = np.array(networks.distance_from_beta(dup_betas))
    with pytest.raises(ValueError):
        common.check_distances_and_betas(dup_distances, dup_betas)
    # negative values of beta
    betas_pos = betas.copy()
    betas_pos[0] = -4
    with pytest.raises(ValueError):
        common.check_distances_and_betas(distances, betas_pos)
    # negative values of distance
    distances_neg = distances.copy()
    distances_neg[0] = -100
    with pytest.raises(ValueError):
        common.check_distances_and_betas(distances_neg, betas)
    # inconsistent distances <-> betas
    betas[1] = 0.03
    with pytest.raises(ValueError):
        common.check_distances_and_betas(distances, betas)


def test_clipped_beta_wt():
    distances = np.array([400, 800, 1600], dtype=np.int_)
    betas = networks.beta_from_distance(distances)
    # try a range of spatial tolerances
    for spatial_tolerance in [0, 50, 400]:
        # calculate curve thresholds
        max_curve_wts = networks.clip_weights_curve(distances, betas, spatial_tolerance)
        # iter
        for dist, beta, max_wt in zip(distances, betas, max_curve_wts):
            # try a range of datapoint distances
            for data_dist in np.array([0, 25, 50, 100, 400, 800, 1600], dtype=np.int_):
                # continue if data distance exceeds current distance threshold - these are ignored in computations
                if data_dist > dist:
                    continue
                # calculate raw vs. clipped
                wt_raw = np.exp(-beta * data_dist)
                wt_clip = common.clipped_beta_wt(beta, max_wt, data_dist)
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
