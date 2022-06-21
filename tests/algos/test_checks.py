# pyright: basic
from __future__ import annotations

import numba
import numpy as np
import numpy.typing as npt
import pytest

from cityseer.algos import checks
from cityseer.metrics import layers, networks
from cityseer.tools import mock


def test_check_numerical_data(primal_graph):
    mock_numerical = mock.mock_numerical_data(primal_graph)
    # check for malformed data
    # single dimension
    with pytest.raises(ValueError):
        corrupt_numerical = mock_numerical["mock_numerical_1"].values
        assert corrupt_numerical.ndim == 1
        checks.check_numerical_data(corrupt_numerical)
    # catch infinites
    with pytest.raises(ValueError):
        mock_numerical.at[0, "mock_numerical_1"] = np.inf
        checks.check_numerical_data(mock_numerical["mock_numerical_1"].values)


def test_check_categorical_data(primal_graph):
    mock_categorical = mock.mock_landuse_categorical_data(primal_graph)
    # check for malformed data
    corrupt_categorical = mock_categorical.categorical_landuses.values.copy()
    # negatives
    with pytest.raises((ValueError, numba.TypingError)):
        corrupt_categorical[0] = -1
        checks.check_categorical_data(corrupt_categorical)
    # NaN
    with pytest.raises((ValueError, numba.TypingError)):
        corrupt_categorical[0] = np.nan
        checks.check_categorical_data(corrupt_categorical)
    # floats
    with pytest.raises((ValueError, numba.TypingError)):
        checks.check_categorical_data(mock_categorical.categorical_landuses.values.astype(np.float_))


def test_check_distances_and_betas():
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025], np.float32)
    distances = networks.distance_from_beta(betas)
    # zero length arrays
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array([]), betas)
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, np.array([]))
    # mismatching array lengths
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(np.array(distances[:-1]), betas)
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas[:-1])
    # check that duplicates are caught
    dup_betas = np.array([0.02, 0.02])
    dup_distances = np.array(networks.distance_from_beta(dup_betas))
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(dup_distances, dup_betas)
    # negative values of beta
    betas_pos = betas.copy()
    betas_pos[0] = -4
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas_pos)
    # negative values of distance
    distances_neg = distances.copy()
    distances_neg[0] = -100
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances_neg, betas)
    # inconsistent distances <-> betas
    betas[1] = 0.03
    with pytest.raises(ValueError):
        checks.check_distances_and_betas(distances, betas)
