# pyright: basic
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from cityseer.algos import checks
from cityseer.metrics import layers, networks
from cityseer.tools import mock


def test_check_numerical_data():
    mock_numerical = mock.mock_numerical_data(50)

    # check for malformed data
    # difficult to catch int arrays without running into numba type checking errors
    # single dimension
    with pytest.raises(ValueError):
        corrupt_numerical = mock_numerical[0]
        assert corrupt_numerical.ndim == 1
        checks.check_numerical_data(corrupt_numerical)
    # catch infinites
    with pytest.raises(ValueError):
        mock_numerical[0][0] = np.inf
        checks.check_numerical_data(mock_numerical)


def test_check_categorical_data():
    mock_categorical = mock.mock_categorical_data(50)
    _data_classes, data_encoding = layers.encode_categorical(mock_categorical)
    # check for malformed data
    # negatives
    with pytest.raises(ValueError):
        data_encoding[0] = -1
        checks.check_categorical_data(data_encoding)
    # NaN
    with pytest.raises(ValueError):
        data_encoding[0] = np.nan
        checks.check_categorical_data(data_encoding)
    # floats
    with pytest.raises(ValueError):
        data_encoding_float: npt.NDArray[np.float32] = np.full(data_encoding.shape[0], np.nan, np.float32)
        data_encoding_float[:] = data_encoding[:].astype(float)
        data_encoding_float[0] = 1.2345
        checks.check_categorical_data(data_encoding_float)


def test_check_distances_and_betas():
    betas: npt.NDArray[np.float32] = np.array([0.02, 0.01, 0.005, 0.0025, 0.0], np.float32)
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
