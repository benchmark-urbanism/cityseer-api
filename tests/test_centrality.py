import pytest
import numpy as np
from cityseer import centrality


def test_custom_decay_betas():

    assert centrality.custom_betas(0.04) == np.array([100])

    assert centrality.custom_betas(0.0025) == np.array([1600])

    assert centrality.custom_betas(0.04, threshold_weight=0.001).round(8) == np.array([172.69388197]).round(8)

    assert np.array_equal(centrality.custom_betas([0.04, 0.0025]), np.array([100, 1600]))

    with pytest.raises(ValueError):
        centrality.custom_betas(-0.04)