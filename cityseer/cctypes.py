from __future__ import annotations

from typing import Union

import numpy as np
import numpy.typing as npt

DistancesType = Union[
    int, float, list[Union[int, float]], tuple[Union[int, float]], npt.NDArray[Union[np.int_, np.float32]]
]
BetasType = Union[float, list[float], tuple[float], npt.NDArray[np.float32]]
QsType = Union[  # pylint: disable=invalid-name
    int,
    float,
    Union[list[int], list[float]],
    Union[tuple[int], tuple[float]],
    Union[npt.NDArray[np.int_], npt.NDArray[np.float32]],
    None,
]
