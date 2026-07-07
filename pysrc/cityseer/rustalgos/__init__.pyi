"""Common utilities for converting between distance, time, and decay parameters, and data validation."""

import numpy as np
import numpy.typing as npt

# Declare the existence of submodules for type checkers
from . import centrality as centrality
from . import data as data
from . import diversity as diversity
from . import graph as graph
from . import viewshed as viewshed

def check_numerical_data(data_arr: npt.NDArray[np.float32]) -> None:
    """
    Validates that all elements in a 2D numerical array are finite.

    Raises
    ------
    ValueError
        If any element is not finite (NaN or infinity).
    """
    ...

def distances_from_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[int]:
    r"""
    Convert decay parameters (betas) to distance thresholds ($d_{max}$).

    Requires betas > 0 and sorted in strictly decreasing order. Uses a default minimum weight threshold.

    Parameters
    ----------
    betas: list[float]
        $\beta$ values (> 0, strictly decreasing) to convert.
    min_threshold_wt: float | None
        Optional cutoff weight $w_{min}$ (default: ~0.0183).

    Returns
    -------
    list[int]
        Corresponding distance thresholds $d_{max}$.

    Raises
    ------
    ValueError
        If inputs are invalid (empty, non-positive, not decreasing).


    Examples
    --------
    ```python
    from cityseer import rustalgos

    betas = [0.01, 0.02]
    distances = rustalgos.distances_from_betas(betas)
    print(distances)  # prints: [400, 200]
    ```

    Uses the formula:

    $$d_{max} = \frac{log(w_{min})}{-\beta}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded distance thresholds,
    for example:

    | $\beta$ | $d_{max}$ |
    |:-------:|:---------:|
    | 0.02 | 200m |
    | 0.01 | 400m |
    | 0.005 | 800m |
    | 0.0025 | 1600m |

    """
    ...

def betas_from_distances(distances: list[int], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Convert distance thresholds ($d_{max}$) to decay parameters (betas).

    Requires distances > 0 and sorted in strictly increasing order. Uses a default minimum weight threshold.

    Parameters
    ----------
    distances: list[int]
        $d_{max}$ values (> 0, strictly increasing) to convert.
    min_threshold_wt: float | None
        Optional cutoff weight $w_{min}$ (default: ~0.0183).

    Returns
    -------
    list[float]
        Corresponding decay parameters $\beta$.

    Raises
    ------
    ValueError
        If inputs are invalid (empty, non-positive, not increasing).


    Examples
    --------
    ```python
    from cityseer import rustalgos

    distances = [200, 400, 800, 1600]
    betas = rustalgos.betas_from_distances(distances)
    print(betas)  # prints: [0.02, 0.01, 0.005, 0.0025]
    ```

    The $\beta$ parameter controls the strength of exponential distance decay:

    $$weight = exp(-\beta \cdot distance)$$

    This reflects a decreasing willingness to walk correspondingly farther distances. For example, if $\beta=0.005$
    were to represent a person's willingness to walk to a bus stop, then a location 100m distant would be weighted at
    60% and a location 400m away would be weighted at 13.5%.

    ![Example beta decays](/images/betas.png)

    Uses the formula:

    $$\beta = -\frac{log(w_{min})}{d_{max}}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ values,
    for example:

    | $d_{max}$ | $\beta$ |
    |:---------:|:-------:|
    | 200m | 0.02 |
    | 400m | 0.01 |
    | 800m | 0.005 |
    | 1600m | 0.0025 |

    Overriding the default $w_{min}$ will adjust the $\beta$ accordingly.

    """
    ...

def distances_from_seconds(
    seconds: list[int],
    speed_m_s: float,
) -> list[int]:
    r"""
    Convert time in seconds to distance thresholds ($d_{max}$) based on speed.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    The default `speed_m_s` of $1.333$ yields the following $d_{max}$ walking thresholds:

    | $seconds$ | $d_{max}$ |
    |:-------:|:---------:|
    | 300 | 400m |
    | 600 | 800m |
    | 1200 | 1600m |

    Setting the `speed_m_s` to a higher or lower number will affect the $d_{max}$ accordingly.]
    Parameters
    ----------
    seconds: list[int]
        Time values in seconds.
    speed_m_s: float
        Speed in meters per second.

    Returns
    -------
    list[int]
        Corresponding distance thresholds $d_{max}$.
    """
    ...

def seconds_from_distances(
    distances: list[int],
    speed_m_s: float,
) -> list[int]:
    r"""
    Convert distance thresholds ($d_{max}$) to time in seconds based on speed.


    :::note
    It is generally not necessary to utilise this function directly.
    :::

    The default `speed_m_s` of $1.33333$ yields the following walking times:

    | $d_{max}$ | $seconds$ |
    |:-------:|:---------:|
    | 400m | 300 |
    | 800m | 600 |
    | 1600m | 1200 |

    Setting the `speed_m_s` to a higher or lower number will affect the walking time accordingly.

    Parameters
    ----------
    distances: list[int]
        Distance thresholds $d_{max}$.
    speed_m_s: float
        Speed in meters per second.

    Returns
    -------
    list[int]
        Corresponding time values in seconds.
    """
    ...

def pair_distances_and_time(
    speed_m_s: float,
    distances: list[int] | None = None,
    minutes: list[float] | None = None,
) -> tuple[list[int], list[int]]:
    r"""
    Resolve distances and seconds from either distances or minutes.

    Exactly one of `distances` or `minutes` must be provided.

    Parameters
    ----------
    speed_m_s: float
        Walking speed in meters per second.
    distances: list[int] | None
        Distance thresholds in metres.
    minutes: list[float] | None
        Time in minutes.

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing (distances, seconds).

    Raises
    ------
    ValueError
        If not exactly one of `distances` or `minutes` is provided, or if inputs are invalid.
    """
    ...

def avg_distances_for_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Calculate the mean distance corresponding to given beta parameters.

    Parameters
    ----------
    betas: list[float]
        $\beta$ parameters.
    min_threshold_wt: float | None
        Optional cutoff weight $w_{min}$.

    Returns
    -------
    list[float]
        The average walking distance for each beta.


    Examples
    --------
    ```python
    from cityseer import rustalgos

    distances = [100, 200, 400, 800, 1600]
    betas = rustalgos.betas_from_distances(distances)
    print("betas", betas)
    # betas [0.04, 0.02, 0.01, 0.005, 0.0025]

    avg = rustalgos.avg_distances_for_betas(betas)
    print("avg", avg)
    ```

    """
    ...

def clip_wts_curve(distances: list[int], betas: list[float], spatial_tolerance: int) -> list[float]:
    r"""
    Calculate upper weight bounds for clipping distance decay curves based on spatial tolerance.

    Used when data point location has uncertainty defined by `spatial_tolerance`.
    Determine the upper weights threshold of the distance decay curve for a given $\beta$ based on the
    `spatial_tolerance` parameter. This is used by downstream functions to determine the upper extent at which weights
    derived for spatial impedance functions are flattened and normalised. This functionality is only intended for
    situations where the location of datapoints is uncertain for a given spatial tolerance.

    :::warning
    Use distance based clipping with caution for smaller distance thresholds. For example, if using a 200m distance
    threshold clipped by 100m, then substantial distortion is introduced by the process of clipping and normalising the
    distance decay curve. More generally, smaller distance thresholds should generally be avoided for situations where
    datapoints are not located with high spatial precision.
    :::

    Parameters
    ----------
    distances: list[int]
        Distance thresholds ($d_{max}$).
    betas: list[float]
        Decay parameters ($\beta$).
    spatial_tolerance: int
        Spatial buffer distance (uncertainty).

    Returns
    -------
    list[float]
        Maximum weights for clipping the decay curve for each beta.
    """
    ...

def clipped_beta_wt(beta: float, max_curve_wt: float, data_dist: float) -> float:
    r"""
    Calculate a single weight using beta decay, clipped by a maximum weight.

    Applies $weight = exp(-\beta \cdot distance)$, ensuring the result does not exceed `max_curve_wt`.

    Parameters
    ----------
    beta: float
        The decay parameter $\beta$.
    max_curve_wt: float
        The maximum allowed weight (from `clip_wts_curve`).
    data_dist: float
        The distance to the data point.

    Returns
    -------
    float
        The calculated (potentially clipped) weight. Returns 0.0 if calculation fails.
    """
    ...
