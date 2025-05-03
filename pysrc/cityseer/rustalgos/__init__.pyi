"""Common geometry and mathematical utilities."""

# Declare the existence of submodules for type checkers
from . import centrality as centrality
from . import data as data
from . import diversity as diversity
from . import graph as graph
from . import viewshed as viewshed

def check_numerical_data(data_arr: list[float]) -> None:
    """
    Checks the integrity of a numerical data array.
    data_arr: list[float]
    """
    ...

def distances_from_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[int]:
    r"""
    Map distance thresholds $d_{max}$ to equivalent decay parameters $\beta$ at the specified cutoff weight $w_{min}$.

    See [`distance_from_beta`](#distance-from-beta) for additional discussion.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    Parameters
    ----------
    distance: list[int]
        $d_{max}$ value/s to convert to decay parameters $\beta$.
    min_threshold_wt: float
        The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

    Returns
    -------
    list[float]
        A numpy array of decay parameters $\beta$.

    Examples
    --------
    ```python
    from cityseer.metrics import networks

    # a list of betas
    distances = [400, 200]
    # convert to betas
    betas = networks.beta_from_distance(distances)
    print(betas)  # prints: array([0.01, 0.02])
    ```

    Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
    the `distances` parameter, then this function will be called in order to extrapolate the decay parameters
    implicitly, using:

    $$\beta = -\frac{log(w_{min})}{d_{max}}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $\beta$ parameters, for
    example:

    | $d_{max}$ | $\beta$ |
    |:---------:|:-------:|
    | 200m | 0.02 |
    | 400m | 0.01 |
    | 800m | 0.005 |
    | 1600m | 0.0025 |

    """
    ...

def betas_from_distances(distances: list[int], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Map decay parameters $\beta$ to equivalent distance thresholds $d_{max}$ at the specified cutoff weight $w_{min}$.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    Parameters
    ----------
    betas: list[float]
        $\beta$ value/s to convert to distance thresholds $d_{max}$.
    min_threshold_wt: float | None
        An optional cutoff weight $w_{min}$ at which to set the distance threshold $d_{max}$.

    Returns
    -------
    distances: list[int]
        A list of distance thresholds $d_{max}$.

    Examples
    --------
    ```python
    from cityseer import rustalgos

    # a list of betas
    betas = [0.01, 0.02]
    # convert to distance thresholds
    d_max = rustalgos.distances_from_betas(betas)
    print(d_max)
    # prints: [400, 200]
    ```

    Weighted measures such as the gravity index, weighted betweenness, and weighted land-use accessibilities are
    computed using a negative exponential decay function in the form of:

    $$weight = exp(-\beta \cdot distance)$$

    The strength of the decay is controlled by the $\beta$ parameter, which reflects a decreasing willingness to walk
    correspondingly farther distances. For example, if $\beta=0.005$ were to represent a person's willingness to walk
    to a bus stop, then a location 100m distant would be weighted at 60% and a location 400m away would be weighted at
    13.5%. After an initially rapid decrease, the weightings decay ever more gradually in perpetuity; thus, once a
    sufficiently small weight is encountered it becomes computationally expensive to consider locations any farther
    away. The minimum weight at which this cutoff occurs is represented by $w_{min}$, and the corresponding maximum
    distance threshold by $d_{max}$.

    ![Example beta decays](/images/betas.png)

    Most `networks` module methods can be invoked with either `distances` or `betas` parameters, but not both. If using
    the `betas` parameter, then this function will be called in order to extrapolate the distance thresholds implicitly,
    using:

    $$d_{max} = \frac{log(w_{min})}{-\beta}$$

    The default `min_threshold_wt` of $w_{min}=0.01831563888873418$ yields conveniently rounded $d_{max}$ walking
    thresholds, for example:

    | $\beta$ | $d_{max}$ |
    |:-------:|:---------:|
    | 0.02 | 200m |
    | 0.01 | 400m |
    | 0.005 | 800m |
    | 0.0025 | 1600m |

    Overriding the default $w_{min}$ will adjust the $d_{max}$ accordingly.

    """
    ...

def distances_from_seconds(
    seconds: list[int],
    speed_m_s: float,
) -> list[int]:
    r"""
    Map seconds to equivalent distance thresholds $d_{max}$.

    :::note
    It is generally not necessary to utilise this function directly.
    :::

    The default `speed_m_s` of $1.333$ yields the following $d_{max}$ walking thresholds:

    | $seconds$ | $d_{max}$ |
    |:-------:|:---------:|
    | 300 | 400m |
    | 600 | 800m |
    | 1200 | 1600m |

    Setting the `speed_m_s` to a higher or lower number will affect the $d_{max}$ accordingly.

    Parameters
    ----------
    seconds: list[int]
        Seconds to convert to distance thresholds $d_{max}$.
    speed_m_s: float
        The walking speed in metres per second.

    Returns
    -------
    list[int]
        A numpy array of distance thresholds $d_{max}$.

    """
    ...

def seconds_from_distances(
    distances: list[int],
    speed_m_s: float,
) -> list[int]:
    r"""
    Map distances into equivalent walking time in seconds.

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
        Distances to convert to seconds.
    speed_m_s: float
        The walking speed in metres per second.

    Returns
    -------
    list[int]
        A numpy array of walking time in seconds.

    """
    ...

def pair_distances_betas_time(
    speed_m_s: float,
    distances: list[int] | None = None,
    betas: list[float] | None = None,
    minutes: list[float] | None = None,
    min_threshold_wt: float | None = None,
) -> tuple[list[int], list[float], list[int]]:
    r"""
    Pair distances, betas, and time, where only one parameter is provided.

    Parameters
    ----------
    speed_m_s: float
        The default walking speed in meters per second can optionally be overridden to configure the distances covered
        by the respective walking times.
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `betas` or `minutes` parameter must be provided instead.
    betas: tuple[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distances` or `minutes` parameter must be provided instead.
    minutes: list[float]
        Walking times in minutes to be used for calculations. The `distance` and `beta` parameters will be determined
        implicitly. If the `minutes` parameter is not provided, then the `distances` or `betas` parameters must be
        provided instead.
    min_threshold_wt: float
        The default `min_threshold_wt` parameter can be overridden to generate custom mappings between the
        `distance` and `beta` parameters. See [`distance_from_beta`](#distance-from-beta) for more information.

    Returns
    -------
    distances: list[int]
        Distances corresponding to the local $d_{max}$ thresholds to be used for calculations. The $\beta$ parameters
        (for distance-weighted metrics) will be determined implicitly. If the `distances` parameter is not provided,
        then the `beta` parameter must be provided instead.
    betas: list[float]
        A $\beta$, or array of $\beta$ to be used for the exponential decay function for weighted metrics. The
        `distance` parameters for unweighted metrics will be determined implicitly. If the `betas` parameter is not
        provided, then the `distance` parameter must be provided instead.
    seconds: list[int]
        Walking times in seconds corresponding to the distances used for calculations.

    Examples
    --------
    :::warning
    Networks should be buffered according to the largest distance threshold that will be used for analysis. This
    protects nodes near network boundaries from edge falloffs. Nodes outside the area of interest but within these
    buffered extents should be set to 'dead' so that centralities or other forms of measures are not calculated.
    Whereas metrics are not calculated for 'dead' nodes, they can still be traversed by network analysis algorithms
    when calculating shortest paths and landuse accessibilities.
    :::

    """
    ...

def avg_distances_for_betas(betas: list[float], min_threshold_wt: float | None = None) -> list[float]:
    r"""
    Calculate the mean distance for a given $\beta$ parameter.

    Parameters
    ----------
    beta: list[float]
        $\beta$ representing a spatial impedance / distance decay for which to compute the average walking distance.
    min_threshold_wt: float
        The cutoff weight $w_{min}$ on which to model the decay parameters $\beta$.

    Returns
    -------
    list[float]
        The average walking distance for a given $\beta$.

    Examples
    --------
    ```python
    from cityseer.metrics import networks
    import numpy as np

    distances = [100, 200, 400, 800, 1600]
    print("distances", distances)
    # distances [ 100  200  400  800 1600]

    betas = networks.beta_from_distance(distances)
    print("betas", betas)
    # betas [0.04   0.02   0.01   0.005  0.0025]

    print("avg", networks.avg_distance_for_beta(betas))
    # avg [ 35.11949  70.23898 140.47797 280.95593 561.91187]
    ```

    """
    ...

def clip_wts_curve(distances: list[int], betas: list[float], spatial_tolerance: int) -> list[float]:
    r"""
    Calculate the upper bounds for clipping weights produced by spatial impedance functions.

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
        An array of distances corresponding to the local $d_{max}$ thresholds to be used for calculations.
    betas: list[float]
        An array of $\beta$ to be used for the exponential decay function for weighted metrics.
    spatial_tolerance: int
        The spatial buffer distance corresponding to the tolerance for spatial inaccuracy.

    Returns
    -------
    max_curve_wts: list[float]
        An array of maximum weights at which curves for corresponding $\beta$ will be clipped.

    """
    ...

def clipped_beta_wt(beta: float, max_curve_wt: float, data_dist: float) -> float: ...
