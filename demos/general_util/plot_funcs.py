from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import minmax_scale

template_cmap = LinearSegmentedColormap.from_list("reds", ["#FAFAFA", "#9a0007", "#ff6659", "#d32f2f"])


def plt_setup():
    """Flush previous matplotlib invocations."""
    plt.close("all")
    plt.cla()
    plt.clf()
    mpl.rcdefaults()  # resets seaborn
    mpl_rc_path = Path(Path.cwd() / "./matplotlib.rc")
    mpl.rc_file(mpl_rc_path)


def _dynamic_view_extent(fig, ax, km_per_inch: float, centre: tuple):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    width_m = width * km_per_inch * 1000
    height_m = height * km_per_inch * 1000
    x_left = centre[0] - width_m / 2
    x_right = centre[0] + width_m / 2
    y_bottom = centre[1] - height_m / 2
    y_top = centre[1] + height_m / 2

    return x_left, x_right, y_bottom, y_top


def _view_idx(xs, ys, x_left, x_right, y_bottom, y_top):
    select = xs > x_left
    select = np.logical_and(select, xs < x_right)
    select = np.logical_and(select, ys > y_bottom)
    select = np.logical_and(select, ys < y_top)
    select_idx = np.where(select)[0]

    return select_idx


def _prepare_v(vals):
    # don't reshape distribution: emphasise larger values if necessary using exponential
    # i.e. amplify existing distribution rather than using a reshaped normal or uniform distribution
    # clip out outliers
    vals = np.clip(vals, np.nanpercentile(vals, 0.1), np.nanpercentile(vals, 99.9))
    # scale colours to [0, 1]
    vals = minmax_scale(vals, feature_range=(0, 1))
    return vals


def plot_scatter(
    fig,
    ax,
    xs,
    ys,
    vals=None,
    bbox_extents: tuple[int, int, int, int] = None,
    centre: tuple[int, int] = (532000, 183000),
    km_per_inch=4,
    s_min=0,
    s_max=0.6,
    c_exp=1,
    s_exp=1,
    cmap=None,
    rasterized=True,
    **kwargs,
):
    """ """
    if vals is not None and vals.ndim == 2:
        raise ValueError("Please pass a single dimensional array")
    if cmap is None:
        cmap = template_cmap
    # get extents relative to centre and ax size
    if bbox_extents:
        print("Found bbox extents, ignoring centre")
        y_bottom, x_left, y_top, x_rightpl = bbox_extents
    else:
        x_left, x_right, y_bottom, y_top = _dynamic_view_extent(fig, ax, km_per_inch, centre=centre)
    select_idx = _view_idx(xs, ys, x_left, x_right, y_bottom, y_top)
    if "c" in kwargs and isinstance(kwargs["c"], (list, tuple, np.ndarray)):
        c = np.array(kwargs["c"])
        kwargs["c"] = c[select_idx]
    elif "c" in kwargs and isinstance(kwargs["c"], str):
        pass
    elif vals is not None:
        v = _prepare_v(vals)
        # apply exponential - still [0, 1]
        c = v**c_exp
        kwargs["c"] = c[select_idx]
    if "s" in kwargs and isinstance(kwargs["c"], (list, tuple, np.ndarray)):
        s = np.array(kwargs["c"])
        kwargs["s"] = s[select_idx]
    elif vals is not None:
        v = _prepare_v(vals)
        s = v**s_exp
        # rescale s to [s_min, s_max]
        s = minmax_scale(s, feature_range=(s_min, s_max))
        kwargs["s"] = s[select_idx]
    im = ax.scatter(
        xs[select_idx], ys[select_idx], linewidths=0, edgecolors="none", cmap=cmap, rasterized=rasterized, **kwargs
    )
    ax.set_xlim(left=x_left, right=x_right)
    ax.set_ylim(bottom=y_bottom, top=y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    return im


def plot_heatmap(
    heatmap_ax,
    heatmap: npt.NDArray[np.float32] = None,
    row_labels: list = None,
    col_labels: list = None,
    set_row_labels: bool = True,
    set_col_labels: bool = True,
    constrain: tuple = (-1, 1),
    text: npt.NDArray[np.float32] = None,
    cmap=None,
    grid_fontsize="x-small",
):
    """
    Modified to permit text only plots
    """
    if heatmap is None and text is None:
        raise ValueError("Pass either a heatmap or a text grid as a parameter")
    if set_row_labels and row_labels is None:
        raise ValueError("Pass row labels if setting to True")
    if set_col_labels and col_labels is None:
        raise ValueError("Pass column labels if setting to True")
    # plot
    if heatmap is not None:
        if cmap is None:
            cmap = template_cmap
        im = heatmap_ax.imshow(heatmap, cmap=cmap, vmin=constrain[0], vmax=constrain[1], origin="upper")
    # when doing text only plots, use a sham heatmap plot
    else:
        arr = np.full((len(row_labels), len(col_labels), 3), 1.0)
        im = heatmap_ax.imshow(arr, origin="upper")
    # set axes
    if row_labels is not None:
        heatmap_ax.set_yticks(list(range(len(row_labels))))
    if col_labels is not None:
        heatmap_ax.set_xticks(list(range(len(col_labels))))
    # row labels
    if row_labels is not None and set_row_labels:
        y_labels = [str(l) for l in row_labels]
        heatmap_ax.set_yticklabels(y_labels, rotation="horizontal")
    else:
        heatmap_ax.set_yticklabels([])
    # col labels
    if col_labels is not None and set_col_labels:
        x_labels = [str(l) for l in col_labels]
        heatmap_ax.set_xticklabels(x_labels, rotation="vertical")
    else:
        heatmap_ax.set_xticklabels([])
    # move x ticks to top
    heatmap_ax.xaxis.tick_top()
    # text
    if text is not None:
        for row_idx in range(text.shape[0]):
            for col_idx in range(text.shape[1]):
                t = text[row_idx][col_idx]
                c = "black"
                if heatmap is not None:
                    v = heatmap[row_idx][col_idx]
                    if isinstance(t, float):
                        t = round(t, 3)
                    # use white colour on darker backgrounds
                    if abs(v) > 0.5:
                        c = "w"
                heatmap_ax.text(
                    col_idx,
                    row_idx,
                    t,
                    ha="center",
                    va="center",
                    color=c,
                    fontsize=grid_fontsize,
                )
    return im
