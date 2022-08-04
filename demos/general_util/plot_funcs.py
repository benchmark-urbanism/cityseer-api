from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from shapely import geometry
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

template_cmap = LinearSegmentedColormap.from_list("reds", ["#FAFAFA", "#9a0007", "#ff6659", "#d32f2f"])


def plot_scatter(
    ax: plt.Axes,
    xs: npt.NDArray[np.float_],
    ys: npt.NDArray[np.float_],
    vals: npt.NDArray[np.float32],
    bbox_extents: tuple[int, int, int, int] | tuple[float, float, float, float],
    c_min: float = 0,
    c_max: float = 1,
    c_exp: float = 1,
    s_min: float = 0,
    s_max: float = 1,
    s_exp: float = 1,
    cmap_key: str = "Reds",
    rasterized: bool = True,
):
    """ """
    # get extents relative to centre and ax size
    min_x, min_y, max_x, max_y = bbox_extents
    # filter
    select = xs > min_x
    select = np.logical_and(select, xs < max_x)
    select = np.logical_and(select, ys > min_y)
    select = np.logical_and(select, ys < max_y)
    select_idx = np.where(select)[0]
    # remove any extreme outliers
    v = np.clip(vals, np.nanpercentile(vals, 0.1), np.nanpercentile(vals, 99.9))
    # shape if wanted
    c = v**c_exp
    c: npt.NDArray[np.float_] = minmax_scale(c, feature_range=(c_min, c_max))
    s = v**s_exp
    s: npt.NDArray[np.float_] = minmax_scale(s, feature_range=(s_min, s_max))
    # plot
    im = ax.scatter(
        xs[select_idx],
        ys[select_idx],
        c=c[select_idx],
        s=s[select_idx],
        linewidths=0,
        edgecolors="none",
        cmap=plt.get_cmap(cmap_key),
        rasterized=rasterized,
    )
    # limits
    ax.set_xlim(left=min_x, right=max_x)
    ax.set_ylim(bottom=min_y, top=max_y)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect(1)
    ax.set_facecolor("white")

    return im


def plot_heatmap(
    heatmap_ax,
    heatmap: npt.NDArray[np.float32] | None = None,
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


def plot_nx_edges(
    ax: plt.Axes,
    nx_multigraph: nx.MultiGraph,
    edge_metrics_key: str,
    bbox_extents: tuple[int, int, int, int] | tuple[float, float, float, float],
    colour: str = "#ef1a33",
    rasterized: bool = True,
):
    """ """
    min_x, min_y, max_x, max_y = bbox_extents  # type: ignore
    # extract data for shaping
    edge_vals: list[str] = []
    edge_geoms: list[geometry.LineString] = []
    for _, _, edge_data in tqdm(nx_multigraph.edges(data=True)):  # type: ignore
        edge_vals.append(edge_data[edge_metrics_key])  # type: ignore
        edge_geoms.append(edge_data["geom"])  # type: ignore
    edge_vals_arr: npt.NDArray[np.float_] = np.array(edge_vals)
    edge_vals_arr = np.clip(edge_vals_arr, np.nanpercentile(edge_vals_arr, 0.1), np.nanpercentile(edge_vals_arr, 99.9))
    # plot using geoms
    n_edges = edge_vals_arr.shape[0]
    for idx in tqdm(range(n_edges)):
        xs = np.array(edge_geoms[idx].coords.xy[0])
        ys = np.array(edge_geoms[idx].coords.xy[1])
        if np.any(xs < min_x) or np.any(xs > max_x):
            continue
        if np.any(ys < min_y) or np.any(ys > max_y):
            continue
        # normalise val
        edge_val = edge_vals_arr[idx]
        norm_val = (edge_val - edge_vals_arr.min()) / (edge_vals_arr.max() - edge_vals_arr.min())
        val_shape = norm_val * 0.95 + 0.05
        ax.plot(xs, ys, linewidth=val_shape, color=colour, rasterized=rasterized)
    ax.axis("off")
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
