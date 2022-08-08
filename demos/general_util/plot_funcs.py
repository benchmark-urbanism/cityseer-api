from __future__ import annotations

import numpy as np

template_cmap = LinearSegmentedColormap.from_list("reds", ["#FAFAFA", "#9a0007", "#ff6659", "#d32f2f"])


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
