import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .columns import cols_cens, cols_cent, cols_lu, labels_cens, labels_cent, labels_lu, template_distances

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

data_path = Path(Path.cwd() / "data/")
weights_path = Path(Path.cwd() / "temp/weights/")
weights_path.mkdir(parents=True, exist_ok=True)


def generate_theme(
    df: pd.DataFrame,
    theme: str,
    bandwise: bool = False,
    add_city_pop_id: bool = False,
    max_dist: int = None,
    verbose: bool = False,
):
    """
    Optional use of distance-wise bands gives slightly more defined delineations for latent dimensions.
    """
    df_copy = df.copy(deep=True)
    if max_dist is None:
        distances = [d for d in template_distances]
    else:
        distances = [d for d in template_distances if d <= max_dist]
    if theme == "all":
        columns = cols_cent + cols_lu + cols_cens
        labels = labels_cent + labels_lu + labels_cens
    elif theme == "cent":
        columns = cols_cent
        labels = labels_cent
    elif theme == "lu":
        columns = cols_lu
        labels = labels_lu
    elif theme == "cens":
        columns = cols_cens
        labels = labels_cens
    else:
        raise ValueError("Invalid theme specified for data theme.")
    if add_city_pop_id:
        # unpack the columns by distances and fetch the data
        # first add generic (non-distance) columns
        labels = ["City Population ID"] + labels
        selected_columns = ["city_pop_id"]
    else:
        selected_columns = []
    # unpack the columns by distances and unpack the data
    for column in columns:
        for d in distances:
            selected_columns.append(f"{column}_{d}")
    # if not bandwise, simply return distance based columns as they are
    if not bandwise:
        X = df_copy.loc[:, selected_columns]
    # but if bandwise, first subtract foregoing distances
    else:
        print("Generating bandwise data")
        for column in columns:
            for d in distances:
                if verbose:
                    print(f"Current distance leading edge: {d}m")
                d_idx = distances.index(d)
                if d_idx == 0:
                    if verbose:
                        print(f"No trailing edge for distance {d}m")
                # subsequent bands subtract the prior band
                else:
                    lag_idx = d_idx - 1
                    lag_dist = distances[lag_idx]
                    if verbose:
                        print(f"Trailing edge: {lag_dist}m, {column}")
                    lead_col = f"{column}_{d}"
                    lag_col = f"{column}_{lag_dist}"
                    df_copy.loc[:, lead_col] = df.loc[:, lead_col] - df.loc[:, lag_col]
        # edited necessary columns in place, only pass those columns back
        X = df_copy.loc[:, selected_columns]
    # add the interpolated census cols
    # if theme in ['all', 'cens']:
    #     X.loc[:, cols_cens_interp] = df.loc[:, cols_cens_interp]

    return X, distances, labels


def train_test_idxs(df, mod):
    """
    Modulo of 300 gives about 11%
    Whereas 200 gives about 25%
    """
    xs = np.copy(df.x)
    xs /= 100
    xs = np.round(xs)
    xs *= 100
    xs_where = xs % mod == 0
    ys = np.copy(df.y)
    ys /= 100
    ys = np.round(ys)
    ys *= 100
    ys_where = ys % mod == 0
    xy_where = np.logical_and(xs_where, ys_where)
    print(f"Isolated {xy_where.sum() / len(df):.2%} test samples.")
    return xy_where
