"""
- Encodes latents for various VAE hyperparameters and saves to data files
- Calculates UDR entangling metric and saves to data files
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, quantile_transform

from demos.general_util import plot_funcs, theme_setup
from demos.vae.vae_util import vae_model


def process_latents_vae():
    """
    Loads VAE weights from disk and saves the latents
    """
    df = pd.read_feather((theme_setup.data_path / "df_north_london.feather").resolve())
    df = df.set_index("uid")
    X_raw, distances, labels = theme_setup.generate_theme(df, "all", bandwise=True, max_dist=None)
    X_trans = StandardScaler().fit_transform(X_raw)
    # iterat the seeds and models
    seeds = list(range(10))
    latent_dim = 6
    epochs = 25
    batch = 256
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    # iterate
    for beta in betas:
        caps = [0, 4, 8, 12, 16, 20]
        if beta == 0:
            caps = [0]
        for cap in caps:
            print(f"...collecting seeds from models for beta {beta} and capacitance {cap}")
            for seed in seeds:
                # generate identifying model key
                model_key = f"VAE_d{latent_dim}_b{beta}_c{cap}_s{seed}"
                # paths
                out_path = theme_setup.weights_path / "vae/data"
                out_path.mkdir(exist_ok=True, parents=True)
                # skip if already prepared
                if Path(out_path / f"model_{model_key}_latent.npy").exists():
                    continue
                dir_path = Path(theme_setup.weights_path / f"vae/seed_{seed}/{model_key}_e{epochs}_b{batch}_train")
                # creates model
                vae = vae_model.VAE(
                    raw_dim=X_trans.shape[1],
                    latent_dim=latent_dim,
                    beta=beta,
                    capacity=cap,
                    epochs=epochs,
                    model_key=model_key,
                    seed=seed,
                )
                # load weights
                vae.load_weights(str(dir_path / "weights")).expect_partial()
                # save the latents
                Z_mu, Z_log_var, Z = vae.encode(X_trans, training=False)
                np.save(out_path / "model_{model_key}_latent", Z)
                np.save(out_path / "model_{model_key}_z_mu", Z_mu)
                np.save(out_path / "model_{model_key}_z_log_var", Z_log_var)


def generate_udr_grid(
    latent_dim: int,
    seeds: list[int],
    kl_threshold: float = 0.01,
    random_state=np.random.RandomState(0),
):
    """
    Calculate UDR and set in ndarray
    """
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    arr = np.full((len(betas), 6, len(seeds)), np.nan)
    mask_count_arr = np.full((len(betas), 6, len(seeds)), np.nan)
    for beta_idx, (beta) in enumerate(betas):
        caps = [0, 4, 8, 12, 16, 20]
        if beta == 0:
            caps = [0]
        for cap_idx, cap in enumerate(caps):
            # gather the latent representations and the latent kl divergences
            inferred_model_reps = []
            kl_vecs = []
            for seed in seeds:
                model_key = f"VAE_d{latent_dim}_b{beta}_c{cap}_s{seed}"
                print(f"...loading data for {model_key}")
                Z = np.load(theme_setup.weights_path / f"vae/data/model_{model_key}_latent.npy")
                inferred_model_reps.append(Z)
                """
                getting average kl divergence per individual latent:

                See equation 3 on page 5 of paper, which matches typical VAE closed-form equation for KL divergence...
                e.g.: https://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html
                −DKL(qϕ(z|x)||pθ(z))=1/2∑(1+log(σ2)−μ2−σ2)
                Seems that this is ordinarily summed over the latent dimensions...
                but notice that the paper is PER latent

                Further, see the code comments in udr.py where the function signature states that the kl divergence
                vector should be: "a vector of the average kl divergence per latent"
                """
                Z_mu = np.load(theme_setup.weights_path / f"vae/data/model_{model_key}_z_mu.npy")
                Z_log_var = np.load(theme_setup.weights_path / f"vae/data/model_{model_key}_z_log_var.npy")
                # single latent kl divergence, i.e. don't sum over latents
                kl_loss = -0.5 * (1 + Z_log_var - np.square(Z_mu) - np.exp(Z_log_var))
                # average kl divergence per latent
                kl_vector = np.mean(kl_loss, axis=0)
                kl_vecs.append(kl_vector)
            # filter out non finite values
            before_len = inferred_model_reps[0].shape[0]
            not_finite_idx = np.zeros(before_len, dtype=bool)
            for inferred_model in inferred_model_reps:
                for l_idx in range(latent_dim):
                    not_finite_idx = np.logical_or(not_finite_idx, ~np.isfinite(inferred_model[:, l_idx]))
            after_len = np.sum(~not_finite_idx)
            if after_len == 0:
                print(f"NO FINITE VALUES: UNABLE TO PROCESS for beta: {beta} and cap: {cap}")
                continue
            elif after_len != before_len:
                print(f"DROPPED {before_len - after_len} NON FINITE SAMPLES...")
            # filter out
            for i, inferred_model in enumerate(inferred_model_reps):
                inferred_model_reps[i] = inferred_model[~not_finite_idx, :]
                if inferred_model_reps[i].shape[0] != after_len:
                    raise ValueError("New array length doesn't match data.")
            print("...calculating UDR")
            udr = udr.compute_udr_sklearn_modified(
                inferred_model_reps,
                kl_vecs,
                random_state,
                correlation_matrix="spearman",  # lasso throws convergence errors
                filter_low_kl=True,
                include_raw_correlations=True,
                kl_filter_threshold=kl_threshold,
            )
            arr[beta_idx][cap_idx] = udr["model_scores"]
            mask_count_arr[beta_idx][cap_idx] = np.sum(np.array(kl_vecs) > kl_threshold, axis=1)

    return arr, mask_count_arr


def gather_loss(seeds, epochs, latent_dims, batch=256):
    vae_losses = [
        "train_loss",
        "val_loss",
        "val_capacity_term",
        "val_kl",
        "val_kl_beta",
        "val_kl_beta_cap",
        "val_rec_loss",
    ]
    vae_history_data = {}
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    # iterate dims, betas, caps
    for latent_dim in latent_dims:
        for beta in betas:
            caps = [0, 4, 8, 12, 16, 20]
            if beta == 0:
                caps = [0]
            for cap in caps:
                data_key = f"e{epochs}_d{latent_dim}_b{beta}_c{cap}"
                # each model gets a data_key in the dictionary - all epochs and seeds will be stored here
                vae_history_data[data_key] = {}
                for loss in vae_losses:
                    # each loss gets a nested data_key of epochs and seeds
                    vae_history_data[data_key][loss] = np.full((epochs, len(seeds)), np.nan)
                # iterate the seeds
                for seed_idx, seed in enumerate(seeds):
                    # in case not all models have completed running
                    model_key = f"VAE_d{latent_dim}_b{beta}_c{cap}_s{seed}"
                    fp = theme_setup.weights_path / f"vae/seed_{seed}/{model_key}_e{epochs}_b{batch}_train/history.json"
                    if Path(fp).is_file():
                        with open(fp) as f:
                            vae_data = json.load(f)
                            # fetch each of the losses
                            for loss in vae_losses:
                                for epoch_idx in range(epochs):
                                    try:
                                        vae_history_data[data_key][loss][epoch_idx, seed_idx] = float(
                                            vae_data[loss][epoch_idx]
                                        )
                                    except IndexError as e:
                                        print(
                                            f"data_key: {data_key}, epoch idx: {epoch_idx} unvailable for loss: {loss}"
                                        )
                    else:
                        print(f"File not found: {fp}")
    return vae_history_data


def format_text(val):
    if val < 1:
        return f"{val:^.2f}"
    elif val < 10:
        return f"{val:^.1f}"
    else:
        return f"{val:^.0f}"


def generate_heatmap(ax, theme, epochs, latent_dim, theme_data, set_row_labels, set_col_labels):
    # empty numpy array for containing
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    arr = np.full((len(betas), 6), np.nan)
    # iterate dims, betas, caps
    for row_idx, beta in enumerate(betas):
        caps = [0, 4, 8, 12, 16, 20]
        if beta == 0:
            caps = [0]
        for col_idx, cap in enumerate(caps):
            # data_key for referencing history data
            data_key = f"e{epochs}_d{latent_dim}_b{beta}_c{cap}"
            loss_dict = theme_data[data_key]
            # iterate the ax rows and ax cols
            # each loss_dict contains the loss keys with numpy values of epochs x seeds
            # take the argmin across the epoch axis then take the mean of the seeds
            # arr[row_idx][col_idx] = np.nanmean(np.nanmin(loss_dict[theme], axis=0))
            arr[row_idx][col_idx] = np.nanmean(loss_dict[theme][-1])
    # remove outliers
    scaled = arr.copy()
    # sklearn preprocessing works only per axis dimension
    scaled = scaled.reshape((-1, 1))
    scaled = quantile_transform(scaled, n_quantiles=9, output_distribution="uniform")
    scaled /= 3
    scaled = scaled.reshape((len(betas), len(caps)))
    # remove text longer than four characters
    text = arr.astype("str")
    for i in range(text.shape[0]):
        for j in range(text.shape[1]):
            if i == 0 and j > 0:
                text[i][j] = ""
            else:
                text[i][j] = format_text(arr[i][j])
    # plot
    plot_funcs.plot_heatmap(
        ax,
        heatmap=scaled,
        row_labels=[r"$\beta=" + f"{b}$" for b in betas],
        set_row_labels=set_row_labels,
        col_labels=[f"$C={c}$" for c in [0, 4, 8, 12, 16, 20]],
        set_col_labels=set_col_labels,
        text=text,
        grid_fontsize="xx-small",
    )
    if theme == "val_loss":
        ax_title = "Combined loss"
    elif theme == "val_rec_loss":
        ax_title = "MSE reconstruction"
    elif theme == "val_kl":
        ax_title = r"$D_{KL}$"
    elif theme == "val_kl_beta":
        ax_title = r"$\beta \cdot D_{KL}$"
    elif theme == "val_kl_beta_cap":
        ax_title = r"$\beta \cdot |D_{KL} - C|$"
    elif theme == "val_capacity_term":
        ax_title = "Capacity term $C$"
    ax.set_xlabel(ax_title)


def generate_arr_heatmap(ax, arr, cmap=None, constrain=(-1, 1)):
    # remove outliers
    betas = [0, 1, 2, 4, 8, 16, 32, 64]
    caps = [0, 4, 8, 12, 16, 20]
    scaled = arr.copy()
    # sklearn preprocessing works only per axis dimension
    scaled = scaled.reshape((-1, 1))
    scaled = quantile_transform(scaled, n_quantiles=9, output_distribution="uniform")
    scaled /= 3
    scaled = scaled.reshape((len(betas), len(caps)))
    # remove text longer than four characters
    text = arr.astype("str")
    for i in range(text.shape[0]):
        for j in range(text.shape[1]):
            # beta == 0
            if i == 0 and j > 0:
                text[i][j] = ""
            else:
                text[i][j] = format_text(arr[i][j])
    # plot
    plot_funcs.plot_heatmap(
        ax,
        heatmap=scaled,
        row_labels=None,
        set_row_labels=False,
        col_labels=[f"$C={c}$" for c in caps],
        set_col_labels=True,
        text=text,
        constrain=constrain,
        cmap=cmap,
        grid_fontsize="xx-small",
    )
