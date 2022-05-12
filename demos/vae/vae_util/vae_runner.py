"""
"""
import json
import logging
import pathlib
from datetime import datetime

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mse_sum(y_true, y_pred):
    # sum over dimensions
    rl = K.sum(K.square(y_pred - y_true), axis=-1)
    # take mean over samples
    return K.mean(rl)


class Trainer:
    def __init__(
        self,
        model=None,
        X_samples=None,
        labels=None,
        distances=None,
        logs_path=None,
        epochs=1,
        batch=256,
        lr=1e-3,
        clip_norm=1.0,
        best_loss=False,
        save_path: pathlib.Path = None,
        test_indices: npt.NDArray[np.float32] = None,
        shuffle_seed: int = 0,
    ):
        self.model = model
        self.labels = labels
        self.distances = distances
        self.epochs = epochs
        self.batch = batch
        self.best_loss = best_loss
        self.train_loss = metrics.Mean(name="train_loss", dtype=np.float32)
        self.val_loss = metrics.Mean(name="val_loss", dtype=np.float32)
        self.history: dict[str, list[float]] = {}
        # optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=clip_norm)
        # logging
        self.writer = None
        if logs_path is not None:
            path = f'{datetime.now().strftime("%Hh%Mm%Ss")}_{model.theme}_e{epochs}_b{batch}_lr{lr}'
            path = str(logs_path / f"{path}")
            logger.info(f"Tensorboard log directory: {path}")
            self.writer = tf.summary.create_file_writer(path)
        # saving
        self.save_path = None
        self.save_path_history = None
        if save_path is not None:
            if not save_path.exists():
                save_path.mkdir(exist_ok=True, parents=True)
            self.save_path = str(pathlib.Path(save_path / "weights"))
            self.save_path_history = str(pathlib.Path(save_path / "history.json"))
        # setup datasets
        self.X_train = X_samples[~test_indices]
        self.X_val = X_samples[test_indices]
        # training dataset
        training_dataset = tf.data.Dataset.from_tensor_slices(self.X_train)
        training_dataset = training_dataset.shuffle(
            buffer_size=self.X_train.shape[0],
            reshuffle_each_iteration=True,
            seed=shuffle_seed,
        )
        self.training_dataset = training_dataset.batch(batch, drop_remainder=False)
        # validation dataset
        validation_dataset = tf.data.Dataset.from_tensor_slices(self.X_val)
        validation_dataset = validation_dataset.shuffle(
            buffer_size=self.X_val.shape[0],
            reshuffle_each_iteration=False,
            seed=shuffle_seed,
        )
        self.validation_dataset = validation_dataset.batch(batch, drop_remainder=False)

    @tf.function
    def training_step(self, batch_x, training):
        with tf.GradientTape() as tape:
            self.model(batch_x, training=training)
            loss = sum(self.model.losses)
        # Update the weights of the VAE.
        grads = tape.gradient(loss, self.model.trainable_weights)
        # guards against exploding gradients
        # grads, global_norm = tf.clip_by_global_norm(grads, self.max_global_norm)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        if training:
            self.train_loss(loss)
        else:
            self.val_loss(loss)

    def train(self, crank=None):
        # process epochs
        least_loss = np.inf
        best_epoch = 0
        batch_counter = 0  # track aggregate counter over different batches
        n_iters = self.epochs
        if crank:
            n_iters = crank
        for epoch_step in range(n_iters):
            logger.info(f"Epoch: {epoch_step + 1}")
            # setup progress bar
            progress_bar = tf.keras.utils.Progbar(self.X_train.shape[0])
            # run custom epoch setup steps
            self.epoch_setup(epoch_step)
            # reset metrics states
            self.train_loss.reset_states()
            self.reset_summary_metrics()
            # run batches
            for batch_step, training_batch in enumerate(self.training_dataset.as_numpy_iterator()):
                # iterate
                self.training_step(training_batch, training=True)
                if not np.isfinite(self.train_loss.result()):
                    logger.warning(f"Invalid loss encountered on batch step {batch_step}")
                    break
                # custom batch operations
                if batch_step % 25 == 0:
                    # process batch level tensorboard
                    self.batch_writes(batch_counter)
                    self.reset_summary_metrics()
                # progress
                progress_bar.add(self.batch, values=[("loss", self.train_loss.result())])
                batch_counter += 1
            # epoch steps
            if not np.isfinite(self.train_loss.result()):
                break
            # reset metrics states
            self.val_loss.reset_states()
            self.reset_summary_metrics()
            # compute validation loss
            for validation_batch in self.validation_dataset.as_numpy_iterator():
                self.training_step(validation_batch, training=False)
            if not np.isfinite(self.val_loss.result()):
                logger.warning(
                    f"Step: {epoch_step + 1}: non finite validation loss encountered: " f"{self.val_loss.result()}"
                )
            else:
                logger.info(f"Step: {epoch_step + 1}: validation loss: {self.val_loss.result()}")
                # process epoch level tensorboard
                self.epoch_writes(epoch_step)
            # write validation history
            self.update_history()
            # save if best weights
            if self.save_path is not None and self.best_loss and self.val_loss.result() < least_loss:
                logger.info(f"Updating least loss - saving to {self.save_path}")
                least_loss = self.val_loss.result()
                best_epoch = epoch_step + 1
                self.model.save_weights(self.save_path, overwrite=True)
        # finalise weights
        if not np.isfinite(self.train_loss.result()):
            logger.error(f"Invalid loss encountered: {self.train_loss.result()}")
        elif self.save_path is not None:
            if self.best_loss:
                logger.info(f"Best loss {least_loss:.2f} from epoch step: {best_epoch}.")
            else:
                logger.info("Saving last weights")
                self.model.save_weights(self.save_path, overwrite=True)
            # save history
            with open(self.save_path_history, "w") as json_out:
                json.dump(self.history, json_out)

    def reset_summary_metrics(self):
        if hasattr(self.model, "summary_metrics"):
            for metric_val in self.model.summary_metrics.values():
                metric_val.reset_states()

    def update_history(self):
        if "val_loss" not in self.history:
            self.history["train_loss"] = [float(self.train_loss.result())]
            self.history["val_loss"] = [float(self.val_loss.result())]
        else:
            self.history["train_loss"].append(float(self.train_loss.result()))
            self.history["val_loss"].append(float(self.val_loss.result()))
        # add metrics - this should be based on validation step
        if hasattr(self.model, "summary_metrics"):
            for metric_name, metric_val in self.model.summary_metrics.items():
                target_name = f"val_{metric_name}"
                if target_name not in self.history:
                    self.history[target_name] = [float(metric_val.result().numpy())]
                else:
                    self.history[target_name].append(float(metric_val.result().numpy()))

    def batch_writes(self, batch_step):
        # log scalars
        if self.writer is not None:
            with self.writer.as_default():
                tf.summary.scalar("training loss", self.train_loss.result(), step=batch_step)
                if hasattr(self.model, "summary_metrics"):
                    for metric_name, metric_val in self.model.summary_metrics.items():
                        tf.summary.scalar(metric_name, metric_val.result(), step=batch_step)

    def epoch_setup(self, epoch_step):
        pass

    def epoch_writes(self, epoch_step):
        if self.writer is not None:
            with self.writer.as_default():
                # write validation loss
                tf.summary.scalar("validation loss", self.val_loss.result(), step=epoch_step)
                if hasattr(self.model, "summary_metrics"):
                    for metric_name, metric_val in self.model.summary_metrics.items():
                        target_name = f"val_{metric_name}"
                        tf.summary.scalar(target_name, metric_val.result(), step=epoch_step)


class VAE_trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def epoch_setup(self, epoch_step):
        # update capacity term
        self.model.kl_divergence.capacity_update(epoch_step)

    def epoch_writes(self, epoch_step):
        if self.writer is not None:
            with self.writer.as_default():
                super().epoch_writes(epoch_step)
                # histograms
                if hasattr(self.model, "sampling"):
                    tf.summary.histogram(
                        "Z mu biases",
                        self.model.sampling.Z_mu_layer.weights[1],
                        step=epoch_step,
                    )
                    tf.summary.histogram(
                        "Z mu weights",
                        self.model.sampling.Z_mu_layer.weights[0],
                        step=epoch_step,
                    )
                    tf.summary.histogram(
                        "Z logvar biases",
                        self.model.sampling.Z_logvar_layer.weights[1],
                        step=epoch_step,
                    )
                    tf.summary.histogram(
                        "Z logvar weights",
                        self.model.sampling.Z_logvar_layer.weights[0],
                        step=epoch_step,
                    )
