import os
import json
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from flax.linen.initializers import constant, orthogonal, variance_scaling
from typing import Sequence
from flax.training.train_state import TrainState
import optax
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle
import pandas as pd
from tqdm import tqdm
import time


class NeuralNetwork(nn.Module):
    output_dim: Sequence[int]

    @nn.compact
    def __call__(self, x: Float[Array, "n_obs n_input"]) -> Float[Array, "n_obs n_output"]:

        y = nn.Dense(512, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(x)
        y = nn.relu(y)
        y = nn.Dense(256, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(128, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(64, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(32, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(self.output_dim, kernel_init=variance_scaling(scale=1.0, mode='fan_in', distribution='truncated_normal'), bias_init=constant(0.0))(y)
        y = y.reshape(-1, self.output_dim)
        
        return y


def loss_fn(state, params, x: Float[Array, "n_obs n_input"], y: Float[Array, "n_obs n_output"]) -> Float[Array, "1"]:
    y_pred = state.apply_fn(params, x)
    return optax.l2_loss(predictions=y_pred, targets=y).mean()


@jax.jit
def _epoch(runner: tuple, epoch: int) -> Tuple[tuple, float]:
    state, x_train, y_train = runner
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(state, state.params, x_train, y_train)
    state = state.apply_gradients(grads=grads)
    runner = (state, x_train, y_train)
    return runner, loss


def train(
        nn_model: nn.Module,
        x: Float[Array, "n_obs n_input"],
        y: Float[Array, "n_obs n_output"],
        n_epochs: int = 20_000,
        lr: float = 1e-4,
        path: Optional[str | Path] = None,
        verbose: bool = True
) -> Tuple[dict, list]:

    save_params = path is not None
    if save_params:
        if not isinstance(path, Path): path = Path(Path(path).as_posix())
        path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(x, jnp.ndarray): x = jnp.asarray(x)
    if not isinstance(y, jnp.ndarray): y = jnp.asarray(y)

    nn_init_rng = jax.random.PRNGKey(42)
    params = nn_model.init(nn_init_rng, jnp.take(x, 0, axis=0))

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )

    state = TrainState.create(
        apply_fn=nn_model.apply,
        params=params,
        tx=tx
    )

    runner = (state, x, y)

    jax.block_until_ready(_epoch(runner, 0))  # Warmup --> don't count compile time in training

    start = time.time()

    if verbose:
        losses = []
        for i in tqdm(range(n_epochs)):
            runner, loss = _epoch(runner, i)
            losses.append(loss)
    else:
        # runner, losses = lax.scan(scan_tqdm(n_epochs)(_epoch), runner, jnp.arange(n_epochs), n_epochs)
        runner, losses = jax.block_until_ready(lax.scan(_epoch, runner, jnp.arange(n_epochs), n_epochs))

    end = time.time()
    print(f"Training took {end - start:.2f} seconds")

    state, _, _ = runner
    trained_params = state.params

    losses = np.asarray(losses)

    if save_params:
        with open(path, 'wb') as f: pickle.dump(trained_params, f)
        print(f"Saved model parameters at {path}")

    return trained_params, losses


def inference(model, params, x):
    return np.asarray(model.apply(params, jnp.asarray(x)))


def plot_predictions(model, params, x_train, x_test, y_train, y_test, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat_train = inference(model, params, x_train)
    y_hat_test = inference(model, params, x_test)

    figs = []
    zipped = zip(y_train.T, y_hat_train.T, y_test.T, y_hat_test.T)
    for i_point, (y_t_train, y_p_train, y_t_test, y_p_test) in enumerate(zipped):
        fig = plt.figure()
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
        plt.scatter(y_t_train, y_p_train, marker='x', c='b', label="Train")
        plt.scatter(y_t_test, y_p_test, marker='x', c='r', label="Test")
        plt.axline((0, 0), slope=1, c='k')
        plt.plot([y_t_train.min(), y_t_train.min()], [y_t_train.max(), y_t_train.max()], c='k')
        plt.xlabel('Observation', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid()
        plt.close()
        figs.append(fig)
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()


def plot_wall(model, params, x, y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, params, x)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100

    fig = plt.figure()
    for disp in y:
        plt.plot(disp, np.arange(1, y_hat.shape[1]+1), c='b', alpha=0.3)
    for i, y_depth in enumerate(y_hat.T):
        mean = y_depth.mean()
        ci = np.quantile(y_depth, [0.025, 0.975])
        xerr = np.array([
            np.maximum(ci.min(), 2e-3)-1e-3,
            np.maximum(ci.max(), 2e-3)+1e-3,
        ]).squeeze()
        plt.errorbar([mean], [i+1], xerr=xerr[:, np.newaxis], fmt='o', color='r', capsize=5)
    plt.xlabel('Displacement [mm]', fontsize=12)
    plt.ylabel('Point # along wall', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.close()

    if path.suffix == ".png":
        fig.savefig(path)
    else:
        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()


def plot_wall_error(model, params, x, y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, params, x)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100

    fig = plt.figure()
    for r in st_residuals:
        plt.plot(r, np.arange(1, y_hat.shape[1]+1), c='b', alpha=0.3)
    plt.xlabel('Error [%]', fontsize=12)
    plt.ylabel('Point # along wall', fontsize=12)
    plt.gca().invert_yaxis()
    plt.grid()
    plt.close()

    if path.suffix == ".png":
        fig.savefig(path)
    else:
        pp = PdfPages(path)
        pp.savefig(fig)
        pp.close()


def plot_variables(model, params, x, y, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    y_hat = inference(model, params, x)
    residuals = y - y_hat
    st_residuals = residuals / np.abs(y+1e-8) * 100

    figs = []
    for i_point, r in enumerate(st_residuals.T):
        fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
        fig.suptitle(f"Point #{i_point+1:d} along wall", fontsize=14)
        sc = axs[0].scatter(x[:, 0], r, c=x[:, 1])
        cbar = fig.colorbar(sc, ax=axs[0])
        cbar.set_label("Cohesion [kPa]", fontsize=10, labelpad=10)
        axs[0].set_xlabel('Phi [deg]', fontsize=12)
        axs[0].set_ylabel('Error [%]', fontsize=12)
        sc = axs[1].scatter(x[:, 1], r, c=x[:, 0])
        cbar = fig.colorbar(sc, ax=axs[1])
        cbar.set_label("Phi [deg]", fontsize=10, labelpad=10)
        axs[1].set_xlabel('Cohesion [kPa]', fontsize=12)
        axs[1].set_ylabel('Error [%]', fontsize=12)
        axs[0].set_ylim(residuals.min(), residuals.max())
        axs[1].set_ylim(residuals.min(), residuals.max())
        axs[0].grid()
        axs[1].grid()
        plt.close()
        figs.append(fig)
    pp = PdfPages(path)
    [pp.savefig(fig) for fig in figs]
    pp.close()


def plot_losses(losses, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    fig = plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, losses.size+1), losses, c="b")
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss [${mm}^{2}$]', fontsize=12)
    plt.grid()
    plt.close()

    fig.savefig(path)


def plot(model, params, x_train, x_test, y_train, y_test, path, losses=None):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())

    path.mkdir(parents=True, exist_ok=True)

    plot_predictions(model, params, x_train, x_test, y_train, y_test, path/"predictions.pdf")
    plot_wall(model, params, x_train, y_train, path/"wall.png")
    plot_wall_error(model, params, x_train, y_train, path/"wall_error.png")
    plot_variables(model, params, x_train, y_train, path/"variables.pdf")

    if losses is not None: plot_losses(losses, path/"losses.png")


if __name__ == "__main__":

    path = os.environ["SRG_DATA_PATH"]
    path = Path(Path(path).as_posix())

    df_path = path / "compiled_data"
    df_files = [f for f in df_path.iterdir()]
    dates = [int("".join(f.stem.split("_")[-2:])) for f in df_files]
    df_file = df_files[dates.index(max(dates))]
    df = pd.read_csv(df_file)

    X_cols = [col for col in df.columns if col.split("_")[0] != "disp" and col != "index"]
    X_cols = [col for col in X_cols if "soilcurko2" not in col and "soilcurko3" not in col]
    X = df[X_cols].values

    idx_locs = list(range(1, 151, 10))
    y_cols = [col for col in df.columns if col.split("_")[0] == "disp" and int(col.split("_")[-1]) in idx_locs]
    y = df[y_cols].values

    # Remove extreme outliers which probably correspond to numerical error in DSheetPiling
    quartiles = np.quantile(y, [0.25, 0.75], axis=0)
    iqr = np.diff(quartiles, axis=0)
    bnd_distance = 0.01
    bounds = quartiles + np.array([-bnd_distance, +bnd_distance])[:, np.newaxis] * iqr[np.newaxis, :]
    # outliers = np.all(np.logical_and(bounds[0]<=y, y<=bounds[1]), axis=1)
    outliers = np.any(np.abs(y)>=1_000, axis=1)
    X, y = X[~outliers], y[~outliers]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    retrain = True

    if retrain:

        model = NeuralNetwork(y.shape[-1])
        params, losses = train(
            nn_model=model,
            x=X_train,
            y=y_train,
            lr=1e-6,
            n_epochs=3_000,
            path=r'results/mlp.pkl',
            verbose=True
        )

        print(jax.devices())

    else:

        with open(r'results/mlp.pkl', 'rb') as f: params = pickle.load(f)
        losses = None

    plot(model, params, X_train, X_test, y_train, y_test, r'figures/srg_mlp', losses)

