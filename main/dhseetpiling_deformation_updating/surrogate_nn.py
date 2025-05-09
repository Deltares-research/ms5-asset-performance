import json
import os
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from jax import lax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
from flax.training.train_state import TrainState
import optax
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pickle


class PlateNeuralNetwork(nn.Module):
    output_dim: Sequence[int]

    @nn.compact
    def __call__(self, x: Float[Array, "n_obs n_input"]) -> Float[Array, "n_obs n_output"]:

        y = nn.Dense(512, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        y = nn.relu(y)
        y = nn.Dense(256, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(32, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(y)
        y = nn.relu(y)
        y = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(y)
        y = y.reshape(-1, self.output_dim)

        return y


def loss_fn(state, params, x: Float[Array, "n_obs n_input"], y: Float[Array, "n_obs n_output"]) -> Float[Array, "1"]:
    y_pred = state.apply_fn(params, x)
    return optax.l2_loss(predictions=y_pred, targets=y).mean()


def _epoch(runner: tuple, epoch: int) -> Tuple[tuple, float]:
    state, x_train, y_train = runner
    loss, grads = jax.value_and_grad(loss_fn, argnums=1)(state, state.params, x_train, y_train)
    state = state.apply_gradients(grads=grads)
    runner = (state, x_train, y_train)
    return runner, loss


def train(nn_model: nn.Module, x: Float[Array, "n_obs n_input"], y: Float[Array, "n_obs n_output"],
          n_epochs: int = 20_000, lr: float = 1e-4) -> Tuple[dict, list]:

    nn_init_rng = jax.random.PRNGKey(42)
    params = nn_model.init(nn_init_rng, jnp.take(x, 0, axis=0))

    state = TrainState.create(
        apply_fn=nn_model.apply,
        params=params,
        tx=optax.adam(lr)
    )

    runner = (state, x, y)
    runner, losses = lax.scan(scan_tqdm(n_epochs)(_epoch), runner, jnp.arange(n_epochs), n_epochs)
    state, _, _ = runner

    return state.params, losses


if __name__ == "__main__":

    data_path = r"results/sample_1000_unpooled.json"
    data_path = Path(Path(data_path).as_posix())
    with open(data_path, "r") as f: data = json.load(f)

    y = data["displacement"]
    y = [[item if item is not None else np.nan for item in row] for row in y]
    y = np.asarray(y)
    idx = np.where(~np.any(np.isnan(y), axis=-1))[0]

    x = (data["Klei_soilphi"], data["Klei_soilcohesion"])
    x = tuple([jnp.asarray(item) for item in x])
    x = jnp.stack(x, axis=-1)

    x = jnp.asarray(x[idx])
    y = jnp.asarray(y[idx])

    model = PlateNeuralNetwork(y.shape[-1])
    params, losses = train(model, x, y, lr=1e-4, n_epochs=20_000)

    with open(r'results/nn_surrogate.pkl', 'wb') as f: pickle.dump(params, f)
    # with open(r'results/nn_surrogate.pkl', 'rb') as f: params = pickle.load(f)

    figs = []
    y_hat = model.apply(params, x)
    y = np.asarray(y)
    y_hat = np.asarray(y_hat)
    for i_loc, (y_t, y_p) in enumerate(zip(y.T, y_hat.T)):

        fig = plt.figure()
        plt.scatter(y_t, y_p, marker='x', c='b')
        plt.axline((0, 0), slope=1, c='k')
        plt.plot([y_t.min(), y_t.min()], [y_t.max(), y_t.max()], c='k')
        plt.xlabel('Observation', fontsize=12)
        plt.ylabel('Prediction', fontsize=12)
        plt.close()
        figs.append(fig)

    pp = PdfPages(r'figures/surrogate/plots.pdf')
    [pp.savefig(fig) for fig in figs]
    pp.close()

