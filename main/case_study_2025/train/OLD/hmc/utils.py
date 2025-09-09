import numpy as np
import torch
import torch.nn as nn
import pytensor
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path


def extract_mlp_weights(model):
    weights = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            weights.append((W, b))
    return weights


def mlp_forward_pt(x, weights):
    h = x
    for i, (W, b) in enumerate(weights):
        W_pt = pt.constant(W.astype("float32"))
        b_pt = pt.constant(b.astype("float32"))
        h = pt.dot(h, W_pt.T) + b_pt
        if i < len(weights) - 1:
            h = pt.switch(h > 0, h, 0)
    return h


def chebysev_forward_pt(x, weights, basis):
    h = x
    for i, (W, b) in enumerate(weights):
        W_pt = pt.constant(W.astype("float32"))
        b_pt = pt.constant(b.astype("float32"))
        h = pt.dot(h, W_pt.T) + b_pt
        if i < len(weights) - 1:
            h = pt.switch(h > 0, h, 0)  # ReLU

    # h now contains the Chebyshev coefficients
    basis_pt = pt.constant(basis.astype("float32"))  # shape (degree+1, n_points)
    out = pt.dot(h, basis_pt)  # (batch, n_points)
    return out


def chebysev_forward_np(x, weights, basis):
    h = x
    for i, (W, b) in enumerate(weights):
        h = np.dot(h, W.T) + b
        if i < len(weights) - 1:
            h = np.where(h > 0, h, 0)  # ReLU
    # h now contains the Chebyshev coefficients
    out = np.dot(h, basis)  # (batch, n_points)
    return out


def posterior_plot(idata, ref_vals, path):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    path.mkdir(parents=True, exist_ok=True)

    # Extract prior and posterior samples of x
    x_prior = idata.prior["x"].stack(sample=("chain", "draw")).values  # shape (n_dim, n_samples)
    x_post = idata.posterior["x"].stack(sample=("chain", "draw")).values

    # Transpose if needed to get shape (n_samples, n_dim)
    if x_prior.shape[0] < x_prior.shape[1]: x_prior = x_prior.T
    if x_post.shape[0] < x_post.shape[1]: x_post = x_post.T

    n_dim = x_post.shape[1]

    figs = []
    for i in range(n_dim):

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(x_prior[:, i], bins=50, density=True, alpha=0.4, label="Prior", color="b")
        ax.hist(x_post[:, i], bins=50, density=True, alpha=0.6, label="Posterior", color="r")
        ax.axvline(ref_vals[i], color="k", linestyle="--", label="Ref")

        ax.set_title(f"x[{i}]")
        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")

        plt.tight_layout()
        plt.close()

        figs.append(fig)

    pp = PdfPages(path/"posterior_plot.pdf")
    [pp.savefig(fig) for fig in figs]
    pp.close()


def summarize(idata, path, var_names=["x"]):

    if not isinstance(path, Path): path = Path(Path(path).as_posix())
    path.mkdir(parents=True, exist_ok=True)

    summary = az.summary(idata, var_names=["x"], hdi_prob=0.95)

    with open(path/"posterior_summary.txt", "w") as f:
        f.write(summary.to_string())

    print(summary.to_string())

