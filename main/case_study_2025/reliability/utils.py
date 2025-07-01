import numpy as np
import torch
from scipy import stats
from pathlib import Path
import matplotlib.pyplot as plt


def plot_errorbar(x, xerr, y, color="b", whiskersize=0.1):
    plt.scatter(x, y, c=color)
    plt.hlines(y, xmin=min(xerr), xmax=max(xerr), colors=color)
    plt.vlines(min(xerr), ymin=y-whiskersize/2, ymax=y+whiskersize/2, colors=color)
    plt.vlines(max(xerr), ymin=y-whiskersize/2, ymax=y+whiskersize/2, colors=color)


def plot_fos_hist(fos, path=None, modelfit="lognormal", ci_alpha=0.05):

    fig = plt.figure()

    pf_mcs = np.mean(fos<1)
    q_mcs = np.quantile(fos, [ci_alpha/2, 1-ci_alpha/2])
    bins = 80 if fos.size >= 1_000 else 50
    plt.hist(fos, bins=80, density=True, color="b", alpha=0.6, edgecolor="k", linewidth=.5, label="MCS ${P}_{f}$ = "+f"{pf_mcs*100:.1e}")
    plot_errorbar(x=fos.mean(), xerr=q_mcs, y=0.7, color="b", whiskersize=0.1)

    if modelfit == "lognormal":
        x_grid = np.linspace(fos.min(), fos.max(), 1_000)
        shape, loc, scale = stats.lognorm.fit(fos)
        pdf_fitted = stats.lognorm.pdf(x_grid, shape, loc, scale)
        expectation_fit = np.trapezoid(pdf_fitted*x_grid, x_grid)
        q_fit = stats.lognorm.ppf([0.025, 0.975], shape, loc, scale)
        pf_fit = stats.lognorm.cdf(1, shape, loc, scale)
        plt.plot(x_grid, pdf_fitted, c="r", label="Fit ${P}_{f}$ = "+f"{pf_fit*100:.1e}")
        plot_errorbar(x=expectation_fit, xerr=q_fit, y=0.5, color="r", whiskersize=0.1)

    plt.axvline(1, c="k", label="Safety margin")
    plt.xlabel("FoS [-]", fontsize=12)
    plt.ylabel("Density [-]", fontsize=12)
    plt.legend(fontsize=12)
    plt.close()

    if path is not None: fig.savefig(path)


if __name__ == "__main__":

    pass

