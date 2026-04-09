"""
Plotting functions for the D-SheetPiling reliability analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from scipy import stats


def plot_beta_forecast(
    results: Dict[float, Dict[str, Any]],
    beta_req: float = None,
) -> plt.Figure:
    """Plot prior and posterior beta over observation time."""
    times = sorted(results.keys())
    beta_prior = [results[t]["prior"]["beta"] for t in times]
    beta_posterior = [results[t]["posterior"]["beta"] for t in times]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times, beta_prior, "b--", linewidth=2, label="Prior")
    ax.plot(times, beta_posterior, "r-o", linewidth=2, markersize=4, label="Posterior")
    if beta_req is not None:
        ax.axhline(beta_req, color="k", linestyle=":", linewidth=1.5, label=f"Requirement ({beta_req})")
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Reliability index [-]")
    ax.set_title("Reliability Index over Time")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig


def plot_beta_forecast_at_time(
    current_time: float,
    results: Dict[float, Dict[str, Any]],
    beta_req: float = None,
) -> plt.Figure:
    """Plot beta forecasts from all observations up to current_time."""
    fig, ax = plt.subplots(figsize=(8, 5))

    times = sorted(results.keys())

    # Prior (from first observation)
    first_t = times[0]
    prior_forecast = results[first_t]["prior"]["beta_forecast"]
    ft_pr = sorted(prior_forecast.keys(), key=float)
    ax.plot([float(t) for t in ft_pr], [prior_forecast[t] for t in ft_pr],
            "b--", linewidth=1.5, alpha=0.5, label="Prior")

    # Posterior forecasts from each observation time
    for t in times:
        post_forecast = results[t]["posterior"]["beta_forecast"]
        ft_po = sorted(post_forecast.keys(), key=float)
        is_current = (t == current_time)
        ax.plot([float(f) for f in ft_po], [post_forecast[f] for f in ft_po],
                "r-" if is_current else "gray",
                linewidth=2 if is_current else 0.8,
                alpha=1.0 if is_current else 0.4,
                label=f"Posterior (t={t:.0f})" if is_current else None)

    if beta_req is not None:
        ax.axhline(beta_req, color="k", linestyle=":", linewidth=1.5, label=f"Req ({beta_req})")

    ax.axvline(current_time, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel("Reliability index [-]")
    ax.set_title(f"Beta Forecast (obs up to t={current_time:.0f})")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.close()
    return fig


def plot_end_of_life(
    results: Dict[float, Dict[str, Any]],
    beta_req: float = 2.3,
    t_start: float = 50.0,
) -> plt.Figure:
    """Bar chart of end-of-life (when beta drops below requirement)."""
    times = sorted(results.keys())

    def find_eol(beta_forecast):
        ft = sorted(beta_forecast.keys(), key=float)
        for t in ft:
            if beta_forecast[t] < beta_req:
                return float(t)
        return float(ft[-1])

    eol_prior = find_eol(results[times[0]]["prior"]["beta_forecast"])
    eol_posterior = [find_eol(results[t]["posterior"]["beta_forecast"]) for t in times]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(times))
    ax.bar(x, eol_posterior, color="tab:red", alpha=0.7, label="Posterior EOL")
    ax.axhline(eol_prior, color="b", linestyle="--", linewidth=2, label=f"Prior EOL ({eol_prior:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.0f}" for t in times], rotation=45)
    ax.set_xlabel("Observation time [years]")
    ax.set_ylabel("End-of-life [years]")
    ax.set_title(f"End of Life (beta < {beta_req})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.close()
    return fig
