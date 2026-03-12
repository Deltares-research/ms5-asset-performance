"""
Settlement computation engine using Terzaghi consolidation theory and
the NEN-Bjerrum settlement model.

Computes time-dependent primary consolidation settlement for a soil layer
under a given stress history, parameterized by compression ratio (CR) and
permeability (k). Supports vectorized evaluation over grids of CR and k
for use in probabilistic analyses.
"""

import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.stats as st
from typing import List, Dict, TypeAlias


FloatArray: TypeAlias = NDArray[np.floating]


def ensure_2d(x, axis=0):
    """Reshape a 1D array to 2D for broadcasting.

    Args:
        x: Input array.
        axis: If 0, returns shape (1, n). If 1, returns shape (n, 1).

    Returns:
        2D array suitable for broadcasting.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return x[np.newaxis, :] if axis == 0 else x[:, np.newaxis]


def get_doc(
        t: float | List[float] | FloatArray,
        h: float =1.,
        cv: float | List[float] | FloatArray = 1.,
        method: str = "Terzaghi"
) -> FloatArray:
    """

    Parameters
    ----------
    t : array_like
        time [days]
    h : float or array_like, optional
        percolation layer thickness [m]. The default is 1.0.
    cv : float or array_like, optional
        consolidation coefficient [m2/day]. The default is 1.0.
    method : str, optional
        method to compute the consolidation curve. The default is 'Terzaghi'.


    Returns
    -------
    U : array_like
        mean degree of consolidation [-]

    """
    if cv.ndim < 2:
        cv = ensure_2d(cv, axis=1)

    if t.ndim < 2:
        t = ensure_2d(t, axis=0)

    Tv = cv  * t / h**2

    if method.title() == 'Terzaghi':
        doc_start = np.sqrt( Tv * 4 / np.pi)
        doc_end = 1 - 8 / np.pi**2 * np.exp( -Tv / 4 * np.pi**2)
        doc_lims = np.stack((doc_start, doc_end), axis=-1)
        doc = np.min(doc_lims, axis=-1)
    else:
        ValueError(f'method {method.title()} not implemented')

    return doc


def get_end_settlement(
        sigma_0: float = 10.0,
        sigma_v: float = 20.0,
        RR: float = 0.02,
        CR: float | List[float] | FloatArray = 0.2,
        Ca: float = 0.,
        sigma_p: float = 0.,
        h: float = 1.
) -> FloatArray:
    '''
    NEN-Bjerrum settlement model for primary consolidation settlement.

    Parameters
    ----------
    sigma_0 : float
        initial effective vertical stress [kPa]
    sigma_v : array_like
        effective vertical stress [kPa]
    Cr : float, optional
        recompression index [-]. The default is 0.02.
    Cc : float, optional
        compression index [-]. The default is 0.2.
    sigma_p : float, optional
        preconsolidation stress [kPa]. The default is 0.0.
    e0 : float, optional
        initial void ratio [-]. The default is 1.0.
    h : float, optional
        layer thickness [m]. The default is 1.0.

    Returns
    -------
    S : array_like
        final settlement [m]

    '''

    sigma_p = max(sigma_0, sigma_p)

    if sigma_v < sigma_p:
        S = h *  RR * np.log10(sigma_v/sigma_0)
    else:
        S = h * (RR * np.log10(sigma_p / sigma_0) + CR * np.log10(sigma_v / sigma_p))

    return S


def get_settlement(
        t: float | List[float] | FloatArray,
        CR: float | List[float] | FloatArray = 0.2,
        k: float | List[float] | FloatArray = 3e-10,
        RR: float = 0.02,
        Ca: float = 0.,
        h: float = 1.,
        sigma_0: float = 10.0,
        sigma_v: float = 20.0,
        sigma_p: float = 0.,
        method: str = "Terzaghi",
        grid_based = False
) -> FloatArray:
    """Compute time-dependent settlement on a (CR, k, t) grid.

    Combines the degree of consolidation (DoC) with the end-of-primary
    settlement to produce settlement = DoC * end_settlement.

    The inputs CR and k are broadcast into a 3D array of shape
    (n_CR, n_k, n_t), enabling vectorized evaluation over the full
    parameter grid at all time steps simultaneously.

    Parameters
    ----------
    t : array_like
        Time points [days].
    CR : array_like
        Compression ratio [-]. Broadcast along axis 0.
    k : array_like
        Permeability [m/s]. Broadcast along axis 1.
    RR : float
        Recompression ratio [-].
    Ca : float
        Secondary compression coefficient [-].
    h : float
        Layer thickness [m].
    sigma_0 : float
        Initial effective vertical stress [kPa].
    sigma_v : float
        Applied vertical stress [kPa].
    sigma_p : float
        Preconsolidation stress [kPa].
    method : str
        Consolidation method (default "Terzaghi").

    Returns
    -------
    NDArray
        Settlement array of shape (n_CR, n_k, n_t) [m].
        Squeezed if any dimension is 1.
    """

    if grid_based:
        CR = ensure_2d(CR, axis=1)[..., np.newaxis]
        k = ensure_2d(k, axis=0)[..., np.newaxis]
        t = ensure_2d(t, axis=0)[np.newaxis, ...]
    else:
        CR = CR[..., np.newaxis]
        k = k[..., np.newaxis]
        t = t[np.newaxis, ...]

    gamma_w = 9.81
    mv = CR / (sigma_v * np.log(10))
    k_day = k * 3_600 * 24 # Time in days, so permeability needs to be scaled.
    cv = k_day / (gamma_w * mv)

    doc = get_doc(t=t, h=h, cv=cv, method=method)

    end_settlement = get_end_settlement(
        sigma_0=sigma_0,
        sigma_v=sigma_v,
        RR=RR,
        CR=CR,
        Ca=Ca,
        sigma_p=sigma_p,
        h=h
    )

    print(doc)

    settlement = doc * end_settlement

    return settlement.squeeze()


if __name__ == "__main__":

    pass
