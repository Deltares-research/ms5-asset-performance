import numpy as np
from scipy.interpolate import UnivariateSpline

def curvature(displacements, dLs):
    """
    Calculate the curvature of the displacements
    """

    displacements = displacements.copy() / 1_000

    # padding = (
    #     (0, 0),  # no padding on axis 0
    #     (0, 0),  # no padding on axis 1
    #     (2, 2),  # pad two columns on axis 2 -> double derivative -> moments have the shape of displacements
    # )
    # displacements_padded = np.pad(displacements.copy(), pad_width=padding, mode='edge')
    # dy2_dx2 = np.diff(displacements_padded, n=2, axis=-1)[..., 1:-1] / (dLs ** 2 + 1e-6)

    #TODO: Fix moment estimation using simple double derivative calculation
    x = np.cumsum(dLs)
    dy2_dx2 = np.zeros_like(displacements)
    for i, disp_chain in enumerate(displacements):
        for j, disp_chain_sample in enumerate(disp_chain):
            spline = UnivariateSpline(x, disp_chain_sample, s=1e-8)
            dy2_dx2[i, j] = spline.derivative(n=2)(x)

    return dy2_dx2


def moments(displacements, wall_props):
    """
    Calculate the moments of the displacements
    """

    EI, _, wall_locs, monitoring_locs = wall_props

    _, keep_idx = np.unique(wall_locs, return_index=True)
    keep_idx = np.sort(keep_idx)
    wall_locs = wall_locs[keep_idx]

    dLs = np.abs(np.diff(wall_locs))
    dLs = np.append(dLs[0], dLs)

    dy2_dx2 = curvature(displacements, dLs)

    moments = - EI * dy2_dx2  # Minus for proper sign in moment convention
    return moments