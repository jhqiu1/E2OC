"""Crowding distance calculation for NSGA-II."""

import numpy as np


def crowdis(objv, levels):
    """Compute crowding distance for each individual.

    Args:
        objv: Objective value matrix, shape (N, M).
        levels: Non-dominated rank for each individual, shape (N,).

    Returns:
        Crowding distance array, shape (N,).
        Boundary solutions get infinite distance.
    """
    N, M = objv.shape
    dis = np.zeros(N)
    max_level = int(np.max(levels))

    for front in range(1, max_level + 1):
        front_idx = np.where(levels == front)[0]
        if len(front_idx) <= 2:
            dis[front_idx] = np.inf
            continue

        for m in range(M):
            sorted_order = np.argsort(objv[front_idx, m])
            sorted_idx = front_idx[sorted_order]

            dis[sorted_idx[0]] = np.inf
            dis[sorted_idx[-1]] = np.inf

            f_min = objv[sorted_idx[0], m]
            f_max = objv[sorted_idx[-1], m]

            if f_max == f_min:
                continue

            for k in range(1, len(sorted_idx) - 1):
                dis[sorted_idx[k]] += (
                    objv[sorted_idx[k + 1], m] - objv[sorted_idx[k - 1], m]
                ) / (f_max - f_min)

    return dis
