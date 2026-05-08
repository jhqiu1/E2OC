"""Non-dominated sorting algorithms — pure Python implementations."""

import numpy as np


def ndsortESS(objv, N, subPop, cv, maxormins):
    """Efficient Sequential Sort (ESS) for non-dominated sorting.

    Deb et al. (2002) — O(M·N²) worst-case.

    Args:
        objv: Objective value matrix, shape (N, M).
        N: Number of individuals to sort.
        subPop: Ignored (kept for API compatibility).
        cv: Constraint violation, shape (N,) or None.
        maxormins: List of 1 (min) or -1 (max) per objective.

    Returns:
        (levels, max_level): levels is 1-based rank array; max_level is
        the number of fronts found.
    """
    if N is None:
        N = objv.shape[0]
    objv = objv[:N]
    n_individuals, M = objv.shape

    feasible = np.ones(N, dtype=bool)
    if cv is not None:
        cv_slice = cv[:N]
        if cv_slice.ndim > 1:
            feasible = np.all(cv_slice <= 0, axis=1)
        else:
            feasible = cv_slice <= 0

    objv_adj = objv.copy()
    for j in range(M):
        if maxormins[j] == -1:
            objv_adj[:, j] = -objv_adj[:, j]

    S = [[] for _ in range(N)]
    n_p = np.zeros(N, dtype=int)
    levels = np.ones(N, dtype=int)

    for i in range(N):
        if not feasible[i]:
            n_p[i] = 0
            continue
        for j in range(N):
            if i == j:
                continue
            if not feasible[j]:
                continue
            if _dominates(objv_adj[i], objv_adj[j]):
                S[i].append(j)
            elif _dominates(objv_adj[j], objv_adj[i]):
                n_p[i] += 1

        if n_p[i] == 0:
            levels[i] = 1

    front = 1
    current_front = [i for i in range(N) if levels[i] == front]

    while current_front:
        next_front = []
        for i in current_front:
            for j in S[i]:
                n_p[j] -= 1
                if n_p[j] == 0:
                    levels[j] = front + 1
                    next_front.append(j)
        front += 1
        current_front = next_front

    return levels, front - 1


def ndsortTNS(objv, N, subPop, cv, maxormins):
    """Tournament Non-dominated Sort — alternate implementation.

    Falls back to ESS for simplicity. TNS is typically used for
    many-objective problems (M >= 10) for efficiency.
    """
    return ndsortESS(objv, N, subPop, cv, maxormins)


def _dominates(a, b):
    """Return True if a dominates b (all <= and at least one <)."""
    return np.all(a <= b) and np.any(a < b)
