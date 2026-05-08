"""Selection operators for evolutionary algorithms."""

import numpy as np


def selecting(style, FitnV, NIND):
    """Select individuals from the population.

    Args:
        style: Selection method ('tour' = tournament, 'dup' = duplicate).
        FitnV: Fitness values, shape (N, 1). Higher = better.
        NIND: Number of individuals to select.

    Returns:
        For 'tour': integer index array of selected individuals.
        For 'dup': boolean mask indicating selected individuals.
    """
    N = FitnV.shape[0]

    if style == "tour":
        selected = np.zeros(NIND, dtype=np.int32)
        for i in range(NIND):
            candidates = np.random.choice(N, 2, replace=False)
            if FitnV[candidates[0], 0] >= FitnV[candidates[1], 0]:
                selected[i] = candidates[0]
            else:
                selected[i] = candidates[1]
        return selected

    elif style == "dup":
        sorted_idx = np.argsort(-FitnV.flatten(), kind="mergesort")
        choose_flag = np.zeros(N, dtype=bool)
        choose_flag[sorted_idx[:NIND]] = True
        return choose_flag

    else:
        raise ValueError(f"Unknown selection style: {style}")
