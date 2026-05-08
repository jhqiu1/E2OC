"""Field descriptor for population encoding."""

import numpy as np


def crtfld(Encoding, varTypes, ranges, borders):
    """Create a field descriptor array.

    The returned object is used by Population to determine variable bounds
    and types during chromosome initialization.

    Args:
        Encoding: Encoding type string ('P', 'RI', etc.).
        varTypes: List of variable types (0=continuous, 1=discrete).
        ranges: 2D array [[lower_bounds], [upper_bounds]].
        borders: 2D array [[lower_inclusion], [upper_inclusion]].

    Returns:
        A list [n_vars, lb, ub, varTypes, lbin, ubin] where:
          - n_vars: number of decision variables
          - lb, ub: lower/upper bound arrays
          - varTypes: variable type list
          - lbin, ubin: boundary inclusion arrays
    """
    n_vars = len(varTypes)
    lb = np.array(ranges[0])
    ub = np.array(ranges[1])
    lbin = np.array(borders[0])
    ubin = np.array(borders[1])
    return [n_vars, lb, ub, varTypes, lbin, ubin]
