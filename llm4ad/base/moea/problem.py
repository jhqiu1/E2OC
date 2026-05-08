"""Base Problem class for E2OC."""

import numpy as np


class Problem:
    """Base class for optimization problems.

    Attributes:
        name: Problem name.
        M: Number of objectives.
        maxormins: List of 1 (minimize) or -1 (maximize) per objective.
        Dim: Number of decision variables.
        varTypes: Variable types (0=continuous, 1=integer/discrete).
        lb: Lower bound (scalar, broadcast to all dims).
        ub: Upper bound (scalar, broadcast to all dims).
        lbin: Lower boundary inclusion (1=inclusive).
        ubin: Upper boundary inclusion (1=inclusive).
    """

    def __init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin):
        self.name = name
        self.M = M
        self.maxormins = maxormins
        self.Dim = Dim
        self.varTypes = varTypes
        self.lb = lb
        self.ub = ub
        self.lbin = lbin
        self.ubin = ubin

    def aimFunc(self, pop):
        """Evaluate objective values for population.

        Subclasses MUST override this. Sets pop.ObjV in-place.

        Args:
            pop: Population object with pop.Phen (phenotype matrix).
        """
        raise NotImplementedError
