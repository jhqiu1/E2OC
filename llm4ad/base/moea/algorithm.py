"""Base MOEA Algorithm class for E2OC."""

import numpy as np


class MoeaAlgorithm:
    """Base class for multi-objective evolutionary algorithms.

    Provides the template for NSGA-II and similar algorithms.
    Subclasses should override run() and reinsertion().
    """

    def __init__(self, problem, population, **kwargs):
        self.name = "base-algorithm"
        self.problem = problem
        self.population = population
        self.currentGen = 0
        self.MAXGEN = 100
        self.logTras = 1
        self.verbose = True
        self.drawing = 0
        self.callback = None
        self.pop = None  # attached during run for callback access

    def initialization(self):
        """Initialize dynamic parameters before evolution starts."""
        self.currentGen = 0

    def call_aimFunc(self, population):
        """Call the problem's aimFunc on the population."""
        self.problem.aimFunc(population)

    def terminated(self, population):
        """Check termination criteria.

        Returns True when currentGen exceeds MAXGEN.
        """
        self.currentGen += 1
        return self.currentGen > self.MAXGEN

    def finishing(self, population):
        """Finalize and return best individual and final population.

        Returns (best_individual, population) tuple.
        The best individual is the one with the lowest non-dominated rank.
        """
        if population.ObjV is None:
            return population[0], population

        from .sorting import ndsortESS

        N = population.sizes
        M = self.problem.M
        if M < 10:
            levels, _ = ndsortESS(
                population.ObjV, N, None, population.CV, self.problem.maxormins
            )
        else:
            from .sorting import ndsortTNS

            levels, _ = ndsortTNS(
                population.ObjV, N, None, population.CV, self.problem.maxormins
            )

        best_idx = np.argmin(levels)
        return population[best_idx], population

    def run(self, prophetPop=None):
        raise NotImplementedError

    def reinsertion(self, population, offspring, NUM):
        raise NotImplementedError
