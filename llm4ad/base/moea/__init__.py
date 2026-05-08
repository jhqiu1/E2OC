"""Custom MOEA framework — pure-Python implementation for E2OC.

Provides MOEA framework classes (Problem, MoeaAlgorithm, Population)
and operators (non-dominated sorting, crowding distance, selection) in
pure Python with no external dependencies.
"""

from .problem import Problem
from .algorithm import MoeaAlgorithm
from .population import Population
from .field import crtfld
from .sorting import ndsortESS, ndsortTNS
from .selection import selecting
from .crowding import crowdis
from .recombination import Recombination
from .mutation import Mutation
