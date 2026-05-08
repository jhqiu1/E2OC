"""Bi-objective TSP problem definition for E2OC's MOEA framework.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import numpy as np
import random
from llm4ad.base import moea as ea


class Problem_biTSP(ea.Problem):
    """Bi-objective TSP problem for use with E2OC's MOEA algorithms.

    Wraps a BiTSP_instance and exposes the Problem interface
    (aimFunc, population initialization).
    """

    def __init__(self, obj_list, instance, ref_points, init_method="random"):
        self.obj_list = obj_list
        self.instance = instance
        self.init_method = init_method
        self.ref_points = ref_points

        self.variable_info = self.instance.variable_info
        self.Dims = self.instance.n

        name = "BITSP"
        self.M = len(self.obj_list)
        maxormins = [1] * self.M
        varTypes = [1] * self.Dims
        lb = self.Dims - 1
        ub = self.Dims - 1
        lbin = [1] * (self.Dims - 1)
        ubin = [1] * (self.Dims - 1)

        ea.Problem.__init__(
            self, name, self.M, maxormins, self.Dims, varTypes, lb, ub, lbin, ubin
        )

    def call_aimFunc(self, solution):
        result_dict = self.instance.evaluate_fitness(solution)
        return [result_dict[obj] for obj in self.obj_list]

    def generate_reference_point(self, m=10):
        max_obj_values = {obj: float("-inf") for obj in self.obj_list}
        samples = [self.instance.generate_random_solution() for _ in range(m)]
        obj_values = [self.call_aimFunc(samples[i]) for i in range(len(samples))]

        for obj_value in obj_values:
            for idx, obj in enumerate(self.obj_list):
                if obj_value[idx] > max_obj_values[obj]:
                    max_obj_values[obj] = obj_value[idx]
        return [max_obj_values[obj] * 1.05 for obj in self.obj_list]

    def generate_initial_solution(self):
        if self.init_method == "random":
            return self.instance.generate_random_solution()
        elif self.init_method == "ref_point":
            return self.ref_points
        else:
            raise ValueError(f"method not implemented: {self.init_method}")

    def generate_initial_solutions(self, num):
        dim = self.instance.n
        k = len(self.obj_list) + dim
        res = np.zeros((num, k))
        for i in range(num):
            res[i][0:dim] = list(range(dim))
            random.shuffle(res[i][0:dim])
            res[i][dim:k] = self.instance.evaluate_fitness(res[i][0:dim])
        return res

    def aimFunc(self, pop):
        Vars = pop.Chrom if pop.Chrom is not None else pop.Phen
        pop.ObjV = np.zeros((len(Vars), self.M))
        for i in range(len(Vars)):
            obj1, obj2 = self.instance.evaluate_fitness(Vars[i])
            pop.ObjV[i, 0] = obj1
            pop.ObjV[i, 1] = obj2
