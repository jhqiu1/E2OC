"""NSGA-II implementation for TSP with operator wrapper for E2OC.

Provides an NSGA-II that accepts user-defined operators as keyword arguments.
Each operator is wrapped in TSPOperatorWrapper so it conforms to the expected
call signature.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import numpy as np
from llm4ad.base import moea as ea
from copy import deepcopy


class NSGA2_TSP(ea.MoeaAlgorithm):
    """NSGA-II for permutation-encoded TSP problems.

    Accepts operator functions via **kwargs. Each operator is called
    once per generation on the offspring chromosome matrix.

    Usage:
        operators = {"tsp_2opt": my_2opt_func, "tsp_oropt": my_oropt_func}
        alg = NSGA2_TSP(problem, population, **operators)
    """

    def __init__(self, problem, population, **kwargs):
        super().__init__(problem, population, **kwargs)

        self.name = "NSGA2-TSP"
        self.problem = problem
        self.population = population
        self.ref_point = self.problem.ref_points

        if problem.M < 10:
            self.ndSort = ea.ndsortESS
        else:
            self.ndSort = ea.ndsortTNS

        self.selFunc = "tour"

        self.Opti_operators = {}
        self.kwargs = kwargs
        for operator_name in self.kwargs:
            operator = kwargs.get(operator_name, None)
            if operator:
                self.Opti_operators[operator_name] = TSPOperatorWrapper(operator)

        print("Algorithm initialization successful")

    def run(self, prophetPop=None):
        population = self.population
        NIND = population.sizes
        self.initialization()

        population.initChrom()

        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]

        self.call_aimFunc(population)

        [levels, criLevel] = self.ndSort(
            population.ObjV, NIND, None, population.CV, self.problem.maxormins
        )
        population.FitnV = (1 / levels).reshape(-1, 1)

        while not self.terminated(population):
            try:
                offspring = population[
                    ea.selecting(self.selFunc, population.FitnV, NIND)
                ]
                for operator_name in self.Opti_operators:
                    operator = self.Opti_operators[operator_name]
                    parent_pops = deepcopy(offspring.Chrom)
                    try:
                        sub_pops = operator.do(parent_pops)
                    except Exception as e:
                        print(
                            f"Operator exception_{operator_name}_{self.currentGen}_{e}"
                        )
                    offspring.Chrom = sub_pops

                self.call_aimFunc(offspring)
                population = self.reinsertion(population, offspring, NIND)

                self.pop = population
                if self.callback is not None:
                    self.callback(self)

            except Exception as e:
                print(f"Warning: Error in generation {self.currentGen}: {e}")
                continue

        return self.finishing(population), True

    def reinsertion(self, population, offspring, NUM):
        population = population + offspring

        [levels, criLevel] = self.ndSort(
            population.ObjV, population.sizes, None, population.CV, self.problem.maxormins
        )

        dis = ea.crowdis(population.ObjV, levels)
        population.FitnV[:, 0] = np.argsort(
            np.lexsort(np.array([dis, -levels])), kind="mergesort"
        )
        chooseFlag = ea.selecting("dup", population.FitnV, NUM)

        return population[chooseFlag]

    def check_legal_pop(self, pops):
        for i in range(len(pops)):
            pop = pops[i]
            if set(pop) != set(range(self.problem.instance.n)):
                return False
        return True

    def cal_HV(self, PF, ref):
        import hvwfg
        ref_region = 1
        for i in range(ref.shape[0]):
            ref_region = ref_region * (ref[i] - 0)
        hv_val = hvwfg.wfg(
            np.array(PF).astype("float"),
            ref.astype("float"),
        )
        hv_val = hv_val / ref_region
        return hv_val


class TSPOperatorWrapper:
    """Wraps a user-provided operator function to expose a .do() method."""

    def __init__(self, operator):
        if operator:
            self.do = operator
