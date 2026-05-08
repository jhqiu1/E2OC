"""Multi-instance TSP Evaluator for E2OC.

This is the MOST IMPORTANT user-implemented file. The evaluator bridges:
  - Generated operator code (from LLM) → compiled into callable function
  - Callable operator → plugged into MOEA (NSGA-II)
  - MOEA run on TSP instances -> Hypervolume (HV) score returned

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Callable, Optional, Any, Union
from copy import deepcopy
import random
from llm4ad.base import moea as ea

from llm4ad.base import Evaluation
from e2oc.utils.code_to_program import string_to_callable


def main_op(
    ref_point: np.ndarray,
    algorithm: Callable,
    problem,
    generation_num: int,
    pop_size: int,
    operators: Dict[str, Callable],
    save_iteration_data: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the optimization algorithm and return complete results.

    Args:
        ref_point: Reference point for HV calculation.
        algorithm: MOEA algorithm class (e.g., NSGA2_TSP).
        problem: Problem instance.
        generation_num: Number of generations.
        pop_size: Population size.
        operators: Dict of operator_name -> callable function.
        save_iteration_data: Whether to save per-generation data.
        save_dir: Directory to save results.

    Returns:
        Dictionary with keys: hv, pareto_front, generations, population_size, etc.
    """
    NIND = pop_size
    n = problem.instance.n

    Encoding = "P"
    var_types = [1] * n
    lower_bound = [0] * n
    upper_bound = [n - 1] * n
    ranges = np.array([lower_bound, upper_bound])
    borders = np.array([[1] * n, [1] * n])

    Field = ea.crtfld(Encoding, var_types, ranges, borders)
    population = ea.Population(Encoding, Field, NIND)

    myAlgorithm = algorithm(problem, population, **operators)
    myAlgorithm.MAXGEN = generation_num
    myAlgorithm.logTras = 1
    myAlgorithm.verbose = True
    myAlgorithm.drawing = 0

    iteration_records = []

    def callback(alg):
        gen = int(alg.currentGen)
        pop = getattr(alg, "pop", None) or getattr(alg, "population", None)
        if pop is None or pop.ObjV is None:
            return

        objv = pop.ObjV
        N, M = objv.shape

        levels, _ = alg.ndSort(objv, N, None, pop.CV, alg.problem.maxormins)
        pareto_mask = levels == 1
        pareto_front = np.unique(objv[pareto_mask], axis=0).tolist()

        stats = {
            "min": np.min(objv, axis=0).tolist(),
            "max": np.max(objv, axis=0).tolist(),
            "mean": np.mean(objv, axis=0).tolist(),
        }

        hv_val = None
        if hasattr(alg, "ref_point") and alg.problem.M > 1:
            try:
                hv_val = alg.cal_HV(pareto_front, alg.ref_point)
            except Exception:
                hv_val = None

        record = {
            "generation": gen,
            "hv": hv_val,
            "timestamp": time.time(),
            "population_size": int(getattr(pop, "sizes", getattr(pop, "size", N))),
            "pareto_front": pareto_front,
            "pop_stats": stats,
        }
        iteration_records.append(record)

        if save_iteration_data and save_dir:
            out_path = os.path.join(save_dir, "iterations.json")
            os.makedirs(save_dir, exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(iteration_records, f, indent=2)

    myAlgorithm.callback = callback

    [BestIndi, population], resultOr = myAlgorithm.run()

    if hasattr(myAlgorithm, "methdoName"):
        if myAlgorithm.methdoName == "MOEAD":
            pareto_front = BestIndi
    else:
        pareto_front = BestIndi.ObjV

    if resultOr:
        hv = myAlgorithm.cal_HV(pareto_front, myAlgorithm.ref_point)
    else:
        hv = 0

    return resulst_calculation(pareto_front, hv, generation_num, pop_size, iteration_records)


def resulst_calculation(pareto_front, hv, generation_num, pop_size, iteration_records=None):
    num_solutions = len(pareto_front)
    avg_obj1 = np.mean(pareto_front[:, 0])
    avg_obj2 = np.mean(pareto_front[:, 1])
    min_obj1 = np.min(pareto_front[:, 0])
    min_obj2 = np.max(pareto_front[:, 1])

    return {
        "hv": hv,
        "pareto_front": pareto_front.tolist(),
        "generations": generation_num,
        "population_size": pop_size,
        "num_solutions": num_solutions,
        "avg_obj1": avg_obj1,
        "avg_obj2": avg_obj2,
        "min_obj1": min_obj1,
        "min_obj2": min_obj2,
        "iteration_records": iteration_records,
    }


def evaluate(
    algorithm: Callable,
    problems: Dict[str, Any],
    operators: Dict[str, Callable],
    n_evals: int = 11,
    generation_num: int = 20,
    ret_all_results: bool = False,
    pop_size: int = 200,
    save_dir: Optional[str] = None,
    save_iteration_data: bool = False,
) -> Any:
    """Evaluate operator combination performance on multiple instances.

    Args:
        algorithm: MOEA class.
        problems: Dict of instance_name -> problem object.
        operators: Dict of operator_name -> callable function.
        n_evals: Number of independent evaluations per instance.
        generation_num: MOEA generations per run.
        ret_all_results: If True, return list of all results.
        pop_size: MOEA population size.
        save_dir: Directory for saving results.
        save_iteration_data: Whether to save per-generation data.

    Returns:
        Average HV across all instances and repeats, or list of all results.
    """
    all_res = []
    for instance_name, problem in problems.items():
        instance_dir = os.path.join(save_dir, instance_name) if save_dir else None
        print(f"=================== instances_{instance_name}")

        for i in range(n_evals):
            print(f"=================== n_evals_{i}")
            np.random.seed(i)
            random.seed(i)

            run_dir = os.path.join(instance_dir, f"run_{i}") if instance_dir else None

            res = main_op(
                ref_point=problem.ref_points,
                algorithm=algorithm,
                problem=problem,
                generation_num=generation_num,
                pop_size=pop_size,
                operators=operators,
                save_iteration_data=save_iteration_data,
                save_dir=run_dir,
            )
            all_res.append(res)

            if run_dir:
                os.makedirs(run_dir, exist_ok=True)
                with open(os.path.join(run_dir, "result.json"), "w") as f:
                    json.dump(res, f, indent=2)

            print(f" Eval {i}: HV={res['hv']:.6f}")

    avg_hv = np.mean([r["hv"] for r in all_res])

    if np.isnan(avg_hv) or avg_hv < 0:
        raise ValueError(f"Invalid HV value: {avg_hv}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        summary = {"avg_hv": avg_hv, "results": all_res}
        with open(os.path.join(save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if ret_all_results:
        return all_res
    else:
        return avg_hv


class MultiEvaluation(Evaluation):
    """Multi-instance evaluator for TSP operator combinations.

    Inherits from llm4ad.base.Evaluation and implements evaluate_program()
    to compile LLM-generated operator code and evaluate it within NSGA-II.

    Args:
        algorithm: MOEA class (e.g., NSGA2_TSP).
        instance_paths: List of paths to problem instance files.
        obj_list: Objective names.
        ref_dict: Instance filename -> reference point mapping.
        exp_name: Experiment name for output.
        template: Code template string for the operator being evolved.
        ev_operator_name: Name of the operator currently being evolved.
        operators: Dict of all operators (default implementations).
        generation_num: MOEA generations per evaluation.
        n_evals: Number of repeated evaluations.
        pop_size: MOEA population size.
        ret_all_results: If True, return full results instead of scalar.
        output_dir: Output directory for results.
        save_iteration_data: Whether to save per-generation data.
    """

    def __init__(
        self,
        algorithm,
        instance_paths,
        obj_list,
        ref_dict,
        exp_name,
        template,
        ev_operator_name,
        operators,
        generation_num=20,
        n_evals=6,
        pop_size=200,
        ret_all_results=False,
        output_dir="experiments",
        save_iteration_data=False,
    ):
        super().__init__(template_program=template, timeout_seconds=1500)
        self.n_evals = n_evals
        self.ret_all_results = ret_all_results
        self.algorithm = algorithm
        self.generation_num = generation_num
        self.exp_name = exp_name
        self.pop_size = pop_size
        self.ev_operator_name = ev_operator_name
        self.operators = operators
        self.output_dir = output_dir
        self.save_iteration_data = save_iteration_data
        self.ref_dict = ref_dict
        self.obj_list = obj_list
        self.instance_name = None
        self.ref_point = None

        self.ini_problems(instance_paths)

        self.evaluation_history = []
        self.best_score = float("-inf")
        self.best_operators = {}
        self.evaluation_count = 0

    def ini_problems(self, instance_paths):
        from problems.tsp.problem_biTSP import Problem_biTSP
        from problems.tsp.tsp_instance import BiTSP_instance

        self.train_paths = []
        self.train_problems = {}
        self.test_paths = []
        self.test_problems = {}

        instance_idx = 1
        train_instance_rate = 1

        for i, path in enumerate(instance_paths):
            instance_name = os.path.basename(path)
            instance = BiTSP_instance()
            ref_points = np.array(self.ref_dict[instance_name])

            if i <= round(len(instance_paths) * train_instance_rate):
                self.train_paths.append(path)
                instance.read_instance(path, instance_idx)
                print(f"Loaded: {instance_name} (nodes={instance.n})")
                self.train_problems[instance_name] = Problem_biTSP(
                    self.obj_list, instance, ref_points
                )
            else:
                self.test_paths.append(path)
                instance.read_instance(path, instance_idx)
                self.test_problems[instance_name] = Problem_biTSP(
                    self.obj_list, instance, ref_points
                )

    def set_operator(self, operator_name: str, operator: Callable):
        if operator_name in self.operators:
            self.operators[operator_name] = operator
        else:
            raise ValueError(f"Invalid operator name: {operator_name}")

    def save_operatorstr_bydict(self, operators: Dict[str, Union[Callable, str]]):
        if operators:
            self.oprerators_str = operators

    def set_operators_by_settingoperators(self):
        for op_name, op_func in self.oprerators_str.items():
            if op_name != self.ev_operator_name:
                if isinstance(op_func, str):
                    callable_func = string_to_callable(op_func, op_name)
                    if callable_func is not None:
                        self.set_operator(op_name, callable_func)
                    else:
                        print(f"Warning: Failed to compile {op_name} from source code")
                else:
                    self.set_operator(op_name, op_func)

    def evaluate_program(
        self, program_str: str, new_callable_func: callable, **kwargs
    ) -> Union[Any, None]:
        """Called by EoH to evaluate a generated operator.

        Args:
            program_str: The generated code as string.
            new_callable_func: The compiled callable function.

        Returns:
            Average HV score across instances and repeats.
        """
        self.operators[self.ev_operator_name] = new_callable_func
        if self.oprerators_str:
            self.set_operators_by_settingoperators()

        print("waiting for evaluate...")

        res = evaluate(
            algorithm=self.algorithm,
            problems=self.train_problems,
            operators=self.operators,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            pop_size=self.pop_size,
            save_iteration_data=self.save_iteration_data,
        )

        print(f"Result Mean: {res}")
        return res

    def evaluate_combination(
        self,
        operators: Optional[Dict[str, Union[Callable, str]]] = None,
        use_train_set: bool = True,
    ) -> float:
        """Evaluate a specific operator combination.

        Args:
            operators: Dict of operator_name -> callable or code string.
            use_train_set: Use training set (True) or test set (False).

        Returns:
            Average HV score.
        """
        if operators:
            for op_name, op_func in operators.items():
                if isinstance(op_func, str):
                    callable_func = string_to_callable(op_func, op_name)
                    if callable_func:
                        self.set_operator(op_name, callable_func)
                    else:
                        print(f"Warning: Failed to compile {op_name} from source code")
                else:
                    self.set_operator(op_name, op_func)
        if self.oprerators_str:
            self.set_operators_by_settingoperators()

        if use_train_set:
            eval_problems = self.train_problems
            dataset_type = "train"
        else:
            eval_problems = self.test_problems
            dataset_type = "test"

        save_dir = os.path.join(
            self.output_dir, self.exp_name, dataset_type, time.strftime("%Y%m%d-%H%M%S")
        )

        all_res = evaluate(
            algorithm=self.algorithm,
            problems=eval_problems,
            operators=self.operators,
            n_evals=self.n_evals,
            generation_num=self.generation_num,
            ret_all_results=True,
            pop_size=self.pop_size,
            save_dir=save_dir,
            save_iteration_data=self.save_iteration_data,
        )

        avg_hv = np.mean([r["hv"] for r in all_res])
        print(f"Average HV for {dataset_type} set: {avg_hv:.4f}")

        self.record_evaluation(avg_hv)
        return avg_hv

    def record_evaluation(self, score):
        self.evaluation_count += 1
        print(f"Total evaluations: {self.evaluation_count}")
        copied_operators = {k: deepcopy(v) for k, v in self.operators.items()}
        self.evaluation_history.append((copied_operators, score))

        if score > self.best_score:
            self.best_score = score
            self.best_operators = copied_operators
            print(f"New local best combination found! Score: {self.best_score:.4f}")
