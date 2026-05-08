"""Operator Rotation: coordinate descent optimization of interdependent operators.

Part of the E2OC four-component co-evolution framework:
  Warm-Start → Language Space → Progressive MCTS Search → Operator Rotation Evolution

The rotation component freezes all operators except one and calls the Generator
to optimize that single operator. It then rotates to the next operator
(coordinate descent). The Generator is a pluggable abstraction — EoH is
one implementation, but other LLM-based methods can be used instead.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

from copy import deepcopy
import os
from typing import Dict, List, Tuple, Optional, Type

from llm4ad.base import Generator


def quick_oc_eva(
    main_args: dict,
    dir: str,
    state_variables: List[str],
    generator_cls: Type[Generator] = None,
) -> Tuple[Dict, float]:
    """Perform one round of operator rotation evolution.

    Iteratively optimizes each operator in sequence while keeping others
    fixed, using the provided Generator to design single operators.

    Args:
        main_args: Configuration dictionary with all parameters.
            Required keys:
                - max_cycles: Coordinate descent cycles
                - operators_to_optimize: List of operator names
                - algorithm_class: MOEA algorithm class
                - instance_paths: Paths to problem instances
                - ref_dict: Reference point dict
                - obj_list: Objective names
                - default_operators_source: Default operator implementations
                - generation_num: MOEA generations per evaluation
                - n_evals: Number of repeated evaluations
                - pop_size: MOEA population size
                - eoh_max_generations: Generator generations
                - eoh_max_sample_nums: Max LLM samples per Generator run
                - eoh_pop_size: Generator population size
                - eoh_num_samplers: Parallel sampling threads
                - eoh_num_evaluators: Parallel evaluation threads
                - log_dir: Directory template for logging
                - exp_name: Experiment name
                - llm: LLM API instance
        dir: Directory identifier for logging.
        state_variables: List of current design templates for each operator.
        generator_cls: Generator class to use (e.g., EoH).
            If None, defaults to EoH.

    Returns:
        Tuple of (updated_operators_dict, best_score).
    """
    if generator_cls is None:
        from llm4ad.method.eoh import EoH as generator_cls

    max_cycles = main_args["max_cycles"]
    operators_to_optimize = main_args["operators_to_optimize"]
    template_dict = state_variables
    current_operators = deepcopy(main_args["default_operators_source"])
    instance_paths = main_args.get("instance_paths", [])
    exp_name = main_args.get("exp_name", "")

    from e2oc.utils.code_to_program import manage_directory
    from examples.tsp_biobjective.evaluator import MultiEvaluation

    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def get_path(*relative_parts):
        return os.path.join(PROJECT_ROOT, *relative_parts)

    best_score = float("-inf")
    for cycle in range(max_cycles):
        print(f"--- Starting coordinate cycle {cycle+1}/{max_cycles} ---")

        for state_index in range(len(operators_to_optimize)):
            operator_name = operators_to_optimize[state_index]
            print(f"Optimizing {operator_name}...")
            template = template_dict[state_index]

            operator_evaluator = MultiEvaluation(
                algorithm=main_args["algorithm_class"],
                instance_paths=instance_paths,
                ref_dict=main_args["ref_dict"],
                exp_name=exp_name,
                template=template,
                obj_list=main_args["obj_list"],
                ev_operator_name=operator_name,
                operators=main_args["default_operators_source"],
                generation_num=main_args["generation_num"],
                n_evals=main_args["n_evals"],
                pop_size=main_args["pop_size"],
            )

            operator_evaluator.save_operatorstr_bydict(current_operators)

            log_path = str(
                get_path(
                    main_args.get("log_dir", "").format(
                        f"cycle{dir}_{cycle+1}_{operator_name}"
                    )
                )
            )
            manage_directory(log_path)

            from llm4ad.method.eoh import EoHProfiler

            profiler = EoHProfiler(
                log_dir=log_path,
                method_name=generator_cls.__name__,
                evaluation_name=operator_name,
                log_style="simple",
            )

            try:
                generator = generator_cls(
                    evaluation=operator_evaluator,
                    profiler=profiler,
                    llm=main_args.get("llm"),
                    debug_mode=True,
                    max_generations=main_args["eoh_max_generations"],
                    max_sample_nums=main_args["eoh_max_sample_nums"],
                    pop_size=main_args["eoh_pop_size"],
                    num_samplers=main_args["eoh_num_samplers"],
                    num_evaluators=main_args["eoh_num_evaluators"],
                )
            except ValueError as e:
                print(f"  Skipping {operator_name}: invalid template - {e}")
                continue

            generator.run()
            ope_pops = generator.get_population()

            if ope_pops and len(ope_pops.population) > 0:
                best_function = ope_pops.population[0]
                best_score_val = best_function.score

                if (
                    best_score_val is not None
                    and best_score_val > float("-inf")
                    and best_score_val > best_score
                ):
                    best_score = best_score_val
                    current_operators[operator_name] = str(best_function)

    return current_operators, best_score
