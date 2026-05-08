"""E2OC Main Entry Point.

Orchestrates the four-component co-evolution process:
  1. Warm-Start: Initialize operator populations via independent evolution
  2. Language Space: Build multi-domain design thought space
  3. Progressive MCTS Search: Explore design strategy combinations
  4. Operator Rotation Evolution: Co-evolve operators in context

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.

Usage:
    python run_e2oc.py
"""

import os
import sys
import time
import inspect

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler.tensorboard_profiler import TensorboardProfiler

from e2oc import MCTS, quick_oc_eva, Storages
from e2oc.utils.code_to_program import manage_directory
from e2oc.utils.warmup_cache import (
    load_warmup_batch,
    _serialize_population_simple,
    save_warmup_batch,
)

from algorithms.nsga2 import NSGA2_TSP
from problems.tsp import Problem_biTSP, BiTSP_instance
from examples.tsp_biobjective.evaluator import MultiEvaluation
from examples.tsp_biobjective.operators import (
    tsp_oropt,
    tsp_3opt,
    tsp_2opt,
    tsp_oropt_template,
    tsp_3opt_template,
    tsp_2opt_template,
)
from examples.tsp_biobjective.prompts import Prompts_TSP
from examples.tsp_biobjective.config import (
    LLM_CONFIG,
    BENCHMARK,
    OBJ_LIST,
    REF_DICT,
    OPERATORS_TO_OPTIMIZE,
    MCTS_CONFIG,
    ROTATION_CONFIG,
    EOH_CONFIG,
    EVAL_CONFIG,
    EXP_DIR,
    LOG_DIR,
)


def get_llm_api(config: dict):
    """Create an LLM API instance from config dict."""
    host = config.get("host", "api.deepseek.com")
    api_key = config.get("api_key", "")
    model = config.get("model", "deepseek-chat")
    timeout = config.get("timeout", 120)

    if "openai.com" in host or config.get("provider") == "openai":
        return OpenAIAPI(
            api_key=api_key,
            model=model,
            base_url=f"https://{host}" if not host.startswith("http") else host,
            timeout=timeout,
        )
    else:
        return HttpsApi(host=host, key=api_key, model=model, timeout=timeout)


def build_instance_paths():
    """Build list of instance file paths from REF_DICT."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    instances_dir = os.path.join(base_dir, "instances", BENCHMARK)
    paths = [
        os.path.join(instances_dir, fname)
        for fname in REF_DICT
        if os.path.isfile(os.path.join(instances_dir, fname))
    ]
    print(f"[Setup] Benchmark: {BENCHMARK}")
    print(f"[Setup] Instances found: {len(paths)}")
    for p in paths:
        print(f"  - {os.path.basename(p)}")
    return paths


def warm_start_operators(
    llm,
    instance_paths,
    operators_to_optimize,
    template_dict,
    default_operators_source,
    cache_dir,
    storages,
):
    """Run EoH warm-start for each operator, or load from cache."""
    num_ops = len(operators_to_optimize)
    for idx, operator_name in enumerate(operators_to_optimize):
        print(f"\n  [{idx+1}/{num_ops}] Operator: {operator_name}")
        cached = load_warmup_batch(cache_dir, operator_name)
        storages.set_template(
            operator_name, cycle=-1, template=template_dict[operator_name + "_template"]
        )

        if cached is not None:
            storages.record_batch(operator=operator_name, individuals=cached, cycle=0)
            print(f"  -> Loaded from cache ({len(cached)} individuals)")
            continue

        print(f"  -> No cache found, running EoH warm-start...")
        template = template_dict[operator_name + "_template"]
        operator_evaluator = MultiEvaluation(
            algorithm=NSGA2_TSP,
            instance_paths=instance_paths,
            ref_dict=REF_DICT,
            exp_name=cache_dir,
            template=template,
            obj_list=OBJ_LIST,
            ev_operator_name=operator_name,
            operators=default_operators_source,
            generation_num=EVAL_CONFIG["generation_num"],
            n_evals=EVAL_CONFIG["n_evals"],
            pop_size=EVAL_CONFIG["pop_size"],
        )
        operator_evaluator.instance_name = list(REF_DICT.keys())[0]
        operator_evaluator.save_operatorstr_bydict(default_operators_source)

        log_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            LOG_DIR.format(f"warmup_{operator_name}"),
        )
        manage_directory(log_path)

        profiler = TensorboardProfiler(
            wandb_project_name="e2oc",
            log_dir=log_path,
            name=f"e2oc_warmup_{operator_name}",
            create_random_path=False,
        )

        eoh = EoH(
            evaluation=operator_evaluator,
            profiler=profiler,
            llm=llm,
            debug_mode=True,
            max_generations=EOH_CONFIG["max_generations"],
            max_sample_nums=EOH_CONFIG["max_sample_nums"],
            pop_size=EOH_CONFIG["pop_size"],
            num_samplers=EOH_CONFIG["num_samplers"],
            num_evaluators=EOH_CONFIG["num_evaluators"],
        )
        eoh.run()

        try:
            pop = eoh.get_population() if hasattr(eoh, "get_population") else eoh._population
        except Exception:
            pop = getattr(eoh, "_population", None)

        if not pop or not getattr(pop, "population", None):
            continue

        individuals = _serialize_population_simple(pop)
        save_warmup_batch(cache_dir, operator_name, individuals)
        storages.record_batch(operator=operator_name, individuals=individuals, cycle=0)


def main():
    print("=" * 60)
    print("E2OC: Evolution of Operator Combination")
    print("Bi-objective TSP")
    print("=" * 60)

    # --- LLM initialization ---
    llm = get_llm_api(LLM_CONFIG)

    # --- Instance paths ---
    instance_paths = build_instance_paths()

    # --- Operator configuration ---
    default_operators_source = {
        "tsp_oropt": inspect.getsource(tsp_oropt),
        "tsp_3opt": inspect.getsource(tsp_3opt),
        "tsp_2opt": inspect.getsource(tsp_2opt),
    }

    template_dict = {
        "tsp_oropt_template": tsp_oropt_template,
        "tsp_3opt_template": tsp_3opt_template,
        "tsp_2opt_template": tsp_2opt_template,
    }

    # --- Storage ---
    base_dir = os.path.dirname(os.path.abspath(__file__))
    storages = Storages(
        operators=OPERATORS_TO_OPTIMIZE,
        q_alpha=0.10,
        save_path=os.path.join(
            base_dir, "outputs_tsp", EXP_DIR, "storage", "storages.json"
        ),
        ucb_beta=1.0,
    )

    # --- Warm-start ---
    print(f"\n{'='*60}")
    print("[Phase 1] Warm-start: initializing operator populations via EoH")
    print(f"  Operators: {OPERATORS_TO_OPTIMIZE}")
    print(f"  Output dir: outputs_tsp/{EXP_DIR}")
    warm_start_operators(
        llm=llm,
        instance_paths=instance_paths,
        operators_to_optimize=OPERATORS_TO_OPTIMIZE,
        template_dict=template_dict,
        default_operators_source=default_operators_source,
        cache_dir=EXP_DIR,
        storages=storages,
    )

    # --- MCTS search ---
    prompt_gen = Prompts_TSP()
    main_args = {
        "ref_dict": REF_DICT,
        "exp_dir": EXP_DIR,
        "storage_dir": EXP_DIR,
        "prompt_gen": prompt_gen,
        "instances_dir": os.path.join(base_dir, "instances", BENCHMARK),
        "instance_name": instance_paths,
        "instance_paths": instance_paths,
        "exp_name": EXP_DIR,
        "log_dir": LOG_DIR,
        "obj_list": OBJ_LIST,
        "algorithm_class": NSGA2_TSP,
        "llm": llm,
        "max_iterations": MCTS_CONFIG["max_iterations"],
        "max_sampling_num": MCTS_CONFIG["max_sampling_num"],
        "max_cycles": ROTATION_CONFIG["max_cycles"],
        "default_operators_source": default_operators_source,
        "operators_to_optimize": OPERATORS_TO_OPTIMIZE,
        "template_dict": template_dict,
        "eoh_max_generations": EOH_CONFIG["max_generations"],
        "eoh_max_sample_nums": EOH_CONFIG["max_sample_nums"],
        "eoh_pop_size": EOH_CONFIG["pop_size"],
        "eoh_num_samplers": EOH_CONFIG["num_samplers"],
        "eoh_num_evaluators": EOH_CONFIG["num_evaluators"],
        "n_evals": EVAL_CONFIG["n_evals"],
        "generation_num": EVAL_CONFIG["generation_num"],
        "pop_size": EVAL_CONFIG["pop_size"],
    }

    print(f"\n{'='*60}")
    print("[Phase 2] MCTS: progressive design strategy search")
    print(f"  Iterations: {MCTS_CONFIG['max_iterations']}")
    print(f"  Simulations/node: {MCTS_CONFIG['max_simulations']}")
    print(f"  Rotation cycles: {ROTATION_CONFIG['max_cycles']}")
    mcts = MCTS(
        main_args,
        llm,
        storages,
        max_iterations=MCTS_CONFIG["max_iterations"],
        max_sampling_num=MCTS_CONFIG["max_sampling_num"],
        max_simulations=MCTS_CONFIG["max_simulations"],
    )
    history = mcts.search()
    mcts.save_results()

    print(f"\n{'='*60}")
    print("E2OC complete! Results saved to outputs_tsp/")


if __name__ == "__main__":
    main()
