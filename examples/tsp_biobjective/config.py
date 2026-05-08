"""User configuration for E2OC on Bi-objective TSP.

Copy this file and modify parameters for your own problem.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import os

# =============================================================================
# 1. LLM Configuration (REQUIRED: fill in your credentials)
# =============================================================================

LLM_CONFIG = {
    "host": os.environ.get("LLM_API_HOST", "api.deepseek.com"),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
    "timeout": 120,
}

# =============================================================================
# 2. Problem Configuration
# =============================================================================

# Problem benchmark name
BENCHMARK = "BiTSP"

# Objective list
OBJ_LIST = ["obj_1", "obj_2"]

# Instance file -> reference point mapping
# Reference point is used for Hypervolume (HV) calculation
REF_DICT = {
    "testdata_tsp_size100.pt": [65, 65],
}

# =============================================================================
# 3. Operators to Evolve
# =============================================================================

# Operator names (must match keys in operators.py)
OPERATORS_TO_OPTIMIZE = ["tsp_oropt", "tsp_3opt", "tsp_2opt"]

# =============================================================================
# 4. Progressive MCTS Search Parameters
# =============================================================================

MCTS_CONFIG = {
    "max_iterations": 2,        # Number of MCTS iterations
    "max_simulations": 1,       # Simulations per node
    "max_sampling_num": 1,      # Max templates sampled per operator
}

# =============================================================================
# 5. Operator Rotation Evolution Parameters
# =============================================================================

ROTATION_CONFIG = {
    "max_cycles": 1,            # Coordinate descent cycles
}

# =============================================================================
# 6. EoH Generator Parameters (Single Operator Design)
# =============================================================================

EOH_CONFIG = {
    "max_generations": 2,       # Max EoH generations
    "max_sample_nums": 5,       # Max LLM samples per EoH run
    "pop_size": 3,              # Population size
    "num_samplers": 1,          # Parallel sampling threads
    "num_evaluators": 1,        # Parallel evaluation threads
}

# =============================================================================
# 7. MOEA Evaluation Parameters
# =============================================================================

EVAL_CONFIG = {
    "n_evals": 2,               # Number of repeated evaluations
    "generation_num": 10,       # MOEA generations per evaluation
    "pop_size": 50,             # MOEA population size
}

# =============================================================================
# 8. Output and Experiment Configuration
# =============================================================================

import time

# Use fixed directory for cache reuse during development.
# Change to "TSP_" + time.strftime("%Y%m%d_%H%M%S") for production.
EXP_DIR = "TSP_test"
LOG_DIR = "outputs_tsp/" + EXP_DIR + "/eoh_log/{}"
