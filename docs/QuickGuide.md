# Quick Guide: Adapting E2OC to a New Problem

This guide walks you through adapting E2OC to your own multi-objective optimization problem. You only need to modify 4 files.

## Overview

The E2OC framework is problem-agnostic. The `e2oc/` and `llm4ad/` packages handle MCTS, evolution, LLM interaction, and storage — you don't touch them.

Your job is to describe **your problem** to the framework through these 4 files under `examples/your_problem/`:

```
examples/your_problem/
├── config.py        # All parameters
├── evaluator.py     # Problem → HV score evaluator
├── operators.py     # Default operators + templates
└── prompts.py       # LLM prompt templates
```

## Step 1: Define Your Operators (`operators.py`)

Create the operator functions that NSGA-II will use (e.g., crossover, mutation). Each operator needs:

### 1a. A default implementation

```python
import numpy as np

def my_crossover(offspring_chromosome):
    """Apply crossover to the offspring population.

    Args:
        offspring_chromosome: 2D numpy array of shape (pop_size, chrom_length).

    Returns:
        2D numpy array of same shape.
    """
    pop_size, chrom_length = offspring_chromosome.shape
    result = offspring_chromosome.copy()
    # ... your crossover logic ...
    return result
```

**Critical constraints for operator functions:**
- Input shape: `(pop_size, chrom_length)` numpy array
- Output shape: same as input
- Do NOT modify input in-place; return a new array
- Catch all exceptions internally; return a copy of the original on failure
- Prefer numpy vectorized operations

### 1b. A code template string

This is what the LLM sees as a reference. It must include the full function signature, docstring, and constraints:

```python
my_crossover_template = '''
import numpy as np

def my_crossover(offspring_chromosome):
    """Your operator description.

    Constraints:
    1. Input: offspring_chromosome is a 2D numpy array of shape (pop_size, chrom_length)
    2. Output: return an array of the same shape and dtype; do NOT modify the input in-place
    3. Must ensure output is a valid solution (e.g., valid permutation)
    4. All exceptions must be caught; return a copy of the original on failure
    5. Use only numpy; prefer vectorized operations over explicit loops
    """
    # Your default implementation here
    return offspring_chromosome.copy()
'''
```

## Step 2: Configure Parameters (`config.py`)

```python
# LLM API credentials
LLM_CONFIG = {
    "host": "api.deepseek.com",
    "api_key": "YOUR_API_KEY_HERE",
    "model": "deepseek-chat",
    "timeout": 120,
}

# Problem definition
BENCHMARK = "YourBenchmark"
OBJ_LIST = ["obj_1", "obj_2"]
REF_DICT = {"instance_001.pt": [ref1, ref2]}  # reference points for HV

# Operators to evolve (must match keys in operators.py)
OPERATORS_TO_OPTIMIZE = ["my_crossover", "my_mutation"]

# MCTS (outer loop)
MCTS_CONFIG = {
    "max_iterations": 30,
    "max_simulations": 1,
    "max_sampling_num": 3,
}

# Rotation (middle loop)
ROTATION_CONFIG = {
    "max_cycles": 5,
}

# EoH (inner loop)
EOH_CONFIG = {
    "max_generations": 5,
    "max_sample_nums": 25,
    "pop_size": 5,
    "num_samplers": 5,
    "num_evaluators": 5,
}

# MOEA evaluation
EVAL_CONFIG = {
    "n_evals": 3,
    "generation_num": 30,
    "pop_size": 100,
}
```

## Step 3: Create the Evaluator (`evaluator.py`)

This is the **most important file**. It must:
1. Accept generated operator code (from LLM)
2. Compile it into a callable function
3. Plug it into your MOEA
4. Run the MOEA on your problem instances
5. Return the Hypervolume (HV) score

### Key class: `MultiEvaluation(evaluation.Evaluation)`

You must implement:
- `__init__()` — set up algorithm, instances, operators
- `ini_problems(instance_paths)` — load problem instances
- `evaluate_program(program_str, new_callable_func)` — compile + evaluate + return score
- `evaluate_combination(operators, use_train_set)` — evaluate a full operator set

### Key function: `main_op()`

Executes **one** MOEA run with given operators and returns HV:

```python
def main_op(ref_point, algorithm, problem, generation_num,
            pop_size, operators):
    # 1. Create MOEA population
    # 2. Initialize NSGA-II with operators
    # 3. Run algorithm (with callback for per-gen stats)
    # 4. Get Pareto front → compute HV
    # 5. Return {"hv": hv, "pareto_front": pf, ...}
```

### Key function: `evaluate()`

Runs `main_op()` across multiple instances and random seeds, returns average HV:

```python
def evaluate(algorithm, problems, operators, n_evals,
             generation_num, pop_size):
    for instance in problems:
        for repeat in range(n_evals):
            res = main_op(...)
    return np.mean([r["hv"] for r in all_results])
```

### Template for adapting to your problem:

1. Replace TSP-specific problem class with your own
2. Replace `NSGA2_TSP` with your MOEA algorithm
3. Adjust the population encoding (chromosome representation)
4. Update objective function evaluation in `aimFunc` equivalent
5. Update reference points for HV calculation

## Step 4: Write Prompts (`prompts.py`)

Create a prompt class that tells the LLM about your problem:

```python
class Prompts_YourProblem:
    def __init__(self):
        self.prompt_task = (
            "Describe your multi-objective optimization problem here. "
            "Explain what the decision variables represent and what the "
            "objectives measure."
        )

    def prompt_newTemplate(self, new_alg, template):
        """Generate a refined prompt template."""
        # Return a string with:
        # - Description of the problem
        # - Reference to the current best algorithm
        # - Formatting requirements
        # - Output format specification
```

## Step 5: Create Your Problem Class

You need a problem class under `problems/your_problem/`. E2OC includes a custom pure-Python MOEA framework at `llm4ad/base/moea/`, no external MOEA installation needed.

```python
from llm4ad.base import moea as ea

class Problem_YourProblem(ea.Problem):
    def __init__(self, obj_list, instance, ref_points):
        # Define decision variable encoding
        self.M = len(obj_list)
        self.Dims = instance.n
        # ... initialize Problem ...

    def aimFunc(self, pop):
        """Evaluate population — called by the MOEA."""
        Vars = pop.Phen
        pop.ObjV = np.zeros((len(Vars), self.M))
        for i in range(len(Vars)):
            # Compute your objectives
            pop.ObjV[i, :] = self.instance.evaluate(Vars[i])
```

And an instance class that reads your data format:

```python
class YourInstance:
    def read_instance(self, filepath):
        # Load your problem instance
        pass

    def generate_random_solution(self):
        # Generate a valid random solution
        pass

    def evaluate_fitness(self, solution):
        # Compute objective values
        return obj1_value, obj2_value
```

## Step 6: Update `run_e2oc.py`

Update the imports in `run_e2oc.py` to use your new problem:

```python
from problems.your_problem import Problem_YourProblem, YourInstance
from examples.your_problem.evaluator import MultiEvaluation
from examples.your_problem.operators import (
    my_crossover, my_mutation,
    my_crossover_template, my_mutation_template,
)
from examples.your_problem.config import *
```

## Evaluation Metric: Hypervolume (HV)

E2OC uses Hypervolume (HV) as the scalar performance metric. HV measures both convergence and diversity of the Pareto front approximation:

- **Higher HV = better** (closer to true Pareto front, better spread)
- Reference point must be **worse than** (dominated by) all possible objective values
- HV is normalized by the reference region volume

## Common Mistakes & Troubleshooting

1. **Operators modify input in-place**: Always `copy()` the input array first.
2. **Invalid solutions**: Ensure your operators produce valid solutions (e.g., valid permutations for TSP).
3. **Missing error handling**: LLM-generated code may fail. Wrap with try/except and return original on failure.
4. **HV returning NaN/0**: Check that your reference point is properly set (should be worse than any solution).
5. **Population API**: The custom MOEA framework at `llm4ad/base/moea/` provides a pure-Python, self-contained engine with no binary dependencies.

## Checklist for New Problem Adaptation

- [ ] `operators.py`: 2+ operators with implementations + templates
- [ ] `config.py`: LLM keys, problem benchmark, all hyperparameters
- [ ] `evaluator.py`: MultiEvaluation + main_op + evaluate functions
- [ ] `prompts.py`: Prompt class with problem description + template refinement
- [ ] `problems/your_problem/`: Problem class + instance class
- [ ] `algorithms/your_algo/`: Your MOEA algorithm (or reuse NSGA-II)
- [ ] Instance data files
- [ ] Updated `run_e2oc.py` imports
