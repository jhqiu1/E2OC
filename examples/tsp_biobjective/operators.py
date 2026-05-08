"""Operator definitions and code templates for Bi-objective TSP.

Each operator has TWO parts:
  1. A default implementation (the actual Python function)
  2. A code template string (what the LLM sees as reference for generation)

When adapting to a new problem:
  - Replace the operator implementations with your own
  - Update the code templates to describe your desired function signature
  - The template MUST include: import statements, function signature, docstring with constraints

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import numpy as np


# =============================================================================
# Default Operator Implementations
# =============================================================================

def tsp_2opt(offspring_chromosome):
    """2-opt Mutation Operator: reverses a segment of the tour."""
    pop_size = offspring_chromosome.shape[0]
    chrom_length = offspring_chromosome.shape[1]

    for i in range(pop_size):
        swap_idx = np.random.choice(chrom_length, 2, replace=False)
        city1, city2 = sorted(swap_idx)
        offspring_chromosome[i, city1 : city2 + 1] = np.flip(
            offspring_chromosome[i, city1 : city2 + 1]
        )

    return offspring_chromosome


def tsp_3opt(offspring_chromosome):
    """3-opt Mutation Operator: reconnects three segments of the tour."""
    pop_size = offspring_chromosome.shape[0]
    chrom_length = offspring_chromosome.shape[1]

    for i in range(pop_size):
        swap_idx = np.random.choice(chrom_length, 3, replace=False)
        city1, city2, city3 = sorted(swap_idx)

        temp = np.copy(offspring_chromosome[i])
        offspring_chromosome[i, city1 : city3 + 1] = np.concatenate(
            (temp[city1 : city2 + 1], np.flip(temp[city2 + 1 : city3 + 1]))
        )

    return offspring_chromosome


def tsp_oropt(offspring_chromosome, dist_matrix=None, apply_if_improves=False):
    """Or-Opt style mutation: removes a segment and reinserts it elsewhere."""
    offspring_chromosome = offspring_chromosome.copy()
    pop_size, chrom_length = offspring_chromosome.shape

    def tour_length(tour, dist_m):
        return sum(dist_m[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour)))

    for i in range(pop_size):
        tour = offspring_chromosome[i]
        start, end = sorted(np.random.choice(chrom_length, 2, replace=False))
        segment = tour[start : end + 1]
        remaining = np.concatenate([tour[:start], tour[end + 1 :]])

        insert_pos = np.random.randint(len(remaining) + 1)
        new_tour = np.insert(remaining, insert_pos, segment)

        if apply_if_improves and dist_matrix is not None:
            current_len = tour_length(tour, dist_matrix)
            new_len = tour_length(new_tour, dist_matrix)
            if new_len < current_len:
                offspring_chromosome[i] = new_tour
        else:
            offspring_chromosome[i] = new_tour

    return offspring_chromosome


# =============================================================================
# Code Templates (shown to LLM as reference for generating new operators)
# =============================================================================

tsp_2opt_template = '''

import numpy as np
from typing import Tuple
import random


def tsp_2opt(offspring_chromosome):
    """2-opt Mutation Operator.

    Constraints:
    1. Input: offspring_chromosome is a 2D numpy array of shape (pop_size, chrom_length)
    2. Output: return an array of the same shape and dtype; do NOT modify the input in-place
    3. Must ensure output is a valid permutation (no duplicate city indices)
    4. All exceptions must be caught; return a copy of the original on failure
    5. Use only numpy; prefer vectorized operations over explicit loops

    """
    pop_size = offspring_chromosome.shape[0]
    chrom_length = offspring_chromosome.shape[1]

    for i in range(pop_size):
        swap_idx = np.random.choice(chrom_length, 2, replace=False)
        city1, city2 = sorted(swap_idx)

        offspring_chromosome[i, city1 : city2 + 1] = np.flip(
            offspring_chromosome[i, city1 : city2 + 1]
        )

    return offspring_chromosome
'''

tsp_3opt_template = '''

import numpy as np
from typing import Tuple
import random

def tsp_3opt(offspring_chromosome):
    """3-opt Mutation Operator.

    Constraints:
    1. Input: offspring_chromosome is a 2D numpy array of shape (pop_size, chrom_length)
    2. Output: return an array of the same shape and dtype; do NOT modify the input in-place
    3. Must ensure output is a valid permutation (no duplicate city indices)
    4. All exceptions must be caught; return a copy of the original on failure
    5. Use only numpy; prefer vectorized operations over explicit loops

    """
    pop_size = offspring_chromosome.shape[0]
    chrom_length = offspring_chromosome.shape[1]

    for i in range(pop_size):
        swap_idx = np.random.choice(chrom_length, 3, replace=False)
        city1, city2, city3 = sorted(swap_idx)

        temp = np.copy(offspring_chromosome[i])
        offspring_chromosome[i, city1 : city3 + 1] = np.concatenate(
            (temp[city1 : city2 + 1], np.flip(temp[city2 + 1 : city3 + 1]))
        )

    return offspring_chromosome
'''

tsp_oropt_template = '''

import numpy as np
from typing import Tuple
import random

def tsp_oropt(offspring_chromosome, dist_matrix=None, apply_if_improves=False):
    """Or-Opt Mutation Operator.

    Constraints:
    1. Input: offspring_chromosome is a 2D numpy array of shape (pop_size, chrom_length)
    2. Output: return an array of the same shape and dtype; do NOT modify the input in-place
    3. Must ensure output is a valid permutation (no duplicate city indices)
    4. All exceptions must be caught; return a copy of the original on failure
    5. Use only numpy; prefer vectorized operations over explicit loops

    """
    offspring_chromosome = offspring_chromosome.copy()
    pop_size, chrom_length = offspring_chromosome.shape

    def tour_length(tour, dist_m):
        return sum(dist_m[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

    for i in range(pop_size):
        tour = offspring_chromosome[i]
        start, end = sorted(np.random.choice(chrom_length, 2, replace=False))
        segment = tour[start:end+1]
        remaining = np.concatenate([tour[:start], tour[end+1:]])

        insert_pos = np.random.randint(len(remaining) + 1)
        new_tour = np.insert(remaining, insert_pos, segment)

        if apply_if_improves and dist_matrix is not None:
            current_len = tour_length(tour, dist_matrix)
            new_len = tour_length(new_tour, dist_matrix)
            if new_len < current_len:
                offspring_chromosome[i] = new_tour
        else:
            offspring_chromosome[i] = new_tour

    return offspring_chromosome
'''
