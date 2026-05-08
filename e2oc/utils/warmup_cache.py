"""Warm-start cache: save and load pre-computed operator populations.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import os
import json


def _serialize_population_simple(eoh_population):
    """Extract [{code:str, score:float, algorithm:str}, ...] from an EoH population."""
    pop = getattr(eoh_population, "population", None) or []
    arr = []
    for ind in pop:
        try:
            code = str(ind)
        except Exception:
            code = ""
        try:
            sc = float(getattr(ind, "score", float("-inf")))
        except Exception:
            sc = float("-inf")
        try:
            alg = str(getattr(ind, "algorithm", ""))
        except Exception:
            alg = ""
        arr.append({"code": code, "score": sc, "algorithm": alg})
    return arr


def warmup_cache_dir(exp_dir):
    return os.path.join("outputs_tsp", exp_dir, "storage", "warmup")


def warmup_cache_path(exp_dir, operator_name):
    return os.path.join(warmup_cache_dir(exp_dir), f"{operator_name}.json")


def save_warmup_batch(exp_dir, operator_name, individuals):
    path = warmup_cache_path(exp_dir, operator_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(individuals, f, ensure_ascii=False, indent=2)


def load_warmup_batch(exp_dir, operator_name):
    path = warmup_cache_path(exp_dir, operator_name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
