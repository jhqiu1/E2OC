"""Utility functions for converting between code strings, JSON files, and callable functions.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import os
import json
import shutil
import numpy as np
from typing import List, Callable

from llm4ad.base.code import TextFunctionProgramConverter


def manage_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")


def load_best_operators_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    best_record = max(data, key=lambda x: x["score"])
    if "operators" in best_record:
        return best_record["operators"]
    else:
        return best_record["operator"]


def load_top_operators_from_json(json_path, top_k=5):
    with open(json_path, "r") as f:
        data = json.load(f)
    sorted_data = sorted(data, key=lambda x: x["score"], reverse=True)
    top_records = sorted_data[:top_k]
    top_operators = []
    for record in top_records:
        if "cycle_best_operators" in record:
            top_operators.append(record["cycle_best_operators"])
        else:
            top_operators.append(record["operator"])
    return top_operators


def function_to_callable(function_code):
    program = TextFunctionProgramConverter.text_to_program(function_code)
    program_str = str(program)
    function_name = TextFunctionProgramConverter.text_to_function(program_str).name
    all_globals_namespace = {}
    exec(program_str, all_globals_namespace)
    program_callable = all_globals_namespace.get(function_name)
    if program_callable is None:
        raise ValueError(f"Function '{function_name}' not found in executed code")
    return program_callable


def string_to_callable(function_string, function_name=None):
    try:
        namespace = {}
        try:
            import numpy as np
            namespace["np"] = np
        except ImportError:
            print("Warning: numpy not available")

        try:
            from typing import Tuple, List, Dict, Any, Union, Optional
            namespace["Tuple"] = Tuple
            namespace["List"] = List
            namespace["Dict"] = Dict
            namespace["Any"] = Any
            namespace["Union"] = Union
            namespace["Optional"] = Optional
        except ImportError:
            print("Warning: typing module not available")

        exec(function_string, namespace)

        if function_name is None:
            lines = function_string.strip().split("\n")
            for line in lines:
                if line.startswith("def "):
                    function_name = line.split("def ")[1].split("(")[0].strip()
                    break

        callable_func = namespace.get(function_name)

        if callable_func is not None and callable(callable_func):
            return callable_func
        else:
            print(f"Warning: Function '{function_name}' not found or not callable in code")
            return None

    except Exception as e:
        print(f"Error converting string to callable: {e}")
        return None
