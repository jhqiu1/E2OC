"""Save best operator results to JSON files.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import json
import os


def save_operators_results(json_file_path, current_best):
    """Save current best operator combination to a JSON file.

    Args:
        json_file_path: Path to the output JSON file.
        current_best: Dictionary containing current best results.
    """
    os.makedirs(os.path.dirname(json_file_path), exist_ok=True)

    if os.path.exists(json_file_path) and os.path.getsize(json_file_path) > 0:
        with open(json_file_path, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    existing_data.append(current_best)

    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    print(f"Best operator results saved to: {json_file_path}")
