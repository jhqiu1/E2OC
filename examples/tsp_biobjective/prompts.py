"""LLM prompt templates for Bi-objective TSP operator design.

This class generates prompts for:
  1. Initial operator generation (prompt_init)
  2. EoH evolution prompts (used by llm4ad)
  3. Template improvement prompts for MCTS phase (prompt_newTemplate)

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""


class Prompts_TSP:
    """Prompt templates for TSP multi-objective operator design."""

    def __init__(self):
        self.prompt_task = (
            "Given a set of cities and the distances between them, you need to determine "
            "the optimal route that minimizes both the total distance (minimizing the makespan) "
            "and the total load (balancing the route diversity). "
            "In order to optimize both the total distance and the diversity of the route, "
            "please design an algorithm that calculates the optimal sequence of cities "
            "considering these objectives."
        )

    def prompt_newTemplate(self, new_alg, template):
        """Generate a refined prompt template for multi-objective optimization algorithms.

        Args:
            new_alg: The advanced algorithm code for reference.
            template: The original prompt template to improve.

        Returns:
            A complete prompt template string with clear instructions.
        """
        description = (
            "You are an AI Python expert specializing in multi-objective optimization. "
            "Your task is to refine the following prompt template to generate more advanced "
            "Python algorithms aiming to maximize Hypervolume (HV).\n\n"
            "REFERENCE ALGORITHM (Analyze its advantages for inspiration):\n"
            f"{new_alg}\n\n"
            "ORIGINAL TEMPLATE TO IMPROVE:\n"
            f"{template}\n\n"
        )

        formatting_rules = (
            "IMPROVEMENT REQUIREMENTS:\n"
            "1. Return a COMPLETE, ready-to-use Python function including:\n"
            "   - All necessary import statements\n"
            "   - Preserved function name, input parameters (with types), and return type\n"
            "   - Full function body implementation\n"
            "2. Apply PEP 8 formatting:\n"
            "   - 4-space indentation (no tabs)\n"
            "   - Maximum 79 characters per line\n"
            "   - Proper spacing around operators\n"
            "   - Snake_case for functions/variables, CamelCase for classes\n"
            "3. Correct any syntax errors in the original template\n"
            "4. Enhance the docstring under 'Args' with 2-3 specific improvement suggestions:\n"
            "   - Focus on HV optimization techniques\n"
            "   - Reference the reference algorithm's strengths\n"
        )

        output_spec = (
            "OUTPUT FORMAT:\n"
            "Return ONLY the final improved template as a single string, formatted as:\n"
            "```python\n"
            "import necessary_libraries\n\n"
            "def function_name(param: type) -> return_type:\n"
            '    """\n'
            "    Enhanced function description.\n\n"
            "    Args:\n"
            "        param (type): Description.\n\n"
            "    Improvement Suggestions:\n"
            "        - Suggestion 1 based on reference algorithm\n"
            "        - Suggestion 2 for HV optimization\n\n"
            "    Returns:\n"
            "        return_type: Description.\n"
            '    """\n'
            "    # Implemented function body\n"
            "    return result\n"
            "```\n"
            "No additional explanations or text outside the code block."
        )

        return description + formatting_rules + output_spec
