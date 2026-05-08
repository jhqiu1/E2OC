"""MCTS-based progressive search for operator design strategies.

Reference:
    Qiu et al., "Evolving Interdependent Operators with Large Language Models
    for Multi-Objective Combinatorial Optimization", ICML 2026.
"""

import random
import math
import os
import datetime
import json
from copy import deepcopy


class Node:
    """Monte Carlo Tree Search node representing a partial operator design state."""

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def ucb_score(self, c=1.41):
        if self.visit_count == 0:
            return float("inf")
        avg_reward = self.total_reward / self.visit_count
        return avg_reward + c * math.sqrt(
            math.log(self.parent.visit_count + 1) / self.visit_count
        )


class TemplateStorage:
    """Manages prompt templates for operator design strategies."""

    def __init__(self, origin_template, llm, main_args):
        self.origin_template = origin_template
        self.template_storage = {}
        self.best_pop = {}
        self.initialize_storage(origin_template)
        self.llm = llm
        self.pri_operators = []
        self.output_dir = main_args["exp_dir"]
        self.main_args = main_args

    def initialize_storage(self, origin_template):
        for temp in origin_template:
            if temp not in self.template_storage:
                self.template_storage[temp] = []
                self.best_pop[temp] = []
            self.template_storage[temp].append(origin_template[temp])

    def sample_template(self, var_ope_index, code):
        operator_name = self.main_args["operators_to_optimize"][var_ope_index]
        raw = self.llm.draw_sample(
            self.main_args["prompt_gen"].prompt_newTemplate(
                code,
                self.main_args["template_dict"][operator_name + "_template"],
            )
        )
        text = raw.strip()
        if text.startswith("```"):
            nl = text.find("\n")
            text = text[nl + 1 :] if nl != -1 else text
            if text.strip().endswith("```"):
                text = text[: text.rfind("```")]
        template = text.strip()
        self.template_storage[operator_name + "_template"].append(template)
        return template

    def update_best_operator(self, state, operators_best, scores, cycle_index, instance_path):
        pop = {"ope": operators_best, "scores": scores}
        self.pri_operators.append(pop)

        best_ope_dict = max(self.pri_operators, key=lambda x: x["scores"])
        best_ope = best_ope_dict["ope"]

        current_best = {
            "timestamp": datetime.datetime.now().isoformat(),
            "cycle": cycle_index,
            "cycle_best_operators": operators_best,
            "score": scores,
            "operator": best_ope,
        }

        json_file_path = os.path.join(
            "outputs_tsp", self.output_dir, f"best_results_{instance_path}.json"
        )
        from e2oc.utils.result_save import save_operators_results
        save_operators_results(json_file_path, current_best)


class MCTS:
    """Progressive design strategy search using Monte Carlo Tree Search.

    Explores combinations of operator design thoughts in the language space
    to identify the best design strategy for multi-operator evolution.

    Args:
        main_args: Dictionary containing all configuration parameters.
        llm: LLM instance for generating templates and operators.
        storages: Storages instance for managing operator history.
        max_iterations: Maximum MCTS iterations (outer loop).
        max_simulations: Number of simulations per node.
        max_sampling_num: Maximum number of templates sampled per operator.
    """

    def __init__(
        self,
        main_args,
        llm,
        storages,
        max_iterations=1000,
        max_simulations=1,
        max_sampling_num=4,
    ):
        self.operators_to_optimize = main_args["operators_to_optimize"]
        self.main_args = main_args
        self.max_iterations = max_iterations
        self.max_simulations = max_simulations
        self.max_sampling_num = max_sampling_num
        self.llm = llm
        self.storages = storages
        self.output_dir = main_args["exp_dir"]
        [self.instance_path] = main_args["instance_name"]

        self.history = {
            "best_solutions": [],
            "variable_spaces": [],
            "exploration_paths": [],
            "best_rewards": [],
        }
        self.history_var = {}

    def select(self, node):
        if not node.children:
            return node
        return max(node.children, key=lambda n: n.ucb_score())

    def initialize_state(self, node, origin_state):
        for next_var_index in range(len(origin_state)):
            var_ope = self.operators_to_optimize[next_var_index]
            if var_ope not in self.history_var:
                self.history_var[var_ope] = [origin_state[var_ope + "_template"]]

        self.template_storage = TemplateStorage(
            self.main_args["template_dict"], self.llm, self.main_args
        )
        self.sample_all_templates()

    def sample_all_templates(self):
        from llm4ad.base.code import TextFunctionProgramConverter

        file_path = os.path.join(
            "outputs_tsp", self.main_args["storage_dir"], "template_init.json"
        )

        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            with open(file_path, "r", encoding="utf-8") as f:
                template_dict = json.load(f)
            if template_dict:
                valid_count = 0
                invalid_count = 0
                for op_name, templates in template_dict.items():
                    valid_templates = []
                    for t in templates:
                        if t and TextFunctionProgramConverter.text_to_function(t) is not None:
                            valid_templates.append(t)
                        else:
                            invalid_count += 1
                    template_dict[op_name] = valid_templates
                if invalid_count > 0:
                    print(f"Filtered out {invalid_count} invalid templates from loaded file")
                if all(len(v) > 0 for v in template_dict.values()):
                    print("Successfully loaded template data from file")
                    self.history_var = template_dict
                    return
                else:
                    print("All templates filtered out, regenerating...")

        print("File not found or empty, performing template sampling")

        for var_ope_index in range(len(self.operators_to_optimize)):
            operation_name = self.operators_to_optimize[var_ope_index]
            alg_set = self.storages.latest[operation_name].individuals
            sorted_alg_set = sorted(alg_set, key=lambda x: (-x["score"], x["code"]))

            for num_index in range(self.max_sampling_num - 1):
                if (
                    len(self.history_var[operation_name]) <= self.max_sampling_num
                    and num_index + 1 < len(sorted_alg_set)
                ):
                    alg_set = sorted_alg_set[num_index]
                    new_temp = self.template_storage.sample_template(
                        var_ope_index, alg_set["code"]
                    )
                    if new_temp and TextFunctionProgramConverter.text_to_function(new_temp) is not None:
                        self.history_var[operation_name].append(new_temp)
                    else:
                        print(f"  Skipping invalid template for {operation_name}")

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.history_var, f, ensure_ascii=False, indent=2)

    def expand(self, generation, node):
        if len(node.state) >= len(self.operators_to_optimize):
            return

        next_var_index = len(node.state)
        var_ope = self.operators_to_optimize[next_var_index]

        for value in self.history_var[var_ope]:
            new_state = node.state + [value]
            child_node = Node(new_state, parent=node)
            node.children.append(child_node)

    def simulate(self, generation, simuIndex, node):
        from e2oc.rotation import quick_oc_eva

        dir = f"{generation}_{simuIndex}"

        current_state = node.state
        while len(current_state) < len(self.operators_to_optimize):
            operator_name = self.operators_to_optimize[len(current_state)]
            next_value = random.choice(self.history_var[operator_name])
            current_state.append(next_value)

        operators_best, scores = quick_oc_eva(self.main_args, dir, current_state)

        instance_name = os.path.basename(self.instance_path)
        self.template_storage.update_best_operator(
            current_state, operators_best, scores, dir, instance_name
        )

        return scores

    def backpropagate(self, node, reward):
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent

    def search(self):
        root = Node(state=[])

        self.initialize_state(root, self.main_args["template_dict"])

        for generation in range(self.max_iterations):
            node = root

            while node.children:
                node = self.select(node)

            self.expand(generation, node)

            best_reward = float("-inf")
            for simuIndex in range(self.max_simulations):
                reward = self.simulate(generation, simuIndex, node)
                if reward > best_reward:
                    best_reward = reward

            self.backpropagate(node, best_reward)

            best_node = self.get_best_complete_node(root)

            if best_node is not None:
                self.history["best_solutions"].append(best_node.state)
                self.history["exploration_paths"].append(self.get_path(best_node))
                self.history["best_rewards"].append(best_reward)
            else:
                self.history["best_solutions"].append(None)
                self.history["exploration_paths"].append(None)
                self.history["best_rewards"].append(None)

            print(f"Generation {generation + 1}: Best Reward = {best_reward}")

        return self.history

    def save_results(self):
        exp_dir = os.path.join("outputs_tsp", self.output_dir)
        os.makedirs(exp_dir, exist_ok=True)
        self.save_to_txt(os.path.join(exp_dir, "mcts_history.txt"))
        self.plot_convergence(exp_dir)

    def get_best_complete_node(self, root):
        best_node = None
        best_reward = float("-inf")

        for node in self.flatten_tree(root):
            if len(node.state) == len(self.operators_to_optimize):
                reward = (
                    node.total_reward / node.visit_count if node.visit_count > 0 else 0
                )
                if reward > best_reward:
                    best_reward = reward
                    best_node = node

        return best_node

    def flatten_tree(self, root):
        nodes = [root]
        for child in root.children:
            nodes.extend(self.flatten_tree(child))
        return nodes

    def get_path(self, node):
        path = []
        while node is not None:
            path.insert(0, node.state)
            node = node.parent
        return path

    def save_to_txt(self, filename):
        with open(filename, "w") as f:
            for i in range(len(self.history["best_solutions"])):
                f.write(f"Generation {i + 1}:\n")
                f.write(f"  Best Solution: {self.history['best_solutions'][i]}\n")
                f.write(f"  Variable Space: {self.history_var}\n")
                f.write(f"  Exploration Path: {self.history['exploration_paths'][i]}\n")
                f.write(f"  Best Reward: {self.history['best_rewards'][i]}\n\n")

    def plot_convergence(self, exp_dir):
        import matplotlib.pyplot as plt
        plt.plot(
            range(1, len(self.history["best_rewards"]) + 1),
            self.history["best_rewards"],
            label="Best Reward",
        )
        plt.xlabel("Generations")
        plt.ylabel("Best Reward")
        plt.title("Convergence Curve")
        plt.legend()

        output_dir = "outputs_tsp"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.savefig(os.path.join(exp_dir, "convergence_curve.jpg"))
        plt.close()
