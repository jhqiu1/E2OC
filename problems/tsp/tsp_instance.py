"""TSP instance base classes and Bi-objective TSP instance.

Provides:
  - mTSP_instance: Base class with common TSP utilities.
  - BiTSP_instance: Reads .pt files with two distance matrices.
"""

import json
import os
import numpy as np


class mTSP_instance:
    """Base class for multi-objective TSP instances.

    Provides distance matrix storage, random solution generation,
    fitness evaluation (2-objective), and JSON-based serialization.
    """

    def __init__(self):
        self.n = None
        self.edge_weight_1 = None
        self.edge_weight_2 = None
        self.variable_info = None
        self.template = None
        self.args = None

    def generate_instance(self, node_counts):
        """Generate multiple symmetric random matrix instances."""
        instances = []
        for n in node_counts:
            mat1 = np.random.uniform(0, 1, size=(n, n))
            mat2 = np.random.beta(a=2, b=5, size=(n, n))

            mat1 = (mat1 + mat1.T) / 2
            mat2 = (mat2 + mat2.T) / 2
            np.fill_diagonal(mat1, 0)
            np.fill_diagonal(mat2, 0)

            instances.append((n, mat1, mat2))
        return instances

    def save_instance(self, base_filename, node_counts, output_dir):
        """Generate and save instances to JSON files."""
        instances = self.generate_instance(node_counts)
        base, ext = os.path.splitext(base_filename)
        if not ext:
            ext = ".json"

        for n, mat1, mat2 in instances:
            data = {
                "n": n,
                "edge_weight_1": mat1.tolist(),
                "edge_weight_2": mat2.tolist(),
            }
            filename = f"{base}_n{n}{ext}"
            filepath = os.path.join(output_dir, filename)
            os.makedirs(output_dir, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Instance with n={n} saved to {filepath}")

    def read_instance(self, filename):
        """Read instance from JSON file and set class attributes."""
        with open(filename, "r") as f:
            data = json.load(f)
        self.n = data["n"]
        self.edge_weight_1 = np.array(data["edge_weight_1"])
        self.edge_weight_2 = np.array(data["edge_weight_2"])
        self.generate_variable_info()
        self.args = {
            "n": self.n,
            "edge_weight_1": self.edge_weight_1,
            "edge_weight_2": self.edge_weight_2,
        }
        print(f"Instance loaded: {filename}. Node count: {self.n}")

    def generate_variable_info(self):
        self.variable_info = {
            "permutation": {
                "start_index": 0,
                "end_index": self.n,
                "length": self.n,
                "lbs": [0 for _ in range(self.n)],
                "ubs": [self.n for _ in range(self.n)],
                "discrete": True,
                "type": "permutation",
            }
        }

    def generate_random_solution(self):
        if self.n is None:
            raise ValueError(
                "Instance not loaded. Please read or generate an instance first."
            )
        return np.random.permutation(self.n)

    def evaluate_fitness(self, solution):
        """Compute total path length under both edge weight matrices.

        Args:
            solution: Permutation of city indices, shape (n,).

        Returns:
            Tuple of (total_distance_1, total_distance_2).
        """
        if len(solution) != self.n:
            raise ValueError(
                f"Solution length {len(solution)} != expected {self.n}."
            )
        if set(solution) != set(range(self.n)):
            raise ValueError("Solution is not a valid permutation of 0 to n-1.")

        total_1 = 0
        total_2 = 0
        for i in range(self.n):
            from_node = int(solution[i])
            to_node = int(solution[(i + 1) % self.n])
            total_1 += self.edge_weight_1[from_node][to_node]
            total_2 += self.edge_weight_2[from_node][to_node]
        return total_1, total_2


class BiTSP_instance(mTSP_instance):
    """Bi-objective TSP instance loaded from .pt files.

    Each instance is a tensor of shape (n, 4):
      - columns 0:2 → first objective coordinates
      - columns 2:4 → second objective coordinates
    """

    def __init__(self):
        super().__init__()
        self.location_1 = None
        self.location_2 = None

    def read_instance(self, filename, idx=0):
        import torch

        instances = torch.load(filename, map_location="cpu")
        instance = instances[idx]

        coords_obj1 = instance[:, :2]
        coords_obj2 = instance[:, 2:]

        self.location_1 = coords_obj1
        self.location_2 = coords_obj2

        def pairwise_distance_matrix(x):
            diff = x.unsqueeze(1) - x.unsqueeze(0)
            return torch.norm(diff, dim=2)

        dist_matrix_obj1 = pairwise_distance_matrix(coords_obj1)
        dist_matrix_obj2 = pairwise_distance_matrix(coords_obj2)

        self.edge_weight_1 = dist_matrix_obj1.numpy()
        self.edge_weight_2 = dist_matrix_obj2.numpy()

        self.n = instance.shape[0]
        self.generate_variable_info()
        self.args = {
            "n": self.n,
            "edge_weight_1": self.edge_weight_1,
            "edge_weight_2": self.edge_weight_2,
        }
        print(f"Instance loaded: {filename}. Node count: {self.n}")

    def generate_greedy_solution(self, weight, init_strategy="random"):
        """Generate a greedy TSP solution using weighted edge cost.

        Args:
            weight: [w1, w2] weights for objectives.
            init_strategy: 'random', 'center', or 'min_sum'.

        Returns:
            np.ndarray of city indices forming a greedy tour.
        """
        n = self.edge_weight_1.shape[0]
        w1, w2 = weight
        total_weight = w1 * self.edge_weight_1 + w2 * self.edge_weight_2

        if init_strategy == "random":
            current = np.random.randint(0, n)
        elif init_strategy == "center":
            center = (self.location_1 + self.location_2) / 2
            centroid = center.mean(axis=0)
            dists = np.linalg.norm(center - centroid, axis=1)
            current = np.argmin(dists)
        elif init_strategy == "min_sum":
            total_dists = total_weight.sum(axis=1)
            current = np.argmin(total_dists)
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        visited = {current}
        tour = [current]

        for _ in range(n - 1):
            unvisited = list(set(range(n)) - visited)
            next_node = min(unvisited, key=lambda j: total_weight[current, j])
            tour.append(next_node)
            visited.add(next_node)
            current = next_node

        return np.array(tour)
