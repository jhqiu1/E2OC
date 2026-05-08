"""Generator — abstract base class for LLM-driven operator design methods.

The Generator is the INNER loop of the E2OC three-layer framework.
It takes a single operator's evaluation function + prompt template and
evolves/optimizes the operator's implementation using an LLM.

Concrete implementations:
  - EoH (Evolution of Heuristics): evolutionary population + LLM sampling
  - Future: FunSearch, ReEvo, etc.

All generators follow the same interface so the middle rotation layer
(and outer MCTS layer) are agnostic to which generator is used.
"""

from abc import ABC, abstractmethod
from typing import Optional


class Generator(ABC):
    """Abstract base class for LLM-based operator generation methods.

    A Generator optimizes ONE operator at a time. The middle rotation
    loop (coordinate descent) calls the Generator repeatedly, each time
    freezing all operators except one.

    Subclasses must implement:
      - run(): Execute the generation process.
      - get_population(): Return the final population of generated programs.

    Optional:
      - get_best_score(): Return the best score found.
      - get_best_function(): Return the best function found.
    """

    @abstractmethod
    def run(self) -> None:
        """Execute the generation process.

        This is the main entry point. It should:
          1. Generate initial population (from prompt template)
          2. Iteratively sample new candidates from LLM
          3. Evaluate candidates via the Evaluation object
          4. Apply survival selection
          5. Repeat until budget exhausted
        """

    @abstractmethod
    def get_population(self):
        """Return the final population of generated programs.

        Returns:
            A Population object whose .population attribute is a list
            of Function/Program objects, each with a .score attribute.
        """

    def get_best_score(self) -> Optional[float]:
        """Return the best (highest) score achieved.

        Override if the generator tracks this differently.
        """
        pop = self.get_population()
        if pop is None or not getattr(pop, "population", None):
            return None
        scores = [ind.score for ind in pop.population if hasattr(ind, "score")]
        return max(scores) if scores else None

    def get_best_function(self):
        """Return the Function/Program with the highest score.

        Override if the generator tracks this differently.
        """
        pop = self.get_population()
        if pop is None or not getattr(pop, "population", None):
            return None
        best = None
        best_score = float("-inf")
        for ind in pop.population:
            if hasattr(ind, "score") and ind.score > best_score:
                best_score = ind.score
                best = ind
        return best
