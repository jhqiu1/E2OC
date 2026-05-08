"""Base Recombination class for E2OC."""


class Recombination:
    """Base class for recombination (crossover) operators."""

    def __init__(self, XOVR=0.7):
        self.XOVR = XOVR

    def do(self, OldChrom):
        raise NotImplementedError
