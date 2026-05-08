"""Base Mutation class for E2OC."""


class Mutation:
    """Base class for mutation operators."""

    def __init__(self, Pm=0.1):
        self.Pm = Pm

    def do(self, Encoding, OldChrom, Field):
        raise NotImplementedError
