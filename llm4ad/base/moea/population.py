"""Population class — chromosome matrix + objective/fitness bookkeeping."""

import numpy as np


class Population:
    """Population of individuals for evolutionary algorithms.

    Attributes:
        Encoding: Encoding type ('P'=permutation, 'RI'=real-integer, etc.).
        Field: Field descriptor array (Field[0] = variable count).
        sizes: Population size (number of individuals).
        Chrom: Chromosome matrix, shape (sizes, Dim).
        Phen: Phenotype matrix, shape (sizes, Dim).
        ObjV: Objective value matrix, shape (sizes, M).
        FitnV: Fitness value matrix, shape (sizes, 1).
        CV: Constraint violation matrix, shape (sizes,) or None.
    """

    def __init__(self, Encoding, Field, NIND=0):
        self.Encoding = Encoding
        self.Field = Field
        self.sizes = NIND
        self.Chrom = None
        self.Phen = None
        self.ObjV = None
        self.FitnV = None
        self.CV = None

    def initChrom(self, NIND=None):
        """Initialize chromosome matrix with random values.

        For permutation encoding ('P'), generates random permutations.
        For real-integer encoding ('RI'), generates uniform random values.
        """
        if NIND is not None:
            self.sizes = NIND

        Dim = int(self.Field[0])

        if self.Encoding == "P":
            self.Chrom = np.zeros((self.sizes, Dim), dtype=np.int32)
            for i in range(self.sizes):
                self.Chrom[i] = np.random.permutation(Dim)
            self.Phen = self.Chrom.copy()

        elif self.Encoding == "RI":
            lb = self.Field[1]
            ub = self.Field[2]
            varTypes = self.Field[3]
            self.Chrom = np.zeros((self.sizes, Dim))
            for j in range(Dim):
                self.Chrom[:, j] = np.random.uniform(lb[j], ub[j], self.sizes)
                if varTypes[j] == 1:
                    self.Chrom[:, j] = np.round(self.Chrom[:, j])
            self.Phen = self.Chrom.copy()

        else:
            Dim = int(self.Field[0])
            self.Chrom = np.random.random((self.sizes, Dim))
            self.Phen = self.Chrom.copy()

    def __getitem__(self, indices):
        """Return a new Population containing the selected individuals."""
        if isinstance(indices, np.ndarray) and indices.dtype == bool:
            n = int(np.sum(indices))
        elif isinstance(indices, (list, np.ndarray)):
            n = len(indices)
        else:
            n = 1
            indices = [indices]

        new_pop = Population(self.Encoding, self.Field, n)

        if self.Chrom is not None:
            new_pop.Chrom = self.Chrom[indices].copy()
        if self.Phen is not None:
            new_pop.Phen = self.Phen[indices].copy()
        if self.ObjV is not None:
            new_pop.ObjV = self.ObjV[indices].copy()
        if self.FitnV is not None:
            new_pop.FitnV = self.FitnV[indices].copy()
        if self.CV is not None:
            new_pop.CV = self.CV[indices].copy()

        return new_pop

    def __add__(self, other):
        """Merge two populations (concatenate all matrices)."""
        new_sizes = self.sizes + other.sizes
        new_pop = Population(self.Encoding, self.Field, new_sizes)

        for attr in ["Chrom", "Phen", "ObjV", "FitnV", "CV"]:
            a = getattr(self, attr)
            b = getattr(other, attr)
            if a is not None and b is not None:
                setattr(new_pop, attr, np.concatenate([a, b], axis=0))
            elif a is not None:
                setattr(new_pop, attr, a.copy())
            elif b is not None:
                setattr(new_pop, attr, b.copy())

        return new_pop

    def __len__(self):
        return self.sizes

    def __repr__(self):
        return f"Population(sizes={self.sizes}, Encoding={self.Encoding})"
