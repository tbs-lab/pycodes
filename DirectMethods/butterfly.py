# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as spla


class Butterfly(object):
    """Class of a butterfly matrix.

    The diagonal random values are defined by exp(r / 10) (|r| <= 0.5).
    This algorithm refers to the folloing article:
        Marc Baboulin et al.
        "Accelerating linear system solutions using randomization
        techniques", 2011,
        URL<https://hal.inria.fr/inria-00593306/document>.

    Notes:
        This class is only for internal usage.

    Attributes:
        R0 (numpy.ndarray): A first diagonal random matrix of size (n / 2).
            A matrix is stored compactly in a 1-dimensional array.
        R1 (numpy.ndarray): A last diagonal random matrix of size (n / 2).
            A matrix is stored compactly in a 1-dimensional array.
    """

    def __init__(self, n):
        """A constructor of a instance.

        Arguments:
            n (int): Size of a matrix. It must be a multiple of 2.
        """
        if n % 2:
            raise ValueError("size of matrix must be a multiple of 2")

        self._R0 = np.exp((np.random.rand(n // 2) - 0.5) / 10)
        self._R1 = np.exp((np.random.rand(n // 2) - 0.5) / 10)

    @property
    def R0(self):
        """A first diagonal random matrix of size (n / 2)."""
        return self._R0

    @property
    def R1(self):
        """A last diagonal random matrix of size (n / 2)."""
        return self._R1

    def __array__(self):
        """Return a reference to self."""
        B = np.block([[np.diag(self.R0), np.diag(self.R1)],
                      [np.diag(self.R0), -np.diag(self.R1)]])
        return np.sqrt(0.5) * B


def build_recursive_butterfly(n, depth):
    """Return a recursive butterfly matrix of a specified depth.

    This algorithm refers to the folloing article:
        Marc Baboulin et al.
        "Accelerating linear system solutions using randomization
        techniques", 2011,
        URL<https://hal.inria.fr/inria-00593306/document>.

    Arguments:
        n (int): Size of a matrix. It must be a multiple of 2^(depth-1).
        depth (int): A recursion depth (> 0).

    Returns:
        numpy.ndarray: Recursive butterfly matrix of depth d.
    """
    if depth < 1:
        raise ValueError("recursion depth must be positive integer")

    if depth == 1:
        return np.array(Butterfly(n))

    W = np.array(Butterfly(n // (2 ** (depth - 1))))
    for _ in range(1, 2 ** (depth - 1)):
        W = spla.block_diag(W, Butterfly(n // (2 ** (depth - 1))))

    return np.dot(W, build_recursive_butterfly(n, depth - 1))
