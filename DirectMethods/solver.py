# -*- coding: utf-8 -*-
import numpy as np
import scipy.linalg as spla

from util import genp_factorize, gepp_factorize
from butterfly import build_recursive_butterfly


def forward(L, b):
    """Return a solution x of a linear system Lx = b with forward
    substitution.

    Arguments:
        L (array_like): A lower triangular matrix with unit diagonal.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Lx = b.
    """
    return spla.solve_triangular(L, b, lower=True, unit_diagonal=True)


def backward(U, b):
    """Return a solution x of a linear system Ux = b with backward
    substitution.

    Arguments:
        U (array_like): A upper triangular matrix.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Ux = b.
    """
    return spla.solve_triangular(U, b, lower=False, unit_diagonal=False)


def gepp(A, b):
    """Return a solution x of a linear system Ax = b with GEPP.

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
    """
    A = np.array(A, dtype=np.float)
    b = np.array(b, dtype=np.float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # PLU factorization using Gaussian Elimination with Partial Pivoting (GEPP)
    P, L, U = gepp_factorize(A)

    # solve
    y = forward(L, np.dot(P.T, b))
    x = backward(U, y)

    return x


def prbt(A, b, depth):
    """Return a solution x of a linear system Ax = b with partial recursive
    butterfly transformation (PRBT).

    This algorithm refers to the folloing article:
        Marc Baboulin et al.
        "Accelerating linear system solutions using randomization
        techniques", 2011,
        URL<https://hal.inria.fr/inria-00593306/document>.

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        depth (int): A recursion depth (> 0).

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
    """
    A = np.array(A, dtype=np.float)
    b = np.array(b, dtype=np.float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    n = A.shape[0]
    augments = 0
    while (n + augments) % (2 ** (depth - 1)):
        augments += 1

    # augment a matrix size adaptively for any size of a system
    A = spla.block_diag(A, np.identity(augments))

    # get two recursive butterfly matrices
    W = build_recursive_butterfly(n + augments, depth)
    V = build_recursive_butterfly(n + augments, depth)

    # partial recursive butterfly transformation
    A_prbt = np.dot(np.dot(W.T, A), V)[:n, :n]

    # LU factorization using Gaussian Elimination with No Pivoting (GENP)
    L, U = genp_factorize(A_prbt)

    # solve
    y = forward(L, np.dot(W.T, b))
    y = backward(U, y)
    x = np.dot(V, y)

    return x
