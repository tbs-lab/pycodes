# -*- coding: utf-8 -*-
import numpy
import scipy.linalg as spla

from factorization import lu, plu
from butterfly import build_recursive_butterfly


def forward_solve(L, b):
    """Return a solution x of a linear system Lx = b with forward
    substitution.

    Arguments:
        L (array_like): A lower triangular matrix with unit diagonal.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Lx = b.
    """
    return spla.solve_triangular(L, b, lower=True, unit_diagonal=True)


def backward_solve(U, b):
    """Return a solution x of a linear system Ux = b with backward
    substitution.

    Arguments:
        U (array_like): A upper triangular matrix.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Ux = b.
    """
    return spla.solve_triangular(U, b, lower=False, unit_diagonal=False)


def plu_solve(A, b):
    """Return a solution x of a linear system Ax = b with GEPP.

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
    """
    A = numpy.array(A, dtype=numpy.float)
    b = numpy.array(b, dtype=numpy.float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # PLU factorization using Gaussian Elimination with Partial Pivoting (GEPP)
    P, L, U = plu(A)

    # solve
    y = forward_solve(L, P.T @ b)
    x = backward_solve(U, y)

    return x


def prbt_solve(A, b, d):
    """Return a solution x of a linear system Ax = b with partial recursive
    butterfly transformation (PRBT).

    This algorithm refers to the folloing article:
        Accelerating linear system solutions using randomization techniques
        (Marc Baboulin et al. 2011),
        URL<https://hal.inria.fr/inria-00593306/document>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        d (int): A recursion depth (> 0).

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
    """
    A = numpy.array(A, dtype=numpy.float)
    b = numpy.array(b, dtype=numpy.float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    n = A.shape[0]
    augments = 0
    while (n + augments) % (2 ** (d - 1)):
        augments += 1

    # augment a matrix size adaptively for any size of a system
    A = spla.block_diag(A, numpy.identity(augments))

    # get two recursive butterfly matrices
    W = build_recursive_butterfly(n + augments, d)
    V = build_recursive_butterfly(n + augments, d)

    # partial recursive butterfly transformation
    A_prbt = (W.T @ A @ V)[:n, :n]

    # LU factorization without pivoting
    L, U = lu(A_prbt)

    # solve
    y = forward_solve(L, W.T @ b)
    y = backward_solve(U, y)
    x = V @ y

    return x
