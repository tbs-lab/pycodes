# -*- coding: utf-8 -*-
import numpy
from scipy.linalg import solve_triangular, block_diag

from lu import lu_no_pivoting
from butterfly import build_recursive_butterfly


def forward_solve(L, b):
    """Return a solution x of a linear system Lx = b with forward substitution.

    Arguments:
        L (numpy.ndarray): A lower triangular matrix with unit diagonal.
        b (numpy.ndarray): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Lx = b.
    """
    return solve_triangular(L, b, lower=True, unit_diagonal=True)


def backward_solve(U, b):
    """Return a solution x of a linear system Ux = b with backward substitution.

    Arguments:
        U (numpy.ndarray): A upper triangular matrix.
        b (numpy.ndarray): A right-hand side vector.

    Returns:
        numpy.ndarray: A solution of a linear system Ux = b.
    """
    return solve_triangular(U, b, lower=False, unit_diagonal=False)


def prbt_solve(A, b, d):
    """Return a solution x of a linear system Ax = b with partial recursive butterfly
    transformation (PRBT).

    This algorithm refers to the folloing article.
        URL: https://hal.inria.fr/inria-00593306/document

    Arguments:
        A (numpy.ndarray): A square coefficient matrix.
        b (numpy.ndarray): A right-hand side vector.
        d (int): A recursion depth (> 0).

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
    """
    if A.shape[0] != A.shape[1]:
        return ValueError("matrix must be square one")

    n = A.shape[0]
    augments = 0
    while (n + augments) % (2 ** (d - 1)):
        augments += 1

    # augment a matrix size adaptively for any size of a system
    A = block_diag(A, numpy.identity(augments))

    # get two recursive butterfly matrices
    W = build_recursive_butterfly(n + augments, d)
    V = build_recursive_butterfly(n + augments, d)

    # partial recursive butterfly transformation
    Ar = (W.T @ A @ V)[:n, :n]

    # LU factorization
    L, U = lu_no_pivoting(Ar)

    # Solve
    y = forward_solve(L, W.T @ b)
    y = backward_solve(U, y)
    x = V @ y

    return x
