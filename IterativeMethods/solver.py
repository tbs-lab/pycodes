# -*- coding: utf-8 -*-
import sys
import numpy
import scipy.linalg as spla


def jacobi_solve(A, b, max_iterations):
    """Return a solution x of a linear system Ax = b with Jacobi iteration
    Method.

    This algorithm refers to the folloing article:
        The Algorithm for The Jacobi Iteration Method - Mathonline
        URL<http://mathonline.wikidot.com/the-algorithm-for-the-jacobi-iteration-method>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        max_iterations (int): A maximum number of iterations.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
        int: A number of iterations.
    """
    A = numpy.array(A, dtype=numpy.float)
    b = numpy.array(b, dtype=numpy.float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # lower and upper triangular matrix of A
    LU = numpy.tril(A, -1) + numpy.triu(A, 1)
    # diagonal vector of A
    d = A.diagonal()
    # solution vector
    x = numpy.ones(b.shape[0])
    # current number of iteration
    k = 0
    # residual 2-norm
    residual = numpy.inf

    while k < max_iterations and residual >= sys.float_info.epsilon:
        _x = x
        x = (b - LU @ _x) / d
        residual = numpy.linalg.norm(x - _x)
        k += 1

    return x, k


def sor_solve(A, b, omega, max_iterations):
    """Return a solution x of a linear system Ax = b with Successive
    Over Relaxation Method. If omega == 1, then this corresponds with
    Gauss-Seidel Method.

    This algorithm refers to the folloing article:
        Andrew Stuart and Jochen Voss,
        "Matrix Analysis and Algorithms", 2009,
        URL<https://www.seehuhn.de/pages/numlinalg.html>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        omega (float): A parameter to accelerate convergence (0 < omega < 2).
        max_iterations (int): A maximum number of iterations.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
        int: A number of iterations.
    """
    A = numpy.array(A, dtype=numpy.float)
    b = numpy.array(b, dtype=numpy.float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # diagonal vector of A
    d = A.diagonal()
    # lower triangular matrix to use in iteration
    L = numpy.tril(A, -1) + numpy.diag(omega * d)
    # upper triangular matrix to use in iteration
    U = numpy.triu(A, 1) + numpy.diag((1 - omega) * d)
    # solution vector
    x = numpy.ones(b.shape[0])
    # current number of iteration
    k = 0
    # residual 2-norm
    residual = numpy.inf

    while k < max_iterations and residual >= sys.float_info.epsilon:
        _x = x
        y = b - U @ _x
        # forward substitution
        x = spla.solve_triangular(L, y, lower=True, unit_diagonal=False)
        residual = numpy.linalg.norm(x - _x)
        k += 1

    return x, k
