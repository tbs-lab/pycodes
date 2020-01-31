# -*- coding: utf-8 -*-
import sys
import np
import scipy.linalg as spla


def jacobi(A, b, max_iterations):
    """Return a solution x of a linear system Ax = b with Jacobi iteration
    Method.

    This algorithm refers to the folloing article:
        Fujio Kako
        "数値解析 4 連立方程式の求解", 2019,
        URL<https://www.ics.nara-wu.ac.jp/~kako/teaching/na/chap4.pdf>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        max_iterations (int): A maximum number of iterations.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
        int: A number of iterations.
    """
    A = np.array(A, dtype=np.float)
    b = np.array(b, dtype=np.float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # lower and upper triangular matrix of A
    LU = np.tril(A, -1) + np.triu(A, 1)
    # inverse diagonal vector of A
    dt = 1 / A.diagonal()
    # solution vector
    x = np.zeros(b.shape[0])
    # current number of iteration
    k = 0
    # residual 2-norm
    residual = np.inf

    while k < max_iterations and residual >= sys.float_info.epsilon:
        _x = x
        x = dt * (b - np.dot(LU, _x))
        residual = np.linalg.norm(x - _x)
        k += 1

    return x, k


def sor(A, b, omega, max_iterations):
    """Return a solution x of a linear system Ax = b with Successive
    Over Relaxation Method. If omega == 1, then this corresponds with
    Gauss-Seidel Method.

    This algorithm refers to the folloing article:
        Fujio Kako
        "数値解析 4 連立方程式の求解", 2019,
        URL<https://www.ics.nara-wu.ac.jp/~kako/teaching/na/chap4.pdf>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        omega (float): A parameter to accelerate convergence (0 < omega < 2).
        max_iterations (int): A maximum number of iterations.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
        int: A number of iterations.
    """
    A = np.array(A, dtype=np.float)
    b = np.array(b, dtype=np.float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # identity matrix
    I = np.identity(A.shape[0])
    # inverse diagonal vector of A
    dt = 1 / A.diagonal()
    # unit lower triangular matrix to use in iteration
    L = I + (omega * dt).reshape(dt.size) * np.tril(A, -1)
    # upper triangular matrix to use in iteration
    U = (1 - omega) * I - (omega * dt).reshape(dt.size) * np.triu(A, 1)
    # constant vector to use in iteration
    c = omega * dt * b
    # solution vector
    x = np.zeros(b.shape[0])
    # current number of iteration
    k = 0
    # residual infinity (maximum) norm
    residual = np.inf

    while k < max_iterations and residual >= sys.float_info.epsilon:
        _x = x
        y = np.dot(U, _x) + c
        # forward substitution
        x = spla.solve_triangular(L, y, lower=True, unit_diagonal=True)
        residual = np.linalg.norm(x - _x, ord=np.inf)
        k += 1

    return x, k


def cg(A, b, max_iterations):
    """Return a solution x of a linear system Ax = b with Conjugate-Gradient
    Method.

    This algorithm refers to the folloing article:
        Kouya Tomonori
        "ソフトウェアとしての数値計算 10 連立一次方程式の解法3 -- Krylov部分空間法", 2007,
        URL<https://na-inet.jp/nasoft/chap10.pdf>

    Arguments:
        A (array_like): A square coefficient matrix.
        b (array_like): A right-hand side vector.
        max_iterations (int): A maximum number of iterations.

    Returns:
        numpy.ndarray: A solution of a linear system Ax = b.
        int: A number of iterations.
    """
    A = np.array(A, dtype=np.float)
    b = np.array(b, dtype=np.float)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")
    if A.shape[0] != b.shape[0]:
        raise ValueError("matrix and vector size must be aligned")

    # solution vector
    x = np.zeros(b.shape[0])
    # residual vector
    r = b - np.dot(A, x)
    # modification vector
    p = r
    # current number of iteration
    k = 0

    while k < max_iterations:
        alpha = np.dot(r, p) / np.dot(p, np(A, p))
        x = x + alpha * p
        _r = r
        r = _r - alpha * np.dot(A, p)
        if np.linalg.norm(r) < sys.float_info.epsilon:
            break
        beta = (np.linalg.norm(r) / np.linalg.norm(_r)) ** 2
        p = r + beta * p
        k += 1

    return x, k
