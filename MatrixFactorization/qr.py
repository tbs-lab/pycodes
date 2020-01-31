# -*- coding: utf-8 -*-
import numpy as np


def build_householder(a):
    """Return factors of householder matrix.

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        a (array_like): A vector.

    Returns:
        v (numpy.ndarray): Householder vector.
            This vector is normalized so that the 1st element is 1.
            See the reference for more details.
        beta (float): A coefficient of householder matrix.
    """
    a = np.array(a, dtype=np.float)
    if a.ndim != 1:
        raise ValueError("vector must be 1 dimensional array")

    v = a.copy()
    alpha = np.linalg.norm(a)

    if a[0] >= 0:
        v[0] -= alpha
    else:
        v[0] += alpha

    beta = 2 * v[0] ** 2 / np.inner(v, v)
    v /= v[0]

    return v, beta


def triangularize(A):
    """Return result of applying householder transformation to a matrix.

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        A (array_like): A matrix.

    Returns:
        A (numpy.ndarray): A upper trianglar matrix.
            This lower trianglar submatrix is overwritten by unit
            householder vectors. See the reference for more details.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape

    # triangularization
    n = cols if rows > cols else rows - 1
    for j in range(n):
        v, beta = build_householder(A[j:, j])
        A[j:, j:] -= np.outer(beta * v, np.dot(v, A[j:, j:]))
        A[j + 1:, j] = v[1:]

    return A


def householder(A):
    """Return factors of QR factorization using householder transformation.

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        A (array_like): A matrix.

    Returns:
        Q (numpy.ndarray): A orthogonal matrix.
        R (numpy.ndarray): A upper trianglar matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape
    Q = np.identity(rows)

    # triangularization
    R = triangularize(A)

    # backward accumulation
    n = cols - 1 if rows > cols else rows - 2
    for j in range(n, -1, -1):
        v = R[j:, j].copy()
        v[0] = 1
        beta = 2 / np.inner(v, v)
        Q[j:, j:] -= np.outer(beta * v, np.dot(v, Q[j:, j:]))

    return Q, np.triu(R)


def build_wy(A):
    """Return factors of WY representaion (I - WY.T).

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013..

    Arguments:
        A (array_like): A upper trianglar matrix.
            This lower trianglar submatrix must be overwritten by unit
            householder vectors. See the reference for more details.

    Returns:
        W (numpy.ndarray): W matrix of WY representation.
        Y (numpy.ndarray): A unit lower trianglar matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape

    # initialization
    Y = np.tril(A)
    Y[np.eye(rows, cols, dtype=bool)] = 1
    W = np.reshape(2 / np.inner(Y[:, 0], Y[:, 0]) * Y[:, 0], (-1, 1))

    # get W matrix
    for j in range(1, cols):
        v = Y[:, j]
        beta = 2 / np.inner(v[j:], v[j:])
        w = beta * (v - np.dot(W, np.dot(Y[j:, :j].T, v[j:])))
        W = np.concatenate([W, w.reshape((-1, 1))], axis=1)

    return W, Y


def wy(A, panels=64):
    """Return factors of QR factorization using WY representaion (I - WY.T).

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        A (array_like): A matrix.
        panels (int, optional): A panel size (column size).

    Returns:
        Q (numpy.ndarray): A orthogonal matrix.
        R (numpy.ndarray): A upper trianglar matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape

    # initialization
    j = 0
    Q = np.identity(rows)
    R = A

    # block householder QR factorization
    n = cols if rows > cols else rows - 1
    while j < n:
        b = min(j + panels, n)
        R[j:, j:b] = triangularize(R[j:, j:b])
        W, Y = build_wy(R[j:, j:b])
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], W), Y.T)
        R[j:, b:] -= np.dot(Y, np.dot(W.T, R[j:, b:]))
        j = b

    return Q, np.triu(R)


def build_compact_wy(A):
    """Return factors of Compact-WY representaion (I - YTY.T).

    This algorithm refers to the folloing article:
        Robert Schreiber and Charles Van Loan,
        "A Storage-Efficient WY Representation for Products of Householder
        Transformations", 1989.

    Arguments:
        A (array_like): A upper trianglar matrix.
            This lower trianglar submatrix must be overwritten by unit
            householder vectors. See the reference for more details.

    Returns:
        Y (numpy.ndarray): A unit lower trianglar matrix.
        T (numpy.ndarray): A upper triangular matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape

    # initialization
    Y = np.tril(A)
    Y[np.eye(rows, cols, dtype=bool)] = 1
    T = np.array([[2 / np.inner(Y[:, 0], Y[:, 0])]])

    # get T matrix
    for j in range(1, cols):
        v = Y[:, j]
        beta = 2 / np.inner(v[j:], v[j:])
        t = (-beta * np.dot(T, np.dot(Y[j:, :j].T, v[j:]))).reshape(-1, 1)
        T = np.block([[T, t], [np.zeros((1, j)), np.array([[beta]])]])

    return Y, T


def compact_wy(A, panels=64):
    """Return factors of QR factorization using Compact-WY
    representaion (I - YTY.T).

    This algorithm refers to the folloing article:
        Robert Schreiber and Charles Van Loan,
        "A Storage-Efficient WY Representation for Products of Householder
        Transformations", 1989.

    Arguments:
        A (array_like): A matrix.
        panels (int, optional): A panel size (column size).

    Returns:
        Q (numpy.ndarray): A orthogonal matrix.
        R (numpy.ndarray): A upper trianglar matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")

    rows, cols = A.shape

    # initialization
    j = 0
    Q = np.identity(rows)
    R = A

    # block householder QR factorization
    n = cols if rows > cols else rows - 1
    while j < n:
        b = min(j + panels, n)
        R[j:, j:b] = triangularize(R[j:, j:b])
        Y, T = build_compact_wy(R[j:, j:b])
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], Y), np.dot(T, Y.T))
        R[j:, b:] -= np.dot(np.dot(Y, T.T), np.dot(Y.T, R[j:, b:]))
        j = b

    return Q, np.triu(R)
