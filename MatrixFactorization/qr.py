# -*- coding: utf-8 -*-
import numpy as np


def build_householder(a):
    """Return factors of householder matrix (I - beta * vv.T).

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        a (array_like): A vector.

    Returns:
        v (numpy.ndarray): Householder vector.
            This vector is modified so that the 1st element is 1.
            See the reference for more details.
        beta (float): A coefficient of householder matrix.
    """
    v = np.array(a, dtype=np.float)
    if v.ndim != 1:
        raise ValueError("vector must be 1 dimensional array")
    # build v
    alpha = np.linalg.norm(v)
    v[0] = v[0] + alpha if v[0] >= 0 else v[0] - alpha
    # build beta
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
    # triangularization
    rows, cols = A.shape
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
        A (numpy.ndarray): A upper trianglar matrix of triangularized A.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    rows, cols = A.shape
    Q = np.identity(rows)
    # triangularization
    A = triangularize(A)
    # backward accumulation
    n = cols - 1 if rows > cols else rows - 2
    for j in range(n, -1, -1):
        v = np.concatenate([[1], A[j + 1:, j]], axis=0)
        beta = 2 / np.inner(v, v)
        Q[j:, j:] -= np.outer(beta * v, np.dot(v, Q[j:, j:]))

    return Q, np.triu(A)


def build_wy(A):
    """Return factors of WY representaion (I - WY.T).

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013..

    Arguments:
        A (array_like): A matrix.

    Returns:
        W (numpy.ndarray): W matrix of WY representation.
        Y (numpy.ndarray): A unit lower trianglar matrix.
            This upper trianglar submatrix is overwritten by
            the result of applying tiangulation to A.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # build Y
    Y = triangularize(A)
    # build W
    rows, cols = A.shape
    v = np.concatenate([[1], Y[1:, 0]], axis=0)
    W = (2 / np.inner(v, v) * v).reshape(-1, 1)
    for j in range(1, cols):
        v = np.concatenate([np.zeros(j), [1], Y[j + 1:, j]], axis=0)
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
        A (numpy.ndarray): A upper trianglar matrix of triangularized A.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    rows, cols = A.shape
    Q = np.identity(rows)
    # block householder QR factorization
    j = 0
    n = cols if rows > cols else rows - 1
    while j < n:
        b = min(j + panels, n)
        W, A[j:, j:b] = build_wy(A[j:, j:b])
        Y = np.tril(A[j:, j:b], k=-1) + np.eye(rows - j, b - j)
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], W), Y.T)
        A[j:, b:] -= np.dot(Y, np.dot(W.T, A[j:, b:]))
        j = b

    return Q, np.triu(A)


def build_compact_wy(A):
    """Return factors of Compact-WY representaion (I - YTY.T).

    This algorithm refers to the folloing article:
        Robert Schreiber and Charles Van Loan,
        "A Storage-Efficient WY Representation for Products of Householder
        Transformations", 1989.

    Arguments:
        A (array_like): A matrix.

    Returns:
        Y (numpy.ndarray): A unit lower trianglar matrix.
            This upper trianglar submatrix is overwritten by
            the result of applying tiangulation to A.
        T (numpy.ndarray): A upper triangular matrix.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # build Y
    Y = triangularize(A)
    # build T
    rows, cols = A.shape
    v = np.concatenate([[1], Y[1:, 0]], axis=0)
    T = np.array([[2 / np.inner(v, v)]])
    for j in range(1, cols):
        v = np.concatenate([np.zeros(j), [1], Y[j + 1:, j]], axis=0)
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
        A (numpy.ndarray): A upper trianglar matrix of triangularized A.
    """
    A = np.array(A, dtype=np.float)
    if A.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    rows, cols = A.shape
    Q = np.identity(rows)
    # block householder QR factorization
    j = 0
    n = cols if rows > cols else rows - 1
    while j < n:
        b = min(j + panels, n)
        A[j:, j:b], T = build_compact_wy(A[j:, j:b])
        Y = np.tril(A[j:, j:b], k=-1) + np.eye(rows - j, b - j)
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], Y), np.dot(T, Y.T))
        A[j:, b:] -= np.dot(np.dot(Y, T.T), np.dot(Y.T, A[j:, b:]))
        j = b

    return Q, np.triu(A)
