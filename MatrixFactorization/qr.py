# -*- coding: utf-8 -*-
import numpy as np


def build_householder(a):
    """Return factors of householder matrix (I - tau * vv.T).

    This algorithm refers to the folloing article:
        G.H Golub and C.F Van Loan,
        "Matrix Computations 4th Edition", 2013.

    Arguments:
        a (array_like): A vector.

    Returns:
        v (numpy.ndarray): Householder vector.
            This vector is modified so that the 1st element is 1.
            See the reference for more details.
        tau (float): A coefficient of householder matrix.
        diagonal (float): A diagonal value which is result of householder
            transformation.
    """
    v = np.array(a, dtype=np.float)
    if v.ndim != 1:
        raise ValueError("vector must be 1 dimensional array")
    # build v and tau
    diagonal = np.linalg.norm(v)
    if v[0] >= 0:
        v[0] += diagonal
        diagonal = -diagonal
    else:
        v[0] -= diagonal
    v /= v[0]
    tau = 2 / np.inner(v, v)

    return v, tau, diagonal


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
    m, n = A.shape
    mn = n if m > n else m - 1
    for j in range(mn):
        v, tau, A[j, j] = build_householder(A[j:, j])
        A[j + 1:, j] = v[1:]
        A[j:, j + 1:] -= np.outer(tau * v, np.dot(v, A[j:, j + 1:]))

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
    R = np.array(A, dtype=np.float)
    if R.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    m, n = R.shape
    Q = np.identity(m)
    # factorization
    mn = n if m > n else m - 1
    for j in range(mn):
        v, tau, R[j, j] = build_householder(R[j:, j])
        R[j + 1:, j] = 0
        R[j:, j + 1:] -= np.outer(tau * v, np.dot(v, R[j:, j + 1:]))
        Q[:, j:] -= np.outer(tau * np.dot(Q[:, j:], v), v)

    return Q, R


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
    Y = np.array(A, dtype=np.float)
    if Y.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    m, n = Y.shape
    mn = n if m > n else m - 1
    v, tau, Y[0, 0] = build_householder(Y[:, 0])
    Y[1:, 0] = v[1:]
    Y[:, 1:] -= np.outer(tau * v, np.dot(v, Y[:, 1:]))
    W = (2 / np.inner(v, v) * v).reshape(-1, 1)
    # build W and Y
    for j in range(1, mn):
        v, tau, Y[j, j] = build_householder(Y[j:, j])
        Y[j + 1:, j] = v[1:]
        Y[j:, j + 1:] -= np.outer(tau * v, np.dot(v, Y[j:, j + 1:]))
        v = np.concatenate([np.zeros(j), v], axis=0)
        w = tau * (v - np.dot(W, np.dot(Y[j:, :j].T, v[j:])))
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
    R = np.array(A, dtype=np.float)
    if R.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    m, n = R.shape
    Q = np.identity(m)
    # block householder QR factorization
    j = 0
    mn = n if m > n else m - 1
    while j < mn:
        b = min(j + panels, mn)
        W, R[j:, j:b] = build_wy(R[j:, j:b])
        Y = np.tril(R[j:, j:b], k=-1) + np.eye(m - j, b - j)
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], W), Y.T)
        R[j:, b:] -= np.dot(Y, np.dot(W.T, R[j:, b:]))
        R[j:, j:b] = np.triu(R[j:, j:b])
        j = b

    return Q, np.triu(R)


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
    Y = np.array(A, dtype=np.float)
    if Y.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    m, n = Y.shape
    mn = n if m > n else m - 1
    v, tau, Y[0, 0] = build_householder(Y[:, 0])
    Y[1:, 0] = v[1:]
    Y[:, 1:] -= np.outer(tau * v, np.dot(v, Y[:, 1:]))
    T = np.array([[2 / np.inner(v, v)]])
    # build Y and T
    for j in range(1, mn):
        v, tau, Y[j, j] = build_householder(Y[j:, j])
        Y[j + 1:, j] = v[1:]
        Y[j:, j + 1:] -= np.outer(tau * v, np.dot(v, Y[j:, j + 1:]))
        v = np.concatenate([np.zeros(j), v], axis=0)
        t = (-tau * np.dot(T, np.dot(Y[j:, :j].T, v[j:]))).reshape(-1, 1)
        T = np.block([[T, t], [np.zeros((1, j)), np.array([[tau]])]])

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
    R = np.array(A, dtype=np.float)
    if R.ndim != 2:
        raise ValueError("matrix must be 2 dimensional array")
    # initialization
    m, n = R.shape
    Q = np.identity(m)
    # block householder QR factorization
    j = 0
    mn = n if m > n else m - 1
    while j < mn:
        b = min(j + panels, mn)
        R[j:, j:b], T = build_compact_wy(R[j:, j:b])
        Y = np.tril(R[j:, j:b], k=-1) + np.eye(m - j, b - j)
        Q[:, j:] -= np.dot(np.dot(Q[:, j:], Y), np.dot(T, Y.T))
        R[j:, b:] -= np.dot(np.dot(Y, T.T), np.dot(Y.T, R[j:, b:]))
        R[j:, j:b] = np.triu(R[j:, j:b])
        j = b

    return Q, R
