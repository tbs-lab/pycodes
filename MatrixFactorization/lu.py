# -*- coding: utf-8 -*-
import np


def genp(A):
    """Return factors of LU factorization using Gaussian Elimination with
    no pivoting (GENP).

    This algorithm refers to the folloing article:
        Andrew M. Stuart and Jochen Voss,
        "Matrix Analysis and Algorithm", 2009
        URL<https://www.seehuhn.de/publications/StuaVo08.html>

    Arguments:
        A (array_like): A square matrix.

    Returns:
        L (numpy.ndarray): A lower triangular matrix with unit diagonal.
        U (numpy.ndarray): A upper triangular matrix.
    """
    A = np.array(A, dtype=np.float)

    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")

    n = A.shape[0]
    L = np.identity(n)
    U = np.array(A)

    for k in range(0, n - 1):
        if not U[k, k]:
            raise ZeroDivisionError("can not divide by zero")

        # solve L matrix
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        # update U matrix
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])

    return L, U


def gepp(A):
    """Return factors of PLU factorization using Gaussian Elimination with
    partial pivoting (GEPP).

    This algorithm refers to the folloing article:
        Andrew M. Stuart and Jochen Voss,
        "Matrix Analysis and Algorithm", 2009
        URL<https://www.seehuhn.de/publications/StuaVo08.html>

    Arguments:
        A (array_like): A square matrix.

    Returns:
        P (numpy.ndarray): A permutation matrix.
        L (numpy.ndarray): A lower triangular matrix with unit diagonal.
        U (numpy.ndarray): A upper triangular matrix.
    """
    A = np.array(A, dtype=np.float)
    # check if square
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square one")

    n = A.shape[0]
    Pt = np.identity(n)
    L = np.identity(n)
    U = np.array(A)

    for k in range(0, n - 1):
        # select a row pivot in maximum absolute values (from k to n - 1)
        pivot = np.abs(U[k:, k]).argmax() + k

        if not U[pivot, k]:
            raise ValueError("this matrix is singular")

        # swap k and pivot rows of each matrices
        Pt[[k, pivot]] = Pt[[pivot, k]]
        U[[k, pivot]] = U[[pivot, k]]
        L[[k, pivot], :k] = L[[pivot, k], :k]

        # solve L matrix
        L[k + 1:, k] = U[k + 1:, k] / U[k, k]
        # update U matrix
        U[k + 1:, k:] -= np.outer(L[k + 1:, k], U[k, k:])

    return Pt.T, L, U
