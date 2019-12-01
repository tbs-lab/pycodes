# -*- coding: utf-8 -*-
import numpy


def lu_no_pivoting(A):
    """Return factors of LU factorization (Doolittle's version) without pivoting.

    This algorithm refers to the folloing article.
        URL: http://mathonline.wikidot.com/the-algorithm-for-doolittle-s-method-for-lu-decompositions

    Arguments:
        A (numpy.ndarray): A square matrix.

    Returns:
        L (numpy.ndarray): A lower triangular matrix with unit diagonal.
        U (numpy.ndarray): A upper triangular matrix.
    """
    if A.shape[0] != A.shape[1]:
        return ValueError("matrix must be square one")

    L = numpy.identity(A.shape[0])
    U = numpy.zeros(A.shape)

    for k in range(A.shape[0]):
        U[k, k:] = A[k, k:] - L[k, 0:k] @ U[0:k, k:]
        L[k + 1:, k] = A[k + 1:, k] - L[k + 1:, 0:k] @ U[0:k, k]
        L[k + 1:, k] /= U[k, k]

    return L, U
