import numpy as np
from numpy.polynomial.legendre import legint, legmul, legval, legder


def L2innerLegendre(c1, c2):
    """Return the L2-inner product of the Legendre polynomials corresponding to the given coefficient vectors."""
    i = legint(legmul(c1, c2))
    return legval(1, i) - legval(-1, i)


def HkinnerLegendre(k):
    """Return the a function that computes the Sobolev-inner product of order k."""
    assert isinstance(k, int) and k >= 0

    def inner(c1, c2):
        """Return the Sobolev-inner product of order k.

        Parameters
        ----------
        c1, c2 : np.ndarray
            Coefficient vectors with respect to the Legendre polynomials.
        """
        ret = L2innerLegendre(c1, c2)
        for j in range(k):
            c1 = legder(c1)
            c2 = legder(c2)
            ret += L2innerLegendre(c1, c2)
        return ret

    return inner


def Gramian(d, inner):
    """Return the Gramian matrix of dimension d with respect to the given inner product."""
    matrix = np.empty((d, d))

    def standard_basis(k):
        return np.eye(1, d, k)[0]

    for i in range(d):
        ei = standard_basis(i)
        for j in range(i + 1):
            ej = standard_basis(j)
            matrix[i, j] = matrix[j, i] = inner(ei, ej)
    return matrix
