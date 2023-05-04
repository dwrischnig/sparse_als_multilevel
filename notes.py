# coding: utf-8
import numpy as np
import scipy.sparse as sps
import sparseqr


def permutation_matrix(permutation: IntArray) -> sps.coo_matrix:
    return sps.coo_matrix((np.ones(len(permutation)), (permutation, np.arange(len(permutation)))))


permutation = np.array([1, 5, 3, 0, 2, 4])
xs = np.random.randn(13, len(permutation))
assert np.all(xs[:, permutation] == xs @ permutation_matrix(permutation))


def qc(matrix: sps.spmatrix) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    Q, R, E, rank = sparseqr.qr(matrix, economy=True)
    Q = Q.tocsr()[:, :rank]
    R = R.tocsr()[:rank]
    E = permutation_matrix(E)
    C = R @ E.T
    assert np.allclose((Q @ C - matrix).data, 0)
    return Q, C


print("Computing sparse QZ: C = QZ")
print("  nnz(C) ==", matrix.nnz)
print("  nnz(Q) ==", Q.nnz)
print("  nnz(Z) ==", C.nnz)
