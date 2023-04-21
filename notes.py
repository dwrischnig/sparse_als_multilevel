# coding: utf-8
import numpy as np
import scipy.sparse as sps
import sparseqr


def permutation_matrix(permutation: IntArray) -> sps.coo_matrix:
    return sps.coo_matrix((np.ones(len(permutation)), (permutation, np.arange(len(permutation)))))


permutation = np.array([1, 5, 3, 0, 2, 4])
xs = np.random.randn(13, len(permutation))
assert np.all(xs[:, permutation] == xs @ permutation_matrix(permutation))

Q, R, E, rank = sparseqr.qr(newCore, economy=True)
E = permutation_matrix(E)
Q = Q.tocsr()[:, :rank]
R = R.tocsr()[:rank]
assert np.allclose((Q @ R @ E.T - newCore).data, 0)

print("Computing sparse QZ: C = QC")
print("  nnz(C) ==", newCore.nnz)
print("  nnz(Q) ==", Q.nnz)
print("  nnz(Z) ==", R.nnz)
