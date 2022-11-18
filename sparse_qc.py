# coding: utf-8
from __future__ import annotations

from collections.abc import Iterator
from functools import wraps

import scipy.sparse as sps
import numpy as np

import autoPDB


def fold_all(generator: Iterator[bool]):
    @wraps(generator)
    def wrapper(*args, **kwargs) -> bool:
        return all(generator(*args, **kwargs))
    return wrapper


@fold_all
def isqpm(matrix, orthogonal=False):
    """Check whether the given array is a quasi-permutation matrix."""
    yield isinstance(matrix, sps.spmatrix)
    yield matrix.ndim == 2
    yield np.all(matrix.data == 1)
    matrix = matrix.tocoo()
    colTest = np.zeros(matrix.shape[1])
    colTest[matrix.col] += 1
    yield np.all(colTest == 1)
    if orthogonal:
        yield (matrix.T @ matrix - sps.eye(matrix.shape[1])).nnz == 0


def deparallelise(matrix):
    """
    Remove duplicate columns from a quasi-permutation matrix.

    The resulting matrix will be left-orthogonal.
    """
    assert isqpm(matrix)

    if __debug__:
        rs, idx, inv = np.unique(matrix.row, return_index=True, return_inverse=True)
        # Let C = matrix.col, R = matrix.row, oR = rs, I = idx and J = inv.
        # Since reindexing is a linear operation, we can interpret I and J as matrices satisfying
        # (I @ R)_k = R[I_k] = oR_k and (J @ oR)_k = oR[J_k] = R_k.
        assert all(matrix.row[idx] == rs) and all(rs[inv] == matrix.row)
        assert len(rs) == len(idx) <= len(matrix.row) == len(inv) == matrix.nnz
    else:
        rs, inv = np.unique(matrix.row, return_inverse=True)

    rank = len(rs)
    s = np.ones(matrix.nnz, matrix.dtype)
    T = sps.coo_matrix((s, (inv, matrix.col)), shape=(rank, matrix.shape[1]))
    if __debug__:
        # Thus J @ I @ R = R and I @ J @ oR = oR.
        # Moreover, since R is an index vector as well, we can write the slicing of the non-zero rows of matrix as R @ matrix = matrix[R].
        # Removing the duplicates, T is given by T = I @ R @ matrix. This can be written as I @ (R @ matrix) = matrix[R][I] or (I @ R) @ matrix = oR @ matrix = matrix[oR].
        # T can, however be implemented more efficiently by noting that T[l,k] = matrix[oR[l], k] = 1 if and only if (oR[l], k) in zip(R, C).
        # Since the elements in C are unique, there exists only one such pair and for every k in C and we can define a unique l_k satisfying (oR[l_k], k) in zip(R, C).
        # Assuming C satisfies
        assert np.all(matrix.col == np.arange(matrix.shape[1]))
        # it holds that C[k] = k for every natural number k.
        # The above condition is thus equivalent to (oR[l_k], k) = (R[k], k) = zip(R, C)[k].
        # The sought index l_k is thus given by l_k = J_k, since oR[l_k] = (J @ I @ R)_k = R_k.
        matrix_csc = matrix.tocsc()
        assert (T - matrix_csc[matrix.row][idx]).nnz == 0
        assert (T - matrix_csc[rs]).nnz == 0
    
    s = np.ones(rank, matrix.dtype)
    k = np.arange(rank)
    N = sps.coo_matrix((s, (rs, k)), shape=(matrix.shape[0], rank))
    if __debug__:
        # We now want to find N such that N @ T = matrix.
        # Such a matrix is given by N = inv(I @ R), since then N @ T = inv(I @ R) @ (I @ R) @ matrix.
        # Since I @ R is orthogonal, we have N = (I @ R).T.
        # The matrix corresponding to N.T = I @ R = oR is given by
        Nt = sps.coo_matrix((s, (k, rs)), shape=(rank, matrix.shape[0]))
        assert (N - Nt.T).nnz == 0
    assert (N @ T - matrix).nnz == 0
    assert (N.T @ N - sps.eye(rank)).nnz == 0
    # TODO: It is easier to explain this the other way around.
    #       Since M = matrix is a quasi-permutation matrix, every column contains exactly one nonzero element.
    #       This means that two columns of M are either orthogonal or identical.
    #       To obtain a basis for the image space of M, it thus suffices to collect all the unique columns.
    #       And since a column is uniquely defined by the position of its nonzero element,
    #       we can simply check for duplicates in the list of row indices R.
    #       Denote these indices by I. Then N is defined by the row indices R[I] and column indices arange(len(I)).
    #       Note, that N is orthogonal by design. Hence, we obtain T as N.T @ M, where N.T @ M simply selects the
    #       unique rows in M, i.e. N.T @ M = M[R[I]].
    #       Now, the explanation of the fast implmentation for T can be copied from above.

    return N, T


def sparse_qc(matrix, minimal=False, precision=1e-8):
    s = np.ones(matrix.nnz, dtype=matrix.dtype)
    k = np.arange(matrix.nnz)
    U = sps.coo_matrix((s, (matrix.row, k)), shape=(matrix.shape[0], matrix.nnz))
    S = sps.dia_matrix((matrix.data[None], [0]), shape=(matrix.nnz, matrix.nnz))
    Vt = sps.coo_matrix((s, (k, matrix.col)), shape=(matrix.nnz, matrix.shape[1]))
    assert (U @ S @ Vt - matrix).nnz == 0
    Q, T = deparallelise(U)
    C = T @ S @ Vt
    assert (Q @ C - matrix).nnz == 0
    if minimal:
        U, s, Vt = np.linalg.svd(C.toarray())
        rank = np.count_nonzero(s >= precision)
        U, s, Vt = U[:, :rank], s[:rank], Vt[:rank]
        U[abs(U) < precision] = 0
        U = sps.coo_matrix(U)
        assert np.all(abs((U.T @ U - sps.eye(rank)).data) < precision)
        Q = Q @ U
        C = s[:, None] * Vt
        C[abs(C) < precision] = 0
        C = sps.coo_matrix(C)
    return Q, C


def disp_sparsity(array):
    m,n = array.shape
    array = np.pad(array, ((0, m % 2), (0, n % 2)))
    m,n = array.shape
    assert m % 2 == 0 and n % 2 == 0
    ret = ""
    for j in range(m // 2):
        for k in range(n // 2):
            block = array[2*j: 2*j+2, 2*k: 2*k+2] != 0
            assert block.shape == (2, 2)
            if np.all(block == [[0, 0], [0, 0]]):
                ret += " "
            if np.all(block == [[0, 0], [0, 1]]):
                ret += "\u2597"
            if np.all(block == [[0, 0], [1, 0]]):
                ret += "\u2596"
            if np.all(block == [[0, 0], [1, 1]]):
                ret += "\u2584"
            if np.all(block == [[0, 1], [0, 0]]):
                ret += "\u259D"
            if np.all(block == [[0, 1], [0, 1]]):
                ret += "\u2590"
            if np.all(block == [[0, 1], [1, 0]]):
                ret += "\u259E"
            if np.all(block == [[0, 1], [1, 1]]):
                ret += "\u259F"
            if np.all(block == [[1, 0], [0, 0]]):
                ret += "\u2598"
            if np.all(block == [[1, 0], [0, 1]]):
                ret += "\u259A"
            if np.all(block == [[1, 0], [1, 0]]):
                ret += "\u258C"
            if np.all(block == [[1, 0], [1, 1]]):
                ret += "\u2599"
            if np.all(block == [[1, 1], [0, 0]]):
                ret += "\u2580"
            if np.all(block == [[1, 1], [0, 1]]):
                ret += "\u259C"
            if np.all(block == [[1, 1], [1, 0]]):
                ret += "\u259B"
            if np.all(block == [[1, 1], [1, 1]]):
                ret += "\u2588"
        ret += "\n"
    return ret

# for ij in range(16):
#     M = np.array(list(f"{bin(ij)[2:]:>04s}")).astype(int).reshape(2,2)
#     print(M)
#     print_sparsity(M)
#     print()
# print_sparsity(np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))


if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    import matplotlib.pyplot as plt

    console = Console()
    rng = np.random.default_rng()
    
    D, r = 100, 4
    # density = 0.02
    density = 0.05
    rho = 2

    S = sps.random(D, r, density=density, random_state=rng)
    console.print(f"S shape: {S.shape}  ({S.nnz} nonzero)")
    console.print("S:")
    console.print(Panel(disp_sparsity(S.T.toarray())[:-1], expand=False))

    console.rule("Standard QC")
    Q, C = sparse_qc(S)
    assert isqpm(Q, orthogonal=True)
    console.print(f"Q shape: {Q.shape}  ({Q.nnz} nonzero)")
    console.print(f"C shape: {C.shape}  ({C.nnz} nonzero)")
    console.print("Q:")
    console.print(Panel(disp_sparsity(Q.T.toarray())[:-1], expand=False))
    Omega = sps.diags(np.arange(1, D+1)**rho)
    Beta = (Q.T @ Omega @ Q).todia()
    assert len(Beta.offsets) == 1 and Beta.offsets[0] == 0
    console.print("Beta:")
    console.print(Panel(disp_sparsity(Beta.T.toarray())[:-1], expand=False))

    console.rule("Minimal QC")
    Q, C = sparse_qc(S, minimal=True)
    console.print(f"Q shape: {Q.shape}  ({Q.nnz} nonzero)")
    console.print(f"C shape: {C.shape}  ({C.nnz} nonzero)")
    console.print("Q:")
    console.print(Panel(disp_sparsity(Q.T.toarray())[:-1], expand=False))
