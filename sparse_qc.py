# coding: utf-8
"""Functions for decomposition, contraction and presentation of sparse matrices."""
from __future__ import annotations

from typing import Any, Callable
from collections.abc import Iterable, Iterator
from functools import wraps

import scipy.sparse as sps
import numpy as np
from numba import jit
import deal
from rich.console import Console
from rich.panel import Panel

import autoPDB  # noqa: F401


def fold_all(generator: Callable[..., Iterable]) -> Callable[..., bool]:
    """Turn a function returning an iterable into a function returning a bool."""

    @wraps(generator)
    def wrapper(*args, **kwargs) -> bool:
        return all(generator(*args, **kwargs))

    return wrapper


def is_sparse_matrix(matrix: Any) -> bool:
    """Check whether the given object is a sparse matrix."""
    return isinstance(matrix, sps.spmatrix)


@fold_all
def is_qpm(matrix: Any, orthogonal: bool = False) -> Iterator:
    """Check whether the given object is a quasi-permutation matrix."""
    yield is_sparse_matrix(matrix)
    yield np.all(matrix.data == 1)
    matrix = matrix.tocoo()
    colTest = np.zeros(matrix.shape[1])
    colTest[matrix.col] += 1
    yield np.all(colTest == 1)
    if orthogonal:
        yield len(matrix.col) == matrix.shape[1]


@deal.pre(is_qpm)
@deal.post(lambda result: is_qpm(result[0], orthogonal=True) and is_qpm(result[1]))
@deal.ensure(lambda matrix, result: result[0].nnz <= matrix.nnz and result[1].nnz <= matrix.nnz)
@deal.has()
@deal.raises(TypeError)  # np.unique (This should not happen.)
def deparallelise(matrix: sps.spmatrix) -> tuple[sps.spmatrix, sps.spmatrix]:
    """Return a sparse QC decomposition of a quasi-permutation matrices (QPM).

    Parameters
    ----------
    matrix : sps.spmatrix
        A quasi-permutation matrix.

    Returns
    -------
    Q : sps.spmatrix
        The original matrix with duplicate columns removed.
        This matrix will be left-orthogonal.
    C : sps.spmatrix
        A sparse matrix such that Q @ C returns the original matrix.
    """
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
        # Since R is an index vector, we can write the slicing of the non-zero rows of matrix as R @ matrix = matrix[R].
        # Removing the duplicates, T is given by T = I @ R @ matrix.
        # This can be written as I @ (R @ matrix) = matrix[R][I] or (I @ R) @ matrix = oR @ matrix = matrix[oR].
        # T can be implemented more efficiently by noting that
        #     T[l,k] = matrix[oR[l], k] = 1 if and only if (oR[l], k) in zip(R, C).
        # Since the elements in C are unique, there exists only one such pair and for every k in C
        # and we can define a unique l_k satisfying (oR[l_k], k) in zip(R, C).
        # Assuming C satisfies
        assert np.all(matrix.col == np.arange(matrix.shape[1]))
        # it holds that C[k] = k for every natural number k.
        # The above condition is thus equivalent to (oR[l_k], k) = (R[k], k) = zip(R, C)[k].
        # The sought index l_k is thus given by l_k = J_k, since oR[l_k] = (J @ I @ R)_k = R_k.
        matrix_csc = matrix.tocsc()
        assert (T - matrix_csc[matrix.row][idx]).nnz == 0  # type: ignore
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


@deal.pre(lambda _: is_sparse_matrix(_.matrix))
@deal.pre(lambda _: _.precision > 0)
@deal.post(lambda result: len(result) == 2 and is_sparse_matrix(result[0]) and is_sparse_matrix(result[1]))
@deal.ensure(lambda _: deal.implies(not _.minimal, _.result[0].nnz <= _.matrix.nnz and _.result[1].nnz <= _.matrix.nnz))
@deal.ensure(
    lambda _: deal.implies(
        _.minimal, _.result[0].nnz <= _.result[0].shape[1] * _.matrix.nnz and _.result[1].nnz <= _.matrix.nnz
    )
)
@deal.has("import")  # The first line in np.linalg.svd is "import numpy as _nx".
@deal.reason(np.linalg.LinAlgError, lambda matrix, minimal, precision: bool(minimal))
def sparse_qc(
    matrix: sps.spmatrix, minimal: bool = False, precision: float = 1e-8
) -> tuple[sps.spmatrix, sps.spmatrix]:
    """Return a sparse QC decomposition of a sparse matrix.

    Parameters
    ----------
    matrix : sps.spmatrix
        A sparse matrix.
    minimal : bool, optional
        Whether or not to compute a minimal QC decomposition.
        False by default.
    precision : float, optional
        The precision to use in a minimal QC decomposition.
        Defaults to 1e-8.

    Returns
    -------
    Q : sps.spmatrix
        A left-orthogonal sparse matrix.
        If minimal is False, Q will be a quasi-permutation matrix.
    C : sps.spmatrix
        A sparse matrix such that Q @ C returns the original matrix.
    """
    matrix = matrix.tocoo()
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


# This is used to compute entries in the measurement stacks.
def kron_dot_qpm(a, b, c, out=None):
    """
    Efficiently compute out[i] = kron(a[i], b[i]) @ c.

    c has to be an orthogonal quasi-permutation matrix.

    Parameters
    ----------
    a, b : array_like
    c : spmatrix
    out : ndarray, optional
        A location into which the result is stored.
        If provided, it must have the correct shape.
        If not provided or None, a freshly-allocated array is returned.

    Returns
    -------
    out : ndarray

    Notes
    -----
    If a and b have more than two dimension, the Kronecker product
    as well as the matrix product are computed only for the second dimension.
    If ``a.shape = (r0,) + r``, ``b.shape = (s0,) + s`` and
    ``c.shape = (r0*s0, t)``, the output has shape ``r + s + (t,)``.
    The time complexity of this algorithm is linear in the output size.
    """
    assert a.ndim > 1 and b.ndim > 1
    assert a.shape[0] == b.shape[0]
    assert c.shape[0] == a.shape[1] * b.shape[1]
    assert is_qpm(c, orthogonal=True)
    c = c.tocoo()
    outShape = (a.shape[0],) + a.shape[2:] + b.shape[2:] + (c.shape[1],)
    if out is None:
        out = np.empty(outShape)
    else:
        assert out.shape == outShape
    aShape = a.shape + (1,) * (b.ndim - 2)
    bShape = b.shape[:2] + (1,) * (a.ndim - 2) + b.shape[2:]
    __kron_dot_qpm(a.reshape(aShape), b.reshape(bShape), c.row, c.col, out)
    return out


@jit(nopython=True)
def __kron_dot_qpm(a, b, cRow, cCol, out):
    bShape1 = b.shape[1]
    for l, k in zip(cRow, cCol):  # noqa: E741
        i = l // bShape1
        j = l % bShape1
        out[..., k] = a[:, i] * b[:, j]


# This is used to compute entries in the regularisation stacks.
def diag_kron_conjugate_qpm(a, b, c):
    """
    Efficiently compute diag(c.T @ kron(diag(a), diag(b)) @ c).

    c has to be an orthogonal quasi-permutation matrix.

    Notes
    -----
    Then the eager evaluation of sps.kron in c.T @ sps.kron(diag(a), diag(b)) @ c
    results in a comlexity of max(len(a) * len(b), c.nnz).
    This algorithm evaluates the Kronecker product lazily and thereby reduces
    the complexity to c.nnz.
    """
    assert a.ndim == 1 and b.ndim == 1
    assert c.shape[0] == a.size * b.size
    assert is_qpm(c, orthogonal=True)
    c = c.tocoo()
    return __diag_kron_conjugate_qpm(a, b, c.row, c.col)


@jit(nopython=True)
def __diag_kron_conjugate_qpm(a, b, cRow, cCol):
    out = np.empty(len(cCol))
    for l, k in zip(cRow, cCol):  # noqa: E741
        i = l // b.shape[0]
        j = l % b.shape[0]
        out[k] = a[i] * b[j]
    return out


def print_sparsity(array: sps.spmatrix | np.ndarray, console: Console):
    """Print the sparsity pattern of a matrix.

    Parameters
    ----------
    array : sps.spmatrix | np.ndarray
        The matrix to visualise.
    console : rich.Console
        The rich console to print to.
    """
    if isinstance(array, sps.spmatrix):
        array = array.toarray()
    assert isinstance(array, np.ndarray)
    m, n = array.shape
    array = np.pad(array, ((0, m % 2), (0, n % 2)))
    m, n = array.shape
    assert m % 2 == 0 and n % 2 == 0
    ret = ""
    for j in range(m // 2):
        for k in range(n // 2):
            block = array[2 * j : 2 * j + 2, 2 * k : 2 * k + 2] != 0
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

    console.print(Panel(ret[:-1], expand=False))


# for ij in range(16):
#     M = np.array(list(f"{bin(ij)[2:]:>04s}")).astype(int).reshape(2,2)
#     print(M)
#     print_sparsity(M)
#     print()
# print_sparsity(np.array([[0, 1, 1, 0], [1, 0, 0, 1]]))


if __name__ == "__main__":
    console = Console()
    rng = np.random.default_rng()

    l, d, r = 5, 10, 4
    D = l * d
    # density = 0.02
    density = 0.05
    rho = 2

    S = sps.random(D, r, density=density, random_state=rng)
    console.print(f"S shape: {S.shape}  ({S.nnz} nonzero)")
    console.print("S:")
    print_sparsity(S.T, console)

    console.rule("Sparse QC")
    Q, C = sparse_qc(S)
    assert is_qpm(Q, orthogonal=True)
    console.print(f"Q shape: {Q.shape}  ({Q.nnz} nonzero)")
    console.print(f"C shape: {C.shape}  ({C.nnz} nonzero)")
    console.print("Q:")
    print_sparsity(Q.T, console)
    omega_1 = np.arange(1, l + 1) ** rho
    omega_2 = rho ** np.arange(d)
    Omega = sps.kron(sps.diags(omega_1), sps.diags(omega_2))
    Beta = (Q.T @ Omega @ Q).todia()
    assert len(Beta.offsets) == 1 and Beta.offsets[0] == 0
    assert np.all(Beta.data == diag_kron_conjugate_qpm(omega_1, omega_2, Q))
    measures_1 = np.random.randn(20, l, 3)
    measures_2 = np.random.randn(20, d, 5)
    operator = (np.einsum("ndx,ney -> nxyde", measures_1, measures_2).reshape(20 * 3 * 5, D) @ Q).reshape(
        20, 3, 5, Q.shape[1]
    )
    assert np.allclose(operator, kron_dot_qpm(measures_1, measures_2, Q))

    console.rule("Minimal QC")
    Q, C = sparse_qc(S, minimal=True)
    console.print(f"Q shape: {Q.shape}  ({Q.nnz} nonzero)")
    console.print(f"C shape: {C.shape}  ({C.nnz} nonzero)")
    console.print("Q:")
    print_sparsity(Q.T, console)

    console.rule("Simulate core move")
    # This is a generalized core move.
    # The left basis Q of the core S is contracted to S before moving the core sparsely.
    # This results in a potentially larger orthogonal basis Unew, which however, can be decomposed as a tensor product.
    U, C = sparse_qc(S)
    Q = np.linalg.svd(np.random.randn(l, l))[0]
    assert np.allclose(Q.T @ Q, np.eye(l))
    QI = sps.kron(Q, sps.eye(d))
    QIU = QI @ U
    console.print(f"QIU shape: {QIU.shape}  ({QIU.nnz} nonzero)")  # type: ignore
    console.print(f"QIU NNZ bound: {U.nnz * Q.shape[1]}")
    print_sparsity(QIU.T, console)
    Unew, Qnew = sparse_qc(QIU)
    assert np.allclose((Unew @ Qnew @ C - QI @ S).data, 0)
    console.print(f"Unew shape: {Unew.shape}  ({Unew.nnz} nonzero)")
    console.print(f"Unew rank bound: {U.nnz * Q.shape[1]}")
    print_sparsity(Unew.T, console)
    console.print(f"Qnew shape: {Qnew.shape}  ({Qnew.nnz} nonzero)")
    print_sparsity(Qnew.T, console)
    # NOTE: The size of Qnew is not important, since it can be contracted into the core.
    console.print(f"Old nnz: {Q.size + U.nnz}")
    assert d == Unew.shape[0] // l
    X = Unew.tocsr()[:d, : Unew.shape[1] // l]
    assert (Unew - sps.kron(sps.eye(l), X)).nnz == 0  # type: ignore
    console.print(f"New nnz: {X.nnz}  ({Unew.nnz} uncompressed)")
    print_sparsity(X.T, console)
