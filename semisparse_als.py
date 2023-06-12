from __future__ import annotations

from functools import cached_property
from typing import cast, NewType
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
import sparseqr
import deal
from loguru import logger

from sparse_qc import fold_all, is_sparse_matrix, sparse_qc, kron_dot_qpm, diag_kron_conjugate_qpm
from lasso_lars import SimpleOperator, lasso_lars_cv, ConvergenceWarning

import autoPDB  # noqa: F401

NonNegativeInt = NewType("NonNegativeInt", int)
PositiveInt = NewType("PositiveInt", int)
IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]
warnings.filterwarnings(action="ignore", category=ConvergenceWarning, module="lasso_lars")


# Let P be the operator that maps the core space to the ambient tensor product space.
# Moreover, let Ω be the diagonal matrix that contains the weight sequence of this tensor product space.
# Recall that the basis for the core space should be chosen such that P.T @ Ω @ P is diagonal.
# This means that we use that basis U defined by
#     P.T @ Ω @ P == U @ Λ @ U.T .
# Due to the TT structure, we can write A = P.T @ Ω @ P as a product of A_left, weights (the weights for the
# current core) and A_right. Thus
#     U @ Λ @ U.T == eigh(einsum("lL, e, rR -> lerLeR", A_left, weights, A_right).reshape(ler, ler))
#                 == eigh(kron(A_left, diag(e), A_right))
#                 == kron(eigh(A_left), diag(e), eigh(A_right)) .
# This shows that the spectral decomposition can be performed on the left and right parts separately.
# After basis transform, we only need to store the diagonal part of the weight matrix.
# This is done in the following deflated sparse Q @ C decomposition.
# NOTE: The orthogonalised old core Q is uniquely defined by the new core C and the choice of basis for C.
#       In the current setting, we want C to be represented in an l^2-orthonormal eigenbasis of Q.T Ω Q.
#       This means that there is no need to think about the sparisty of Q and C - it is uniquely defined!


def permutation_matrix(permutation: IntArray) -> sps.coo_matrix:
    return sps.coo_matrix((np.ones(len(permutation)), (permutation, np.arange(len(permutation)))))


def qc(matrix: sps.spmatrix) -> tuple[sps.csr_matrix, sps.csr_matrix]:
    Q, R, E, rank = sparseqr.qr(matrix, economy=True)
    Q = Q.tocsr()[:, :rank]
    R = R.tocsr()[:rank]
    E = permutation_matrix(E)
    C = R @ E.T
    assert np.allclose((Q @ C - matrix).data, 0)
    return Q, C


def deflated_sparse_qc_kron(
    matrix: sps.spmatrix,
    left_weights: FloatArray,
    right_weights: FloatArray,
    rank_inflation: int = 2,
    precision: float = 1e-12,
) -> tuple[sps.spmatrix, sps.spmatrix, sps.spmatrix, FloatArray]:
    """Compute a basis for the range of the matrix.

    The returned basis is l2-orthonormal and orthogonal with respect to the weighted l2-inner product
    induced by `weights`.

    Returns
    -------
    A tuple (Q, P, C, w) where
        - Q (sps.spmatrix, QPM): Overcomplete generating system of the range of `matrix`.
        - P (sps.spmatrix): Basis of the range.
        - C (sps.spmatrix): Matrix such that `Q @ C == matrix`.
        - w (FloatArray): Subspace weights `diag(w) == Q.T @ diag(weights) @ Q`.
    """
    logger.debug("Compute sparse ")
    assert matrix.ndim == 2
    dimension, max_rank = matrix.shape
    assert len(left_weights) * len(right_weights) == dimension
    old_nnz = matrix.nnz
    logger.debug(f"Old nnz: {old_nnz:d}")

    q1, c1 = sparse_qc(matrix)
    assert q1.shape[1] == q1.nnz <= old_nnz
    assert np.allclose((q1 @ c1 - matrix).data, 0)
    q2, c2 = qc(c1)
    rank = q2.shape[1]
    assert rank <= q1.shape[1] and rank <= max_rank
    assert np.allclose((q2 @ c2 - c1).data, 0)

    # Inflate the rank
    q1_weights = diag_kron_conjugate_qpm(left_weights, right_weights, q1)
    full_probabilities = 1 / q1_weights
    full_probabilities /= np.sum(full_probabilities)
    rank_inflation = min(rank_inflation, len(full_probabilities), q2.shape[0] - q2.shape[1])
    indices = np.random.choice(
        len(full_probabilities),
        rank_inflation,
        replace=False,
        p=full_probabilities,
    )
    E = lambda k: np.eye(q2.shape[0], 1, k=-k)
    q2, *_ = sparseqr.qr(sps.hstack([q2] + [E(index) for index in indices]), economy=True)
    c2 = q2.T @ c1
    assert np.allclose((q2 @ c2 - c1).data, 0)
    rank += rank_inflation
    assert q2.shape == (q1.shape[1], rank)

    # Let
    #     - X = q1.T @ kron(diag(left_weights), diag(right_weights)) @ q1
    #     - Y = q2.T @ X @ q2
    # We want to choose a transform q3 such that q3.T @ Y @ q3 is diagonal.
    diag_x = q1_weights
    y = q2.T @ sps.diags(diag_x) @ q2
    y = y.toarray()
    es, q3 = np.linalg.eigh(y)
    assert np.allclose(q3 * es @ q3.T, y)
    c3 = q3.T @ c2
    q3[abs(q3) < precision] = 0
    c3[abs(c3) < precision] = 0
    q3 = sps.csr_matrix(q3)
    c3 = sps.csr_matrix(c3)
    q23 = q2 @ q3
    assert (q1 @ q23).nnz == q23.nnz
    logger.debug(f"New nnz: {q23.nnz:d} (upper bound: {old_nnz * rank:d})")
    return q1, q23, c3, es


def is_orthogonal(matrix: sps.spmatrix) -> bool:
    assert isinstance(matrix, sps.spmatrix)
    return matrix.shape[0] >= matrix.shape[1] and np.allclose((matrix.T @ matrix - sps.eye(matrix.shape[1])).data, 0)


class SemiSparseALS(object):
    @deal.ensure(lambda self, *args, result: self.has_consistent_components())
    @deal.ensure(lambda self, *args, result: self.has_consistent_stacks())
    def __init__(
        self,
        measures: list[FloatArray],
        values: FloatArray,
        weights: FloatArray,
        weight_sequences: list[FloatArray],
        components: list[FloatArray] | None = None,
        corePosition: NonNegativeInt | None = None,
    ):
        self.__initialised = False
        self.measures = measures
        for measure in self.measures:
            measure *= (weights ** (0.5 / len(measures)))[:, None]
        self.values = np.sqrt(weights) * values
        self.weight_sequences = weight_sequences
        if components is None:
            assert corePosition is None
            components = [np.eye(dimension, 1)[None] for dimension in self.dimensions]
            components[-1] *= np.mean(self.values)
            corePosition = cast(NonNegativeInt, self.order - 1)
        else:
            assert corePosition is not None
        self.__corePosition = corePosition
        self.components = components
        self.__stack = [None] * self.order + [np.ones((self.sampleSize, 1))]
        self.__weightStack = [None] * self.order + [np.ones(1)]
        # This uses that for corePosition == 0, stack[corePosition-1] = stack[-1].
        while self.corePosition > 0:
            self.move_core("left")
        self.regularisationParameters = [0] * self.order
        self.componentDensities = [1.0] * self.order
        self.sweepDirection = "right"
        self.__initialised = True
        self.rank_inflation = 2

    @property
    def measures(self) -> list[FloatArray]:
        return self.__measures

    @measures.setter
    def measures(self, measures: list[FloatArray]):
        if hasattr(self, "values") or hasattr(self, "weight_sequences"):
            raise AttributeError(
                "'SparseALS' object attribute 'measures' must be set before 'values' and 'weight_sequences"
            )
        if hasattr(self, "measures"):
            raise AttributeError("'SparseALS' object attribute 'measures' is read-only")
        if not (
            len(measures) > 0
            and all(m.ndim == 2 and m.shape[0] > 0 and m.shape[1] > 0 and np.all(np.isfinite(m)) for m in measures)
        ):
            raise ValueError("'measures' must be a sequence of finite, two-dimensional arrays")
        if not all(m.shape[0] == measures[0].shape[0] for m in measures[1:]):
            raise ValueError("arrays in 'measures' must have the same first dimensions")
        if not all(np.all(m[:, 0] == 1) for m in measures):
            raise ValueError("first basis function in 'measures' must be constant")
        self.__measures = measures.copy()

    @property
    def values(self) -> FloatArray:
        return self.__values

    @values.setter
    def values(self, values: FloatArray):
        if hasattr(self, "values"):
            raise AttributeError("'SparseALS' object attribute 'values' is read-only")
        if not (values.ndim == 1 and values.shape[0] > 0 and np.all(np.isfinite(values))):
            raise ValueError("'values' must be a sequence of finite, one-dimensional array")
        if not hasattr(self, "measures"):
            raise ValueError("'SparseALS' object attribute 'measures' must be set before 'values'")
        if values.shape != (self.measures[0].shape[0],):
            raise ValueError("'values' is incompatible with 'SparseALS' object attributes 'measures'")
        self.__values = values

    @property
    def weight_sequences(self) -> list[FloatArray]:
        return self.__weight_sequences

    @weight_sequences.setter
    def weight_sequences(self, weight_sequences: list[FloatArray]):
        if hasattr(self, "weight_sequences"):
            raise AttributeError("'SparseALS' object attribute 'weight_sequences' is read-only")
        if not (
            len(weight_sequences) > 0
            and all(
                w.ndim == 1 and w.shape[0] > 0 and np.all(np.isfinite(w)) and np.all(w >= 0) for w in weight_sequences
            )
        ):
            raise ValueError("'weight_sequences' must be a sequence of finite, non-negative, one-dimensional arrays")
        if not hasattr(self, "measures"):
            raise ValueError("'SparseALS' object attribute 'measures' must be set before 'weight_sequences'")
        if not (
            len(weight_sequences) == len(self.measures)
            and all(w.shape == (m.shape[1],) for w, m in zip(weight_sequences, self.measures))
        ):
            raise ValueError("'weight_sequences' is incompatible with 'SparseALS' object attributes 'measures'")
        # Let measures[:, i] denote a single rank-one measure (shape: (order, dimension)).
        # The entries of the tensor kron(measures[i]) must be bounded by those of kron(weight_sequences), i.e.
        #      abs(kron(measures[:, i]) / kron(weight_sequences)) <= 1  (where <= holds element-wise)
        # <--> kron(abs(measures[:, i])) / kron(weight_sequences) <= 1  (where <= holds element-wise)
        # <--> kron(abs(measures[:, i]) / weight_sequences) <= 1        (where <= holds element-wise)
        order, sample_size, dimension = self.measures.shape
        weight_sequences = np.asarray(weight_sequences)
        self.weight_sequence_sharpness = abs(self.measures) / weight_sequences[:, None, :]
        assert self.weight_sequence_sharpness.shape == (order, sample_size, dimension)
        # <--> max(kron(abs(measures[:, i]) / weight_sequences)) <= 1
        # <--> prod(max(abs(measures[:, i]) / weight_sequences, axis=1)) <= 1.
        self.weight_sequence_sharpness = np.product(np.max(self.weight_sequence_sharpness, axis=2), axis=0)
        assert self.weight_sequence_sharpness.shape == (sample_size,)
        # Since this has to hold for every i, ...
        self.weight_sequence_sharpness = np.max(self.weight_sequence_sharpness)
        if self.weight_sequence_sharpness > 1 + 1e-3:
            raise ValueError("'weight_sequences' must be larger than the sup norm of the basis functions")
        self.__weight_sequences = weight_sequences

    @property
    def corePosition(self) -> NonNegativeInt:
        """Return the position of the core tensor."""
        return cast(NonNegativeInt, self.__corePosition)

    @corePosition.setter
    @deal.pre(lambda self, newCorePosition: 0 <= newCorePosition < self.order)
    def corePosition(self, newCorePosition: int):
        if newCorePosition < self.__corePosition:
            for position in range(newCorePosition, self.__corePosition):
                component = self.__components[position]
                rightRank = component.shape[1]
                component.shape = (-1, self.dimensions[position] * rightRank)
        else:
            for position in range(self.__corePosition, newCorePosition):
                component = self.__components[position]
                leftRank = component.shape[0]
                component.shape = (leftRank * self.dimensions[position], -1)
        self.__corePosition = newCorePosition

    @property
    def sampleSize(self) -> PositiveInt:
        """Return the sample size  of the problem."""
        return cast(PositiveInt, self.values.size)

    @cached_property
    def dimensions(self) -> list[PositiveInt]:
        """Return the dimensions of the tensor train."""
        return [cast(PositiveInt, w.size) for w in self.weight_sequences]

    @property
    def ranks(self) -> list[PositiveInt]:
        """Return the representation ranks of the tensor train."""
        return [cast(PositiveInt, self.cshape(position)[2]) for position in range(self.order - 1)]

    @property
    def order(self) -> PositiveInt:
        """Return the order of the tensor train."""
        return cast(PositiveInt, len(self.dimensions))

    @property
    def parameters(self) -> NonNegativeInt:
        """Return the number of parameters."""
        return cast(NonNegativeInt, sum(component.nnz for component in self.__components))

    @deal.pre(lambda self, position: 0 <= position < self.order)
    @deal.post(lambda shape: len(shape) == 3 and shape[0] > 0 and shape[1] > 0 and shape[2] > 0)
    def cshape(self, position: int) -> tuple[PositiveInt, PositiveInt, PositiveInt]:
        """Return the shape of the component at the given position."""
        dimenion = self.dimensions[position]
        component = self.__components[position]
        if position < self.corePosition:
            # components[position].shape == (<left rank> * <dimension>, <right rank>)
            assert component.shape[0] % dimenion == 0
            leftRank = component.shape[0] // dimenion
            rightRank = component.shape[1]
        else:
            # components[position].shape == (<left rank>, <dimension> * <right rank>)
            assert component.shape[1] % dimenion == 0
            leftRank = component.shape[0]
            rightRank = component.shape[1] // dimenion
        return leftRank, dimenion, rightRank

    @property
    def components(self) -> list[FloatArray]:
        """Return the list of component tensors."""
        return [self.__components[position].toarray().reshape(self.cshape(position)) for position in range(self.order)]

    @components.setter
    def components(self, components: list[FloatArray]):
        if hasattr(self, "components"):
            raise AttributeError("'SparseALS' object attribute 'components' is read-only")
        if not (hasattr(self, "corePosition") and 0 <= self.corePosition < self.order):
            raise AttributeError("'SparseALS' object attribute 'corePosition' is out of bounds")
        if len(components) != self.order:
            raise ValueError("inconsistent length of 'components'")
        rightRank = 1
        for position in range(self.order):
            component = components[position]
            leftRank, dimension, newRightRank = component.shape
            if leftRank != rightRank:
                raise ValueError("inconsistent rank")
            if dimension != self.dimensions[position]:
                raise ValueError("inconsistent dimension")
            if not np.all(np.isfinite(component)):
                raise ValueError("components must be finite")
            rightRank = newRightRank
        if rightRank != 1:
            raise ValueError("inconsistent rank")
        self.__components = [cast(sps.spmatrix, None)] * self.order
        for position in range(self.order):
            component = components[position]
            self.set_component(position, sps.csr_matrix(component.reshape(-1)), component.shape)

    @deal.pre(lambda self, position, unfolding: 0 <= position < self.order and 0 <= unfolding < 4)
    def get_component(self, position: int, unfolding: int) -> sps.spmatrix:
        """Return the specified unfolding of the specified component tensor."""
        component = self.__components[position]
        if unfolding == 0:
            return component.reshape((1, -1), copy=True)
        elif unfolding == 1:
            return component.reshape((self.cshape(position)[0], -1), copy=True)
        elif unfolding == 2:
            return component.reshape((-1, self.cshape(position)[2]), copy=True)
        else:
            return component.reshape((-1, 1), copy=True)

    @fold_all
    def is_valid_component(self, position: int, component: sps.spmatrix, shape: tuple[int, int, int]):
        """Check if the given component is valid."""
        yield 0 <= position < self.order
        yield is_sparse_matrix(component)
        yield np.product(component.shape) == np.product(shape)
        yield len(shape) == 3 and np.product(shape) > 0
        if self.__initialised and position < self.corePosition:
            yield is_orthogonal(component.reshape(-1, shape[2]))
        elif self.__initialised and position > self.corePosition:
            yield is_orthogonal(component.reshape(shape[0], -1).T)
        else:
            yield np.all(np.isfinite(component.data))

    @deal.pre(is_valid_component)
    def set_component(self, position: int, component: sps.spmatrix, shape: tuple[int, int, int]):
        if position < self.corePosition:
            self.__components[position] = component.reshape((shape[0] * shape[1], shape[2]), copy=True)
        else:
            self.__components[position] = component.reshape((shape[0], shape[1] * shape[2]), copy=True)

    @fold_all
    def has_consistent_components(self):
        if not self.__initialised:
            return
        leftRank = 1
        for position in range(self.corePosition):
            component = self.__components[position]
            yield is_sparse_matrix(component)
            yield is_orthogonal(component)
            dimension = self.dimensions[position]
            rightRank = component.shape[1]
            yield component.shape == (leftRank * dimension, rightRank)
            leftRank = rightRank
        rightRank = 1
        for position in reversed(range(self.corePosition, self.order)):
            component = self.__components[position]
            yield is_sparse_matrix(component)
            yield deal.implies(position > self.corePosition, is_orthogonal(component.T))
            leftRank = component.shape[0]
            dimension = self.dimensions[position]
            yield component.shape == (leftRank, dimension * rightRank)
            rightRank = leftRank

    @fold_all
    def has_consistent_stacks(self):
        if not self.__initialised:
            return
        yield len(self.__stack) == self.order + 1
        yield len(self.__weightStack) == self.order + 1
        yield self.__stack[self.corePosition] is None
        yield self.__weightStack[self.corePosition] is None
        for position in range(self.corePosition):
            leftRank = self.cshape(position)[0]
            stackEntry = self.__stack[position - 1]
            yield isinstance(stackEntry, np.ndarray) and stackEntry.shape == (self.sampleSize, leftRank)
            stackEntry = self.__weightStack[position - 1]
            yield isinstance(stackEntry, np.ndarray) and stackEntry.shape == (leftRank,)
        for position in range(self.corePosition + 1, self.order):
            rightRank = self.cshape(position)[2]
            stackEntry = self.__stack[position + 1]
            yield isinstance(stackEntry, np.ndarray) and stackEntry.shape == (self.sampleSize, rightRank)
            stackEntry = self.__weightStack[position + 1]
            yield isinstance(stackEntry, np.ndarray) and stackEntry.shape == (rightRank,)

    @fold_all
    def is_canonicalised(self):
        for position in range(self.corePosition):
            yield is_sparse_matrix(self.__components[position])
            yield is_orthogonal(self.__components[position])
        for position in range(self.corePosition + 1, self.order):
            yield is_sparse_matrix(self.__components[position])
            yield is_orthogonal(self.__components[position].T)

    @deal.ensure(lambda self, *args, result: self.has_consistent_components())
    @deal.ensure(lambda self, *args, result: self.has_consistent_stacks())
    def move_core(self, direction: str):
        """Move the core.

        Parameters
        ----------
        direction : str
            The direction to move the core, either "left" or "right".

        Raises
        ------
        ValueError
            If the direction is neither "left" nor "right".
        """
        if direction == "left":
            if self.corePosition == 0:
                raise ValueError("Can not move further in direction 'left'.")
            k = self.corePosition
            logger.info(f"Move core: {k:d} → {k - 1:d}")
            newCore = self.get_component(position=k - 1, unfolding=2)
            leftRank, leftDimension, oldMiddleRank = self.cshape(k - 1)
            assert newCore.shape == (leftRank * leftDimension, oldMiddleRank)
            oldCore = self.get_component(position=k, unfolding=1)
            oldMiddleRank, rightDimension, rightRank = self.cshape(k)
            assert oldCore.shape == (oldMiddleRank, rightDimension * rightRank)
            logger.debug(f"Old nnz: {newCore.nnz:d} + {oldCore.nnz:d}")

            # TODO: Is it still important, that the old cores are retrieved before the core position changes?
            self.corePosition = k - 1
            Q, P, C, weights = deflated_sparse_qc_kron(
                oldCore.T, left_weights=self.weight_sequences[k], right_weights=self.__weightStack[k + 1]
            )
            # The returned values satisfy:
            #     - Q @ C == oldCore.T
            #         <-->  oldCore = C.T @ Qt  where  Qt = Q.T  ✓
            #     - diag(weights) == Q.T @ kron(diag(weight_sequences[k]), diag(weightStack[k + 1])) @ Q
            #         <-->  einsum("ldr, d, r, Ldr -> lL", Qt, weight_sequences[k], weightStack[k + 1], Qt)
            #               == diag(weights)  ✓
            #               (l and L denote the internal rank index of Qt.)
            QP = Q @ P
            newMiddleRank = QP.shape[1]
            assert QP.T.shape == (newMiddleRank, rightDimension * rightRank)
            assert C.T.shape == (oldMiddleRank, newMiddleRank)
            oldCore = QP.T
            newCore = newCore @ C.T
            self.set_component(position=k, component=oldCore, shape=(newMiddleRank, rightDimension, rightRank))
            self.set_component(position=k - 1, component=newCore, shape=(leftRank, leftDimension, newMiddleRank))
            logger.debug(f"New nnz: {newCore.nnz:d} + {oldCore.nnz:d}")

            # Compute out[i] = outer(measures[k][i], stack[k + 1][i]) @ oldCore.T == outer(measures[k][i], stack[k + 1][i]) @ Q @ P
            self.__stack[k] = kron_dot_qpm(self.measures[k], self.__stack[k + 1], Q) @ P
            self.__stack[k - 1] = None
            self.__weightStack[k] = weights
            self.__weightStack[k - 1] = None

        elif direction == "right":
            if self.corePosition == self.order - 1:
                raise ValueError("Can not move further in direction 'right'.")
            k = self.corePosition
            logger.info(f"Move core: {k:d} → {k + 1:d}")
            oldCore = self.get_component(position=k, unfolding=2)
            leftRank, leftDimension, oldMiddleRank = self.cshape(k)
            assert oldCore.shape == (leftRank * leftDimension, oldMiddleRank)
            newCore = self.get_component(position=k + 1, unfolding=1)
            oldMiddleRank, rightDimension, rightRank = self.cshape(k + 1)
            assert newCore.shape == (oldMiddleRank, rightDimension * rightRank)
            logger.debug(f"Old nnz: {newCore.nnz:d} + {oldCore.nnz:d}")

            # TODO: Is it still important, that the old cores are retrieved before the core position changes?
            self.corePosition = k + 1
            Q, P, C, weights = deflated_sparse_qc_kron(
                oldCore, left_weights=self.__weightStack[k - 1], right_weights=self.weight_sequences[k]
            )
            # The returned values satisfy:
            #     - Q @ C == oldCore  ✓
            #     - diag(weights) == Q.T @ kron(diag(weightStack[k - 1]), diag(weight_sequences[k])) @ Q
            #         <-->  einsum("ldr, l, d, ldR -> rR", Q, weightStack[k - 1], weight_sequences[k], Q)
            #               == diag(weights)  ✓
            #               (r and R denote the internal rank index of Q.)
            QP = Q @ P
            newMiddleRank = QP.shape[1]
            assert QP.shape == (leftRank * leftDimension, newMiddleRank)
            assert C.shape == (newMiddleRank, oldMiddleRank)
            oldCore = QP
            newCore = C @ newCore
            self.set_component(position=k, component=oldCore, shape=(leftRank, leftDimension, newMiddleRank))
            self.set_component(position=k + 1, component=newCore, shape=(newMiddleRank, rightDimension, rightRank))
            logger.debug(f"New nnz: {newCore.nnz:d} + {oldCore.nnz:d}")

            # Compute out[i] = outer(stack[k - 1][i], measures[k][i]) @ oldCore == outer(stack[k - 1][i], measures[k][i]) @ Q @ P
            self.__stack[k] = kron_dot_qpm(self.__stack[k - 1], self.measures[k], Q) @ P
            self.__stack[k + 1] = None
            self.__weightStack[k] = weights
            self.__weightStack[k + 1] = None

    @deal.pre(lambda self, set: self.is_canonicalised())
    def microstep(self, set: slice = slice(None)) -> float:
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        k = self.corePosition

        weights = np.kron(self.__weightStack[k - 1], self.weight_sequences[k])  # type: ignore
        weights = np.kron(weights, self.__weightStack[k + 1])
        lOp = self.__stack[k - 1][set]  # type: ignore
        eOp = self.measures[k][set]
        rOp = self.__stack[k + 1][set]  # type: ignore
        operator = (lOp[:, :, None, None] * eOp[:, None, :, None] * rOp[:, None, None, :]).reshape(lOp.shape[0], -1)
        # nl, ne, nr -> nler
        operator /= weights[None]
        assert np.all(np.isfinite(operator))
        operator = SimpleOperator(operator)
        # TODO: We could also use a tensor product operator.
        #       This would be more efficient.
        # TODO: Check that these operators provide the API defined by sps.linalg.LinearOperator
        #       and that lasso_lars_cv works for all these operators.
        cv = 10
        max_features = self.__components[k].nnz + 1
        model = lasso_lars_cv(operator, self.values[set], cv=cv, max_features=max_features)
        assert model.alpha_ >= 0
        assert len(model.active_) > 0
        # assert np.linalg.norm(model.coef_) > 0
        coreData = model.coef_ / weights[model.active_]
        # assert np.linalg.norm(coreData) > 0
        if np.linalg.norm(coreData) == 0:
            coreData[coreData == 0] = 1e-12 * np.sqrt(np.mean(self.values[set] ** 2))
        coreRow = np.zeros(len(model.active_), dtype=np.int32)
        coreCol = model.active_
        core = sps.coo_matrix((coreData, (coreRow, coreCol)), shape=(1, len(weights)))
        self.set_component(position=k, component=core, shape=(lOp.shape[1], eOp.shape[1], rOp.shape[1]))
        self.regularisationParameters[k] = model.alpha_
        self.componentDensities[k] = len(model.active_) / len(weights)
        return model.cv_error_

    def step(self, set: slice = slice(None)) -> float:
        validationError = self.microstep(set)
        if self.order == 1:
            return
        limit = {"left": 0, "right": self.order - 1}[self.sweepDirection]
        if self.corePosition == limit:
            turn = {"left": "right", "right": "left"}[self.sweepDirection]
            self.sweepDirection = turn
        self.move_core(self.sweepDirection)
        return validationError

    def residual(self, set: slice = slice(None)) -> np.floating:
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        k = self.corePosition

        lOp = self.__stack[k - 1][set]  # type: ignore
        eOp = self.measures[k][set]
        rOp = self.__stack[k + 1][set]  # type: ignore
        erOp = (eOp[:, :, None] * rOp[:, None, :]).reshape(eOp.shape[0], -1)  # ne, nr -> ner
        core = self.get_component(position=k, unfolding=1)
        assert core.shape == (lOp.shape[1], erOp.shape[1])
        if core.shape[0] < core.shape[1]:
            prediction = np.einsum("nl, nl -> n", erOp @ core.T, lOp)
        else:
            prediction = np.einsum("nx, nx -> n", lOp @ core, erOp)
        return np.linalg.norm(prediction - self.values[set]) / np.linalg.norm(self.values[set])
