from functools import cached_property
from typing import NewType
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps

from sparse_qc import isqpm, sparse_qc, kron_dot_qpm, diag_kron_conjugate_qpm
from lasso_lars import SimpleOperator, lasso_lars_cv, ConvergenceWarning


NonNegativeInt = NewType("NonNegativeInt", int)
PositiveInt = NewType("PositiveInt", int)
FloatArray = NDArray[np.float_]
warnings.filterwarnings(action='ignore', category=ConvergenceWarning, module='lasso_lars')


class SparseALS(object):
    def __init__(self, measures, values, weights):
        assert values.ndim == 1 and values.shape[0] > 0
        self.__values = values
        assert all(w.ndim == 1 for w in weights)
        self.__weights = weights
        assert len(measures) == self.order > 0
        for m in range(self.order):
            assert measures[m].shape == (self.sampleSize, self.dimensions[m])
        self.__measures = measures
        self.assert_valid_basics()
        self.__components = [sps.eye(dimension, 1) for dimension in self.dimensions[:-1]]
        self.__components.append(np.mean(values) * sps.eye(1, self.dimensions[-1]))
        self.__ranks = np.empty(self.order + 1, dtype=int)
        self.__corePosition = self.order - 1
        self.assert_valid_components()
        self.assert_canonicalised()
        self.__stack = [None] * self.order + [ np.ones((self.sampleSize, 1)) ]
        self.__weightStack = [None] * self.order + [ np.ones(1) ]
        # This uses that for corePosition == 0, stack[corePosition-1] = stack[-1].
        while self.corePosition > 0:
            self.move_core("left")
        self.assert_valid_components()
        self.assert_canonicalised()
        self.assert_valid_stacks()
        self.lambdas = [0] * self.order
        self.densities = [1] * self.order
        self.sweepDirection = "right"

    def assert_valid_basics(self) -> bool:
        """Ensure a valid basic internal state."""
        assert self.__values.ndim == 1 and self.sampleSize == self.__values.size > 0
        assert self.order == len(self.__measures) == len(self.__weights) > 0
        assert len(self.dimensions) == self.order
        for position in range(self.order):
            assert self.__weights[position].shape == (self.dimensions[position],)
            assert self.__measures[position].shape == (self.sampleSize, self.dimensions[position])

    def assert_valid_components(self) -> bool:
        """Ensure the components constitute a valid tensor train."""
        assert len(self.__components) == self.order
        leftRanks = [self.rank(position, 0) for position in range(self.order)] + [1]
        rightRanks = [1] + [self.rank(position, 2) for position in range(self.order)]
        assert leftRanks == rightRanks

    def assert_valid_stacks(self) -> bool:
        """Ensure the stacks are consistent."""
        assert isinstance(self.corePosition, int)
        assert 0 <= self.corePosition < self.order
        assert len(self.__stack) == self.order + 1
        assert len(self.__weightStack) == self.order + 1
        assert self.__stack[self.corePosition] is None
        assert self.__weightStack[self.corePosition] is None
        for position in range(self.corePosition):
            leftRank = self.rank(position, 0)
            assert self.__stack[position-1].shape == (self.sampleSize, leftRank)
            assert self.__weightStack[position-1].shape == (leftRank,)
        for position in range(self.corePosition+1, self.order):
            rightRank = self.rank(position, 2)
            assert self.__stack[position+1].shape == (self.sampleSize, rightRank)
            assert self.__weightStack[position+1].shape == (rightRank,)

    def assert_canonicalised(self) -> bool:
        """Ensure the tensor train is canonicalised."""
        assert isinstance(self.corePosition, int)
        assert 0 <= self.corePosition < self.order
        for position in range(self.corePosition):
            component = self.get_component(position, unfolding=2)
            assert isqpm(component, orthogonal=True)
        for position in range(self.corePosition + 1, self.order):
            component = self.get_component(position, unfolding=1)
            assert isqpm(component.T, orthogonal=True)

    @property
    def corePosition(self) -> NonNegativeInt:
        """The position of the core tensor.

        All component tensors left of the core tensor are left-orthogonal.
        All component tensors right of the core tensor are right-orthogonal.
        """
        return self.__corePosition

    @corePosition.setter
    def corePosition(self, newCorePosition):
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
        self.assert_valid_components()

    @property
    def sampleSize(self) -> NonNegativeInt:
        """The sample size  of the problem."""
        return self.__values.size

    @cached_property
    def dimensions(self) -> list[PositiveInt]:
        """The dimensions of the tensor train."""
        return [w.size for w in self.__weights]

    @property
    def order(self) -> PositiveInt:
        """The order of the tensor train."""
        return len(self.dimensions)

    @property
    def parameters(self) -> NonNegativeInt:
        """The number of parameters."""
        return sum(component.nnz for component in self.__components)

    def rank(self, position, mode):
        assert 0 <= position < self.order
        assert 0 <= mode < 3
        if mode == 1: return self.dimensions[position]
        if position < self.corePosition and mode == 2:
            # components[position].shape == (<left rank> * <dimension>, <right rank>)
            return self.__components[position].shape[1]
        elif position < self.corePosition and mode == 0:
            assert self.__components[position].shape[0] % self.dimensions[position] == 0
            return self.__components[position].shape[0] // self.dimensions[position]
        elif mode == 0:  # It must hold that position >= self.corePosition.
            # components[position].shape == (<left rank>, <dimension> * <right rank>)
            return self.__components[position].shape[0]
        else:
            assert self.__components[position].shape[1] % self.dimensions[position] == 0
            return self.__components[position].shape[1] // self.dimensions[position]

    def component_shape(self, position):
        return (self.rank(position, 0), self.rank(position, 1), self.rank(position, 2))

    @property
    def ranks(self) -> list[PositiveInt]:
        """The representation rank of the tensor train."""
        for position in range(self.order):
            self.__ranks[position] = self.rank(position, 0)
        self.__ranks[self.order] = 1
        return self.__ranks

    def get_component(self, position, unfolding):
        assert 0 <= position < self.order
        assert 0 <= unfolding < 4
        component = self.__components[position]
        if unfolding == 0:
            return component.reshape((1, -1), copy=True)
        elif unfolding == 1:
            return component.reshape((self.rank(position, 0), -1), copy=True)
        elif unfolding == 2:
            return component.reshape((-1, self.rank(position, 2)), copy=True)
        else:
            return component.reshape((-1, 1), copy=True)

    def set_component(self, position, component, shape):
        assert 0 <= position < self.order
        assert isinstance(shape, tuple) and len(shape) == 3
        assert shape[1] == self.dimensions[position]
        if position < self.corePosition:
            self.__components[position] = component.reshape((shape[0] * shape[1], shape[2]), copy=True)
        else:
            self.__components[position] = component.reshape((shape[0], shape[1] * shape[2]), copy=True)

    def get_tensor(self) -> list[FloatArray]:
        out = []
        for position in range(self.order):
            out.append(self.__components[position].toarray().reshape(
                self.ranks[position],
                self.dimensions[position],
                self.ranks[position+1]
            ))
        return out
    
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
        self.assert_valid_basics()
        self.assert_valid_components()

        if direction == "left":
            if self.corePosition == 0:
                raise ValueError(f"Can not move further in direction 'left'.")
            k = self.corePosition
            newCore = self.get_component(position=k-1, unfolding=2)
            leftRank, leftDimension, oldMiddleRank = self.component_shape(k-1)
            assert newCore.shape == (leftRank * leftDimension, oldMiddleRank)
            oldCore = self.get_component(position=k, unfolding=1)
            oldMiddleRank, rightDimension, rightRank = self.component_shape(k)
            assert oldCore.shape == (oldMiddleRank, rightDimension * rightRank)

            # TODO: Is it still important, that the old cores are retrieved before the core position changes?
            self.corePosition = k-1
            Q, C = sparse_qc(oldCore.T)  # oldCore = C.T @ Q.T
            newMiddleRank = Q.shape[1]
            assert Q.T.shape == (newMiddleRank, rightDimension * rightRank)
            assert C.T.shape == (oldMiddleRank, newMiddleRank)
            self.set_component(position=k, component=Q.T, shape=(newMiddleRank, rightDimension, rightRank))
            self.set_component(position=k-1, component=newCore @ C.T, shape=(leftRank, leftDimension, newMiddleRank))

            # Since Q.shape == (dimension * rightRank, newRank), we need kron(measures, stack).
            self.__stack[k] = kron_dot_qpm(self.__measures[k], self.__stack[k+1], Q)
            self.__stack[k-1] = None
            self.__weightStack[k] = diag_kron_conjugate_qpm(self.__weights[k], self.__weightStack[k+1], Q)
            self.__weightStack[k-1] = None

        elif direction == "right":
            if self.corePosition == self.order - 1:
                raise ValueError(f"Can not move further in direction 'right'.")
            k = self.corePosition
            oldCore = self.get_component(position=k, unfolding=2)
            leftRank, leftDimension, oldMiddleRank = self.component_shape(k)
            assert oldCore.shape == (leftRank * leftDimension, oldMiddleRank)
            newCore = self.get_component(position=k+1, unfolding=1)
            oldMiddleRank, rightDimension, rightRank = self.component_shape(k+1)
            assert newCore.shape == (oldMiddleRank, rightDimension * rightRank)

            # TODO: Is it still important, that the old cores are retrieved before the core position changes?
            self.corePosition = k+1
            Q, C = sparse_qc(oldCore)  # oldCore = Q @ C
            newMiddleRank = Q.shape[1]
            assert Q.shape == (leftRank * leftDimension, newMiddleRank)
            assert C.shape == (newMiddleRank, oldMiddleRank)
            self.set_component(position=k, component=Q, shape=(leftRank, leftDimension, newMiddleRank))
            self.set_component(position=k+1, component=C @ newCore, shape=(newMiddleRank, rightDimension, rightRank))

            # Since Q.shape == (leftRank * dimension, r), we need kron(stack, measures).
            self.__stack[k] = kron_dot_qpm(self.__stack[k-1], self.__measures[k], Q)
            self.__stack[k+1] = None
            self.__weightStack[k] = diag_kron_conjugate_qpm(self.__weightStack[k-1], self.__weights[k], Q)
            self.__weightStack[k+1] = None

    def microstep(self, set=slice(None)):
        self.assert_valid_basics()
        self.assert_valid_components()
        self.assert_canonicalised()
        self.assert_valid_stacks()
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        k = self.corePosition
        l,e,r = self.component_shape(k)

        weights = np.kron(np.kron(self.__weightStack[k-1], self.__weights[k]), self.__weightStack[k+1])
        lOp = self.__stack[k-1][set]
        eOp = self.__measures[k][set]
        rOp = self.__stack[k+1][set]
        operator = (lOp[:, :, None, None] * eOp[:, None, :, None] * rOp[:, None, None, :]).reshape(lOp.shape[0], -1)
        # nl, ne, nr -> nler
        operator /= weights[None]
        assert np.all(np.isfinite(operator))
        operator = SimpleOperator(operator)
        # TODO: We could also use a tensor product operator.
        #       This would be more efficient.
        # TODO: Check that these operators provide the API defined by sps.linalg.LinearOperator
        #       and that lasso_lars_cv works for all these operators.
        model = lasso_lars_cv(operator, self.__values[set], cv=10)
        assert model.alpha_ >= 0
        assert len(model.active_) > 0
        coreData = model.coef_ / weights[model.active_]
        coreRow = np.zeros(len(model.active_), dtype=np.int32)
        coreCol = model.active_
        core = sps.coo_matrix((coreData, (coreRow, coreCol)), shape=(1, len(weights)))
        self.set_component(position=k, component=core, shape=(lOp.shape[1], eOp.shape[1], rOp.shape[1]))
        self.lambdas[k] = model.alpha_
        self.densities[k] = len(model.active_) / len(weights)
    
    def step(self, set=slice(None)):
        self.microstep(set)
        if self.order == 1:
            return
        limit = {"left": 0, "right": self.order - 1}[self.sweepDirection]
        if self.corePosition == limit:
            turn = {"left": "right", "right": "left"}[self.sweepDirection]
            self.sweepDirection = turn
        self.move_core(self.sweepDirection)

    def residual(self, set=slice(None)):
        self.assert_valid_basics()
        self.assert_valid_components()
        self.assert_canonicalised()
        self.assert_valid_stacks()
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        k = self.corePosition

        lOp = self.__stack[k-1][set]
        eOp = self.__measures[k][set]
        rOp = self.__stack[k+1][set]
        erOp = (eOp[:, :, None] * rOp[:, None, :]).reshape(eOp.shape[0], -1)  # ne, nr -> ner
        core = self.get_component(position=k, unfolding=1)
        assert core.shape == (lOp.shape[1], erOp.shape[1])
        if core.shape[0] < core.shape[1]:
            prediction = np.einsum("nl, nl -> n", erOp @ core.T, lOp)
        else:
            prediction = np.einsum("nx, nx -> n", lOp @ core, erOp)
        return np.linalg.norm(prediction - self.__values[set]) / np.linalg.norm(self.__values[set])
