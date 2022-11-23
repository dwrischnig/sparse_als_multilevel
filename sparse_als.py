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
        self.__leftStack = [ np.ones((self.sampleSize, 1)) ] + [None] * self.corePosition
        self.__rightStack = [ np.ones((self.sampleSize, 1)) ]
        self.__leftWeightStack = [ np.ones(1) ] + [None] * self.corePosition
        self.__rightWeightStack = [ np.ones(1) ]
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
        leftRanks = []
        rightRanks = [1]
        for position in range(self.corePosition):
            component = self.__components[position]
            assert isinstance(component, sps.spmatrix)
            assert np.all(np.isfinite(component.data))
            # components[position].shape == (<left rank> * <dimension>, <right rank>)
            leftRanks.append(component.shape[0] // self.dimensions[position])
            rightRanks.append(component.shape[1])
        for position in range(self.corePosition, self.order):
            component = self.__components[position]
            assert isinstance(component, sps.spmatrix)
            assert np.all(np.isfinite(component.data))
            # components[position].shape == (<left rank>, <dimension> * <right rank>)
            leftRanks.append(component.shape[0])
            rightRanks.append(component.shape[1] // self.dimensions[position])
        leftRanks.append(1)
        assert np.all(np.array(leftRanks) == rightRanks)
        assert np.all(np.array(leftRanks) == self.ranks)

    def assert_valid_stacks(self) -> bool:
        """Ensure the stacks are consistent."""
        assert isinstance(self.corePosition, int)
        assert 0 <= self.corePosition < self.order
        ranks = self.ranks
        assert len(self.__leftStack) == len(self.__leftWeightStack) == self.corePosition + 1
        for position in range(self.corePosition + 1):
            assert self.__leftStack[position].shape == (self.sampleSize, ranks[position])
            assert self.__leftWeightStack[position].shape == (ranks[position],)
        assert len(self.__rightStack) == len(self.__rightWeightStack) == self.order - self.corePosition
        for position in range(self.order - self.corePosition):
            assert self.__rightStack[position].shape == (self.sampleSize, ranks[self.order - position])
            assert self.__rightWeightStack[position].shape == (ranks[self.order - position],)

    def assert_canonicalised(self) -> bool:
        """Ensure the tensor train is canonicalised."""
        assert isinstance(self.corePosition, int)
        assert 0 <= self.corePosition < self.order
        for position in range(self.corePosition):
            component = self.__components[position]
            assert isqpm(component)
        for position in range(self.corePosition + 1, self.order):
            component = self.__components[position]
            assert isqpm(component.T)

    @property
    def corePosition(self) -> NonNegativeInt:
        """The position of the core tensor.

        All component tensors left of the core tensor are left-orthogonal.
        All component tensors right of the core tensor are right-orthogonal.
        """
        return self.__corePosition
    
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

    @property
    def ranks(self) -> list[PositiveInt]:
        """The representation rank of the tensor train."""
        self.__ranks[0] = self.__ranks[-1] = 1
        # ranks[0] = ranks[-1] = 1
        # ranks[position+1] = components[position].shape[-1] = components[position + 1].shape[0]
        for position in range(self.corePosition):
            # components[position].shape == (<left rank> * <dimension>, <right rank>)
            self.__ranks[position + 1] = self.__components[position].shape[1]
        for position in range(self.corePosition, self.order):
            # components[position].shape == (<left rank>, <dimension> * <right rank>)
            self.__ranks[position] = self.__components[position].shape[0]
        return self.__ranks

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
            rank = self.ranks
            dimension = self.dimensions[self.corePosition]
            k = self.corePosition

            oldCore = self.__components[k]
            assert oldCore.shape == (rank[k], dimension * rank[k+1])
            # This is already the correct matricisation.
            Q, C = sparse_qc(oldCore.T)
            r = Q.shape[1]
            assert Q.shape == (dimension * rank[k+1], r)
            assert C.shape == (r, rank[k])
            self.__components[k] = Q.T
            newCore = self.__components[k-1]
            assert newCore.shape == (rank[k-1] * dimension, rank[k])
            newCore = newCore @ C.T
            assert newCore.shape == (rank[k-1] * dimension, r)
            # This is the wrong matricisation.
            newCore = newCore.reshape(rank[k-1], dimension * r)
            self.__components[k-1] = newCore
            self.__corePosition = k-1

            self.__rightStack.append(
                kron_dot_qpm(self.__measures[k], self.__rightStack[-1], Q)
            )
            self.__rightWeightStack.append(
                diag_kron_conjugate_qpm(self.__weights[k], self.__rightWeightStack[-1], Q)
            )
            self.__leftStack.pop()
            self.__leftWeightStack.pop()

        elif direction == "right":
            if self.corePosition == self.order - 1:
                raise ValueError(f"Can not move further in direction 'right'.")
            rank = self.ranks
            dimension = self.dimensions[self.corePosition]
            k = self.corePosition

            oldCore = self.__components[k]
            assert oldCore.shape == (rank[k], dimension * rank[k+1])
            # This is the wrong matricisation.
            oldCore = oldCore.reshape(rank[k] * dimension, rank[k+1])
            Q, C = sparse_qc(oldCore)
            r = Q.shape[1]
            assert Q.shape == (rank[k] * dimension, r)
            # This is the correct matricisation.
            self.__components[k] = Q
            # TODO: Define a setter for components, that ensures consistency!
            #       set_left_orthogonal_component(k, Q) -- checks the shape and matricisation
            #       set_right_orthogonal_component(k, Q)
            assert C.shape == (r, rank[k+1])
            newCore = self.__components[k+1]
            assert newCore.shape == (rank[k+1], dimension * rank[k+2])
            newCore = C @ newCore
            assert newCore.shape == (r, dimension * rank[k+2])
            # This is the correct matriciation.
            self.__components[k+1] = newCore
            self.__corePosition = k+1

            self.__leftStack.append(
                kron_dot_qpm(self.__leftStack[-1], self.__measures[k], Q)
            )
            self.__leftWeightStack.append(
                diag_kron_conjugate_qpm(self.__leftWeightStack[-1], self.__weights[k], Q)
            )
            self.__rightStack.pop()
            self.__rightWeightStack.pop()

    def microstep(self, set=slice(None)):
        self.assert_valid_basics()
        self.assert_valid_components()
        self.assert_canonicalised()
        self.assert_valid_stacks()
        if not isinstance(set, slice):
            assert np.ndim(set) == 1
        k = self.corePosition

        lw = self.__leftWeightStack[-1]
        ew = self.__weights[k]
        rw = self.__rightWeightStack[-1]
        weights = np.einsum("l,e,r -> ler", lw, ew, rw).reshape(-1)

        lOp = self.__leftStack[-1][set]
        eOp = self.__measures[k][set]
        rOp = self.__rightStack[-1][set]
        operator = np.einsum("nl,ne,nr -> nler", lOp, eOp, rOp).reshape(eOp.shape[0], -1)
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
        core = core.reshape(len(lw), len(ew) * len(rw))
        self.__components[k] = core
        
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

        lOp = self.__leftStack[-1][set]
        eOp = self.__measures[k][set]
        rOp = self.__rightStack[-1][set]
        erOp = np.einsum("ne,nr -> ner", eOp, rOp).reshape(eOp.shape[0], -1)
        core = self.__components[k]
        assert core.shape == (lOp.shape[1], erOp.shape[1])
        if core.shape[0] < core.shape[1]:
            prediction = np.einsum("nl, nl -> n", erOp @ core.T, lOp)
        else:
            prediction = np.einsum("nx, nx -> n", lOp @ core, erOp)
        return np.linalg.norm(prediction - self.__values[set]) / np.linalg.norm(self.__values[set])