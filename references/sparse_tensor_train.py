"""The elements and tangent spaces of a model class of tensor trains."""
from __future__ import annotations

import sys
from typing import Any, Callable, NewType
from functools import cache, cached_property, wraps
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
from scipy.linalg import block_diag


NonNegativeInt = NewType("NonNegativeInt", int)
PositiveInt = NewType("PositiveInt", int)
NonNegativeFloat = NewType("NonNegativeFloat", float)
BoolArray = NDArray[Any]
# TODO: NDArray[bool] produces the error: ... Type "dtype[bool]" cannot be assigned to type "generic"
FloatArray = NDArray[np.float_]
ObjectArray = NDArray[Any]
# TODO: NDArray[object] produces the error: ... Type "dtype[object]" cannot be assigned to type "generic"


RoundingCondition = Callable[[PositiveInt, FloatArray], BoolArray]


def default_rounding_condition(position: PositiveInt, singularValues: FloatArray) -> BoolArray:
    """Round singular values that are numerically zero.

    Parameters
    ----------
    position : PositiveInt
        The position of the edge. (Edge position is between components[position-1] and components[position].)
    singularValues : FloatArray

    Returns
    -------
    mask : BoolArray
        A mask indicating which singular values should be kept.
    """
    assert len(singularValues) > 0
    mask = singularValues > 1e-12 * np.linalg.norm(singularValues)
    mask[0] = True
    return mask


def matricisation(component: FloatArray, left: tuple[NonNegativeInt], right: tuple[NonNegativeInt]) -> FloatArray:
    """Compute the prescribed matricisation.

    Parameters
    ----------
    component : FloatArray
        The tensor for which to compute the matricisation.

    Returns
    -------
    FloatArray
    """
    component = np.transpose(component, left + right)
    return component.reshape(np.prod(component.shape[:len(left)]), -1)


def orthogonal_complement(basis: FloatArray) -> FloatArray:
    """Compute the orthogonal complement.

    Parameters
    ----------
    basis : FloatArray
        The orthonormal basis for which the orthogonal complement shall be computed.

    Returns
    -------
    FloatArray
    """
    assert basis.ndim == 2 and basis.shape[0] >= basis.shape[1]
    dimension, defect = basis.shape
    assert np.allclose(basis.T @ basis, np.eye(defect))
    kernelProjection = np.eye(dimension) - basis @ basis.T
    weights, complement = np.linalg.eigh(kernelProjection)
    assert np.allclose(weights[:defect], 0) and np.allclose(weights[defect:], 1)
    complement = complement[:, defect:]
    rank = dimension - defect
    assert complement.shape == (dimension, rank) and np.allclose(complement @ complement.T, kernelProjection)
    return complement


def fold_all(generator: Iterator[bool]):
    @wraps(generator)
    def wrapper(*args, **kwargs) -> bool:
        return all(generator(*args, **kwargs))
    return wrapper


class Element(__Element):
    """An element of the tensor train manifold.

    Attributes
    ----------
    components : list[FloatArray]
        The list of component tensors.
    corePosition : NonNegativeInt
    """

    def copy(self) -> Element:
        """Return a copy of the element."""
        assert self.is_valid()
        cls = self.__class__
        tt = cls.__new__(cls)
        tt.components = [component.copy() for component in self.components]
        tt.__corePosition = self.corePosition
        tt.univariateGramians = [gramian.copy() for gramian in self.univariateGramians]
        return tt

    @classmethod
    def from_components(cls, components: list[FloatArray]) -> Element:
        """Construct a tensor train from a list of component tensors."""
        tt = cls.__new__(cls)
        tt.components = [component.copy() for component in components]
        tt.__corePosition = 0
        if not tt.is_valid():
            raise ValueError("The components do not constitute a valid tensor train.")
        tt.round()
        tt.univariateGramians = [np.eye(dimension) for dimension in tt.dimensions]
        return tt

    @fold_all
    def is_valid(self) -> bool:
        """Check if the components and core position constitute a valid tensor train."""
        yield len(self.components) > 0
        for component in self.components:
            yield isinstance(component, np.ndarray)
            yield component.ndim == 3
            yield component.shape[0] > 0
            yield component.shape[1] > 0
            yield component.shape[2] > 0
            yield np.all(np.isfinite(component))
        yield self.components[0].shape[0] == 1
        for position in range(len(self.components) - 1):
            leftRank = self.components[position - 1].shape[2]
            rightRank = self.components[position].shape[0]
            yield leftRank == rightRank
        yield self.components[-1].shape[-1] == 1
        yield isinstance(self.corePosition, int)
        yield 0 <= self.corePosition < self.order

    @fold_all
    def is_canonicalised(self) -> bool:
        for position in range(self.corePosition):
            component = self.components[position]
            l, e, r = component.shape
            component = component.reshape(l * e, r)
            yield np.allclose(component.T @ component, np.eye(r))
        for position in range(self.corePosition + 1, self.order):
            component = self.components[position]
            l, e, r = component.shape
            component = component.reshape(l, e * r)
            yield np.allclose(component @ component.T, np.eye(l))

    @property
    def order(self) -> PositiveInt:
        """The order of the tensor train."""
        return len(self.components)

    @property
    def dimensions(self) -> list[PositiveInt]:
        """The dimensions of the tensor train."""
        return [component.shape[1] for component in self.components]

    @property
    def ranks(self) -> list[PositiveInt]:
        """The representation rank of the tensor train."""
        return [1] + [component.shape[2] for component in self.components]

    @property
    def parameters(self) -> NonNegativeInt:
        """The number of parameters."""
        return sum(component.size for component in self.components)

    @property
    def corePosition(self) -> NonNegativeInt:
        """The position of the core tensor.

        All component tensors left of the core tensor are left-orthogonal.
        All component tensors right of the core tensor are right-orthogonal.
        """
        return self.__corePosition

    def move_core(
        self, direction: str, rounding_condition: RoundingCondition = default_rounding_condition
    ) -> tuple[FloatArray, BoolArray]:
        """Move the core.

        Parameters
        ----------
        direction : str
            The direction to move the core, either "left" or "right".
        rounding_condition : RoundingCondition, optional
            The rounding condition to use, by default default_rounding_condition

        Returns
        -------
        tuple[FloatArray, BoolArray]
            The list singular values of the matricised component and
            a list of bools that indicates which singular values have been kept in the core move.

        Raises
        ------
        ValueError
            If the direction is neither "left" nor "right".
        RuntimeError
            If the rounding condition returns an invalid mask, i.e. if the mask is not a boolean array of the samea
            length as the singular value array or if it does not mark a single singular value to be kept.
        """
        assert self.is_valid()

        if direction == "left":
            if self.corePosition == 0:
                return np.linalg.norm(self.components[0]), np.full(1, True)
            core = self.components[self.corePosition]
            l, e, r = core.shape
            core = core.reshape(l, e * r)
            u, singularValues, vt = np.linalg.svd(core, full_matrices=False)
            # assert np.allclose((u * singularValues) @ vt, core)
            mask = rounding_condition(self.corePosition, singularValues)
            if (
                not isinstance(mask, np.ndarray)
                or mask.shape != (len(singularValues),)
                or mask.dtype != bool
                or not np.any(mask)
            ):
                raise RuntimeError("The rounding condition returned an invalid mask.")
            u, s, vt = u[:, mask], singularValues[mask], vt[mask]
            self.components[self.corePosition] = vt.reshape(len(s), e, r)
            self.__corePosition -= 1
            self.components[self.corePosition] = np.einsum("leR,Rr -> ler", self.components[self.corePosition], u * s)
        elif direction == "right":
            if self.corePosition == self.order - 1:
                return np.linalg.norm(self.components[self.order - 1]), np.full(1, True)
            core = self.components[self.corePosition]
            l, e, r = core.shape
            core = core.reshape(l * e, r)
            u, singularValues, vt = np.linalg.svd(core, full_matrices=False)
            # assert np.allclose(u @ (singularValues[:,None] * vt), core)
            mask = rounding_condition(self.corePosition + 1, singularValues)
            if (
                not isinstance(mask, np.ndarray)
                or mask.shape != (len(singularValues),)
                or mask.dtype != bool
                or not np.any(mask)
            ):
                raise RuntimeError("The rounding condition returned an invalid mask.")
            u, s, vt = u[:, mask], singularValues[mask], vt[mask]
            self.components[self.corePosition] = u.reshape(l, e, len(s))
            self.__corePosition += 1
            self.components[self.corePosition] = np.einsum(
                "lL,Ler -> ler", s[:, None] * vt, self.components[self.corePosition]
            )
        else:
            raise ValueError(f"Unknown direction. Expected 'left' or 'right' but got '{direction}'")
        return singularValues, mask

    def canonicalise(
        self, side: str = "left", rounding_condition: RoundingCondition = default_rounding_condition
    ) -> NonNegativeFloat:
        """Move the core to the left-most or right-most element.

        Parameters
        ----------
        side : str, optional
            The side to which the core is moved, by default "left"
        rounding_condition : RoundingCondition, optional
            The rounding condition to use during core moves, by default default_rounding_condition

        Returns
        -------
        NonNegativeFloat
            An upper bound on the rounding error.
        """
        self.__corePosition = {"left": self.order - 1, "right": 0}[side]
        limit = {"left": 0, "right": self.order - 1}[side]
        roundingError = 0
        while self.corePosition != limit:
            singularValues, mask = self.move_core(side, rounding_condition)
            roundingError += np.sum(singularValues[~mask] ** 2)
        assert self.is_valid() and self.is_canonicalised()
        return np.sqrt(roundingError)

    def round(self, rounding_condition: RoundingCondition = default_rounding_condition) -> NonNegativeFloat:
        """Round the tensor train according to the specified condition.

        After rounding, the tensor train will be right-canonicalised.
        """
        roundingError = self.canonicalise("left") ** 2
        roundingError += self.canonicalise("right", rounding_condition) ** 2
        return np.sqrt(roundingError)

    def singular_values(self) -> list[FloatArray]:
        self.canonicalise("left")
        allSingularValues = []
        while self.corePosition != self.order - 1:
            singularValues, mask = self.move_core("right")
            allSingularValues.append(singularValues[mask])
        norm = np.array([np.linalg.norm(allSingularValues[0])])
        return [norm] + allSingularValues + [norm]

    @classmethod
    def retraction(
        cls,
        tangentVector: FloatArray,
        tangentSpace: TangentSpace,
        rounding_condition: RoundingCondition = default_rounding_condition,
    ) -> tuple[Element, NonNegativeFloat]:
        """Retract a tangent vector back to the model class.

        Parameters
        ----------
        tangentVector : FloatArray
        tangentSpace : TangentSpace
        rounding_condition : RoundingCondition

        Returns
        -------
        element : Element
            The retracted element.
        errorBound : NonNegativeFloat
            An upper bound for the retraction error.
        """
        projections = tangentSpace.summandProjections
        slices = [0] + np.cumsum([projection.shape[1] for projection in projections]).tolist()

        def laplace_component(position: NonNegativeInt) -> FloatArray:
            baseComponent = tangentSpace.baseElement.components[position]
            l, d, r = baseComponent.shape
            core = projections[position] @ tangentVector[slices[position] : slices[position + 1]]  # noqa: E203
            core.shape = (l, d, r)
            # Indexing of rightComponents is shifted by 1, since the list is missing the first component tensor.
            if position == 0:
                component = np.empty((1, d, 2 * r))
                component[:, :, :r] = tangentSpace.left_component(position)
                component[:, :, r:] = core
            elif position < tangentSpace.baseElement.order - 1:
                component = np.empty((2 * l, d, 2 * r))
                component[:l, :, :r] = tangentSpace.left_component(position)
                component[:l, :, r:] = core
                component[l:, :, :r] = 0
                component[l:, :, r:] = tangentSpace.right_component(position)
            else:
                assert position == tangentSpace.baseElement.order - 1
                component = np.empty((2 * l, d, 1))
                component[:l, :, :] = core
                component[l:, :, :] = tangentSpace.right_component(position)
            return component

        components = [laplace_component(position) for position in range(tangentSpace.baseElement.order)]
        element = cls.from_components(components)
        element.univariateGramians = tangentSpace.baseElement.univariateGramians
        errorBound = element.round(rounding_condition)
        return element, errorBound

    @cached_property
    def tangentSpace(self):
        """Return the tangent space of the model class at this element."""
        return sys.modules[self.__module__].TangentSpace(self)


class TangentSpace(__TangentSpace):
    """Tangent space of the tensor train manifold.

    Attributes
    ----------
    summandBases : list[FloatArray]
        ...
    """

    def __init__(self, baseElement: Element) -> None:
        assert baseElement.is_valid()
        self.__leftBaseElement = baseElement.copy()
        self.__leftBaseElement.canonicalise("left")
        self.__rightBaseElement = self.__leftBaseElement.copy()
        self.__rightBaseElement.canonicalise("right")

        # TODO: Normalise the gramians before use?
        # univariateGramians = self.baseElement.univariateGramians
        # for position in range(len(univariateGramians)):
        #     univariateGramians[position] = univariateGramians[position] / np.linalg.norm(univariateGramians[position], 2)

        @cache
        def left_stack(position: NonNegativeInt) -> FloatArray:
            assert 0 <= position < self.baseElement.order
            if position == 0:
                return np.ones((1, 1))
            entry = np.einsum("lL,ldr -> Ldr", left_stack(position - 1), self.left_component(position - 1))
            entry = np.einsum("dD,Ldr -> LDr", self.baseElement.univariateGramians[position - 1], entry)
            return np.einsum("LDr,LDR -> rR", entry, self.left_component(position - 1))

        @cache
        def right_stack(position: NonNegativeInt) -> FloatArray:
            assert 0 <= position < self.baseElement.order
            if position == self.baseElement.order - 1:
                return np.ones((1, 1))
            entry = np.einsum("ldr,rR -> ldR", self.right_component(position + 1), right_stack(position + 1))
            entry = np.einsum("ldR,dD -> lDR", entry, self.baseElement.univariateGramians[position + 1])
            return np.einsum("lDR,LDR -> lL", entry, self.right_component(position + 1))

        @cache
        def summand_projection(position: NonNegativeInt) -> FloatArray:
            # Recall, that every tangent vector X can be represented as a sum of orthogonal tensor trains,
            # where all summands but the last have to satisfy an orthogonality condition.
            # Denote by U the (l*d,r)-matricisation of the `position`-th component of `self.baseElement` and
            # by M the (r*d,r)-matricisation of the `position`-th component of the `position`-th summand of X.
            # Then every column vector M[:,k] must be orthogonal to the space spanned by the orthonormal basis U, i.e.
            #     U.T @ M[:,k] = 0
            # for every k=1,..,r.
            # This means that M lives in the r-fold tensor product of the orthogonal complement of U.
            if position < self.baseElement.order - 1:
                l, d, r = self.baseElement.components[position].shape
                summandBasis = orthogonal_complement(matricisation(self.left_component(position), (0,1), (2,)))
                assert summandBasis.shape == (l*d, l*d - r)
                return block_diag(*((summandBasis,) * r))
            else:
                return np.eye(self.coreComponent.size)

        spectrum = []
        self.summandProjections = []
        for position in range(self.baseElement.order):
            gramian = np.einsum("lL,dD -> lLdD", left_stack(position), self.baseElement.univariateGramians[position])
            gramian = np.einsum("lLdD,rR -> ldrLDR", gramian, right_stack(position))
            gramian = gramian.reshape(np.prod(gramian.shape[:3]), -1)
            es, vs = np.linalg.eigh(summand_projection(position).T @ gramian @ summand_projection(position))
            self.summandProjections.append(summand_projection(position) @ vs)
            assert np.linalg.norm(self.summandProjections[-1].T @ gramian @ self.summandProjections[-1] - np.diag(es)) <= 1e-12 * np.linalg.norm(es)
            #  self.summandProjections.append(summand_projection(position))
            spectrum.extend(np.sqrt(np.maximum(es, np.finfo(es.dtype).tiny)))
        self.__spectrum = np.array(spectrum)

        assert self.is_valid()

    def left_component(self, position: NonNegativeInt) -> FloatArray:
        assert position < self.__rightBaseElement.order - 1
        return self.__rightBaseElement.components[position]

    def right_component(self, position: NonNegativeInt) -> FloatArray:
        assert position > 0
        return self.__leftBaseElement.components[position]

    @property
    def coreComponent(self) -> FloatArray:
        return self.__rightBaseElement.components[-1]

    @fold_all
    def is_valid(self) -> bool:
        yield self.__leftBaseElement.is_valid() and self.__rightBaseElement.is_valid()
        yield self.__leftBaseElement.is_canonicalised() and self.__rightBaseElement.is_canonicalised()
        yield self.__leftBaseElement.corePosition == 0
        yield self.__rightBaseElement.corePosition == self.__rightBaseElement.order - 1
        yield self.dimension == sum(projection.shape[1] for projection in self.summandProjections)

    @cached_property
    def baseElement(self) -> Element:
        return self.__rightBaseElement.copy()

    @property
    def spectrum(self):
        """Return the spectrum of the RKHS inner product in L2."""
        return self.__spectrum

    @property
    def dimension(self):
        """The dimension of the tangent space."""
        return len(self.spectrum)

    @property
    def base(self):
        """The base point of the tangent space, represented as a tangent vector."""
        core = self.coreComponent
        baseVector = np.zeros(self.dimension)
        baseVector[-core.size :] = self.summandProjections[-1].T @ core.reshape(-1)  # noqa: E203
        return baseVector

    def evaluation_operator(self, points: list[FloatArray]) -> FloatArray:
        """Assemble an operator that evaluates a tangent vector using the univaraite measures.

        Parameters
        ----------
        rankOneMeasures : list[FloatArray]
            The CP component tensors for the measurement operator on the full tensor space.
            univariateMeasure[k].shape == (numMeasures, baseElement.dimensions[k])
            rankOneMeasures contains measurements for all components but the first.
            These should be rank-one measures.

        Returns
        -------
        FloatArray
            ... shape == (numMeasures, codomainDimension, tangentSpaceDimension)
        """
        numMeasures = len(points)
        assert numMeasures > 0 and all(len(point) == self.baseElement.order - 1 for point in points)
        # rankOneMeasures[mode, sample, dimension] = points[sample, mode, dimension]
        rankOneMeasures = list(map(np.array, zip(*points)))
        assert len(rankOneMeasures) == self.baseElement.order - 1
        for measure, dimension in zip(rankOneMeasures, self.baseElement.dimensions[1:]):
            assert measure.dtype == float and measure.shape == (numMeasures, dimension)
        rankOneMeasures.insert(0, None)  # The first component is not measured.
        codomainDimension = self.baseElement.dimensions[0]

        @cache
        def left_stack(position: NonNegativeInt) -> FloatArray:
            assert 0 <= position < self.baseElement.order
            if position == 0:
                return np.ones((numMeasures, 1))
            elif position == 1:
                return np.einsum("nl, ldr -> ndr", left_stack(0), self.left_component(0))
            entry = np.einsum("ne, ler -> nlr", rankOneMeasures[position - 1], self.left_component(position - 1))
            return np.einsum("ndl, nlr -> ndr", left_stack(position - 1), entry)
                
        @cache
        def right_stack(position: NonNegativeInt) -> FloatArray:
            assert 0 <= position < self.baseElement.order
            if position == self.baseElement.order - 1:
                return np.ones((numMeasures, 1))
            entry = np.einsum("ne, ler -> nlr", rankOneMeasures[position + 1], self.right_component(position + 1))
            return np.einsum("nlr, nr -> nl", entry, right_stack(position + 1))

        def summand_operator(position: NonNegativeInt) -> FloatArray:
            if position == 0:
                identity = np.eye(codomainDimension)
                measure = np.einsum("nl, de, nr -> ndler", left_stack(0), identity, right_stack(0))
            else:
                measure = np.einsum("ndl, ne -> ndle", left_stack(position), rankOneMeasures[position])
                measure = np.einsum("ndle, nr -> ndler", measure, right_stack(position))
            assert measure.shape == (numMeasures, codomainDimension) + self.baseElement.components[position].shape
            return measure.reshape(numMeasures, codomainDimension, -1) @ self.summandProjections[position]

        operator = np.concatenate([summand_operator(position) for position in range(self.baseElement.order)], axis=2)
        assert operator.shape == (numMeasures, codomainDimension, self.dimension)
        return operator
