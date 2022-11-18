"""The elements and tangent spaces of a model class of extended tensor trains."""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial

from .model import FloatArray
from .tensor_train import Element as __Element, TangentSpace as __TangentSpace, RoundingCondition, default_rounding_condition
from .aux import HkinnerLegendre, Gramian


class Element(__Element):
    def copy(self) -> Element:
        element  = super().copy()
        element.rankOneMeasures = self.rankOneMeasures
        return element

    @classmethod
    def mean_approximation(
        cls, points: FloatArray, values: FloatArray, basisName: str, kernelName: str
    ) -> None:
        """Initialize the model according to the given data."""
        if basisName.startswith("Hermite "):
            basisDimension = int(basisName[len("Hermite "):])
        else:
            assert basisName.startswith("Legendre ")
            basisDimension = int(basisName[len("Legendre "):])
        assert basisDimension > 0
        assert kernelName.startswith("Mixed Sobolev ")
        kernelOrder = int(kernelName[len("Mixed Sobolev "):])
        assert kernelOrder >= 0

        sampleSize, order = points.shape
        assert values.shape[0] == sampleSize
        sampleSize, targetDimension = values.shape
        mean = np.mean(values, axis=0)
        components = [mean.reshape(1, targetDimension, 1)]
        components += [np.eye(1, basisDimension, k=0).reshape(1, basisDimension, 1)] * order
        element = cls.from_components(components)

        if basisName.startswith("Hermite "):
            # Use normalized probabilist's Hermite polynomials.
            factorials = factorial(np.arange(basisDimension), exact=True)
            factors = np.sqrt((1 / factorials).astype(float))
            def measures(points: FloatArray) -> FloatArray:
                assert points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == element.order - 1
                return np.transpose(hermeval(points, np.diag(factors)), (1, 2, 0))
            element.rankOneMeasures = measures
            element.univariateGramians = [np.eye(targetDimension)]
            for dimension in element.dimensions[1 :]:
                diag = np.ones(dimension)
                for order in range(1, kernelOrder + 1):
                    diag += np.maximum(np.arange(dimension) - order, 0) ** 2
                element.univariateGramians.append(np.diag(diag))
        else:
            assert basisName.startswith("Legendre ")
            # Use normalized Legendre polynomials.
            factors = np.sqrt(2 * np.arange(basisDimension) + 1)
            def measures(points: FloatArray) -> FloatArray:
                assert points.ndim == 2 and points.shape[0] > 0 and points.shape[1] == element.order - 1
                assert np.max(abs(points)) < 1
                return np.transpose(legval(points, np.diag(factors)), (1, 2, 0))
            element.rankOneMeasures = measures
            element.univariateGramians = [np.eye(targetDimension)] + [Gramian(basisDimension, HkinnerLegendre(kernelOrder))] * order

        return element

    @classmethod
    def retraction(
        cls,
        tangentVector: FloatArray,
        tangentSpace: TangentSpace,
        rounding_condition: RoundingCondition = default_rounding_condition,
    ) -> tuple[Element, NonNegativeFloat]:
        element, errorBound = super().retraction(tangentVector, tangentSpace, rounding_condition)
        element.rankOneMeasures = tangentSpace.baseElement.rankOneMeasures
        return element, errorBound


class TangentSpace(__TangentSpace):
    def evaluation_operator(self, points: FloatArray) -> FloatArray:
        """Assemble an operator that evaluates a tangent vector at the given points.

        Parameters
        ----------
        points : ObjectArray
            ... shape == (numPoints,)
            Contains any object that the model can handle.
            extended_tensor_train models expect the point sin the domain and evaluate the function representeed by the
            coefficient tensor.

        Returns
        -------
        FloatArray
            ... shape == (numPoints, codomainDimension, tangentSpaceDimension)
        """
        sampleSize, order = points.shape
        basisDimension = self.baseElement.univariateGramians[1].shape[0]
        measures = self.baseElement.rankOneMeasures(points)
        assert measures.shape == (sampleSize, order, basisDimension)
        # maxes = np.max(abs(measures), axis=0)
        # import matplotlib.pyplot as plt
        # for mode in range(len(maxes)):
        #     plt.plot(maxes[mode])
        # plt.yscale('log')
        # plt.show()
        # from IPython import embed; embed()
        return super().evaluation_operator(measures)
