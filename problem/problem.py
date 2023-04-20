"""Provides an abstract base class for problems for which data sets can be generated."""
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import NewType

import numpy as np
from numpy.typing import NDArray

PositiveInt = NewType("PositiveInt", int)
NonnegativeInt = NewType("NonnegativeInt", int)
FloatArray = NDArray[np.float_]


class Problem(metaclass=ABCMeta):
    """Represents a problems for which a data set can be generated."""

    def __init__(self, parameters: dict):
        ...

    @abstractproperty
    def order(self) -> PositiveInt:
        """The number of variables the model depends on."""
        ...

    @abstractproperty
    def dimension(self) -> PositiveInt:
        """The dimension of the models output."""
        ...

    @abstractmethod
    def compute_sample(self, salt: int, size: int, offset: int) -> tuple[FloatArray, FloatArray]:
        """Compute a sample of model evaluations."""
        ...
