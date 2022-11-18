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

    def __init__(self, parameters: dict) -> None:
        pass

    @abstractproperty
    def order(self) -> int:
        """Return the number of variables the model depends on."""
        pass

    @abstractproperty
    def dimension(self) -> int:
        """Return the dimension of the output of the model."""
        pass

    @abstractmethod
    def compute_sample(self, salt: NonnegativeInt, size: PositiveInt, offset: NonnegativeInt) -> tuple[FloatArray]:
        """Compute a sample of model evaluations."""
        pass
