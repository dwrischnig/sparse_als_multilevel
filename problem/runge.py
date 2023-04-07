"""Provides an abstract base class for problems for which data sets can be generated."""
from typing import cast

import numpy as np

from .problem import PositiveInt, FloatArray, Problem


class RungeProblem(Problem):
    """Represents a problems for which a data set can be generated."""

    @property
    def order(self) -> PositiveInt:
        """Return the number of variables the model depends on."""
        return cast(PositiveInt, 1)

    @property
    def dimension(self) -> PositiveInt:
        """Return the dimension of the output of the model."""
        return cast(PositiveInt, 1)

    def compute_sample(self, salt: int, size: int, offset: int) -> tuple[FloatArray, FloatArray]:
        """Compute a sample of model evaluations."""
        assert salt >= 0 and size > 0 and offset >= 0

        def runge(points):
            assert points.shape[1] == 1
            return 1 / (1 + 25 * points**2)

        rng = np.random.default_rng(salt)
        points = rng.uniform(-1, 1, (size, self.order))[offset:]
        values = runge(points)
        assert points.shape == (size - offset, self.order) and values.shape == (size - offset, self.order)
        return points, values
