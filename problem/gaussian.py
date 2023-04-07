from typing import cast

import numpy as np

from .problem import PositiveInt, FloatArray, Problem


class GaussianProblem(Problem):
    """A Gaussian density."""

    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)
        assert parameters["order"] > 0
        self.__order = cast(PositiveInt, parameters["order"])

    @property
    def dimension(self) -> PositiveInt:
        return cast(PositiveInt, 1)

    @property
    def order(self) -> PositiveInt:
        return self.__order

    def compute_sample(self, salt: int, size: int, offset: int) -> tuple[FloatArray, FloatArray]:
        assert salt >= 0 and size > 0 and offset >= 0

        def gaussian(points):
            # NOTE: The gaussian function gets peakier with increasing order!
            return np.exp(-np.linalg.norm(points, axis=1) ** 2)

        rng = np.random.default_rng(salt)
        points = rng.standard_normal((size, self.order))[offset:]
        values = gaussian(points)
        assert points.shape == (size - offset, self.order) and values.shape == (size - offset,)
        return points, values.reshape(size - offset, 1)
