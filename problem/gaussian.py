import numpy as np

from .problem import NonnegativeInt, PositiveInt, Problem, FloatArray


class GaussianProblem(Problem):
    """A Gaussian density."""

    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)
        self.order = parameters["order"]
        self.dimension = 1

    def compute_sample(self, salt: NonnegativeInt, size: PositiveInt, offset: NonnegativeInt) -> tuple[FloatArray]:
        def gaussian(points):
            # NOTE: The gaussian function gets peakier with increasing order!
            return np.exp(-np.linalg.norm(points, axis=1) ** 2)

        rng = np.random.default_rng(salt)
        points = rng.standard_normal((size, self.order))[offset:]
        values = gaussian(points)
        assert points.shape == (size - offset, self.order) and values.shape == (size - offset,)
        return points, values.reshape(size - offset, 1)
