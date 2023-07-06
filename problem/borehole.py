from typing import cast

import numpy as np

from problem import PositiveInt, FloatArray, Problem


class BoreholeProblem(Problem):
    """
    The borehole function.

    Reference: https://uqworld.org/t/borehole-function/60
    """

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        self.__order = cast(PositiveInt, 8)
        self.__dimension = cast(PositiveInt, 1)
        self.parameterIntervals = np.array(
            [
                (0.05, 0.15),  # Radius of borehole [m]
                (100, 50_000),  # Radius of influence [m]
                (63_070, 115_600),  # Transmissivity of upper aquifer [m2/year]
                (990, 1_100),  # Potentiometric head of upper aquifer [m]
                (63.1, 116),  # Transmissivity of lower aquifer [m2/year]
                (700, 820),  # Potentiometric head of lower aquifer [m]
                (1_120, 1_680),  # Length of borehole [m]
                (9_885, 12_045),  # Hydraulic conductivity of borehole [m/year]
            ]
        ).T

    @property
    def order(self) -> PositiveInt:
        return self.__order

    @property
    def dimension(self) -> PositiveInt:
        return self.__dimension

    def compute_sample(self, salt: int, size: int, offset: int) -> tuple[FloatArray, FloatArray]:
        assert salt >= 0 and size > 0 and offset >= 0

        def transform(points: FloatArray) -> FloatArray:
            assert points.ndim == 2
            assert np.all(-1 <= points) and np.all(points <= 1)
            points = (points + 1) / 2
            intervalLength = self.parameterIntervals[1] - self.parameterIntervals[0]
            points = points * intervalLength + self.parameterIntervals[0]
            assert np.all(self.parameterIntervals[0][None] <= points) and np.all(
                points <= self.parameterIntervals[1][None]
            )
            return points

        def borehole(points: FloatArray) -> FloatArray:
            assert points.ndim == 2
            rw, r, Tu, Hu, Tl, Hl, L, Kw = transform(points).T
            nominator = 2 * np.pi * Tu * (Hu - Hl)
            log = np.log(r / rw)
            denominator = log * (1 + (2 * L * Tu) / (log * rw**2 * Kw) + Tu / Tl)
            return nominator / denominator

        rng = np.random.default_rng(salt)
        points = rng.uniform(-1, 1, (size, self.order))[offset:]
        values = borehole(points)
        assert points.shape == (size - offset, self.order) and values.shape == (size - offset,)
        return points, values.reshape(size - offset, 1)


if __name__ == "__main__":
    import argparse, json
    from pathlib import Path

    parameters = {
        "problem": {"name": "borehole"},
        "sampling": {"distribution": "uniform"},
        "learning": {"distribution": "uniform"},
    }

    descr = """Sample solutions for the given problem."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("SAMPLES", type=int, help="the number of samples to compute")
    args = parser.parse_args()

    problem = BoreholeProblem({})
    points, values = problem.compute_sample(0, args.SAMPLES, 0)
    values = values[:, 0]

    path = Path(__file__).parent / "borehole"
    path.mkdir(exist_ok=True)
    with open(path / "parameters.json", "w") as f:
        json.dump(parameters, f, indent=4)
    data_path = path / "data"
    data_path.mkdir(exist_ok=True)
    np.savez_compressed(data_path / "functional_identity.npz", samples=points, values=values)
