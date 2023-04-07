from typing import cast

import numpy as np
from joblib import Parallel, delayed

from .problem import PositiveInt, FloatArray, Problem
from .parametric_pde_sampling.problem.darcy import Problem as _DarcyProblem
from .parametric_pde_sampling.compute_orthogonalization import get_mass_matrix, get_stiffness_matrix, cholesky


class DarcyProblem(Problem):
    def __init__(self, parameters: dict):
        super().__init__(parameters)
        assert parameters["order"] > 0
        self.__order = cast(PositiveInt, parameters["order"])
        self.distribution = parameters["distribution"]
        self.transformation = parameters["transformation"]
        self.jobs = parameters["jobs"]

        if self.distribution not in ["lognormal", "uniform"]:
            raise ValueError(f"Unknown distribution '{self.distribution}'")

        diffusivity = {}
        if self.distribution == "lognormal":
            diffusivity["mean"] = parameters.get("diffusivity mean", 0.0)
            diffusivity["scale"] = parameters.get("diffusivity scale", 0.6079271018540267)
            diffusivity["decay rate"] = parameters.get("diffusivity decay rate", 2)
        else:
            diffusivity["mean"] = parameters.get("diffusivity mean", 1.0)
            diffusivity["scale"] = parameters.get("diffusivity scale", 0.6079271018540267)
            diffusivity["decay rate"] = parameters.get("diffusivity decay rate", 2)

        self.FEProblem = _DarcyProblem(
            {
                "problem": {"name": "darcy"},
                "fe": {
                    "degree": 1,
                    "mesh": parameters.get("mesh size", 8),
                },
                "expansion": {
                    "size": self.order,
                    "mean": diffusivity["mean"],
                    "scale": diffusivity["scale"],
                    "decay rate": diffusivity["decay rate"],
                },
                "sampling": {"distribution": {"lognormal": "normal", "uniform": "uniform"}[self.distribution]},
            }
        )

        self.FEDimension = self.FEProblem.space.dim()
        if self.transformation in ["none", "orthogonalisation"]:
            assert self.FEDimension > 0
            self.__dimension = cast(PositiveInt, self.FEDimension)
        elif self.transformation == "integral":
            self.__dimension = cast(PositiveInt, 1)
        else:
            raise ValueError(f"Unknown transformation '{self.transformation}'")

    @property
    def order(self) -> PositiveInt:
        return self.__order

    @property
    def dimension(self) -> PositiveInt:
        return self.__dimension

    def compute_sample(self, salt: int, size: int, offset: int) -> tuple[FloatArray, FloatArray]:
        assert salt >= 0 and size > 0 and offset >= 0
        rng = np.random.default_rng(salt)
        if self.distribution == "lognormal":
            points = rng.standard_normal((size, self.order))[offset:]
        elif self.distribution == "uniform":
            points = rng.uniform(-1, 1, (size, self.order))[offset:]
        else:
            raise ValueError("'compute_sample' is only implemented for uniform and lognormal distributions")

        values = np.array(Parallel(n_jobs=self.jobs)(map(delayed(self.FEProblem.solution), points)), dtype=np.float64)
        assert values.shape == (size - offset, self.FEDimension)

        assert self.transformation in ["none", "integral", "orthogonalisation"]
        if self.transformation == "none":
            pass
        elif self.transformation == "integral":
            massMatrix = get_mass_matrix(self.FEProblem.space)
            integralOperator = np.ones((1, self.FEDimension)) @ massMatrix
            values = values @ integralOperator.T
        elif self.transformation == "orthogonalisation":
            S = get_stiffness_matrix(self.FEProblem.space)
            P, L = cholesky(S)
            values = L.dot(values.T).T

        assert points.shape == (size - offset, self.order) and values.shape == (size - offset, self.dimension)
        return points, values
