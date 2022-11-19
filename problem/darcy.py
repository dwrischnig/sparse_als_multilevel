import numpy as np
from joblib import Parallel, delayed

from .problem import NonnegativeInt, PositiveInt, Problem
from .parametric_pde_sampling.problem.darcy import Problem as _DarcyProblem
from .parametric_pde_sampling.compute_orthogonalization import get_mass_matrix, get_stiffness_matrix, cholesky


class DarcyProblem(Problem):
    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)
        self.__order = parameters["order"]
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
            self.__dimension = self.FEDimension
        elif self.transformation == "integral":
            self.__dimension = 1
        else:
            raise ValueError(f"Unknown transformation '{self.transformation}'")

    @property
    def order(self):
        return self.__order

    @property
    def dimension(self):
        return self.__dimension

    def compute_sample(self, salt: NonnegativeInt, size: PositiveInt, offset: NonnegativeInt) -> tuple[np.ndarray]:
        rng = np.random.default_rng(salt)
        assert self.distribution in ["lognormal", "uniform"]
        if self.distribution == "lognormal":
            points = rng.standard_normal((size, self.order))[offset:]
        elif self.distribution == "uniform":
            points = rng.uniform(-1, 1, (size, self.order))[offset:]

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
