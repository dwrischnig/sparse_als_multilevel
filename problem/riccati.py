import numpy as np
from scipy.linalg import solve_continuous_are

from .problem import NonnegativeInt, PositiveInt, Problem


class RiccatiProblem(Problem):
    """Value function of the Riccati equation."""
    default_parameters = {
        "order": 8,
        "diffusivity": 1.0,
        "cost parameter": 0.1,
        "boundary type": "Neumann"
    }

    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)
        self.order = parameters["order"]
        self.dimension = 1
        self.diffusivity = parameters["diffusivity"]
        self.costParameter = parameters["cost parameter"]
        self.boundaryType = parameters["boundary type"]

    def compute_sample(self, salt: NonnegativeInt, size: PositiveInt, offset: NonnegativeInt) -> tuple[np.ndarray]:
        rng = np.random.default_rng(salt)
        points = rng.uniform(-1, 1, (size, self.order))[offset:]
        *_, Pi = self.__riccati_matrices(self.order, self.diffusivity, self.costParameter, self.boundaryType)
        values = np.einsum("ni,ij,nj -> n", points, Pi, points)
        assert points.shape == (size - offset, self.order) and values.shape == (size - offset,)
        return points, values.reshape(size - offset, 1)

    def __riccati_matrices(self, _n, _nu=1.0, _lambda=0.1, _boundary="Neumann"):
        """
        Builds the Riccati matrices for the optimization problem

            minimize integral(yQy + uRu, dt)
            subject to y' = Ay + Bu

        A is the 1-dimensional diffusion operator and Bu is a uniform forcing of size u on the interval [-0.4, 0.4].
        The solution Pi of the Riccati equation represents the value function as v(x) = x Pi x.

        Parameters
        ----------
        _n : int
            spatial discretization points that are considered
        _nu : float
            diffusion constant
        _lambda : float
            cost parameter
        _boundary : 'Dirichlet' or 'Neumann'
            the boundary condition to use

        Author: Leon Sallandt
        """
        assert _boundary in ["Dirichlet", "Neumann"]
        domain = (-1, 1)
        s = np.linspace(*domain, num=_n)  # gridpoints
        A = -2 * np.eye(_n) + np.eye(_n, k=1) + np.eye(_n, k=-1)
        Q = np.eye(_n)
        if _boundary == "Dirichlet":
            h = (domain[1] - domain[0]) / (_n + 1)
        elif _boundary == "Neumann":
            h = (domain[1] - domain[0]) / (_n - 1)  # step size in space
            A[[0, -1], [1, -2]] *= 2
            Q[[0, -1], [0, -1]] /= 2
        A *= _nu / h**2
        Q *= h
        B = ((-0.4 < s) & (s < 0.4)).astype(float).reshape(-1, 1)
        R = _lambda * np.eye(1)
        Pi = solve_continuous_are(A, B, Q, R)
        return A, B, Q, R, Pi
