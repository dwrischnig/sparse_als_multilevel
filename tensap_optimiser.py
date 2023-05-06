import numpy as np
import tensap as ts
from loguru import logger

from sparse_als import FloatArray


class TensapOptimiser(object):
    def __init__(self, points: FloatArray, values: FloatArray, weights: FloatArray, basis: str, basis_dimension: int):
        assert points.ndim == 2 and values.shape == (points.shape[0],) and weights.shape == (points.shape[0],)
        self.points = points
        self.values = values
        self.weights = weights
        if basis == "Legendre":
            univariateBasis = ts.PolynomialFunctionalBasis(ts.LegendrePolynomials(), range(basis_dimension))
        elif basis == "Hermite":
            univariateBasis = ts.PolynomialFunctionalBasis(ts.HermitePolynomials(), range(basis_dimension))
        else:
            raise NotImplementedError(f"Unknown basis: {basis}")
        self.basis = ts.FunctionalBases([univariateBasis] * self.modes)
        self.result = None

    @property
    def modes(self):
        return self.points.shape[1]

    @property
    def parameters(self):
        if self.result is None:
            return 0
        return self.result.tensor.sparse_storage()

    def residual(self, set) -> float:
        if self.result is None:
            return 1
        assert self.result.bases is self.basis
        prediction = self.result.eval(self.points[set])
        assert prediction.shape == (self.points[set].shape[0],)
        return np.linalg.norm(np.sqrt(self.weights[set]) * (prediction - self.values[set])) / np.linalg.norm(
            np.sqrt(self.weights[set]) * self.values[set]
        )

    def optimise(self, set):
        solver = ts.TreeBasedTensorLearning.tensor_train(self.modes, ts.SquareLossFunction())

        solver.bases = self.basis
        solver.bases_eval = self.basis.eval(self.points[set])
        assert len(solver.bases_eval) == self.modes
        for mode in range(self.modes):
            assert solver.bases_eval[mode].shape[0] == self.points[set].shape[0]
            solver.bases_eval[mode] *= (self.weights[set] ** (0.5 / self.modes))[:, None]
        solver.training_data = [None, np.sqrt(self.weights[set]) * self.values[set]]

        solver.tolerance["on_stagnation"] = 1e-6
        solver.tolerance["on_error"] = 1e-6

        solver.initialization_type = "canonical"

        solver.linear_model_learning.regularization = False
        solver.linear_model_learning.basis_adaptation = True
        solver.linear_model_learning.error_estimation = True

        solver.test_error = False

        solver.rank_adaptation = True
        solver.rank_adaptation_options["max_iterations"] = 20
        solver.rank_adaptation_options["theta"] = 0.8
        solver.rank_adaptation_options["early_stopping"] = True
        solver.rank_adaptation_options["early_stopping_factor"] = 10

        solver.tree_adaptation = False
        # solver.tree_adaptation_options['max_iterations'] = 1e2
        # solver.tree_adaptation_options['force_rank_adaptation'] = True

        solver.alternating_minimization_parameters["stagnation"] = 1e-10
        solver.alternating_minimization_parameters["max_iterations"] = 50

        solver.display = True
        solver.alternating_minimization_parameters["display"] = False

        solver.model_selection = True
        solver.model_selection_options["type"] = "cv_error"

        fnc, out = solver.solve()
        self.result = fnc
