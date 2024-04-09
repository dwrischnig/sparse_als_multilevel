import argparse

import numpy as np
from numpy.polynomial.legendre import legval
from numpy.linalg import norm
import time
from semisparse_als import SemiSparseALS
from loguru import logger
import json

parser = argparse.ArgumentParser(description='description')
parser.add_argument('--L', type=int, default=2, help='(maximum) level')
parser.add_argument('--nx', type=int, default=5, help='nx')
parser.add_argument('--maxiter', type=int, default=50, help='maximum iterations')
args = parser.parse_args()

def evaluate_tensortrain_legendre(components, points):
    order = len(components)
    basis_dimension = components[0].shape[1]
    evaluated_legendre = legval(points, np.diag(np.sqrt(2 * np.arange(basis_dimension) + 1))).T

    result = np.einsum("mi,hij->mj", evaluated_legendre[0], components[0])

    for mode in range(1, order - 1):
        result = np.einsum("mh,mi,hij->mj", result, evaluated_legendre[mode], components[mode])

    return np.einsum("mh,mi,hij->m", result, evaluated_legendre[-1], components[-1])

def run_sals(points, values):
    dataSize, truncation = points.shape
    assert points.ndim == 2 and dataSize % 10 == 0 and values.shape == (dataSize,)

    weights = np.ones(dataSize, dtype=float)

    weight_sequence = np.sqrt(2 * np.arange(basis_dimension) + 1)
    uniform_weight_factor = 1.05
    weight_sequence[1:] = [uniform_weight_factor * weight for weight in weight_sequence[1:]]
    weight_sequence = [weight_sequence] * truncation

    measures = legval(points, np.diag(np.sqrt(2 * np.arange(basis_dimension) + 1))).T  # shape: (order, sample_size, dimension) Phi?
    assert measures.shape == (truncation, dataSize, basis_dimension)

    sals = SemiSparseALS(measures, values, weights, weight_sequence, perform_checks=True)
    training_set = slice(0, dataSize, None)
    training_errors = [sals.residual(training_set)]
    times = [time.process_time()]

    for iteration in range(args.maxiter): 
        try:
            sals.step(training_set)
        except StopIteration:
            break

        times.append(time.process_time())
        training_errors.append(sals.residual(training_set))

        if iteration - np.argmin(training_errors) > 3:
            logger.warning(f"Terminating: training errors stagnated for over 3 iterations")
            break
        if times[-1] - times[-2] > 120:
            logger.warning(f"Terminating: previous iteration took longer than 120 seconds")
            break

    return {
        "components": sals.components,
        "training_error": training_errors[-1],
        "duration": time.process_time() - times[0]
        }


#parameter files
ml_parameter_file = f"sampling/ml_L{args.L}_nx{args.nx}.json"
with open(ml_parameter_file) as f:
    ml_parameters = json.load(f)

sl_parameter_file = f"sampling/sl_L{args.L}_nx{args.nx}.json"
with open(sl_parameter_file) as f:
    sl_parameters = json.load(f)

basis_dimension = int(max(ml_parameters["sparsity"] + [sl_parameters["sparsity"]]))

# multi level SALS    
ml_sals_solutions = []
ml_parameters["salsTrainingError"] = []
ml_parameters["salsDuration"] = []
ml_parameters["sample_sizes"] = []

for index in range(args.L):
    level = ml_parameters["level"][index]

    fe_samples = np.load(f"sampling/ml_L{args.L}_nx{args.nx}_level{level}.npz")
    points = fe_samples["points"]
    values = fe_samples["values"]

    result = run_sals(points, values)

    ml_sals_solutions.append(result["components"])
    ml_parameters["salsTrainingError"].append(result["training_error"])
    ml_parameters["salsDuration"].append(result["duration"])
    ml_parameters["sample_sizes"].append(values.shape[0])

ml_parameters["duration"] = sum(ml_parameters["salsDuration"]) + sum(ml_parameters["samplingDuration"])


# single level SALS
fe_samples = np.load(f"sampling/sl_L{args.L}_nx{args.nx}.npz")
points = fe_samples["points"]
values = fe_samples["values"]

result = run_sals(points, values)

sl_sals_solution = result["components"]
sl_parameters["salsTrainingError"] = result["training_error"]
sl_parameters["salsDuration"] = result["duration"]
sl_parameters["sample_size"] = values.shape[0]

sl_parameters["duration"] = sl_parameters["salsDuration"] + sl_parameters["samplingDuration"]



# test set
test_set = np.load(f"sampling/test_L5_nx80.npz")
test_points = test_set["points"]
test_values = test_set["values"]

ml_legendre_series_values = sum([evaluate_tensortrain_legendre(solution, test_points) for solution in ml_sals_solutions])
ml_parameters["residual"] = norm(test_values - ml_legendre_series_values) / norm(test_values)

sl_legendre_series_values = evaluate_tensortrain_legendre(sl_sals_solution, test_points)
sl_parameters["residual"] = norm(test_values - sl_legendre_series_values) / norm(test_values)

with open(ml_parameter_file, "w") as f:
    json.dump(ml_parameters, f)

with open(sl_parameter_file, "w") as f:
    json.dump(sl_parameters, f)