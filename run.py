# coding: utf-8
"""
Run optimisation according to the given parameters and store the results in a database.
"""
from __future__ import annotations

import os
import argparse
import time
import json

from loguru import logger
import numpy as np
from numpy.polynomial.hermite_e import hermeval
from numpy.polynomial.legendre import legval
from scipy.special import factorial
from scipy.stats import norm
from colored import fg, attr

from sparse_als import FloatArray
import autoPDB  # noqa: F401


descr = """Run optimisation with the given parameters."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("PROBLEM", type=str, help="The problem to solve.")
parser.add_argument("TRAINING_SET_SIZE", type=int, help="The training set size.")
parser.add_argument("TEST_SET_SIZE", type=int, help="The test set size.")
parser.add_argument(
    "-q",
    "--qoi",
    dest="quantity_of_interest",
    type=str,
    choices=["integral", "pod_mode", "identity"],
    help="The QoI to learn.",
)
parser.add_argument("-x", "--trial", dest="trial_size", type=int, default=10, help="The trial size.")
parser.add_argument(
    "-d", "--dimension", dest="basis_dimension", type=int, default=20, help="The univariate basis dimension."
)
parser.add_argument(
    "-a", "--algorithm", dest="algorithm", type=str, default="sals", help="The approximation algorithm to use."
)
args = parser.parse_args()
args.problem = args.PROBLEM
args.trainingSetSize = args.TRAINING_SET_SIZE
args.testSetSize = args.TEST_SET_SIZE
args.trialSize = args.trial_size
args.basisDimension = args.basis_dimension
args.maxIterations = 50
assert args.trainingSetSize > 0
assert args.testSetSize > 0
assert args.trialSize > 0
assert args.basisDimension > 0
dataSize = args.trialSize * args.trainingSetSize + args.testSetSize


logger.info("Importing optimiser")
if args.algorithm == "sals":
    from sparse_als import SparseALS as ALS
elif args.algorithm == "ssals":
    from semisparse_als import SemiSparseALS as ALS
elif args.algorithm == "tensap":
    from tensap_optimiser import TensapOptimiser
else:
    raise NotImplementedError(f"Unknown algorithm: {args.algorithm}")


def load_parameters(problemDir):
    problemFile = f"{problemDir}/parameters.json"
    try:
        with open(problemFile, "r") as f:
            problemInfo = json.load(f)
    except FileNotFoundError:
        raise IOError(f"Can not read file '{problemFile}'")
    except json.JSONDecodeError:
        raise IOError(f"'{problemFile}' is not a valid JSON file")
    return problemInfo


logger.info("Loading problem")
problem_dir = args.PROBLEM
if not os.path.isdir(problem_dir):
    raise IOError(f"'{problem_dir}' is not a directory")
problem_dir_contents = os.listdir(problem_dir)
if not "parameters.json" in problem_dir_contents:
    raise IOError(f"'{problem_dir}' does not contain a 'parameters.json'")

problem_info = load_parameters(problem_dir)
assert problem_info["sampling"]["distribution"] == problem_info["learning"]["distribution"]

if problem_info["learning"]["distribution"] == "uniform":
    basis_density_1d = lambda x: np.ones(x.shape[0]) / 2
    weight_function = lambda x: np.ones(x.shape[0], dtype=float)

    # The normalized Legendre polynomials.
    factors = np.sqrt(2 * np.arange(args.basisDimension) + 1)

    def evaluate_basis(points: FloatArray) -> FloatArray:
        assert points.ndim == 2 and points.size > 0
        assert np.max(abs(points)) <= 1
        return legval(points, np.diag(factors)).T  # shape: (order, sample_size, dimension)

    args.basis = "Legendre"
    domain = (-1, 1)
elif problem_info["learning"]["distribution"] == "normal":
    sampling_variance = problem_info["sampling"]["variance"]
    assert problem_info["learning"]["variance"] == 1
    sample_density_1d = lambda x: norm.pdf(x, scale=np.sqrt(sampling_variance))
    basis_density_1d = lambda x: norm.pdf(x, scale=1)
    weight_function = lambda x: np.product(basis_density_1d(x) / sample_density_1d(x), axis=1)

    # The normalized probabilist's Hermite polynomials.
    factorials = factorial(np.arange(args.basisDimension), exact=True)
    assert isinstance(factorials, np.ndarray) and factorials.dtype == np.int_
    factors = np.sqrt((1 / factorials).astype(float))

    def evaluate_basis(points: FloatArray) -> FloatArray:
        assert points.ndim == 2 and points.size > 0
        return hermeval(points, np.diag(factors)).T  # shape: (order, sample_size, dimension)

    args.basis = "Hermite"
    domain = (-args.basis_dimension, args.basis_dimension)
else:
    raise NotImplementedError(f"Unknown distribution: {problem_info['learning']['distribution']}")


logger.info("Loading data")
data_dir = f"{problem_dir}/data"
if not os.path.isdir(data_dir):
    raise IOError(f"'{data_dir}' is not a directory")
if args.quantity_of_interest == "integral":
    functional_file = "functional_integral.npz"
elif args.quantity_of_interest in ["pod_mode", "identity"]:
    functional_file = "functional_identity.npz"
else:
    raise NotImplementedError(f"Unknown quantity of interest: {args.quantity_of_interest}")

z = np.load(f"{data_dir}/{functional_file}")
points = z["samples"]
values = z["values"]

if args.quantity_of_interest == "pod_mode":
    assert values.ndim == 2

    def orthogonalise(values):
        assert values.ndim == 2
        gramian = values.T @ values / values.shape[0]
        es, vs = np.linalg.eigh(gramian)
        # assert np.allclose(vs * es @ vs.T, gramian)
        assert np.all(es[:-1] <= es[1:])
        result = values @ vs
        # transformed_gramian = result.T @ result / result.shape[0]
        # assert np.allclose(transformed_gramian - np.diag(es), 0)
        return result

    logger.info("Orthogonalising data")
    # NOTE: The values are FE coefficients for solutions of the Darcy equation with the given parameters.
    #       Since the FE mesh is uniformly refined, the L2-norm is equivalent to an h-weighted l2-norm with a small
    #       equivalence constant of 6. We now (empirically) orthogonalise the FE basis.
    #       This allows us to learn the different physical basis coefficients independently of each other.
    values = orthogonalise(values)

    # We start by learning the coefficient for the basis function that contributes the most to the L2-norm.
    coefficient = -1
    values = values[:, coefficient]


assert points.ndim == 2 and points.shape[0] >= dataSize and values.shape == (points.shape[0],)
assert domain[0] <= np.min(points) and np.max(points) <= domain[1]
points = points[:dataSize]
values = values[:dataSize]
weights = weight_function(points)
assert weights.shape == (dataSize,)


reference_points = np.linspace(*domain, num=10_000)
reference_density = basis_density_1d(reference_points)
reference_measures = evaluate_basis(reference_points[:, None])
assert reference_measures.shape == (1, 10_000, args.basisDimension)
reference_measures = reference_measures[0]
reference_gramian = (reference_measures.T * reference_density)[..., None] * reference_measures[None]
assert reference_gramian.shape == (args.basisDimension, 10_000, args.basisDimension)
reference_gramian = np.trapz(reference_gramian, reference_points, axis=1)
assert np.allclose(reference_gramian, np.eye(args.basisDimension), atol=1e-4)
reference_measures = np.sqrt(weight_function(reference_points[:, None]))[:, None] * reference_measures
assert reference_measures.shape == (10_000, args.basisDimension)
reference_variation = abs(reference_measures)
if args.basis == "Hermite":
    assert np.all(0 < np.argmax(reference_variation, axis=0)) and np.all(np.argmax(reference_variation, axis=0) < 9_999)
# weight_sequence = np.ones(args.basisDimension)  # constant weight sequence
# weight_sequence = np.arange(1, args.basisDimension + 1)  # weight sequence of polynomial degrees (+1)
weight_sequence = reference_variation.max(axis=0)  # optimal weights
weight_sequence = [weight_sequence] * points.shape[1]


measures = evaluate_basis(points)
assert measures.shape == (points.shape[1], dataSize, args.basisDimension)
testSet = slice(None, args.testSetSize, None)


def colorise_success(value, condition):
    """Colorise the value green or red according to the provided condition."""
    if condition:
        return f"{fg('dark_sea_green_2')}{value:.2e}{attr('reset')}"
    else:
        return f"{fg('misty_rose_3')}{value:.2e}{attr('reset')}"


def print_parameters(sparseALS):
    """Print the parameters of the ALS scheme."""
    ω_sharpness = sparseALS.weight_sequence_sharpness
    ω_sharpness = colorise_success(ω_sharpness, ω_sharpness >= 0.5)

    parameters = {
        "dimensions": f"{sparseALS.dimensions}",
        "ranks": f"{sparseALS.ranks}",
        "sample size": f"{sparseALS.sampleSize}",
        "ω-sharpness": ω_sharpness,
    }
    tab = " " * 2
    maxParameterLen = max(len(p) for p in parameters)
    logger.info(f"Algorithm: {sparseALS.__class__.__name__}")
    logger.info("-" * 125)
    for parameter, value in parameters.items():
        offset = " " * (maxParameterLen - len(parameter))
        logger.info(f"{tab}{parameter} = {offset}{value}")
    logger.info("-" * 125)


def print_state(iteration, sparseALS):
    """Print the current state of the ALS scheme."""
    itrStr = f"{iteration:{len(str(args.maxIterations))}d}"
    trnStr = colorise_success(trainingErrors[-1], trainingErrors[-1] <= np.min(trainingErrors) + 1e-8)
    valStr = colorise_success(testErrors[-1], testErrors[-1] <= np.min(testErrors) + 1e-8)

    def display_float(flt):
        r = f"{flt:.0f}"
        if r[-3:] == "inf":
            r = r[:-3] + "\u221e"
        return r

    with np.errstate(divide="ignore"):
        regularisationParameters = np.rint(np.log10(sparseALS.regularisationParameters))
    regularisationParameters = "10^[" + ", ".join(display_float(param) for param in regularisationParameters) + "]"
    componentDensities = "[" + ", ".join(f"{int(100*d+0.5):2d}" for d in sparseALS.componentDensities) + "]%"
    logger.info(f"[{itrStr}]  Residuals: trn={trnStr}, tst={valStr}")
    tab = " " * (len(itrStr) + 4)
    logger.info(f"{tab}Regularisation parameters: {regularisationParameters}")
    logger.info(f"{tab}Component densities: {componentDensities}")
    logger.info(f"{tab}Ranks: {sparseALS.ranks}")


run_dir = f"{problem_dir}/{args.quantity_of_interest}_d{args.basisDimension}_s{args.testSetSize}/{args.algorithm}_t{args.trainingSetSize}"
os.makedirs(run_dir, exist_ok=True)
if args.algorithm in ["sals", "ssals"]:
    trial: int
    for trial in range(args.trialSize):
        logger.info("=" * 125)
        fileName = f"{run_dir}/{trial}.npz"
        if os.path.exists(fileName):
            logger.info(f"Cache file exists: {fileName}")
            z = np.load(fileName)
            if len(z["times"]) < args.maxIterations + 1:
                logger.warning(f"Cache file contains incomplete data: {len(z['times'])} / {args.maxIterations + 1}")
            continue
        logger.info(f"Computing '{fileName}'")

        sparseALS = ALS(measures, values, weights, weight_sequence, perform_checks=True)
        print_parameters(sparseALS)
        logger.info(f"Trial: {trial+1:>{len(str(args.trialSize))}d} / {args.trialSize}")
        start = args.testSetSize + trial * args.trainingSetSize
        trainingSet = slice(start, start + args.trainingSetSize, None)

        dofs = [sparseALS.parameters]
        trainingErrors = [sparseALS.residual(trainingSet)]
        validationErrors = [np.inf]
        testErrors = [sparseALS.residual(testSet)]
        times = [time.process_time()]
        print_state(0, sparseALS)
        logger.info("Optimising")
        for iteration in range(1, args.maxIterations + 1):
            try:
                validationError = sparseALS.step(trainingSet)
                print_state(iteration, sparseALS)
            except StopIteration:
                break
            times.append(time.process_time())
            trainingErrors.append(sparseALS.residual(trainingSet))
            validationErrors.append(validationError)
            testErrors.append(sparseALS.residual(testSet))
            dofs.append(sparseALS.parameters)
            np.savez_compressed(
                fileName,
                times=times,
                trainingErrors=trainingErrors,
                validationErrors=validationErrors,
                testErrors=testErrors,
                dofs=dofs,
            )
            if iteration - np.argmin(trainingErrors) - 1 > 3:
                logger.warning(f"Terminating: training errors stagnated for over 3 iterations")
                break
            if times[-1] - times[-2] > 60:
                logger.warning(f"Terminating: previous iteration took longer than 60 seconds")
                break
    logger.info("=" * 125)

elif args.algorithm == "tensap":
    trial: int
    for trial in range(args.trialSize):
        logger.info("=" * 125)
        fileName = f"{run_dir}/{trial}.npz"
        if os.path.exists(fileName):
            logger.info(f"Cache file exists: {fileName}")
            continue
        logger.info(f"Computing '{fileName}'")

        logger.info("Algorithm: tensap")
        logger.info("-" * 125)
        logger.info(f"  sample size = {dataSize}")
        logger.info("-" * 125)
        tensapOptimiser = TensapOptimiser(points, values, weights, args.basis, args.basis_dimension)
        logger.info("=" * 125)
        logger.info(f"Trial: {trial+1:>{len(str(args.trialSize))}d} / {args.trialSize}")
        start = args.testSetSize + trial * args.trainingSetSize
        trainingSet = slice(start, start + args.trainingSetSize, None)

        try:
            trainingErrors = [tensapOptimiser.residual(trainingSet)]
            testErrors = [tensapOptimiser.residual(testSet)]
            dofs = [tensapOptimiser.parameters]
            times = [time.process_time()]
            logger.info("Optimising")
            tensapOptimiser.optimise(trainingSet)
            times.append(time.process_time())
            trainingErrors.append(tensapOptimiser.residual(trainingSet))
            testErrors.append(tensapOptimiser.residual(testSet))
            dofs.append(tensapOptimiser.parameters)
            logger.info(f"Final training set error: {trainingErrors[-1]:.2e}")
            logger.info(f"Final test set error:     {testErrors[-1]:.2e}")
            logger.info(f"Final dofs:               {dofs[-1]:>8d}")
            np.savez_compressed(
                fileName,
                times=times,
                trainingErrors=trainingErrors,
                testErrors=testErrors,
                dofs=dofs,
            )
        except Exception:
            logger.info("Tensap crashed")
    logger.info("=" * 125)

else:
    raise NotImplementedError(f"Unknown algorithm: {args.algorithm}")
