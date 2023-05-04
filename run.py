# coding: utf-8
"""
Run optimisation according to the given parameters and store the results in a database.
"""
from __future__ import annotations

import os
import argparse
import time

from loguru import logger
import numpy as np
from numpy.polynomial.legendre import legval
from colored import fg, attr

from sparse_als import FloatArray
import autoPDB  # noqa: F401


descr = """Run optimisation with the given parameters."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("PROBLEM", type=str, help="The problem to solve.")
parser.add_argument("TRAINING_SET_SIZE", type=int, help="The training set size.")
parser.add_argument("TEST_SET_SIZE", type=int, help="The test set size.")
parser.add_argument("-x", "--trial", dest="trial_size", type=int, default=10, help="The trial size.")
parser.add_argument(
    "-d", "--dimension", dest="basis_dimension", type=int, default=20, help="The univariate basis dimension."
)
parser.add_argument("-r", "--regularity", dest="regularity", type=int, default=0, help="The regularity to assume.")
parser.add_argument(
    "-a", "--algorithm", dest="algorithm", type=str, default="sals", help="The approximation algorithm to use."
)
args = parser.parse_args()
args.problem = args.PROBLEM
args.trainingSetSize = args.TRAINING_SET_SIZE
args.testSetSize = args.TEST_SET_SIZE
args.trialSize = args.trial_size
args.basisDimension = args.basis_dimension
args.maxIterations = 100
assert args.trainingSetSize > 0
assert args.testSetSize > 0
assert args.trialSize > 0
assert args.basisDimension > 0
dataSize = args.trialSize * args.trainingSetSize + args.testSetSize


logger.info("Importing optimiser")
if args.algorithm == "sals":
    from sparse_als import SparseALS as ALS
if args.algorithm == "ssals":
    from semisparse_als import SemiSparseALS as ALS
elif args.algorithm == "tensap":
    from tensap_optimiser import TensapOptimiser
else:
    raise NotImplementedError(f"Unknown algorithm: {args.algorithm}")


os.makedirs(".cache", exist_ok=True)


# TODO: Assume the domains of the problems are either [-1, 1] or [-np.inf, np.inf].
#       Otherwise it is cumbersome to define orthonormal bases.
#       Just have a property returning the list of distributions (These imply the domains implicitly.)
#       The sampling of the points should be done here and the model should only contain a __call__ function.


if args.problem == "runge":
    from problem.runge import RungeProblem as ProblemClass

    problem = ProblemClass({})
    points, values = problem.compute_sample(salt=0, size=dataSize, offset=0)
    values = values[:, 0]
    weights = np.ones(points.shape[0])
    args.basis = "Legendre"
elif args.problem == "darcy":
    logger.info("Loading data")
    z = np.load("problem/darcy_lognormal_10.npz")
    points = z["points"]
    values = z["values"]

    def orthogonalise(values):
        assert values.ndim == 2
        gramian = values.T @ values / values.shape[0]
        es, vs = np.linalg.eigh(gramian)
        assert np.allclose(vs * es @ vs.T, gramian)
        assert np.all(es[:-1] <= es[1:])
        result = values @ vs
        transformed_gramian = result.T @ result / result.shape[0]
        assert np.allclose(transformed_gramian - np.diag(es), 0)
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

    logger.info("Rejecting sample points")
    bound = 3
    mask = np.all(points >= -bound, axis=1) & np.all(points <= bound, axis=1)
    points = points[mask][:dataSize] / bound
    assert -1 <= np.min(points) and np.max(points) <= 1
    values = values[mask][:dataSize]
    weights = np.product(np.exp(-(points**2) / 2) / np.sqrt(2 * np.pi), axis=1)
    args.basis = "Legendre"
else:
    raise NotImplementedError(f"Unknown problem: {args.problem}")

assert points.ndim == 2 and points.shape[0] == dataSize
assert values.shape == (dataSize,)
assert weights.shape == (dataSize,)

if args.basis == "Legendre":
    # The normalized Legendre polynomials.
    factors = np.sqrt(2 * np.arange(args.basisDimension) + 1)
    # G = Gramian(args.basisDimension, HkinnerLegendre(args.regularity))
    # G = np.diag(factors) @ G @ np.diag(factors) / 2
    # weight_sequence, basis = np.linalg.eigh(G)
    # assert np.all(weight_sequence > 0)
    # assert np.allclose(basis * weight_sequence @ basis.T, G)
    # if args.regularity == 0:
    #     assert np.allclose(weight_sequence, 1)
    # assert np.all(weight_sequence > 0)
    # assert np.allclose(basis * weight_sequence @ basis.T, G)
    # For args.regularity == 0, can not guarantee that the first basis funciton will be constant.
    # Moreover, we can not guarantee that the weight_sequence constitute an upper bound for the infty norm.
    # We therefore use the sequence of (the powers of) the L^\infty-norms.
    weight_sequence = [factors ** (args.regularity + 1)] * points.shape[1]

    def evaluate_basis(points: FloatArray) -> FloatArray:
        assert points.ndim == 2 and points.size > 0
        assert np.max(abs(points)) < 1
        measures = legval(points, np.diag(factors))
        # return np.einsum("de,enm -> mnd", basis.T, measures)
        return measures.T

    # NOTE: The Hermite basis is not implemented, since the weight sequence depends on the choice of weight function.
    #       One possible choice would be to draw the sample according to N(0, 2) and use the PDF of the standard
    #       Gaussian distribution as a weight function. Although the basis functions are not uniformly bounded with
    #       respect to the weighted L^\infty-norm, their maximal points (and theirby their maximal values) can be easily
    #       computed. ((H_n ρ)' = (H_n' - x)ρ = 0 <-> H_n' - x = 0)
    #       This gives a weight sequence satisfying ω_n ≥ ||w B_n||_L∞.
    #       But the sought function has to be sparse with respect to this weight sequence!
    #       Since ω_n ∈ O(exp(n)), a sufficient condition is for the sought function to be analytic and bounded on a
    #       ball of sufficiently large radius.
    # elif args.basis == "Hermite":
    #     # As a basis use the normalized probabilist's Hermite polynomials.
    #     factorials = factorial(np.arange(args.basisDimension), exact=True)
    #     assert isinstance(factorials, np.ndarray) and factorials.dtype == np.int_
    #     factors = np.sqrt((1 / factorials).astype(float))

    #     def evaluate_basis(points: FloatArray) -> FloatArray:
    #         assert points.ndim == 2 and points.size > 0
    #         return hermeval(points, np.diag(factors)).T

else:
    raise NotImplementedError(f"Unknown basis: {args.basis}")

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
    ω_sharpness = sparseALS.weight_sequence_sharpness()
    ω_sharpness = "[" + ", ".join(colorise_success(c, c <= 1) for c in ω_sharpness) + "]"

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
    logger.info(f"[{itrStr}]  Residuals: trn={trnStr}, val={valStr}")
    tab = " " * (len(itrStr) + 4)
    logger.info(f"{tab}Regularisation parameters: {regularisationParameters}")
    logger.info(f"{tab}Component densities: {componentDensities}")
    logger.info(f"{tab}Ranks: {sparseALS.ranks}")


if args.algorithm in ["sals", "ssals"]:
    sparseALS = ALS(measures, values, weights, weight_sequence)
    print_parameters(sparseALS)
    trial: int
    for trial in range(args.trialSize):
        logger.info("=" * 125)
        logger.info(f"Trial: {trial+1:>{len(str(args.trialSize))}d} / {args.trialSize}")
        start = args.testSetSize + trial * args.trainingSetSize
        trainingSet = slice(start, start + args.trainingSetSize, None)

        dofs = [sparseALS.parameters]
        trainingErrors = [sparseALS.residual(trainingSet)]
        testErrors = [sparseALS.residual(testSet)]
        times = [time.process_time()]
        print_state(0, sparseALS)
        logger.info("Optimising")
        for iteration in range(1, args.maxIterations + 1):
            try:
                sparseALS.step(trainingSet)
                print_state(iteration, sparseALS)
            except StopIteration:
                break
            times.append(time.process_time())
            trainingErrors.append(sparseALS.residual(trainingSet))
            testErrors.append(sparseALS.residual(testSet))
            dofs.append(sparseALS.parameters)
            if iteration - np.argmin(trainingErrors) - 1 > 3:
                break

        np.savez_compressed(
            f".cache/{args.problem}_{args.algorithm}_t{args.trainingSetSize}_s{args.testSetSize}_z{args.trialSize}-{trial}.npz",
            times=times,
            trainingErrors=trainingErrors,
            testErrors=testErrors,
            dofs=dofs,
        )
    logger.info("=" * 125)

elif args.algorithm == "tensap":
    logger.info("Initialising tensap optimiser")
    tensapOptimiser = TensapOptimiser(points, values, args.basis, args.basis_dimension)

    trial: int
    for trial in range(args.trialSize):
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
            tensapOptimiser.reset()
            logger.info(f"Final training set error: {trainingErrors[-1]:.2e}")
            logger.info(f"Final test set error:     {testErrors[-1]:.2e}")
            logger.info(f"Final dofs:               {dofs[-1]:>8d}")
            np.savez_compressed(
                f".cache/{args.problem}_{args.algorithm}_t{args.trainingSetSize}_s{args.testSetSize}_z{args.trialSize}-{trial}.npz",
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
