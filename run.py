# coding: utf-8
"""
Run optimisation according to the given parameters and store the results in a database.
"""
from __future__ import annotations

import os
import argparse
import time

import numpy as np
from numpy.polynomial.hermite_e import hermeval
from scipy.special import factorial
from numpy.polynomial.legendre import legval
from tqdm import tqdm, trange
from colored import fg, bg, attr

from gramian import HkinnerLegendre, Gramian
from sparse_als import FloatArray, SparseALS
import autoPDB  # noqa: F401


maxIterations = 100


def log(message: str):
    """Write date, time and message to stdout, respecting tqdm progress bars."""
    tqdm.write(time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime()) + message)


descr = """Run optimisation with the given parameters."""
parser = argparse.ArgumentParser(description=descr)
parser.add_argument("PROBLEM", type=str, help="The problem to solve.")
parser.add_argument("TRAINING_SET_SIZE", type=int, help="The training set size.")
parser.add_argument("TEST_SET_SIZE", type=int, help="The test set size.")
parser.add_argument("-x", "--trial", dest="trial_size", type=int, default=10, help="The trial size.")
parser.add_argument(
    "-d", "--dimension", dest="basis_dimension", type=int, default=10, help="The univariate basis dimension."
)
parser.add_argument("-k", "--kernel", type=str, default="H1mix", help="The RKHS to use.")
args = parser.parse_args()
args.problem = args.PROBLEM
args.trainingSetSize = args.TRAINING_SET_SIZE
args.testSetSize = args.TEST_SET_SIZE
args.trialSize = args.trial_size
args.basisDimension = args.basis_dimension
assert args.trainingSetSize > 0
assert args.testSetSize > 0
assert args.trialSize > 0
assert args.basisDimension > 0
dataSize = args.trialSize * args.trainingSetSize + args.testSetSize

os.makedirs(".cache", exist_ok=True)

if len(args.kernel) >= 5 and args.kernel[0] == "H" and args.kernel[-3:] == "mix":
    kernelOrder = int(args.kernel[1:-3])
    assert kernelOrder >= 0
else:
    raise NotImplementedError(f"Unknown kernel: {args.kernel}")

if args.problem == "runge":
    from problem.runge import RungeProblem as ProblemClass

    problem = ProblemClass({})
    args.basis = "Legendre"
elif args.problem == "gaussian":
    from problem.gaussian import GaussianProblem as ProblemClass

    problem = ProblemClass({"order": 2})
    args.basis = "Hermite"
elif args.problem == "riccati":
    from problem.riccati import RiccatiProblem as ProblemClass

    problem = ProblemClass(ProblemClass.default_parameters)
    args.basis = "Legendre"
elif args.problem == "darcy":
    from problem.darcy import DarcyProblem as ProblemClass

    problem = ProblemClass({"order": 10, "distribution": "uniform", "transformation": "integral", "jobs": 8})
    args.basis = "Legendre"
else:
    raise NotImplementedError(f"Unknown problem: {args.problem}")

if args.basis == "Hermite":
    # The normalized probabilist's Hermite polynomials.
    factorials = factorial(np.arange(args.basisDimension), exact=True)
    factors = np.sqrt((1 / factorials).astype(float))

    def basisval(points: FloatArray) -> FloatArray:
        assert points.ndim == 2 and points.size > 0
        return hermeval(points, np.diag(factors)).T

    weights = np.ones(args.basisDimension)
    for order in range(1, kernelOrder + 1):
        weights += np.maximum(np.arange(args.basisDimension) - order, 0) ** 2
elif args.basis == "Legendre":
    # The normalized Legendre polynomials.
    factors = np.sqrt(2 * np.arange(args.basisDimension) + 1)
    G = Gramian(args.basisDimension, HkinnerLegendre(kernelOrder))
    weights, basis = np.linalg.eigh(G)
    assert np.all(weights > 0)
    assert np.allclose(basis * weights @ basis.T, G)

    def basisval(points: FloatArray) -> FloatArray:
        assert points.ndim == 2 and points.size > 0
        assert np.max(abs(points)) < 1
        measures = legval(points, np.diag(factors))
        return np.einsum("de,enm -> mnd", basis.T, measures)

else:
    raise NotImplementedError(f"Unknown basis: {args.basis}")


points, values = problem.compute_sample(salt=0, size=dataSize, offset=0)
assert values.shape == (dataSize, 1)
values = values[:, 0]
assert points.ndim == 2 and points.shape[0] == dataSize
measures = basisval(points)
assert measures.shape == (points.shape[1], dataSize, args.basisDimension)
weights = [weights] * len(measures)
testSet = slice(None, args.testSetSize, None)


def print_parameters(sparseALS):
    Cs = []
    for position in range(sparseALS.order):
        lInfNorm = np.max(abs(measures[position]), axis=0)
        Cs.append(np.max(lInfNorm / weights[position]))
    fail = lambda x: f"{fg('misty_rose_3')}{x:.2e}{attr('reset')}"
    success = lambda x: f"{fg('dark_sea_green_2')}{x:.2e}{attr('reset')}"
    colorize = lambda x: success(x) if x <= 10 else fail(x)
    Cs = "[" + ", ".join(colorize(C) for C in Cs) + "]"

    parameters = {
        "dimensions": f"{sparseALS.dimensions}",
        "ranks": f"{sparseALS.ranks}",
        "sample size": f"{sparseALS.sampleSize}",
        "RKHS constants": Cs,
    }
    tab = " " * 2
    maxParameterLen = max(len(p) for p in parameters)
    print("-" * 125)
    for parameter, value in parameters.items():
        offset = " " * (maxParameterLen - len(parameter))
        print(f"{tab}{parameter} = {offset}{value}")
    print("-" * 125)


def print_state(iteration, sparseALS):
    itrStr = f"{iteration:{len(str(maxIterations))}d}"
    update_str = (
        lambda prev, new: f"{fg('dark_sea_green_2') if new <= prev+1e-8 else fg('misty_rose_3')}{new:.2e}{attr('reset')}"
    )
    trnStr = update_str(np.min(trainingErrors), trainingErrors[-1])
    valStr = update_str(np.min(testErrors), testErrors[-1])

    def disp_float(flt):
        r = f"{flt:.0f}"
        if r[-3:] == "inf":
            r = r[:-3] + "\u221e"
        return r

    with np.errstate(divide="ignore"):
        lambdas = "10^[" + ", ".join(disp_float(l) for l in np.rint(np.log10(sparseALS.lambdas))) + "]"
    densities = "[" + ", ".join(f"{int(100*d+0.5):2d}" for d in sparseALS.densities) + "]%"
    tqdm.write(f"[{itrStr}]  Residuals: trn={trnStr}, val={valStr}")
    tab = " " * (len(itrStr) + 4)
    tqdm.write(f"{tab}Lambdas: {lambdas}")
    tqdm.write(f"{tab}Densities: {densities}")
    tqdm.write(f"{tab}Ranks: {sparseALS.ranks}")


log("Initialising sparse ALS")
sparseALS = SparseALS(measures, values, weights)
print_parameters(sparseALS)
trial: int
for trial in trange(args.trialSize, desc="Trial"):
    start = args.testSetSize + trial * args.trainingSetSize
    trainingSet = slice(start, start + args.trainingSetSize, None)

    log("Optimising")
    dofs = [sparseALS.parameters]
    trainingErrors = [sparseALS.residual(trainingSet)]
    testErrors = [sparseALS.residual(testSet)]
    times = [time.process_time()]
    print_state(0, sparseALS)
    for iteration in range(1, maxIterations + 1):
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
        f".cache/{args.problem}_t{args.trainingSetSize}_s{args.testSetSize}_z{args.trialSize}-{trial}.npz",
        times=times,
        trainingErrors=trainingErrors,
        testErrors=testErrors,
        dofs=dofs,
    )
