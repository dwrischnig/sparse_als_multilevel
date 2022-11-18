"""Run optimisation according to the given parameters and store the results in a database."""
# coding: utf-8
from __future__ import annotations

import argparse
from pathlib import PurePath
import sys
import time

import numpy as np
from tqdm import tqdm, trange

from aux import git_diff, git_root, git_sha, log, load_class, latest_commit, short_sha
import database
from import_crawler import dependencies, module_path
import autoPDB  # noqa: F401


def log(message: str):
    """Write date, time and message to stdout, respecting tqdm progress bars."""
    tqdm.write(time.strftime("[%Y-%m-%d %H:%M:%S] ", time.localtime()) + message)


def draw_sample(problemClass, parameters, sampleSize, salt=0):
    return problemClass(**parameters).compute_sample(salt=salt, size=sampleSize)


if __name__ == "__main__":
    descr = """Run optimisation with the given parameters."""
    parser = argparse.ArgumentParser(description=descr)
    parser.add_argument("PARAMETERS", type=str, help="Path to the parameter file.")
    parser.add_argument("TRAINING_SET_SIZE", type=int, help="The training set size.")
    parser.add_argument("TRIALS", type=int, help="The trial size.")
    parser.add_argument("--dirty", dest="DIRTY", action="store_true", help="Allow a dirty working directory.")
    args = parser.parse_args()
    args.PARAMETERS = optimisation_parameters(args.PARAMETERS)
    args.PARAMETERS["training set size"] = args.TRAINING_SET_SIZE
    args.PARAMETERS["trial size"] = args.TRIALS

    dataParameters = args.PARAMETERS["data"]
    dataPath = dataParameters.pop("path")

    lossParameters = args.PARAMETERS["loss"]
    lossPath = lossParameters.pop("path")

    modelParameters = args.PARAMETERS["model"]
    modelPath = modelParameters.pop("path")

    updateParameters = args.PARAMETERS["update"]
    updatePath = updateParameters.pop("path")

    def get_module(cls):
        return ".".join(cls.split(".")[:-1])

    modules = [
        PurePath(__file__).stem,
        get_module(dataPath),
        get_module(lossPath),
        get_module(modelPath),
        get_module(updatePath),
    ]
    modules = {
        module_path(module)
        for module in dependencies(*modules, include={git_root()}, exclude=set(sys.path) - {git_root()})
    }

    clean = all(git_diff(module) == "" for module in modules)
    if not clean and not args.DIRTY:
        raise RuntimeError("Working directory is not clean.")

    # sha = git_sha()
    shas = {git_sha(module) for module in modules}
    shas = {sha for sha in shas if sha}
    sha = short_sha(latest_commit(shas))

    trainingSetSize = args.PARAMETERS["training set size"]
    testSetSize = args.PARAMETERS["test set size"]
    trialSize = args.PARAMETERS["trial size"]
    dataSize = trialSize * trainingSetSize + testSetSize
    points, values = draw_sample(dict(path=dataPath, parameters=dict(dataParameters), sha=sha, clean=clean), dataSize)
    testSet = slice(None, testSetSize, None)

    log(f"Loading loss: {lossPath}")
    LossClass = load_class(lossPath)

    log(f"Loading model: {modelPath}")
    ModelClass = load_class(modelPath)

    log(f"Loading update: {updatePath}")
    UpdateClass = load_class(updatePath)

    def experiment_attributes(**update):
        """Return a copy of experimentBlueprint with data, loss, model and update set appropriately."""
        data = database.get(database.Data, path=dataPath, parameters=dict(dataParameters), sha=sha, clean=clean)
        assert data is not None
        experiment = {
            "data": data,
            "loss": database.get_or_create(database.Loss, path=lossPath, parameters=dict(lossParameters)),
            "model": database.get_or_create(database.Model, path=modelPath, parameters=dict(modelParameters)),
            "update": database.get_or_create(database.Update, path=updatePath, parameters=dict(updateParameters)),
            "sha": sha,
            "clean": clean,
            "trainingSetSize": trainingSetSize,
            "testSetSize": testSetSize,
        }
        experiment.update(update)
        return experiment

    def error(model: ModelClass) -> float:
        """Compute the test set error of the given model."""
        return np.linalg.norm(model(points[testSet]) - values[testSet]) / np.linalg.norm(values[testSet])

    trial: int
    for trial in trange(trialSize, desc="Trial"):
        start = testSetSize + trial * trainingSetSize
        trainingSet = slice(start, start + trainingSetSize, None)

        with database.db_session:
            if database.Experiment.exists(**experiment_attributes(trial=trial)):
                log(f"Trial {trial} already exists.")
                continue

        log("Initialising")
        loss = LossClass(**lossParameters)
        model = ModelClass.mean_approximation(points[trainingSet], values[trainingSet], **modelParameters)
        update = UpdateClass(model, loss, points[trainingSet], values[trainingSet])
        update.verbose = True
        for key in updateParameters:
            if not hasattr(update, key):
                log(f"ERROR: '{UpdateClass.__name__}' does not have attribute '{key}'")
                exit(1)
            setattr(update, key, updateParameters[key])

        log("Optimising")
        dofs = [update.model.parameters]
        errors = [error(update.model)]
        times = [time.process_time()]
        log(f"Initial error: {errors[0]:.2e}")
        log(f"Initial parameters: {dofs[0]}")
        for iteration in range(update.maxIterations):
            log(f"Iteration: {iteration+1:d}")
            try:
                update.step()
            except StopIteration:
                break
            times.append(time.process_time())
            errors.append(error(update.model))
            dofs.append(update.model.parameters)
            log(f"Error: {errors[-1]:.2e}")

        log(f"Updating database")
        with database.db_session:
            database.Experiment(**experiment_attributes(trial=trial, dofs=dofs, errors=errors, times=times))
