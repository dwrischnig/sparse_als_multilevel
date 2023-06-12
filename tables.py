"""Display parameter, error and running time metrics for selected experiments."""
# from __future__ import annotations
import re
from pathlib import Path
from dataclasses import dataclass, fields
from typing import Any, Callable, Optional

import numpy as np
from numpy.typing import NDArray
from rich.table import Table

import autoPDB  # noqa: F401


IntArray = NDArray[np.integer]
FloatArray = NDArray[np.floating]


@dataclass
class Experiment:
    problem: str
    algorithm: str
    training_set_size: int
    test_set_size: int
    trial_size: int
    trial: int
    dofs: IntArray
    test_set_errors: FloatArray
    training_set_errors: FloatArray
    times: FloatArray
    validation_set_errors: Optional[FloatArray] = None


ExperimentKey = Callable[[Experiment], Any]


def string_to_dict(string, pattern):
    regex = re.sub(r"{(.+?)}", r"(?P<_\1>.+)", pattern)
    values = list(re.search(regex, string).groups())
    keys = re.findall(r"{(.+?)}", pattern)
    return dict(zip(keys, values))


def load_experiments(data_path: str | Path, pattern: str) -> list[Experiment]:
    field_types = {field.name: field.type for field in fields(Experiment)}
    experiments = []
    data_path = Path(data_path)
    for path in data_path.glob("*.npz"):
        try:
            parameters = string_to_dict(path.name, pattern)
            for key in parameters:
                parameters[key] = field_types[key](parameters[key])
        except (AttributeError, ValueError):
            continue
        z = np.load(path)
        assert set(z.keys()) >= {"testErrors", "trainingErrors", "times", "dofs"}
        parameters["test_set_errors"] = z["testErrors"]
        if parameters["algorithm"] in ["sals", "ssals"]:
            assert "validationErrors" in z
            parameters["validation_set_errors"] = z["validationErrors"]
        parameters["training_set_errors"] = z["trainingErrors"]
        parameters["dofs"] = z["dofs"]
        parameters["times"] = z["times"]
        experiments.append(Experiment(**parameters))
    return experiments


def extract_data(
    experiments: list[Experiment],
    row_key: ExperimentKey,
    column_key: ExperimentKey,
    value_key: ExperimentKey,
):
    rows = sorted({row_key(e) for e in experiments})
    columns = sorted({column_key(e) for e in experiments})
    values = np.full((len(rows), len(columns), 2), np.nan)
    for row_idx, row in enumerate(rows):
        row_experiments = [e for e in experiments if row_key(e) == row]
        for column_idx, column in enumerate(columns):
            entry_values = [value_key(e) for e in row_experiments if column_key(e) == column]
            if len(entry_values) > 0:
                # values[row_idx, column_idx] = (np.mean(entry_values), np.std(entry_values))
                values[row_idx, column_idx] = np.quantile(entry_values, [0.05, 0.95])
    return rows, columns, values


def create_table(
    experiments: list[Experiment],
    row_key: ExperimentKey,
    column_key: ExperimentKey,
    value_key: ExperimentKey,
    title: str,
):
    rows, columns, values = extract_data(experiments, row_key, column_key, value_key)

    def row_strings(values, bold_mask):
        def value_string(value):
            assert value.shape == (2,) and np.isnan(value[0]) == np.isnan(value[1])
            if np.isnan(value[0]):
                return ""
            # return f"{value[0]:.2e} \u00B1 {value[1]:.0e}"
            return f"[{value[0]:.2e}, {value[1]:.2e}]"

        def bold(string):
            return f"[bold]{string}[/bold]"

        value_strings = [value_string(value) for value in values]
        res = np.where(bold_mask, np.vectorize(bold, otypes=[str])(value_strings), value_strings)
        return res

    min_values = np.nanmin(values[:, :, 0], axis=0)
    table = Table(title=title, title_style="bold", show_header=True, header_style="dim")
    table.add_column(style="dim")  # row_key
    for column in columns:
        table.add_column(str(column), justify="right")
    for label, row_values in zip(rows, values):
        table.add_row(label, *row_strings(row_values, row_values[:, 0] == min_values))
    return table


def get_optimal_index(experiment: Experiment, criterion: str) -> int:
    if getattr(experiment, criterion) is not None:
        assert len(experiment.validation_set_errors) == len(experiment.test_set_errors)
        return np.argmin(getattr(experiment, criterion))
    else:
        assert experiment.algorithm != "sals"
        return -1


if __name__ == "__main__":
    from operator import attrgetter

    from rich.console import Console

    data_path = Path(__file__).parent.absolute() / ".cache"
    pattern = "{problem}_{algorithm}_t{training_set_size}_s{test_set_size}_z{trial_size}-{trial}.npz"

    experiments = load_experiments(data_path, pattern)

    # problem = "darcy_lognormal_2"
    # problem = "darcy_lognormal_5"
    # problem = "darcy_lognormal_10"
    problem = "darcy_rauhut"
    experiments = [e for e in experiments if e.problem == problem]

    console = Console()

    # title = f"{problem} ({{0}} \u00B1 standard deviation)"
    title = f"{problem} ({{0}} — 5% and 95% quantiles)"
    console.print()
    console.print(
        create_table(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda e: e.test_set_errors[get_optimal_index(e, "validation_set_errors")],
            title=title.format("Errors"),
        )
    )
    console.print()
    console.print(
        create_table(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda experiment: experiment.times[-1] - experiment.times[0],
            title=title.format("Running times"),
        )
    )
    console.print()
    console.print(
        create_table(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda e: e.dofs[get_optimal_index(e, "validation_set_errors")],
            title=title.format("Parameters"),
        )
    )
    console.print()
