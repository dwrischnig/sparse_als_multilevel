import numpy as np
from numpy.typing import NDArray
import matplotlib
import matplotlib.pyplot as plt


FloatArray = NDArray[np.floating]


def heatmap(data, row_labels, col_labels, ax=None, fig=None, fontdict={}, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        left=True,
        top=True,
        right=False,
        bottom=False,
        labelleft=True,
        labeltop=True,
        labelright=False,
        labelbottom=False,
        pad=10,
        length=0,
    )

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(
        col_labels,
        rotation=0,
        ha="center",
        rotation_mode="anchor",
        backgroundcolor=fig.get_facecolor(),
        **fontdict,
    )
    ax.set_yticklabels(row_labels, ha="right", va="center", backgroundcolor=fig.get_facecolor(), **fontdict)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=fig.get_facecolor(), linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}", fontdict=lambda x: {}, threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    TODO: Now we use a fontdict function, that takes the value and generates the fontdict!
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(**fontdict(im.norm(data[i, j])))
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def scientific_latex(x, places=2):
    assert places > 0
    if np.isnan(x):
        return ""
    s = "-" * int(x < 0)  # sign
    x = abs(x)
    m = f"{x:.{places}e}"  # mantissa
    assert m[2 + places : 4 + places] in ["e+", "e-"]
    e = int(m[3 + places :])  # exponent
    m = m[: 2 + places]
    return f"{s}{m}\cdot 10^{{{e}}}"


@matplotlib.ticker.FuncFormatter
def scientific_latex_formatter(x, pos):
    return f"${scientific_latex(x)}$"


def scientific_latex_range(x, places=2):
    assert len(x) == 2
    return f"[{scientific_latex(x[0], places=places)}, {scientific_latex(x[1], places=places)}]"


@matplotlib.ticker.FuncFormatter
def scientific_latex_range_formatter(x, pos):
    return f"${scientific_latex_range(x)}$"


def create_table(title, rows, columns, values, fig=None, ax=None):
    if ax is None:
        ax = plt.gca()

    if fig is None:
        fig = plt.gcf()

    fig.set_facecolor((0, 0, 0, 0))
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_facecolor((0, 0, 0, 0))

    ax.set_title(
        # rf"Lognormal Darcy ($\frac{{\sigma}}{{\mu}} = {scientific_latex(6.60e-06/3.52e-02)}$)",
        title,
        backgroundcolor=fig.get_facecolor(),
        fontdict={
            "fontsize": 24,
            "color": "white",
            "fontweight": "bold",
        },
        zorder=2,
        # y=1.15,
        pad=20,
    )

    # heatmap_values = values[..., 0]
    heatmap_values = np.exp(np.log(values).mean(axis=-1))
    im = heatmap(
        heatmap_values,
        rows,
        columns,
        ax=ax,
        cmap=traffic_lights,
        norm=matplotlib.colors.LogNorm(vmin=heatmap_values.min(), vmax=heatmap_values.max()),
        aspect=1 / 4,
        fontdict={
            "fontsize": 24,
            "color": "white",
        },
    )

    texts = annotate_heatmap(
        im,
        values,
        valfmt=scientific_latex_range_formatter,
        fontdict=lambda x: {
            "fontsize": 16,
            "color": "black",
            "fontweight": "bold",
        },
        threshold=5e-3,
    )  # 2.5 == 0+(5-0)/2

    fig.tight_layout()


if __name__ == "__main__":
    from operator import attrgetter
    from pathlib import Path

    from tables import load_experiments, extract_data, get_optimal_index
    from traffic_lights_cm import traffic_lights

    def rename(rows: list[str], columns: list[str], values: FloatArray):
        assert values.shape == (len(rows), len(columns), 2)
        ssals_index = rows.index("ssals")
        assert ssals_index >= 0
        rows = rows[:]
        del rows[ssals_index]
        values = np.delete(values, ssals_index, axis=0)
        sals_index = rows.index("sals")
        assert sals_index >= 0
        rows[sals_index] = "SALS (ours)"
        columns = columns[:]
        for idx in range(len(columns)):
            columns[idx] = f"$n = {columns[idx]}$"
        return rows, columns, values

    data_path = Path(__file__).parent.absolute() / ".cache"
    figure_path = Path(__file__).parent.absolute() / "figures"
    figure_path.mkdir(mode=0o777, parents=False, exist_ok=True)
    pattern = "{problem}_{algorithm}_t{training_set_size}_s{test_set_size}_z{trial_size}-{trial}.npz"

    # problem = "darcy_lognormal_2"
    # problem = "darcy_lognormal_5"
    # problem = "darcy_lognormal_10"
    problem = "darcy_rauhut"

    # rename = lambda *args: args

    experiments = load_experiments(data_path, pattern)
    experiments = [e for e in experiments if e.problem == problem]

    rows, columns, values = rename(
        *extract_data(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda experiment: experiment.test_set_errors[get_optimal_index(experiment, "validation_set_errors")],
        )
    )

    fig, ax = plt.subplots(figsize=(15, 4), dpi=300)
    create_table("Uniform Darcy (relative error — 5% and 95% quantiles)", rows, columns, values, fig, ax)
    plt.savefig(
        figure_path / f"table_{problem}-error.png", dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )

    rows, columns, values = rename(
        *extract_data(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda experiment: experiment.times[-1] - experiment.times[0],
        )
    )

    fig, ax = plt.subplots(figsize=(15, 4), dpi=300)
    create_table("Uniform Darcy (running times — 5% and 95% quantiles)", rows, columns, values, fig, ax)
    plt.savefig(
        figure_path / f"table_{problem}-time.png", dpi=300, edgecolor="none", bbox_inches="tight", transparent=True
    )

    rows, columns, values = rename(
        *extract_data(
            experiments,
            attrgetter("algorithm"),
            attrgetter("training_set_size"),
            lambda experiment: experiment.dofs[get_optimal_index(experiment, "validation_set_errors")],
        )
    )

    fig, ax = plt.subplots(figsize=(15, 4), dpi=300)
    create_table("Uniform Darcy (parameters — 5% and 95% quantiles)", rows, columns, values, fig, ax)
    plt.savefig(
        figure_path / f"table_{problem}-parameters.png",
        dpi=300,
        edgecolor="none",
        bbox_inches="tight",
        transparent=True,
    )
