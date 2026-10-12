"""Mean curves with explicit SD or SEM bands for notebooks and exporters."""

import numpy as np
from scipy.stats import sem


def plot_mean_band(
    ax,
    curves,
    *,
    x=None,
    spread="sd",
    minimum=None,
    alpha=0.08,
    band_linewidth=None,
    **line_options,
):
    """Draw a seed mean and band; SD is population SD, SEM uses sample SD."""
    values = np.asarray(curves)
    mean = values.mean(axis=0)
    radius = values.std(axis=0, ddof=1 if spread == "sem" else 0)
    if spread == "sem":
        radius /= np.sqrt(values.shape[0])
    if x is None:
        x = np.arange(values.shape[1])
    lower = mean - radius
    if minimum is not None:
        lower = np.maximum(lower, minimum)
    (line,) = ax.plot(x, mean, **line_options)
    options = {} if band_linewidth is None else {"linewidth": band_linewidth}
    ax.fill_between(x, lower, mean + radius, color=line.get_color(), alpha=alpha, **options)
    return line


def errorbar_mean_sem(ax, x, samples, **style):
    """One mean and sample SEM for each condition's seed measurements."""
    return ax.errorbar(
        x, [sample.mean() for sample in samples], yerr=[sem(sample) for sample in samples], **style
    )
