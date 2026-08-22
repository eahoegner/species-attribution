# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Species-level contributions by time period (burden-based, official-consistent input)
#
# This reads `002_run_magicc_burden_based.py`'s output. Output files are prefixed `consistent_`.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

import attribution_common as ac

# %% [markdown]
# ## Configuration

# %%
EMBARGOED = True
"""Set True once running against real (embargoed) ScenarioMIP scenarios. Must match
002's own EMBARGOED setting."""
DATA_DIR = Path("../data/embargoed") if EMBARGOED else Path("../data")

BASE_SCENARIOS = ac.load_base_scenarios(DATA_DIR)
REGION = ac.REGION

BURDEN_SCM_OUTPUT_DB_DIR = DATA_DIR / "consistent_burden_scm_output_db"
BURDEN_GSAT_DB_DIR = DATA_DIR / "consistent_burden_gsat_db"
"""02's output dirs."""

PLOTS_DIR = Path("../data/plots")
"""Always plain data/plots/, regardless of EMBARGOED - plots are not considered
sensitive, only raw emissions/MAGICC output."""
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "consistent_"
"""Prepended to every plot filename below, to distinguish these official-input-convention
plots from 114's own (102-sourced) outputs."""

FORCING_CATEGORIES = {
    "CO2": "Effective Radiative Forcing|CO2",
    "CH4": "Effective Radiative Forcing|CH4",
    "N2O": "Effective Radiative Forcing|N2O",
    "F-Gases": "Effective Radiative Forcing|F-Gases",
    "Montreal Protocol Halogen Gases": "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Tropospheric Ozone": "Effective Radiative Forcing|Tropospheric Ozone",
    "Stratospheric Ozone": "Effective Radiative Forcing|Stratospheric Ozone",
    "Stratospheric H2O": "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O",
    "Aerosol-Radiation Interactions": "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Aerosol-Cloud Interactions": "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Black Carbon on Snow": "Effective Radiative Forcing|Black Carbon on Snow",
    "Land Use": "Effective Radiative Forcing|Land-use Change",
    "Contrails and Aviation-Induced Cirrus": "Effective Radiative Forcing|Aviation|Contrail and Cirrus",
    "Solar": "Effective Radiative Forcing|Solar",
    "Volcanic": "Effective Radiative Forcing|Volcanic",
}
"""Must match 002's FORCING_CATEGORIES - kept in sync manually."""

STACK_BUCKETS = {
    "CO2": ["CO2"],
    "CH4": ["CH4"],
    "N2O": ["N2O"],
    "Montreal Protocol Halogen Gases": ["Montreal Protocol Halogen Gases"],
    "F-Gases": ["F-Gases"],
    "Tropospheric Ozone": ["Tropospheric Ozone"],
    "Stratospheric H2O": ["Stratospheric H2O"],
    "Aerosols Direct Effect": ["Aerosol-Radiation Interactions"],
    "Aerosols Indirect Effect": ["Aerosol-Cloud Interactions"],
}

TIME_BLOCKS = [
    ("Historical\n(to 2025)", None, 2025),
    ("2025-2050", 2025, 2050),
    ("2050-2075", 2050, 2075),
    ("2075-2100", 2075, 2100),
    ("Total\n(to 2100)", None, 2100),
]
"""(label, start_year, end_year) - start_year=None means "absolute value at end_year"
(cumulative since 1750), otherwise "value at end_year minus value at start_year"."""

SCENARIO_COLORS = ac.SCENARIO_COLORS
"""Fixed per-scenario colors, for visual consistency between notebooks (and outside
this repo) - same scenario always means the same color."""

# %% [markdown]
# ## Load shared data

# %%
burden_scm_output = OpenSCMDB(
    backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=BURDEN_SCM_OUTPUT_DB_DIR
).load(out_columns_type=int)
burden_scm_output.columns.name = "year"

burden_gsat_by_channel = OpenSCMDB(
    backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=BURDEN_GSAT_DB_DIR
).load(out_columns_type=int)
burden_gsat_by_channel.columns.name = "year"
burden_gsat_by_channel = burden_gsat_by_channel.loc[
    (burden_gsat_by_channel.index.get_level_values("variable") == "Surface Air Temperature Change")
    & (burden_gsat_by_channel.index.get_level_values("region") == REGION)
]


def category_erf_matrix(base_scenario, category):
    return ac.load_erf(burden_scm_output, base_scenario, FORCING_CATEGORIES[category], region=REGION)


def _raw_category_gsat_matrix(base_scenario, category):
    scenario = f"{base_scenario}_forcing_only_{category}"
    mask = burden_gsat_by_channel.index.get_level_values("scenario") == scenario
    sub = burden_gsat_by_channel.loc[mask].reset_index().set_index("run_id")
    year_columns = [c for c in sub.columns if isinstance(c, int)]
    return sub[year_columns].sort_index()


def category_gsat_matrix(base_scenario, category):
    """Rebased to the IPCC-standard 1850-1900 reference period. Shares
    `base_scenario`'s real total run's own assessment-matching shift
    (`ac.gsat_assessment_shift`) rather than independently recalibrating this one
    category to the full AR6-assessed 0.85K on its own."""
    shift = ac.gsat_assessment_shift(_raw_total_gsat_matrix(base_scenario), n_channels=len(FORCING_CATEGORIES))
    return ac.rebase_gsat_channel_matrix(_raw_category_gsat_matrix(base_scenario, category), shift)


def bucket_matrix(base_scenario, bucket, category_matrix_fn):
    members = STACK_BUCKETS[bucket]
    return sum(category_matrix_fn(base_scenario, c) for c in members)


def total_erf_matrix(base_scenario):
    return ac.load_erf(burden_scm_output, base_scenario, "Effective Radiative Forcing", region=REGION)


def _raw_total_gsat_matrix(base_scenario):
    mask = (
        (burden_scm_output.index.get_level_values("scenario") == base_scenario)
        & (burden_scm_output.index.get_level_values("variable") == "Surface Air Temperature Change")
        & (burden_scm_output.index.get_level_values("region") == REGION)
    )
    sub = burden_scm_output.loc[mask].reset_index().set_index("run_id")
    year_columns = [c for c in sub.columns if isinstance(c, int)]
    return sub[year_columns].sort_index()


def total_gsat_matrix(base_scenario):
    """Rebased to the IPCC-standard 1850-1900 reference period."""
    return ac.rebase_gsat_matrix(_raw_total_gsat_matrix(base_scenario))


# %% [markdown]
# ## Plotting helper

# %%
def period_delta(matrix, start_year, end_year):
    """Member-level-first: per-member delta (or absolute value if start_year is None),
    *then* median/17-83rd pct - never the other way round."""
    return matrix[end_year] if start_year is None else (matrix[end_year] - matrix[start_year])


def plot_species_by_period(base_scenario, bar_color, category_matrix_fn, total_matrix_fn, title, ylabel, out_path):
    panels = ["Total", *STACK_BUCKETS]
    ncols = 2
    nrows = -(-len(panels) // ncols)  # ceil
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 3.0 * nrows), constrained_layout=True
    )
    axes = axes.flatten()

    x = range(len(TIME_BLOCKS))
    tick_labels = [label for label, _, _ in TIME_BLOCKS]

    for ax, panel in zip(axes, panels):
        matrix = total_matrix_fn(base_scenario) if panel == "Total" else bucket_matrix(
            base_scenario, panel, category_matrix_fn
        )

        medians, lower_err, upper_err = [], [], []
        for _, start_year, end_year in TIME_BLOCKS:
            delta = period_delta(matrix, start_year, end_year)
            median = delta.median()
            medians.append(median)
            lower_err.append(median - delta.quantile(0.17))
            upper_err.append(delta.quantile(0.83) - median)

        ax.bar(
            x, medians, yerr=[lower_err, upper_err], color=bar_color, edgecolor="black", linewidth=0.6,
            width=0.6, capsize=3, error_kw={"elinewidth": 1.2, "capthick": 1.2},
        )
        ax.axhline(0, color=ac.COLOR_WARMING, linewidth=1.2, linestyle="--", zorder=4)
        ax.set_title(panel, fontsize=10, color=ac.COLOR_PRIMARY_TEXT)
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.tick_params(axis="y", labelsize=8, colors=ac.COLOR_MUTED)
        ax.yaxis.grid(True, color=ac.COLOR_GRIDLINE, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_color("black")

    for ax in axes[len(panels):]:
        ax.axis("off")

    fig.supylabel(ylabel, fontsize=11, color=ac.COLOR_PRIMARY_TEXT)
    fig.suptitle(title, fontsize=14, color=ac.COLOR_PRIMARY_TEXT)
    fig.savefig(out_path, dpi=200, facecolor=ac.COLOR_SURFACE, bbox_inches="tight")
    return fig, axes


# %% [markdown]
# ## ERF by category and time period, per scenario

# %%
for base_scenario in BASE_SCENARIOS:
    short_name = ac.scenario_short_name(base_scenario)
    plot_species_by_period(
        base_scenario,
        SCENARIO_COLORS[short_name],
        category_erf_matrix,
        total_erf_matrix,
        title=f"ERF by category and period, {ac.scenario_display_label(base_scenario)} (median, 17-83rd pct)",
        ylabel="ERF (W/m$^2$)",
        out_path=PLOTS_DIR / f"{OUTPUT_PREFIX}species_by_period_erf_{short_name}.png",
    )
    plt.close("all")

# %% [markdown]
# ## GSAT contribution by category and time period, per scenario

# %%
for base_scenario in BASE_SCENARIOS:
    short_name = ac.scenario_short_name(base_scenario)
    plot_species_by_period(
        base_scenario,
        SCENARIO_COLORS[short_name],
        category_gsat_matrix,
        total_gsat_matrix,
        title=f"GSAT contribution by category and period, {ac.scenario_display_label(base_scenario)}, rebased to 1850-1900 (median, 17-83rd pct)",
        ylabel=r"$\Delta$ GSAT (°C)",
        out_path=PLOTS_DIR / f"{OUTPUT_PREFIX}species_by_period_gsat_{short_name}.png",
    )
    plt.close("all")

# %% [markdown]
# # Export underlying data to CSV
#
# Full quantile distributions (not just the median + 17-83rd pct shown in the plots
# above) for every (scenario, panel, period) combination already plotted - "Total" is
# the real single-run total (not a derived sum, same quantity `total_matrix_fn` plots),
# the other panels are the 9 `STACK_BUCKETS`. One combined CSV per metric (ERF/GSAT),
# covering all 7 scenarios, rather than one file per scenario/PNG.

# %%
import pandas as pd

EXPORT_QUANTILES = [0.05, 0.10, 1 / 6, 0.33, 0.50, 0.67, 5 / 6, 0.90, 0.95]
"""Same 9-quantile convention used elsewhere in this project (e.g. the official
assessed-warming/ERF quantile CSVs, `repro_compare_erf_quantiles.py`)."""


def _export_period_quantiles(category_matrix_fn, total_matrix_fn, out_path):
    rows = []
    panels = ["Total", *STACK_BUCKETS]
    for base_scenario in BASE_SCENARIOS:
        short_name = ac.scenario_short_name(base_scenario)
        for panel in panels:
            matrix = (
                total_matrix_fn(base_scenario) if panel == "Total" else bucket_matrix(base_scenario, panel, category_matrix_fn)
            )
            for label, start_year, end_year in TIME_BLOCKS:
                delta = period_delta(matrix, start_year, end_year)
                for q in EXPORT_QUANTILES:
                    rows.append(
                        {
                            "scenario": short_name,
                            "panel": panel,
                            "period": label.replace("\n", " "),
                            "quantile": q,
                            "value": delta.quantile(q),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} rows)")


_export_period_quantiles(category_erf_matrix, total_erf_matrix, PLOTS_DIR / f"{OUTPUT_PREFIX}species_by_period_erf.csv")
_export_period_quantiles(category_gsat_matrix, total_gsat_matrix, PLOTS_DIR / f"{OUTPUT_PREFIX}species_by_period_gsat.csv")
