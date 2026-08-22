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
# # Burden-based decomposition
#
# This reads `002_run_magicc_burden_based.py`'s output, a MAGICC run supplied emissions from 2015 onward, matching the official CMIP7 ScenarioMIP workflow's input convention).
#
# Output files are prefixed `consistent_` to indicate consistency with the official ScenarioMIP processing pipeline.
#
# GSAT is rebased to the IPCC-standard 1850-1900 reference period, via `ac.rebase_gsat_matrix`.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

import attribution_common as ac

# %% [markdown]
# ## Configuration

# %%
EMBARGOED = True
"""Set True once running against real (embargoed) ScenarioMIP scenarios - reads from
data/embargoed/ instead of plain data/. Must match 002's own EMBARGOED setting."""
DATA_DIR = Path("../data/embargoed") if EMBARGOED else Path("../data")

YEAR = 2023
REGION = ac.REGION

BURDEN_SCM_OUTPUT_DB_DIR = DATA_DIR / "consistent_burden_scm_output_db"
BURDEN_GSAT_DB_DIR = DATA_DIR / "consistent_burden_gsat_db"
"""002's output dirs."""
PLOTS_DIR = Path("../data/plots")
"""Always plain data/plots/, regardless of EMBARGOED - plots and summary tables are not
considered sensitive, only raw emissions/MAGICC output."""
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "consistent_"
"""Prepended to every plot filename below."""

SCENARIOS = ac.load_base_scenarios(DATA_DIR)
"""Auto-discovered from 001's base_scenarios.json manifest - whatever base scenarios
001 actually processed, no need to know/hardcode the real names. Falls back to
["historical"] if 001 hasn't been run yet."""
COMBINED_LABEL = "Combined"

SCENARIO_COLORS = dict(zip((ac.scenario_short_name(s) for s in SCENARIOS), ac.CATEGORICAL_PALETTE))
"""Fixed categorical color assignment for visual consistency - used below
to color each scenario's own "Total" bar in the bar charts, matching that
scenario's color in other plots."""

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
"""Must match 002's FORCING_CATEGORIES - kept in sync manually since
it's a config constant, not shared machinery."""

PLOT_BUCKETS = {
    "CO2": ["CO2"],
    "N2O": ["N2O"],
    "CH4": ["CH4"],
    "Montreal Protocol Halogen Gases": ["Montreal Protocol Halogen Gases"],
    "Tropospheric Ozone": ["Tropospheric Ozone"],
    "F-Gases": ["F-Gases"],
    "Stratospheric H2O": ["Stratospheric H2O"],
    "Aerosol-Cloud Interactions": ["Aerosol-Cloud Interactions"],
    "Aerosol-Radiation Interactions": ["Aerosol-Radiation Interactions"],
    "Black Carbon on Snow": ["Black Carbon on Snow"],
    "Other": ["Stratospheric Ozone", "Solar", "Volcanic", "Land Use"],
}
"""Display-only grouping of FORCING_CATEGORIES for the bar charts below.

Contrails and Aviation-Induced Cirrus is dropped entirely - it isn't actually 
implemented (MAGICC reports exactly 0 for this variable in every scenario)."""

BUCKET_COLORS = {
    "CO2": "#707070",
    "CH4": "LightSkyBlue",
    "N2O": "orange",
    "Montreal Protocol Halogen Gases": "cyan",
    "F-Gases": "blue",
    "Tropospheric Ozone": "yellow",
    "Stratospheric H2O": "magenta",
    "Aerosol-Radiation Interactions": "purple",
    "Aerosol-Cloud Interactions": "#B19CD9",
    "Black Carbon on Snow": "saddlebrown",
    "Other": "red",
}
"""Deliberately has no "Total" entry - each scenario's "Total" bar (pinned 
to the top of the bar charts below rather than sorted in among the individual 
categories) is colored per-scenario via SCENARIO_COLORS instead, overridden 
onto a copy of this dict inside summarize_base_scenario."""

PLOT_YEARS = [2050, 2100]
"""Years shown side by side in the two-panel bar charts below."""

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


def _to_matrix(sub):
    """Reshape a (scenario/variable/region/run_id)-indexed slice into a plain
    run_id-indexed, year-column matrix, as `ac.rebase_gsat_matrix` expects."""
    sub = sub.reset_index().set_index("run_id")
    year_columns = [c for c in sub.columns if isinstance(c, int)]
    return sub[year_columns].sort_index()


def _raw_total_gsat_matrix(scenario):
    mask = (
        (burden_scm_output.index.get_level_values("scenario") == scenario)
        & (burden_scm_output.index.get_level_values("variable") == "Surface Air Temperature Change")
        & (burden_scm_output.index.get_level_values("region") == REGION)
    )
    return _to_matrix(burden_scm_output.loc[mask])


def load_total_gsat(scenario):
    """Rebased to the IPCC-standard 1850-1900 reference period - see `113`'s module
    docstring and `ac.rebase_gsat_matrix` for why GSAT (not ERF) needs this."""
    return ac.rebase_gsat_matrix(_raw_total_gsat_matrix(scenario))


# %% [markdown]
# ## Per-scenario ERF/GSAT-by-category summary, additivity checks, figures


# %%
def summarize_base_scenario(base_scenario):
    """Build the ERF-by-category and GSAT-by-category tables, additivity check and
    plot for `base_scenario`. Returns the GSAT-by-category summary DataFrame."""
    total_gsat = load_total_gsat(base_scenario)
    categories = list(FORCING_CATEGORIES.keys())
    gsat_shift = ac.gsat_assessment_shift(_raw_total_gsat_matrix(base_scenario), n_channels=len(categories))
    bucket_colors = {**BUCKET_COLORS, "Total": SCENARIO_COLORS[ac.scenario_short_name(base_scenario)]}

    def gsat_series(category, year=YEAR):
        """Rebased to the IPCC-standard 1850-1900 reference period, sharing the real
        total run's own assessment-matching shift (`gsat_shift`) rather than
        independently recalibrating this one channel - see
        `ac.rebase_gsat_channel_matrix`'s docstring for why that matters."""
        scenario = f"{base_scenario}_forcing_only_{category}"
        mask = burden_gsat_by_channel.index.get_level_values("scenario") == scenario
        matrix = ac.rebase_gsat_channel_matrix(_to_matrix(burden_gsat_by_channel.loc[mask]), gsat_shift)
        return ac.member_series(matrix, year)

    def erf_series_at(category, year):
        return ac.member_series(
            ac.load_erf(burden_scm_output, base_scenario, FORCING_CATEGORIES[category], region=REGION), year
        )

    erf_series = {
        c: ac.member_series(ac.load_erf(burden_scm_output, base_scenario, variable, region=REGION), YEAR)
        for c, variable in FORCING_CATEGORIES.items()
    }
    gsat_contribution_series = {c: gsat_series(c) for c in categories}
    combined_series = gsat_series(COMBINED_LABEL)
    total_gsat_series = ac.member_series(total_gsat, YEAR)

    summary = pd.DataFrame(
        {
            "ERF mean (W/m^2)": {c: s.mean() for c, s in erf_series.items()},
            "ERF median (W/m^2)": {c: s.median() for c, s in erf_series.items()},
            "GSAT contribution mean (K)": {c: s.mean() for c, s in gsat_contribution_series.items()},
            "GSAT contribution median (K)": {c: s.median() for c, s in gsat_contribution_series.items()},
        }
    )
    summary.index.name = "category"
    summary = summary.sort_values("GSAT contribution mean (K)", ascending=False)

    sum_of_parts_series = sum(gsat_contribution_series.values())
    additivity_residual_series = sum_of_parts_series - combined_series
    reconstruction_residual_series = combined_series - total_gsat_series

    combined_stats = ac.distribution_summary(combined_series)
    total_stats = ac.distribution_summary(total_gsat_series)

    print(f"=== {ac.scenario_short_name(base_scenario)} ({YEAR}) ===")
    print(summary)
    print()
    print("Additivity residual (sum of isolated channels - Combined QEXTRA run), per member:")
    print(ac.distribution_summary(additivity_residual_series))
    print()
    print("Reconstruction residual (Combined QEXTRA run - real all-forcings run), per member:")
    print(ac.distribution_summary(reconstruction_residual_series))
    print()
    print(f"Combined (QEXTRA reconstruction) GSAT: mean={combined_stats['mean']:.4f}, median={combined_stats['median']:.4f}")
    print(f"Real all-forcings run GSAT:            mean={total_stats['mean']:.4f}, median={total_stats['median']:.4f}")
    print()

    def erf_bucket_series_at(year):
        erf_series_at_year = {c: erf_series_at(c, year) for c in categories}
        bucket_series = {bucket: sum(erf_series_at_year[c] for c in members) for bucket, members in PLOT_BUCKETS.items()}
        bucket_series["Total"] = ac.member_series(
            ac.load_erf(burden_scm_output, base_scenario, "Effective Radiative Forcing", region=REGION), year
        )
        return bucket_series

    def gsat_bucket_series_at(year):
        gsat_series_at_year = {c: gsat_series(c, year) for c in categories}
        bucket_series = {bucket: sum(gsat_series_at_year[c] for c in members) for bucket, members in PLOT_BUCKETS.items()}
        bucket_series["Total"] = ac.member_series(total_gsat, year)
        return bucket_series

    erf_medians_by_year, erf_lower_by_year, erf_upper_by_year = {}, {}, {}
    gsat_medians_by_year, gsat_lower_by_year, gsat_upper_by_year = {}, {}, {}
    for year in PLOT_YEARS:
        erf_bucket_series = erf_bucket_series_at(year)
        medians = pd.Series({b: s.median() for b, s in erf_bucket_series.items()})
        p17 = pd.Series({b: s.quantile(0.17) for b, s in erf_bucket_series.items()})
        p83 = pd.Series({b: s.quantile(0.83) for b, s in erf_bucket_series.items()})
        erf_medians_by_year[year] = medians
        erf_lower_by_year[year] = medians - p17
        erf_upper_by_year[year] = p83 - medians

        gsat_bucket_series = gsat_bucket_series_at(year)
        medians = pd.Series({b: s.median() for b, s in gsat_bucket_series.items()})
        p17 = pd.Series({b: s.quantile(0.17) for b, s in gsat_bucket_series.items()})
        p83 = pd.Series({b: s.quantile(0.83) for b, s in gsat_bucket_series.items()})
        gsat_medians_by_year[year] = medians
        gsat_lower_by_year[year] = medians - p17
        gsat_upper_by_year[year] = p83 - medians

    years_label = "_".join(str(y) for y in PLOT_YEARS)

    # "Total" pinned to the top regardless of its magnitude (it's the real run's own
    # total, not one of the individual categories being ranked) - the rest sort
    # ascending by the LAST year (GSAT's, since both panels share one order), same
    # convention as plot_category_bars_grouped_years' own default.
    shared_order = list(
        gsat_medians_by_year[PLOT_YEARS[-1]].drop("Total").sort_values(ascending=True).index
    ) + ["Total"]

    ac.plot_category_bars_two_metrics(
        panels=[
            {
                "values_by_year": erf_medians_by_year,
                "lower_err_by_year": erf_lower_by_year,
                "upper_err_by_year": erf_upper_by_year,
                "xlabel": "Effective Radiative Forcing (W/m$^2$)",
            },
            {
                "values_by_year": gsat_medians_by_year,
                "lower_err_by_year": gsat_lower_by_year,
                "upper_err_by_year": gsat_upper_by_year,
                "xlabel": "Contribution to GSAT Change (K)",
            },
        ],
        years=PLOT_YEARS,
        title=f"{ac.scenario_short_name(base_scenario)}: contributions by forcing agent (median, 17-83rd pct)",
        out_path=PLOTS_DIR / f"{OUTPUT_PREFIX}burden_{base_scenario}_by_category_{years_label}.png",
        bucket_colors=bucket_colors,
        order=shared_order,
    )
    plt.close("all")

    return summary[["GSAT contribution mean (K)", "GSAT contribution median (K)"]]


# %%
YEAR = 2100
per_scenario_summaries = {base_scenario: summarize_base_scenario(base_scenario) for base_scenario in SCENARIOS}

# %% [markdown]
# ## Cross-scenario comparison

# %%
cross_scenario_median = pd.DataFrame(
    {base_scenario: summary["GSAT contribution median (K)"] for base_scenario, summary in per_scenario_summaries.items()}
)
print(cross_scenario_median)
