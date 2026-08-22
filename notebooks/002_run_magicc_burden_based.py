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
# # Run MAGICC for the burden-based decomposition, official-consistent input convention
#
# 1. Run MAGICC consistent with the official CMIP7 ScenarioMIP workflow (`gcages.cmip7_scenariomip.scm_running.CMIP7ScenarioMIPSCMRunner`).
#
# `scenarios_osr` is truncated to 2015-2100 before being handed to MAGICC. MAGICC's own
# output spans its full configured run range (1750-2100). Produces ERF and GSAT change
# output variables for all scenarios that can be used downstream for the decomposition.
#
# 2. Take the ERFs produced by the full and CMIP7-consistent MAGICC run
# and feed them back into MAGICC's forcing-driven `QEXTRA`mode to compute the GSAT
# change attributable to those forcings.

# %% [markdown]
# ## Imports

# %%
import logging
import os
import warnings
from pathlib import Path

import attribution_common as ac

warnings.filterwarnings("ignore", message=".*Extending solar RF.*")
warnings.filterwarnings("ignore", message=".*magicc logged a WARNING message.*")
logging.getLogger("pymagicc").setLevel(logging.ERROR)

# %% [markdown]
# ## Configuration

# %%
EMBARGOED = True
"""Set True once running against real (embargoed) ScenarioMIP scenarios, so this
notebook's outputs are written under data/embargoed/ instead of plain data/. Leave
False for historical-only runs."""
DATA_DIR = Path("../data/embargoed") if EMBARGOED else Path("../data")

SCENARIOS_DB_DIR = DATA_DIR / "scenarios_with_counterfactuals_db"
"""Written by 001_prepare_counterfactuals.py. This notebook doesn't need the
counterfactuals themselves, just the base scenarios also stored in that db - must match
001's own EMBARGOED setting for this path to resolve correctly."""

SCENARIOS = ac.load_base_scenarios(DATA_DIR)
"""Auto-discovered from 001's base_scenarios.json manifest - whatever base scenarios
001 actually processed, no need to know/hardcode the real names. Falls back to
["historical"] if 001 hasn't been run yet."""

MAGICC_SUPPLY_START_YEAR = 2015
"""Matches the official CMIP7 ScenarioMIP workflow's `get_complete_scenarios_for_magicc`
convention - see the module docstring above and `attribution_common.load_scenarios`."""

CONSISTENT_BURDEN_SCM_OUTPUT_DB_DIR = DATA_DIR / "consistent_burden_scm_output_db"
CONSISTENT_BURDEN_GSAT_DB_DIR = DATA_DIR / "consistent_burden_gsat_db"
CONSISTENT_FORCING_CHANNELS_DIR = DATA_DIR / "consistent_burden_forcing_channels"

REGION = ac.REGION
N_TRIAL_MEMBERS = None
"""Set to a small int (e.g. 10) for fast iteration. None = full ensemble."""
MAX_PROCESSES = 5
BATCH_SIZE_SCENARIOS = 15

OUTPUT_VARIABLES = (
    # GSAT/GMST - for the additivity check (sum of isolated channels vs. the real,
    # all-forcings-together run).
    "Surface Air Temperature Change",
    "Surface Air Ocean Blended Temperature Change",
    # Totals, for cross-checks.
    "Effective Radiative Forcing",
    "Effective Radiative Forcing|Anthropogenic",
    "Effective Radiative Forcing|Greenhouse Gases",
    "Effective Radiative Forcing|Ozone",
    "Effective Radiative Forcing|Aerosols",
    # The forcing-agent categories themselves - see FORCING_CATEGORIES below. Matches
    # IPCC AR6 WG1 Ch.7 Fig 7.6/7.7.
    "Effective Radiative Forcing|CO2",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|F-Gases",
    "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
    "Effective Radiative Forcing|Tropospheric Ozone",
    "Effective Radiative Forcing|Stratospheric Ozone",
    "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Direct Effect|BC", # currently not used
    "Effective Radiative Forcing|Aerosols|Direct Effect|OC", # currently not used
    "Effective Radiative Forcing|Aerosols|Direct Effect|SOx", # currently not used
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
    "Effective Radiative Forcing|Black Carbon on Snow",
    "Effective Radiative Forcing|Land-use Change",
    "Effective Radiative Forcing|Aviation|Contrail and Cirrus",
    "Effective Radiative Forcing|Solar",
    "Effective Radiative Forcing|Volcanic",
    # Concentrations, for diagnostics.
    "Atmospheric Concentrations|CO2",
    "Atmospheric Concentrations|CH4",
    "Atmospheric Concentrations|N2O",
)

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

COMBINED_LABEL = "Combined"

# %% [markdown]
# ## Load scenarios
#
# Truncated to `MAGICC_SUPPLY_START_YEAR` (2015) onward - the one deliberate difference
# from 102, which supplies the full 1750-2100 range. Years 2015-2022 in this data are
# already a composite of real history and scenario data (from `101_prepare_counterfactuals.py`'s
# own merge) - confirmed elsewhere in this project to be numerically identical to what
# the official `get_complete_scenarios_for_magicc` produces for that same window.

# %%
scenarios_osr_full = ac.load_scenarios(SCENARIOS, SCENARIOS_DB_DIR)
scenarios_osr = scenarios_osr_full.loc[:, MAGICC_SUPPLY_START_YEAR:]
print("scenarios:", sorted(scenarios_osr.index.get_level_values("scenario").unique()))
print("year range supplied to MAGICC:", scenarios_osr.columns.min(), scenarios_osr.columns.max())

# %% [markdown]
# ## Step 1: single default-config run per scenario

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(ac.MAGICC_EXECUTABLE_PATH)

climate_models_cfgs = ac.load_magicc_cfgs(n_members=N_TRIAL_MEMBERS)
print("ensemble size:", len(climate_models_cfgs["MAGICC7"]))

burden_output_db = ac.run_scms_to_db(
    scenarios_osr,
    SCENARIOS,
    climate_models_cfgs,
    OUTPUT_VARIABLES,
    CONSISTENT_BURDEN_SCM_OUTPUT_DB_DIR,
    max_processes=MAX_PROCESSES,
    batch_size_scenarios=BATCH_SIZE_SCENARIOS,
)

result = burden_output_db.load(out_columns_type=int)
result.columns.name = "year"
print("output year range:", result.columns.min(), result.columns.max())
gsat = result.loc[result.index.get_level_values("variable") == "Surface Air Temperature Change"]
last_year = gsat.columns.max()
print(gsat.groupby(gsat.index.get_level_values("scenario"))[last_year].agg(["mean", "median"]))

# %% [markdown]
# ## Step 2: per-category QEXTRA rewiring


# %%
def process_base_scenario_for_burden_analysis(base_scenario):
    """Write per-member FILE_EXTRA_RF inputs for every category (plus Combined) for
    `base_scenario`, then run MAGICC's climate module on each via QEXTRA. Returns the
    per-channel GSAT/ERF OpenSCMDB."""
    out_dir = CONSISTENT_FORCING_CHANNELS_DIR / base_scenario

    series = {label: ac.load_erf(result, base_scenario, variable, region=REGION) for label, variable in FORCING_CATEGORIES.items()}
    if N_TRIAL_MEMBERS is not None:
        series = {label: df.loc[df.index < N_TRIAL_MEMBERS] for label, df in series.items()}

    channel_db = ac.run_qextra_channels(
        scenarios_osr=result,
        driving_scenario_name=base_scenario,
        channel_series=series,
        combined_label=COMBINED_LABEL,
        climate_models_cfgs=climate_models_cfgs,
        forcing_files_dir=out_dir,
        out_db_dir=CONSISTENT_BURDEN_GSAT_DB_DIR,
        max_processes=MAX_PROCESSES,
    )

    last_year_local = series[next(iter(series))].columns.max()
    print(f"--- {base_scenario}: ERF by category, {last_year_local} (mean, W/m^2) ---")
    print({label: df[last_year_local].mean() for label, df in series.items()})
    return channel_db


# %%
for base_scenario in SCENARIOS:
    process_base_scenario_for_burden_analysis(base_scenario)

# %% [markdown]
# ## Step 3: verify against official CMIP7 ScenarioMIP published quantiles
#
# This is the actual consistency check this notebook exists for: with the emissions
# supply now matching the official convention, the median ERF should match the published
# `erf-timeseries-quantiles_<model>.csv` files closely - unlike an earlier iteration, whose 
# numbers diverged.

# %%
import pandas as pd  # noqa: E402
from pandas_openscm.grouping import groupby_except  # noqa: E402

CLIMATE_ASSESSMENT_DIR = Path("../input_files/climate-assessment")
VALIDATION_YEARS = (2050, 2100)
MEDIAN = 0.5

erf_for_validation = result.loc[result.index.get_level_values("variable").isin(OUTPUT_VARIABLES)]
median_by_scenario = groupby_except(erf_for_validation, "run_id").quantile(MEDIAN)
median_by_scenario.index = median_by_scenario.index.droplevel(
    [lvl for lvl in median_by_scenario.index.names if lvl not in ("model", "scenario", "variable")]
)

validation_rows = []
for short, meta in ac.SCENARIO_METADATA.items():
    model, scenario = meta["model"], meta["scenario"]
    official_file = CLIMATE_ASSESSMENT_DIR / f"erf-timeseries-quantiles_{model}.csv"
    if not official_file.exists():
        continue
    official_df = pd.read_csv(official_file)
    official_df = official_df[(official_df["scenario"] == scenario) & (official_df["quantile"] == MEDIAN)].set_index("variable")

    for variable in official_df.index.unique():
        if variable not in OUTPUT_VARIABLES:
            continue
        try:
            ours_row = median_by_scenario.xs((model, scenario, variable), level=("model", "scenario", "variable")).iloc[0]
        except KeyError:
            continue
        official_row = official_df.loc[variable]
        if isinstance(official_row, pd.DataFrame):
            official_row = official_row.iloc[0]
        for year in VALIDATION_YEARS:
            if str(year) not in official_row.index or year not in ours_row.index:
                continue
            ours_val = float(ours_row[year])
            official_val = float(official_row[str(year)])
            abs_diff = ours_val - official_val
            rel_diff_pct = abs_diff / abs(official_val) * 100 if official_val != 0 else float("nan")
            validation_rows.append(
                {
                    "marker": short,
                    "variable": variable,
                    "year": year,
                    "consistent_002": ours_val,
                    "official": official_val,
                    "abs_diff": abs_diff,
                    "rel_diff_pct": rel_diff_pct,
                }
            )

validation_table = pd.DataFrame(validation_rows)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)
print(validation_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nmax |rel_diff_pct| across all rows:", validation_table["rel_diff_pct"].abs().max())
