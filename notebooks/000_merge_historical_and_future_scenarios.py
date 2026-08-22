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
# # Merge historical and future-only scenario data
#
# IAMC-format scenario files typically only cover the future projection period (e.g. from 2023 onwards) - for emissions-driven species attribution, we need to supply a continuous timeseries to MAGICC through the run's end year. This merges each future scenario with the (separately sourced) historical emissions record in preparation for `001_prepare_counterfactuals.py`.

# %% [markdown]
# ## Imports

# %%
import warnings
from pathlib import Path
import os

from pandas_openscm.io import load_timeseries_csv
import pandas as pd

# %% [markdown]
# ## Configuration

# %%
MARKERS = (
    # (model, scenario, ScenarioMIP name, final_version)
    ("REMIND-MAgPIE 3.5-4.11", "SSP1 - Very Low Emissions", "vl", 5),
    ("AIM 3.0", "SSP2 - Low Overshoot_a", "ln", 23),
    ("MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "SSP2 - Low Emissions", "l", 21),
    ("COFFEE 1.6", "SSP2 - Medium-Low Emissions", "ml", 14),
    ("IMAGE 3.4", "SSP2 - Medium Emissions", "m", 25),
    ("WITCH 6.0", "SSP5 - Medium-Low Emissions_a", "hl", 32),
    ("GCAM 8s", "SSP3 - High Emissions", "h", 3),
)

MARKERS_BY_SCENARIOMIP_NAME = {
    scenariomip_name: {
        "model": model,
        "scenario": scenario,
        "version": version,
    }
    for model, scenario, scenariomip_name, version in MARKERS
}

# %%
HISTORICAL_FILE = Path("../data/global-workflow-history_202511261223_202511040855_202512032146_202512021030_7e32405ade790677a6022ff498395bff00d9792d_202511040855_202512071232_202511040855_202511040855_0002_0002.csv")
"""Historical emissions, e.g. 1750 through (at least) the future scenarios' start
year."""

FUTURE_SCENARIOS_PATH = Path("../data/embargoed/Round 8.1 (v20260325)")
"""Future-only scenario data (from 2023 onwards): one row per (model, scenario,
region, variable, unit), same variable/unit convention as HISTORICAL_FILE."""

FUTURE_SCENARIOS_FILE = Path("../data/embargoed/scenariomip_cmip7_v20260325.csv")
"""Where the merged, continuous scenarios are written - point
001_prepare_counterfactuals.py's SCENARIOS_FILE here."""

INDEX_COLUMNS = ["model", "scenario", "region", "variable", "unit"]

# %% [markdown]
# ## Load data, filter for marker scenarios, and merge with history

# %%
# Load historical data as a multi-index DataFrame with years as columns
historical = pd.read_csv(HISTORICAL_FILE, index_col=INDEX_COLUMNS)
historical.columns = historical.columns.astype(int)  # Ensure year columns are integers
historical.columns.name = "year"

# Drop the 2023 column from historical to avoid duplication
historical = historical.drop(columns=[2023])

# %%
# Initialize a list to store merged DataFrames
merged_dfs = []

# Loop through scenario files
for filename in os.listdir(FUTURE_SCENARIOS_PATH):
    if filename.endswith(".csv"):
        file_path = os.path.join(FUTURE_SCENARIOS_PATH, filename)
        scenario_df = pd.read_csv(file_path, index_col=INDEX_COLUMNS).dropna(axis=1)
        scenario_df.columns = scenario_df.columns.astype(int)
        scenario_df.columns.name = "year"

        # Reset index to access "region" and "variable" as columns
        scenario_df = scenario_df.reset_index()

        # Filter rows based on criteria
        filtered_df = scenario_df[
            (~scenario_df["variable"].str.contains("AR6GWP100", na=False)) &
            (scenario_df["variable"].str.startswith("Emissions|")) &
            (~(scenario_df["variable"] == "Emissions|CO2"))
        ]

        # Further filter by scenario and model
        filtered_df = filtered_df[
            filtered_df["scenario"].isin(
                [MARKERS_BY_SCENARIOMIP_NAME[name]["scenario"]
                 for name, info in MARKERS_BY_SCENARIOMIP_NAME.items()
                 if info["model"] in filtered_df["model"].values]
            )
        ]

        # Merge with historical data on "region" and "variable"
        merged_df = pd.merge(
            filtered_df,
            historical,
            on=["region", "variable", "unit"],
            how="left",
            suffixes=("", "_hist")
        )

        # Drop duplicate year columns from historical (if any)
        merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

        # Reindex to ensure all years from 1750 to 2100 are included
        all_years = range(1750, 2101)
        merged_df = merged_df.reindex(columns=[*merged_df.columns[:5], *all_years])

        # Set the index back to the original multi-index structure
        merged_df = merged_df.set_index(INDEX_COLUMNS)

        merged_dfs.append(merged_df)

# Concatenate all merged DataFrames
final_df = pd.concat(merged_dfs)

# %%
final_df

# %% [markdown]
# ## Save

# %%
final_df = final_df.sort_index(axis="columns")

FUTURE_SCENARIOS_FILE.parent.mkdir(parents=True, exist_ok=True)
final_df.reset_index().to_csv(FUTURE_SCENARIOS_FILE, index=False)

# %% [markdown]
# ## Check
#
# Scenario names and year coverage only - deliberately nothing about the values
# themselves.

# %%
print(sorted(final_df.index.get_level_values("scenario").unique()))
print(f"year range: {final_df.columns.min()}-{final_df.columns.max()}")
