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
# # Merge historical and future-only scenario data (100-series, unembargoed)
#
# Start of a new, parallel 100-series workflow: "hold one species/basket at its 2023
# level through 2100" counterfactuals (as opposed to the 000-010 series' "leave one
# species out from preindustrial" counterfactuals). Uses the now-unembargoed full
# marker-scenario emissions in `input_files/emissions/infilled_{model}.csv` - one file
# per marker, each already containing only that marker's own scenario (no
# cross-model/cross-scenario filtering needed, unlike `000`'s embargoed multi-scenario
# source files).
#
# Merges each marker's future-only projection (2023 onwards) with the (separately
# sourced) historical emissions record, so `101_prepare_counterfactuals.py` gets a
# continuous 1750-2100 timeseries per marker.

# %% [markdown]
# ## Imports

# %%
import os
from pathlib import Path

import pandas as pd

# %% [markdown]
# ## Configuration

# %%
HISTORICAL_FILE = Path("../data/historical_emissions.csv")
"""Historical emissions, 1750 through (at least) 2022 - always public."""

FUTURE_SCENARIOS_PATH = Path("../input_files/emissions")
"""Unembargoed future-only scenario data (from 2023 onwards): one `infilled_{model}.csv`
per marker, each holding only that marker's own (model, scenario) - same
variable/unit convention as HISTORICAL_FILE."""

MERGED_SCENARIOS_FILE = Path("../data/scenariomip_cmip7_unembargoed.csv")
"""Where the merged, continuous marker scenarios are written - point
101_prepare_counterfactuals.py's MERGED_SCENARIOS_FILE here."""

INDEX_COLUMNS = ["model", "scenario", "region", "variable", "unit"]

# %% [markdown]
# ## Load data and merge with history

# %%
historical = pd.read_csv(HISTORICAL_FILE, index_col=INDEX_COLUMNS)
historical.columns = historical.columns.astype(int)
historical.columns.name = "year"

# Drop the 2023 column from historical to avoid duplication with the future files,
# which start at 2023.
historical = historical.drop(columns=[2023])

# %%
merged_dfs = []

for filename in sorted(os.listdir(FUTURE_SCENARIOS_PATH)):
    if not filename.endswith(".csv"):
        continue
    file_path = FUTURE_SCENARIOS_PATH / filename
    scenario_df = pd.read_csv(file_path, index_col=INDEX_COLUMNS).dropna(axis=1)
    scenario_df.columns = scenario_df.columns.astype(int)
    scenario_df.columns.name = "year"

    scenario_df = scenario_df.reset_index()

    filtered_df = scenario_df[
        (~scenario_df["variable"].str.contains("AR6GWP100", na=False))
        & (scenario_df["variable"].str.startswith("Emissions|"))
        & (~(scenario_df["variable"] == "Emissions|CO2"))
    ]

    merged_df = pd.merge(
        filtered_df,
        historical,
        on=["region", "variable", "unit"],
        how="left",
        suffixes=("", "_hist"),
    )
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    all_years = range(1750, 2101)
    merged_df = merged_df.reindex(columns=[*merged_df.columns[:5], *all_years])
    merged_df = merged_df.set_index(INDEX_COLUMNS)

    merged_dfs.append(merged_df)

final_df = pd.concat(merged_dfs)

# %%
final_df

# %% [markdown]
# ## Save

# %%
final_df = final_df.sort_index(axis="columns")

MERGED_SCENARIOS_FILE.parent.mkdir(parents=True, exist_ok=True)
final_df.reset_index().to_csv(MERGED_SCENARIOS_FILE, index=False)

# %% [markdown]
# ## Check
#
# Scenario names and year coverage only - deliberately nothing about the values
# themselves.

# %%
print(sorted(final_df.index.get_level_values("scenario").unique()))
print(f"year range: {final_df.columns.min()}-{final_df.columns.max()}")
