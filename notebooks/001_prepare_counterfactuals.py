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
# # Prepare counterfactuals (production)
#
# Takes the full length 1750-2100 emissions sets prepared in `000_merge_historical_and_future_scenarios.py` and produces the counterfactual emissions sets necessary for the emissions-driven decomposition (notebooks `004` through `008`).
#
# 1. **Load** the already-merged historical+future scenarios file (`MERGED_SCENARIOS_FILE`).
#    Falls back to historical-only if that file isn't found.
# 3. **Build counterfactuals**: for every base scenario x every attributed species x
#    every counterfactual-branch year in `START_YEARS`, hold that species at a pinned
#    level (zero for CO2, the `PIN_YEAR` historical value for everything else) from the
#    branch year onward - `f"{base_scenario}_no_{label}_{start_year}"`.

# %% [markdown]
# ## Imports

# %%
import json
import re
import warnings
from pathlib import Path

import pandas as pd
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.io import load_timeseries_csv

# %% [markdown]
# ## Configuration

# %%
EMBARGOED = True
"""Set True once running against real (embargoed) ScenarioMIP scenarios, so OUT_DB_DIR
is written under data/embargoed/ and inherits the source data's tool-access protection
instead of living in plain data/. Leave False for historical-only runs (not embargoed)."""
DATA_DIR = Path("../data/embargoed") if EMBARGOED else Path("../data")

HISTORICAL_FILE = Path("../data/historical_emissions.csv")
"""Historical emissions, 1750 through (at least) any future scenario's start year -
always public, not gated by EMBARGOED."""

MERGED_SCENARIOS_FILE = Path("../data/embargoed/scenariomip_cmip7_v20260325.csv")
"""Merged with history, marker scenarios, already produced by 000_merge_historical_and_future_scenarios.iypnb"""

REFERENCE_FILE = Path("../data/historical_emissions.csv")
"""Used to look up each non-CO2 species' PIN_YEAR value."""

OUT_DB_DIR = DATA_DIR / "scenarios_with_counterfactuals_db"
"""Output OpenSCMDB - one file per scenario."""

INDEX_COLUMNS = ["model", "scenario", "region", "variable", "unit"]
DB_GROUPBY_COLUMNS = ["scenario"]

PIN_YEAR = 1750
"""Year in REFERENCE_FILE whose value is used to pin non-CO2 species - always 1750
(preindustrial), independent of START_YEARS."""

START_YEARS = [1750]
"""Years from which the counterfactuals start applying. One counterfactual set per
start year, e.g. [1750, 1990] creates both "..._no_CH4_1750" and "..._no_CH4_1990"."""

CO2_VARIABLES = {
    "Emissions|CO2|Energy and Industrial Processes",
    "Emissions|CO2|AFOLU",
}
"""Variables treated as CO2, i.e. pinned to zero rather than to the PIN_YEAR value."""

# %% [markdown]
# ## Step 1: load the merged historical+future scenarios (if available)

# %%
if MERGED_SCENARIOS_FILE.exists():
    combined = load_timeseries_csv(MERGED_SCENARIOS_FILE, index_columns=INDEX_COLUMNS, out_columns_type=int)
else:
    print(f"{MERGED_SCENARIOS_FILE} not found - proceeding historical-only.")
    combined = load_timeseries_csv(HISTORICAL_FILE, index_columns=INDEX_COLUMNS, out_columns_type=int)

combined.columns.name = "year"
combined = combined.sort_index(axis="columns")

print(sorted(combined.index.get_level_values("scenario").unique()))
print(f"year range: {combined.columns.min()}-{combined.columns.max()}")

# %% [markdown]
# ## Step 2: build counterfactuals


# %%
def build_species_to_attribute(variables):
    """Map each attribution label to the emissions variables it covers. `variables` is
    the set of variables present in the loaded scenario data, used to pick up
    halogenated-gas variables by pattern and to drop labels absent from this dataset."""
    variables = set(variables)

    aerosol_vars = ["Emissions|BC", "Emissions|OC", "Emissions|Sulfur", "Emissions|NOx", "Emissions|NH3"]
    montreal_gas_vars = [
        "Emissions|CH3CCl3", "Emissions|CH3Br", "Emissions|CH3Cl", "Emissions|CCl4", "Emissions|CHCl3",
        "Emissions|Halon2402", "Emissions|CH2Cl2", "Emissions|Halon1202", "Emissions|Halon1301",
        "Emissions|HCFC141b", "Emissions|HCFC142b", "Emissions|Halon1211", "Emissions|HCFC22",
    ]
    f_gas_vars = {"Emissions|CF4", "Emissions|SF6", "Emissions|NF3", "Emissions|SO2F2"}

    species_to_attribute = {
        "CO2 Fossil": ["Emissions|CO2|Energy and Industrial Processes"],
        "CO2 AFOLU": ["Emissions|CO2|AFOLU"],
        "CH4": ["Emissions|CH4"],
        "N2O": ["Emissions|N2O"],
        "CO": ["Emissions|CO"],
        "VOC": ["Emissions|VOC"],
        "BC": ["Emissions|BC"],
        "OC": ["Emissions|OC"],
        "Sulfur": ["Emissions|Sulfur"],
        "NOx": ["Emissions|NOx"],
        "NH3": ["Emissions|NH3"],
        "Aerosols": aerosol_vars,
        "Montreal Protocol Halogen Gases": sorted(
            {v for v in variables if "Montreal" in v} | (variables & set(montreal_gas_vars))
        ),
        "F-Gases": sorted(
            {v for v in variables if "HFC" in v or re.match(r"Emissions\|c?C\d*F\d*", v)} | (variables & f_gas_vars)
        ),
    }
    return {
        label: [v for v in species_vars if v in variables]
        for label, species_vars in species_to_attribute.items()
        if any(v in variables for v in species_vars)
    }


# %%
scenarios = combined

reference = load_timeseries_csv(REFERENCE_FILE, index_columns=INDEX_COLUMNS, out_columns_type=int)
pin_values = reference[PIN_YEAR].groupby("variable").first()

species_to_attribute = build_species_to_attribute(scenarios.index.get_level_values("variable"))

# SOx and NH3 leave-one-out counterfactuals are deliberately NOT generated: they compete
# for the same ammonia pool in nitrate formation (RF_NO3_LAMBDASO2), so removing either
# one triggers a compensating increase in ammonium-nitrate aerosol (also cooling) that
# can outweigh the precursor's own direct/indirect effect - the marginal leave-one-out
# delta measures that net, confounded effect, not the precursor's own radiative forcing
# (confirmed empirically: SOx's own marginal came out net *warming*, the wrong sign).
# Use the burden-based analysis for SOx/NH3 instead.
species_to_attribute.pop("Sulfur", None)
species_to_attribute.pop("NH3", None)

# Combined CO2 counterfactual (Fossil + AFOLU zeroed together) - CO2's own leave-one-out
# treats CO2 as a single species, matching MAGICC's single CO2_SWITCHFROMCONC2EMIS_YEAR.
if "CO2 Fossil" in species_to_attribute and "CO2 AFOLU" in species_to_attribute:
    species_to_attribute["CO2"] = species_to_attribute["CO2 Fossil"] + species_to_attribute["CO2 AFOLU"]

species_to_attribute


# %%
def pin_value_for(variable):
    """Return the value `variable` should be held at from `start_year` onwards."""
    if variable in CO2_VARIABLES:
        return 0.0
    if variable not in pin_values.index:
        warnings.warn(f"No {PIN_YEAR} reference value found for {variable!r}, pinning to 0", stacklevel=2)
        return 0.0
    return pin_values[variable]


def make_counterfactual(scenario_data, base_scenario, species, label, start_year):
    """Return a copy of `scenario_data` with `species` held at its pinned value from `start_year` onwards."""
    cf = scenario_data.copy()
    year_columns = [c for c in cf.columns if c >= start_year]
    for variable in species:
        mask = cf.index.get_level_values("variable") == variable
        cf.loc[mask, year_columns] = pin_value_for(variable)
    new_index = cf.index.to_frame(index=False)
    new_index["scenario"] = f"{base_scenario}_no_{label}_{start_year}"
    cf.index = pd.MultiIndex.from_frame(new_index)
    return cf


# %%
counterfactuals = [
    make_counterfactual(
        scenarios.loc[scenarios.index.get_level_values("scenario") == base_scenario],
        base_scenario,
        species,
        label,
        start_year,
    )
    for start_year in START_YEARS
    for label, species in species_to_attribute.items()
    for base_scenario in scenarios.index.get_level_values("scenario").unique()
]

scenarios_with_counterfactuals = pd.concat([scenarios, *counterfactuals])

# %% [markdown]
# ## Save

# %%
db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=OUT_DB_DIR)
db.save(scenarios_with_counterfactuals, groupby=DB_GROUPBY_COLUMNS, allow_overwrite=True)

# %% [markdown]
# ## Base-scenario manifest
#
# Downstream notebooks need to know which scenarios in OUT_DB_DIR are *base* scenarios 
# (as opposed to the "..._no_X_YYYY" counterfactuals derived from them),
# without anyone having to hardcode or inspect the actual scenario names. 
# Written next to OUT_DB_DIR so it inherits the same EMBARGOED-gated location.

# %%
base_scenarios = sorted(scenarios.index.get_level_values("scenario").unique())
BASE_SCENARIOS_FILE = DATA_DIR / "base_scenarios.json"
BASE_SCENARIOS_FILE.write_text(json.dumps(base_scenarios, indent=2))

# %% [markdown]
# ## Check
#
# Reload one base scenario and one of its counterfactuals and plot the affected
# variable to confirm the pinning worked as expected.

# %%
import matplotlib.pyplot as plt

check_base_scenario = scenarios.index.get_level_values("scenario")[0]
check_label, check_species = next(iter(species_to_attribute.items()))
check_scenario = f"{check_base_scenario}_no_{check_label}_{START_YEARS[0]}"

check_selector = pd.Index([check_base_scenario, check_scenario], name="scenario")
check_data = db.load(check_selector)
check_data = check_data.loc[check_data.index.get_level_values("variable").isin(check_species)]

fig, ax = plt.subplots()
for (scenario, variable), row in check_data.groupby(level=["scenario", "variable"]).first().iterrows():
    ax.plot(row.index, row.values, label=f"{scenario} | {variable}")
ax.legend()
ax.set_xlabel("year")
ax.set_ylabel("emissions")
