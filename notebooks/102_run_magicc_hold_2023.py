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
# # Run MAGICC on the "hold at 2023" counterfactuals (100-series)
#
# Reads `101_prepare_counterfactuals.py`'s output (`hold_2023_scenarios_with_counterfactuals_db`
# / `hold_2023_base_scenarios.json`) and runs MAGICC on every base scenario + basket
# counterfactual pair - kept separate from `101` so re-running MAGICC never requires
# rebuilding the counterfactual emissions, and vice versa.
#
# Because MAGICC's own default `*_SWITCHFROMCONC2EMIS_YEAR` is 2015 for CO2/CH4/N2O/
# FGAS/MHALO (confirmed in `MAGCFG_DEFAULTALL.CFG`), every one of these baskets is
# *already* emissions-driven from 2015 onward in the plain default config - unlike
# `004`'s from-preindustrial leave-one-out (which needs the switch forced back to 1750
# plus CH4/N2O budget-closure re-anchoring to avoid an early-industrial mismatch). Since
# this counterfactual only ever diverges from the base scenario starting in 2023 (well
# inside that already-emissions-driven window), no switches or overrides are needed here
# - this notebook reuses the same official-consistent default config as `002`.
#
# One base scenario + counterfactual pair is run together per basket (matching `004`'s
# pattern), each requesting only the output variables needed for that basket (total ERF,
# GSAT, and the basket's own ERF contribution(s)) - see `BASKET_OUTPUT_VARIABLES`.

# %% [markdown]
# ## Imports

# %%
import json
import logging
import os
import warnings
from pathlib import Path

import attribution_common as ac
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

warnings.filterwarnings("ignore", message=".*Extending solar RF.*")
warnings.filterwarnings("ignore", message=".*magicc logged a WARNING message.*")
logging.getLogger("pymagicc").setLevel(logging.ERROR)

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = Path("../data")
"""Unembargoed - matches 101's own DATA_DIR."""

OUT_COUNTERFACTUALS_DB_DIR = DATA_DIR / "hold_2023_scenarios_with_counterfactuals_db"
"""Written by 101_prepare_counterfactuals.py."""

BASE_SCENARIOS = json.loads((DATA_DIR / "hold_2023_base_scenarios.json").read_text())
"""Auto-discovered from 101's own manifest - whatever base scenarios 101 actually
processed, no need to know/hardcode the real names."""

BASKET_LABELS = ["CO2", "CH4", "N2O", "F-Gases", "Montreal Halogens", "Aerosols", "Other"]
"""Must match 101's own BASKETS keys (order doesn't matter) - not re-derived from
gcages here since running MAGICC only needs each basket's label (for the counterfactual
scenario name/output db dir), not its member species. Aerosols = BC, OC, Sulfur, NH3;
Other = NOx, CO, VOC (the tropospheric-ozone-forming precursors) - split from the
former single "Residual" basket."""

MAGICC_SUPPLY_START_YEAR = 2015
"""Matches the official CMIP7 ScenarioMIP workflow convention (see `002`) - MAGICC's own
default switch year for CO2/CH4/N2O/FGAS/MHALO (2015) means supplying from 2015 onward
already gives an emissions-driven run for every basket here, no explicit switches
needed."""

N_TRIAL_MEMBERS = None
"""Set to a small int (e.g. 10) for fast iteration. None = full ensemble."""
MAX_PROCESSES = 5
BATCH_SIZE_SCENARIOS = 15

CORE_OUTPUT_VARIABLES = ("Surface Air Temperature Change", "Effective Radiative Forcing")

BASKET_OUTPUT_VARIABLES = {
    "CO2": (*CORE_OUTPUT_VARIABLES, "Effective Radiative Forcing|CO2"),
    "CH4": (
        *CORE_OUTPUT_VARIABLES,
        "Effective Radiative Forcing|CH4",
        "Effective Radiative Forcing|Tropospheric Ozone",
        "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O",
    ),
    "N2O": (*CORE_OUTPUT_VARIABLES, "Effective Radiative Forcing|N2O"),
    "F-Gases": (*CORE_OUTPUT_VARIABLES, "Effective Radiative Forcing|F-Gases"),
    "Montreal Halogens": (*CORE_OUTPUT_VARIABLES, "Effective Radiative Forcing|Montreal Protocol Halogen Gases"),
    "Aerosols": (
        *CORE_OUTPUT_VARIABLES,
        "Effective Radiative Forcing|Aerosols|Direct Effect",
        "Effective Radiative Forcing|Aerosols|Indirect Effect",
        "Effective Radiative Forcing|Black Carbon on Snow",
    ),
    "Other": (
        *CORE_OUTPUT_VARIABLES,
        "Effective Radiative Forcing|Tropospheric Ozone",
    ),
}


def out_db_dir(label):
    return DATA_DIR / f"hold_2023_scm_output_db_{ac.slugify(label)}"


# %% [markdown]
# ## Run MAGICC per basket (base + counterfactual together)
#
# Plain official-consistent default config (no switches/overrides, matching `002`) -
# see the module docstring for why that's sufficient here.

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(ac.MAGICC_EXECUTABLE_PATH)

climate_models_cfgs = ac.load_magicc_cfgs(n_members=N_TRIAL_MEMBERS)
print("ensemble size:", len(climate_models_cfgs["MAGICC7"]))

for label in BASKET_LABELS:
    output_variables = BASKET_OUTPUT_VARIABLES[label]

    for base_scenario in BASE_SCENARIOS:
        counterfactual_scenario = f"{base_scenario}_hold_2023_{ac.slugify(label)}"
        needed_scenarios = [base_scenario, counterfactual_scenario]

        print(f"=== {label} ({base_scenario}) === output: {output_variables}")

        scenarios_osr_full = ac.load_scenarios(needed_scenarios, OUT_COUNTERFACTUALS_DB_DIR)
        scenarios_osr = scenarios_osr_full.loc[:, MAGICC_SUPPLY_START_YEAR:]

        ac.run_scms_to_db(
            scenarios_osr,
            needed_scenarios,
            climate_models_cfgs,
            output_variables,
            out_db_dir(label),
            max_processes=MAX_PROCESSES,
            batch_size_scenarios=BATCH_SIZE_SCENARIOS,
        )

# %% [markdown]
# ## Check: GSAT by basket

# %%
for label in BASKET_LABELS:
    output_db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=out_db_dir(label))
    result = output_db.load(out_columns_type=int)
    gsat = result.loc[result.index.get_level_values("variable") == "Surface Air Temperature Change"]
    last_year = gsat.columns.max()
    print(f"--- {label} ---")
    print(gsat.groupby(gsat.index.get_level_values("scenario"))[last_year].agg(["mean", "median"]))
