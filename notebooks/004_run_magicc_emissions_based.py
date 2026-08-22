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
# # Run MAGICC for the emissions-based (leave-one-out) decomposition (with the 1751/CH4-N2O fixes)
#
#
# 1. **Supplied emissions start from 1751, not 1750** (`MAGICC_SUPPLY_START_YEAR`
#    below), supplying 1750 values destabilises aerosol trajectories, in particular OC.
# 2. **CH4/N2O budget-closure re-anchoring applied to whichever of the two isn't already
#    being switched to emissions-driven mode** (`CH4_HARDWIRED_BUDGET_OVERRIDES`/
#    `N2O_HARDWIRED_BUDGET_OVERRIDES` below) - the same fix `102`/`202` (burden-based)
#    still carry as a documented-but-unapplied item, now applied here: whenever a gas
#    stays concentration-driven throughout (as it always was, unfixed, in `103`), MAGICC's
#    default natural-emissions budget window disagrees with what our real emissions imply
#    for that window, producing the ~3-19% CH4/N2O ERF divergence from official numbers
#    documented in `load_scenarios`'s docstring, fix 2. This is a *different* mechanism
#    (and a *different* re-anchoring window - 2008/10 and 1978/10, not 1800/10) from the
#    existing `CH4_BUDGET_OVERRIDES`/`N2O_BUDGET_OVERRIDES` fix below, which only applies
#    when that gas is itself switched to emissions-driven mode (an early-industrial
#    emissions-driven-run mismatch, not a divergence-from-official-numbers one) - every
#    species config now gets exactly one of the two overrides per gas, never neither.
#
# Output written to new `fixed_` -prefixed db directories, so `103`'s own (unfixed)
# output stays available for comparison.
#
# Consolidates the leave-one-out running logic from `002_run_maggic.py` /
# `012_run_reactive_precursor_maggic.py` / `014_run_ghg_switch_comparison_maggic.py` /
# `016_run_ch4_corrected_maggic.py` / `018_run_co2_leaveoneout_maggic.py` into one
# dictionary-driven runner, using only the now-converged, validated configuration for
# each species (see `data/species_interaction_overview.md`):
#
# - Switch year **1750** throughout (matches the actual scenario start year) - unaffected
#   by fix 1 above (that only changes what year *data* starts at, not the configured
#   switch year; MAGICC falls back to its own bundled pre-1751 value for whichever gas
#   is switched at 1750, same as for BC/OC/SOx).
# - **Switch only what's needed per species** - never all-GHG-switched-together, which
#   was found this project to contaminate results via a secondary, temperature-mediated
#   pathway through species with no direct chemical link to the one being attributed
#   (Brewer-Dobson-circulation-scaled lifetimes responding to the small warming
#   difference other switched species create). NOx/CO/VOC only need CH4 switched (their
#   effect runs entirely through CH4); CH4 itself additionally needs F-Gases/Montreal
#   Halogens switched (its shared-OH-sink effect on HFCs/HCFCs is abundance-changing);
#   N2O and CO2 each need only themselves switched.
# - **CH4's and N2O's budget-closure re-anchoring applied whenever CH4/N2O is switched**
#   (`CH4_BUDGET_OVERRIDES`/`N2O_BUDGET_OVERRIDES` below) - regardless of whether CH4/N2O
#   is the species being attributed, or just an intermediate abundance-changing pathway
#   (e.g. for NOx/CO/VOC). This is the fix for the wrong-signed CH4/N2O transient found
#   this project - MAGICC's default natural-emissions mass-balance reference window
#   (calibrated against modern conditions) is wildly mismatched with an early-industrial
#   emissions-driven simulation; re-anchoring it near 1791-1800 fixes this
#   mechanistically at any switch year. `feed_yrstart` is left at MAGICC's own default
#   for both gases (1927/1925) - the more conservative of the two options tested (see
#   `species_interaction_overview.md` for the ~8% sensitivity this carries).

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
False for historical-only runs. Must match 101's own EMBARGOED setting, since
SCENARIOS_DB_DIR needs to resolve to wherever 101 actually wrote its output."""
DATA_DIR = Path("../data/embargoed") if EMBARGOED else Path("../data")

SCENARIOS_DB_DIR = DATA_DIR / "scenarios_with_counterfactuals_db"
"""Written by 101_prepare_counterfactuals.py."""

BASE_SCENARIOS = ac.load_base_scenarios(DATA_DIR)
"""Auto-discovered from 101's base_scenarios.json manifest - whatever base scenarios
101 actually processed, no need to know/hardcode the real names. Falls back to
["historical"] if 101 hasn't been run yet."""

MAGICC_SUPPLY_START_YEAR = 1751
"""Drop year 1750 from what's actually handed to MAGICC. Avoids aerosol runaway effects."""

SWITCH_YEAR = 1750

SWITCH_KEY_MAP = {
    "CH4": "CH4_SWITCHFROMCONC2EMIS_YEAR",
    "N2O": "N2O_SWITCHFROMCONC2EMIS_YEAR",
    "CO2": "CO2_SWITCHFROMCONC2EMIS_YEAR",
    "FGAS": "FGAS_SWITCHFROMCONC2EMIS_YEAR",
    "MHALO": "MHALO_SWITCHFROMCONC2EMIS_YEAR",
}

CH4_BUDGET_OVERRIDES = {
    "ch4_incl_ch4ox": 1,
    "ch4_lastbudgetyear": 1800,
    "ch4_budget_avgyears": 10,
    "ch4_feed_yrstart": 1927.0,  # MAGICC's default
}
N2O_BUDGET_OVERRIDES = {
    "n2o_lastbudgetyear": 1800,
    "n2o_budget_avgyears": 10,
    "n2o_feed_yrstart": 1925.0,  # MAGICC's default
}

CH4_HARDWIRED_BUDGET_OVERRIDES = {
    "ch4_lastbudgetyear": 2008,
    "ch4_budget_avgyears": 10,
}
N2O_HARDWIRED_BUDGET_OVERRIDES = {
    "n2o_lastbudgetyear": 1978,
    "n2o_budget_avgyears": 10,
}
"""Applied instead of CH4_BUDGET_OVERRIDES/N2O_BUDGET_OVERRIDES whenever that gas is 
NOT being switched to emissions-driven mode for a given species run. Found by a 
brute-force window scan and specific to the cmip7-historical emissions data. 
Natural-emissions budget-closure window to best reproduce what a concentration-driven
("hardwired history") run gives past 2015, closing the ~3-19% CH4/N2O ERF divergence
this otherwise leaves past 2015 in every species config where CH4/N2O itself isn't 
the one being switched, but we supply MAGICC with emissions from pre-2015."""

CORE_CHANNELS = (
    "Surface Air Temperature Change",
    "Effective Radiative Forcing|Tropospheric Ozone",
    "Effective Radiative Forcing|CH4",
    "Effective Radiative Forcing|CH4 Oxidation Stratospheric H2O",
)
NOX_CHANNELS = (
    *CORE_CHANNELS,
    "Effective Radiative Forcing|N2O",
    "Effective Radiative Forcing|Aerosols|Direct Effect",
    "Effective Radiative Forcing|Aerosols|Indirect Effect",
)
"""NOx additionally forms nitrate aerosol (Direct + Indirect) and has a documented,
if negligible, N2O overlap channel; CO/VOC have no direct aerosol-forming pathway of 
their own, so they stay on the 3-channel CORE_CHANNELS. Unlike SOx/NH3 below, NOx's own 
nitrate-forming pathway is direct (its own emissions are the nitrate precursor), not a 
competition-mediated side effect of removing some other species, so it doesn't carry the 
same sign-flip risk."""

EMISSIONS_BASED_SPECIES = {
    # NOx/CO/VOC: effect runs entirely through CH4 - own Tropospheric Ozone channel
    # (direct) plus the CH4/Stratospheric H2O channels (via the shared OH sink).
    "NOx": {"label": "NOx", "switches": ["CH4"], "output_variables": NOX_CHANNELS},
    "CO": {"label": "CO", "switches": ["CH4"], "output_variables": CORE_CHANNELS},
    "VOC": {"label": "VOC", "switches": ["CH4"], "output_variables": CORE_CHANNELS},
    # CH4 itself: same three channels, plus its own ERF and the HFC/HCFC channel (needs
    # F-Gases/Montreal Halogens switched too).
    "CH4": {
        "label": "CH4",
        "switches": ["CH4", "FGAS", "MHALO"],
        "output_variables": (
            *CORE_CHANNELS,
            "Effective Radiative Forcing|F-Gases",
            "Effective Radiative Forcing|Montreal Protocol Halogen Gases",
        ),
    },
    # N2O: own ERF, plus a new check on whether its Stratospheric Ozone channel is
    # measurable via leave-one-out (previously an open, never-checked gap).
    "N2O": {
        "label": "N2O",
        "switches": ["N2O"],
        "output_variables": (
            "Surface Air Temperature Change",
            "Effective Radiative Forcing|N2O",
            "Effective Radiative Forcing|Stratospheric Ozone",
        ),
    },
    # CO2: own ERF, plus CH4/N2O ERF to re-confirm the spectral-overlap term stays
    # negligible.
    "CO2": {
        "label": "CO2",
        "switches": ["CO2"],
        "output_variables": (
            "Surface Air Temperature Change",
            "Effective Radiative Forcing|CO2",
            "Effective Radiative Forcing|CH4",
            "Effective Radiative Forcing|N2O",
        ),
    },
}

N_TRIAL_MEMBERS = None

MAX_PROCESSES = 5
BATCH_SIZE_SCENARIOS = 15


def overrides_for(switches):
    """Every species config gets always one CH4 override and one N2O override. 
    Whichever of CH4/N2O is switched to emissions-driven mode gets the existing 
    early-industrial-run CH4_BUDGET_OVERRIDES/N2O_BUDGET_OVERRIDES; whichever stays 
    concentration-driven gets the newly-applied CH4_HARDWIRED_BUDGET_OVERRIDES/
    N2O_HARDWIRED_BUDGET_OVERRIDES instead, to match what the default 
    concentration-driven run would give past 2015."""
    overrides = {}
    if "CH4" in switches:
        overrides.update(CH4_BUDGET_OVERRIDES)
    else:
        overrides.update(CH4_HARDWIRED_BUDGET_OVERRIDES)
    if "N2O" in switches:
        overrides.update(N2O_BUDGET_OVERRIDES)
    else:
        overrides.update(N2O_HARDWIRED_BUDGET_OVERRIDES)
    return overrides


def out_db_dir(species_key):
    return DATA_DIR / f"fixed_emissions_scm_output_db_{ac.slugify(species_key)}"


# %% [markdown]
# ## Run each species' leave-one-out pair (counterfactual + baseline in same config)

# %%
os.environ["MAGICC_EXECUTABLE_7"] = str(ac.MAGICC_EXECUTABLE_PATH)

for species_key, spec in EMISSIONS_BASED_SPECIES.items():
    magicc_switches = [SWITCH_KEY_MAP[s] for s in spec["switches"]]
    overrides = overrides_for(spec["switches"])

    for base_scenario in BASE_SCENARIOS:
        counterfactual_scenario = f"{base_scenario}_no_{spec['label']}_{SWITCH_YEAR}"
        needed_scenarios = [base_scenario, counterfactual_scenario]

        print(f"=== {species_key} ({base_scenario}) === switches @ {SWITCH_YEAR}: {magicc_switches}, overrides: {overrides}")

        scenarios_osr_full = ac.load_scenarios(needed_scenarios, SCENARIOS_DB_DIR)
        scenarios_osr = scenarios_osr_full.loc[:, MAGICC_SUPPLY_START_YEAR:]
        climate_models_cfgs = ac.load_magicc_cfgs(
            n_members=N_TRIAL_MEMBERS, switches=magicc_switches, overrides=overrides
        )
        print("ensemble size:", len(climate_models_cfgs["MAGICC7"]))

        ac.run_scms_to_db(
            scenarios_osr,
            needed_scenarios,
            climate_models_cfgs,
            spec["output_variables"],
            out_db_dir(species_key),
            max_processes=MAX_PROCESSES,
            batch_size_scenarios=BATCH_SIZE_SCENARIOS,
        )

# %% [markdown]
# ## Check

# %%
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

for species_key in EMISSIONS_BASED_SPECIES:
    output_db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=out_db_dir(species_key))
    result = output_db.load(out_columns_type=int)
    gsat = result.loc[result.index.get_level_values("variable") == "Surface Air Temperature Change"]
    last_year = gsat.columns.max()
    print(f"--- {species_key} ---")
    print(gsat.groupby(gsat.index.get_level_values("scenario"))[last_year].agg(["mean", "median"]))
