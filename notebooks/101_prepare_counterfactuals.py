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
# # Prepare "hold at 2023" counterfactuals (100-series)
#
# Parallel workflow to `001_prepare_counterfactuals.py`, but with a different
# counterfactual: instead of leaving a species out entirely from preindustrial
# (`..._no_X_1750`), each counterfactual here **holds one basket of species' emissions
# constant at their real 2023 level from 2023 through 2100**, with every other species
# following the real marker scenario unchanged (`..._hold_2023_<basket>`).
#
# Baskets: CO2 (Fossil + AFOLU together), CH4, N2O, F-Gases, Montreal Protocol Halogen
# Gases (via `gcages`' `EMISSIONS_VARIABLES` table - see Step 2), and Residual
# (everything else: BC, OC, Sulfur, NOx, NH3, CO, VOC).
#
# Only builds and checks the counterfactual emissions themselves - running MAGICC on
# them is `102_run_magicc_hold_2023.py`'s job (kept separate so re-plotting/re-checking
# the emissions never requires re-running MAGICC, and vice versa).

# %% [markdown]
# ## Imports

# %%
import json
from pathlib import Path

import attribution_common as ac
import pandas as pd
from gcages.databases.emissions_variables import EMISSIONS_VARIABLES as GCAGES_EMISSIONS_VARIABLES
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.io import load_timeseries_csv

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = Path("../data")
"""Unembargoed - the full marker-scenario emissions (`input_files/emissions`) are no
longer under embargo, so this 100-series workflow writes straight to plain `data/`."""

MERGED_SCENARIOS_FILE = Path("../data/scenariomip_cmip7_unembargoed.csv")
"""Written by 100_merge_historical_and_future_scenarios.py."""

INDEX_COLUMNS = ["model", "scenario", "region", "variable", "unit"]

MARKERS_TO_RUN = list(ac.SCENARIO_METADATA)
"""Marker short codes (keys of `ac.SCENARIO_METADATA`) to process - all seven."""

HOLD_YEAR = 2023
"""Species/basket emissions are pinned to their own HOLD_YEAR value, held constant from
HOLD_YEAR through 2100. Also the year real future-scenario data begins (historical data
is merged up to, but not including, 2023 - see `100`)."""

OUT_COUNTERFACTUALS_DB_DIR = DATA_DIR / "hold_2023_scenarios_with_counterfactuals_db"

# %% [markdown]
# ## Step 1: load the merged historical+future scenarios, filter to the markers to run

# %%
combined = load_timeseries_csv(MERGED_SCENARIOS_FILE, index_columns=INDEX_COLUMNS, out_columns_type=int)
combined.columns.name = "year"
combined = combined.sort_index(axis="columns")

marker_scenario_names = [ac.SCENARIO_METADATA[m]["scenario"] for m in MARKERS_TO_RUN]
scenarios = combined.loc[combined.index.get_level_values("scenario").isin(marker_scenario_names)]

print("markers to run:", MARKERS_TO_RUN, "->", marker_scenario_names)
print("year range:", scenarios.columns.min(), "-", scenarios.columns.max())

# %% [markdown]
# ## Step 2: define the baskets
#
# CO2/CH4/N2O are single variables (CO2 combines Fossil + AFOLU, matching MAGICC's
# single `CO2_SWITCHFROMCONC2EMIS_YEAR`). F-Gases and Montreal Protocol Halogen Gases
# come from `gcages`' own `EMISSIONS_VARIABLES` table (its `rcmip` column groups species
# under `Emissions|F-Gases|...`/`Emissions|Montreal Gases|...`; its `cmip7_scenariomip`
# column gives the matching flat variable name used in our own scenario data) - the
# authoritative source for this grouping, rather than re-deriving it by regex. Residual
# is everything else actually present in the data (BC, OC, Sulfur, NOx, NH3, CO, VOC).


# %%
def fgas_and_mhalo_variables(variables):
    """Return (f_gas_vars, montreal_halogen_vars), each the subset of `variables`
    (cmip7_scenariomip-convention variable names) gcages' EMISSIONS_VARIABLES table
    groups under Emissions|F-Gases|... / Emissions|Montreal Gases|... respectively."""
    variables = set(variables)
    df = GCAGES_EMISSIONS_VARIABLES
    fgas = set(df.loc[df["rcmip"].str.startswith("Emissions|F-Gases|"), "cmip7_scenariomip"]) & variables
    mhalo = set(df.loc[df["rcmip"].str.startswith("Emissions|Montreal Gases|"), "cmip7_scenariomip"]) & variables
    return sorted(fgas), sorted(mhalo)


# %%
variables_present = set(scenarios.index.get_level_values("variable"))

co2_vars = [v for v in ["Emissions|CO2|Energy and Industrial Processes", "Emissions|CO2|AFOLU"] if v in variables_present]
ch4_vars = [v for v in ["Emissions|CH4"] if v in variables_present]
n2o_vars = [v for v in ["Emissions|N2O"] if v in variables_present]
fgas_vars, mhalo_vars = fgas_and_mhalo_variables(variables_present)

assigned = set(co2_vars) | set(ch4_vars) | set(n2o_vars) | set(fgas_vars) | set(mhalo_vars)
residual_vars = sorted(v for v in variables_present if v.startswith("Emissions|") and v not in assigned)

BASKETS = {
    "CO2": co2_vars,
    "CH4": ch4_vars,
    "N2O": n2o_vars,
    "F-Gases": fgas_vars,
    "Montreal Halogens": mhalo_vars,
    "Residual": residual_vars,
}
BASKETS = {label: species for label, species in BASKETS.items() if species}
BASKETS

# %% [markdown]
# ## Step 3: build the counterfactuals
#
# Each basket's own emissions are pinned to their HOLD_YEAR value from HOLD_YEAR
# onwards - unlike `001`, the pin value comes from the scenario's own data at HOLD_YEAR,
# not a separate preindustrial reference file (CO2 included: held at its real 2023
# level, not zeroed).


# %%
def make_hold_year_counterfactual(scenario_data, base_scenario, species, label, hold_year):
    """Return a copy of `scenario_data` with `species` held at its own hold_year value
    from hold_year onwards."""
    cf = scenario_data.copy()
    year_columns = [c for c in cf.columns if c >= hold_year]
    for variable in species:
        mask = cf.index.get_level_values("variable") == variable
        pin_value = cf.loc[mask, hold_year].iloc[0]
        cf.loc[mask, year_columns] = pin_value
    new_index = cf.index.to_frame(index=False)
    new_index["scenario"] = f"{base_scenario}_hold_2023_{ac.slugify(label)}"
    cf.index = pd.MultiIndex.from_frame(new_index)
    return cf


# %%
counterfactuals = [
    make_hold_year_counterfactual(
        scenarios.loc[scenarios.index.get_level_values("scenario") == base_scenario],
        base_scenario,
        species,
        label,
        HOLD_YEAR,
    )
    for label, species in BASKETS.items()
    for base_scenario in scenarios.index.get_level_values("scenario").unique()
]

scenarios_with_counterfactuals = pd.concat([scenarios, *counterfactuals])

# %% [markdown]
# ## Save counterfactual emissions

# %%
db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=OUT_COUNTERFACTUALS_DB_DIR)
db.save(scenarios_with_counterfactuals, groupby=["scenario"], allow_overwrite=True)

base_scenarios = sorted(scenarios.index.get_level_values("scenario").unique())
BASE_SCENARIOS_FILE = DATA_DIR / "hold_2023_base_scenarios.json"
BASE_SCENARIOS_FILE.write_text(json.dumps(base_scenarios, indent=2))

# %% [markdown]
# ## Check: counterfactual emissions time series, all six baskets
#
# One multi-panel figure per base scenario - one panel per basket, each plotting every
# variable in that basket twice: solid for the real scenario, dashed for its
# `_hold_2023_<basket>` counterfactual (same color per variable across the two lines).
# Large baskets (F-Gases/Montreal Halogens/Residual) skip the per-variable legend - would
# be unreadable - and just show the solid-vs-dashed convention once, via `LEGEND_MAX_SPECIES`.

# %%
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PLOTS_DIR = Path("../data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_XLIM = (1950, 2100)
LEGEND_MAX_SPECIES = 6
"""Baskets with more variables than this skip the per-variable legend."""


def plot_basket_timeseries(base_scenario):
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)

    for ax, (label, species) in zip(axes.flat, BASKETS.items()):
        counterfactual_scenario = f"{base_scenario}_hold_2023_{ac.slugify(label)}"

        for i, variable in enumerate(species):
            color = color_cycle[i % len(color_cycle)]
            base_row = scenarios_with_counterfactuals.loc[
                (scenarios_with_counterfactuals.index.get_level_values("scenario") == base_scenario)
                & (scenarios_with_counterfactuals.index.get_level_values("variable") == variable)
            ].iloc[0]
            cf_row = scenarios_with_counterfactuals.loc[
                (scenarios_with_counterfactuals.index.get_level_values("scenario") == counterfactual_scenario)
                & (scenarios_with_counterfactuals.index.get_level_values("variable") == variable)
            ].iloc[0]

            varname = variable.split("|")[-1]
            plot_label = varname if len(species) <= LEGEND_MAX_SPECIES else None
            ax.plot(base_row.index, base_row.values, color=color, linestyle="-", linewidth=1.2, label=plot_label)
            ax.plot(cf_row.index, cf_row.values, color=color, linestyle="--", linewidth=1.2)

        ax.axvline(HOLD_YEAR, color="grey", linestyle=":", linewidth=0.8)
        ax.set_xlim(*TIMESERIES_XLIM)
        ax.set_title(label)
        ax.set_xlabel("year")
        ylabel = "emissions"
        if len(species) == 1:
            unit_mask = (scenarios_with_counterfactuals.index.get_level_values("scenario") == base_scenario) & (
                scenarios_with_counterfactuals.index.get_level_values("variable") == species[0]
            )
            ylabel = scenarios_with_counterfactuals.loc[unit_mask].index.get_level_values("unit")[0]
        ax.set_ylabel(ylabel)
        if len(species) <= LEGEND_MAX_SPECIES:
            ax.legend(fontsize=7, loc="best")

    style_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="original"),
        Line2D([0], [0], color="black", linestyle="--", label="hold 2023"),
    ]
    fig.legend(handles=style_handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Counterfactual emissions check: {ac.scenario_display_label(base_scenario)}", y=1.1)

    out_path = PLOTS_DIR / f"hold_2023_counterfactual_timeseries_{ac.scenario_short_name(base_scenario)}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


for base_scenario in base_scenarios:
    plot_basket_timeseries(base_scenario)
