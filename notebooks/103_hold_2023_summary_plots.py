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
# # Cross-scenario summary plots of the "hold at 2023" basket attribution (100-series)
#
# Reads `101_prepare_counterfactuals.py`'s output. For each basket, the basket's own
# GSAT contribution at a given year is `GSAT(base scenario) - GSAT(basket held at 2023
# from 2023)`, i.e. the marginal warming actually caused by that basket's real
# post-2023 emissions trajectory (relative to it having stayed flat at 2023) - computed
# member-level first (`ac.compute_delta`, aligned by `run_id`), *then* median, same
# discipline as `006`. Plotted the same way as `006`'s
# `consistent_summary_burden_attribution_2050_2100.png` (one stacked bar group per
# scenario, sub-bars per year), just with seven much coarser baskets instead of 15
# forcing categories.

# %% [markdown]
# ## Imports

# %%
import json
from pathlib import Path

import attribution_common as ac
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB

# %% [markdown]
# ## Configuration

# %%
DATA_DIR = Path("../data")
"""Unembargoed - matches 101's own DATA_DIR."""

BASE_SCENARIOS = json.loads((DATA_DIR / "hold_2023_base_scenarios.json").read_text())
"""Written by 101_prepare_counterfactuals.py (a differently-named manifest than the
000-010 series' base_scenarios.json, since this is a separate, parallel db)."""

HOLD_YEAR = 2023
"""Must match 101's own HOLD_YEAR - the year every counterfactual starts diverging
from its base scenario, and so the natural baseline for the "real ΔGSAT" reference
marker below."""

YEARS_TO_PLOT = [2050, 2100]
REGION = ac.REGION

BASKETS = ["CO2", "CH4", "N2O", "F-Gases", "Montreal Halogens", "Aerosols", "Other"]
"""Aerosols = BC, OC, Sulfur, NH3; Other = NOx, CO, VOC (tropospheric-ozone-forming
precursors) - split from the former single "Residual" basket."""

PLOTS_DIR = Path("../data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "hold_2023_"

BUCKET_COLORS = {
    # Matches 006's FORCING_CATEGORIES colors where the basket is a 1:1 match (same
    # species, same color, everywhere). "Aerosols" reuses the fainter purple originally
    # picked for the combined "Residual" basket, before it was split into Aerosols/Other.
    # This insertion order sets the stack draw order (nearest zero = first) - the
    # legend's own order is set independently below via LEGEND_ORDER.
    "CO2": "#707070",
    "CH4": "LightSkyBlue",
    "N2O": "orange",
    "F-Gases": "blue",
    "Montreal Halogens": "cyan",
    "Aerosols": "#B19CD9",
    "Other": "yellow",
}

DISPLAY_LABELS = {"Other": "CO, VOC, NOx"}
"""Internal basket label -> the label shown on the plot - kept separate from BASKETS
itself so the internal key still matches the on-disk scenario/db naming
(`_hold_2023_other`, `hold_2023_scm_output_db_other`)."""

LEGEND_ORDER = ["CO, VOC, NOx", "Aerosols", "CO2", "CH4", "N2O", "F-Gases", "Montreal Halogens"]
"""Top-to-bottom legend order, in display labels - independent of BUCKET_COLORS'
own insertion order (which sets the stack draw order instead)."""


def basket_output_db_dir(label):
    return DATA_DIR / f"hold_2023_scm_output_db_{ac.slugify(label)}"


# %% [markdown]
# ## Load each basket's SCM output

# %%
basket_scm_output = {}
for label in BASKETS:
    df = OpenSCMDB(
        backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=basket_output_db_dir(label)
    ).load(out_columns_type=int)
    df.columns.name = "year"
    basket_scm_output[label] = df


# %% [markdown]
# ## Compute per-(scenario, year) basket ΔGSAT and reference markers
#
# Member-level discipline throughout: `ac.compute_delta` aligns base vs. counterfactual
# by `run_id` and subtracts *before* any median is taken; the "Sum of basket ΔGSAT"
# reference marker likewise sums the baskets' per-member deltas first, then takes
# the median of that sum - never `sum(median(a), median(b), ...)`.


# %%
def basket_delta_matrix(label, base_scenario):
    counterfactual_scenario = f"{base_scenario}_hold_2023_{ac.slugify(label)}"
    return ac.compute_delta(
        basket_scm_output[label], base_scenario, counterfactual_scenario, "Surface Air Temperature Change", region=REGION
    )


# %%
bucket_values = {label: {} for label in BASKETS}
reference_markers_values = {"Sum of basket ΔGSAT": {}}

for base_scenario in BASE_SCENARIOS:
    short_name = ac.scenario_short_name(base_scenario)
    delta_matrices = {label: basket_delta_matrix(label, base_scenario) for label in BASKETS}

    for year in YEARS_TO_PLOT:
        for label in BASKETS:
            bucket_values[label][(short_name, year)] = delta_matrices[label][year].median()

        summed = sum(delta_matrices[label][year] for label in BASKETS)
        reference_markers_values["Sum of basket ΔGSAT"][(short_name, year)] = summed.median()

# %% [markdown]
# ## Order-of-operations spot-check
#
# `median(sum(per-member basket deltas))` ("Sum of basket ΔGSAT" below) vs.
# `sum(median(per-member basket deltas))` (the bar height itself, i.e. each basket's
# delta median'd independently *then* summed). Median doesn't commute with summation,
# so these two aren't guaranteed to agree exactly - this checks that our
# member-level-first convention isn't quietly distorting the stacked bar. Not a check
# against the real scenario's own ΔGSAT (these are independent single-basket
# leave-one-out runs, one per basket, with no reason to sum to that - see `101`'s
# docstring on nonlinearity/committed warming).

# %%
for base_scenario in BASE_SCENARIOS:
    short_name = ac.scenario_short_name(base_scenario)
    for year in YEARS_TO_PLOT:
        summed = reference_markers_values["Sum of basket ΔGSAT"][(short_name, year)]
        stacked = sum(bucket_values[label][(short_name, year)] for label in BASKETS)
        print(f"{short_name} {year}: median(sum of baskets)={summed:+.4f} K, sum(median of baskets)={stacked:+.4f} K, diff={summed - stacked:+.4f} K")

# %% [markdown]
# ## Plot

# %%
scenario_order = [ac.scenario_short_name(s) for s in BASE_SCENARIOS]
scenario_labels = [ac.scenario_display_label(s) for s in BASE_SCENARIOS]

plot_bucket_values = {DISPLAY_LABELS.get(label, label): values for label, values in bucket_values.items()}
plot_bucket_colors = {DISPLAY_LABELS.get(label, label): color for label, color in BUCKET_COLORS.items()}

ac.plot_scenario_year_stacked_bars(
    bucket_values=plot_bucket_values,
    bucket_colors=plot_bucket_colors,
    reference_markers=[],
    scenario_order=scenario_order,
    scenario_labels=scenario_labels,
    years=YEARS_TO_PLOT,
    legend_order=LEGEND_ORDER,
    title=r"Decomposition of $\Delta$GSAT change vs a constant " + f"{HOLD_YEAR} emissions baseline by scenario",
    ylabel=r"$\Delta$GSAT change (°C)",
    out_path=PLOTS_DIR / f"{OUTPUT_PREFIX}summary_attribution_{'_'.join(str(y) for y in YEARS_TO_PLOT)}.png",
);

# %% [markdown]
# ## Export underlying data to CSV
#
# The full quantile distribution (not just the median shown in the bars) behind the
# plot above, member-level first (`basket_delta_matrix`, aligned by `run_id`) *then*
# quantile - same discipline as everywhere else in this project. Same 9-quantile
# convention as `006`'s own export.

# %%
import pandas as pd  # noqa: E402

EXPORT_QUANTILES = [0.05, 0.10, 1 / 6, 0.33, 0.50, 0.67, 5 / 6, 0.90, 0.95]

export_rows = []
for base_scenario in BASE_SCENARIOS:
    short_name = ac.scenario_short_name(base_scenario)
    for label in BASKETS:
        display_label = DISPLAY_LABELS.get(label, label)
        for year in YEARS_TO_PLOT:
            delta = basket_delta_matrix(label, base_scenario)[year]
            for q in EXPORT_QUANTILES:
                export_rows.append(
                    {
                        "scenario": short_name,
                        "basket": display_label,
                        "year": year,
                        "quantile": q,
                        "GSAT_change (°C)": delta.quantile(q),
                    }
                )

export_df = pd.DataFrame(export_rows)
export_path = PLOTS_DIR / f"{OUTPUT_PREFIX}summary_attribution.csv"
export_df.to_csv(export_path, index=False)
print(f"wrote {export_path} ({len(export_df)} rows)")
