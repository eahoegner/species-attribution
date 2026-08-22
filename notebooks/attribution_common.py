"""Shared helpers for the production pipeline (000-010): MAGICC config/run wrappers,
GSAT rebasing, and plotting utilities. A plain module, not a jupytext notebook - import
from it, don't execute it directly.
"""

import copy
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from gcages.ar6.post_processing import get_temperatures_in_line_with_assessment
from gcages.renaming import SupportedNamingConventions, convert_variable_name
from gcages.scm_running import run_scms
from pandas_openscm.db import FeatherDataBackend, FeatherIndexBackend, OpenSCMDB
from pandas_openscm.index_manipulation import update_index_levels_func

# ---------------------------------------------------------------------------
# Shared MAGICC install / ensemble constants
# ---------------------------------------------------------------------------

MAGICC_VERSION = "v7.6.0a3"
MAGICC_DIR = Path("/Users/hoegner/Projects/magicc/magicc-v7.6.0a3")
MAGICC_EXECUTABLE_PATH = MAGICC_DIR / "bin" / "magicc-darwin-arm64"
MAGICC_PROB_DISTRIBUTION_PATH = MAGICC_DIR / "configs" / "magicc-ar7-fast-track-drawnset-v0-3-0.json"
MAGICC_START_YEAR = 1750
CLIMATE_MODEL_NAME = f"MAGICC{MAGICC_VERSION}"

DB_GROUPBY_COLUMNS = ["model", "scenario", "variable"]
REGION = "World"

# Diverging blue<->red pair (cooling<->warming), reference palette default.
COLOR_WARMING = "#e34948"
COLOR_COOLING = "#2a78d6"
COLOR_PRIMARY_TEXT = "#0b0b0b"
COLOR_SECONDARY_TEXT = "#52514e"
COLOR_MUTED = "#898781"
COLOR_GRIDLINE = "#e1e0d9"
COLOR_BASELINE = "#c3c2b7"
COLOR_SURFACE = "#ffffff"

# Categorical palette, fixed order (light-mode only - every plot in this project is a
# static matplotlib PNG, no dark-mode variant needed). This exact order is validated for
# adjacent-pair CVD safety in stacked bars/lines specifically (dataviz skill,
# references/palette.md) - assign slots in this order, never re-cycle or reorder per
# category/scenario. Slot 8 (red) is reserved/unused by default to avoid colliding with
# COLOR_WARMING above.
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]


# ---------------------------------------------------------------------------
# DB / scenario loading
# ---------------------------------------------------------------------------


def assign_level(df, **levels):
    """Return a copy of `df` with the given index level(s) set to a constant value."""
    new_index = df.index.to_frame(index=False)
    for level, value in levels.items():
        new_index[level] = value
    df = df.copy()
    df.index = type(df.index).from_frame(new_index)
    return df


class GroupedSaveDB:
    """Wraps an OpenSCMDB so `.save()` always groups/overwrites consistently."""

    def __init__(self, db, groupby=DB_GROUPBY_COLUMNS):
        self._db = db
        self._groupby = groupby

    def load_metadata(self, *args, **kwargs):
        return self._db.load_metadata(*args, **kwargs)

    def save(self, *args, **kwargs):
        return self._db.save(*args, **kwargs, groupby=self._groupby, allow_overwrite=True)


def load_base_scenarios(data_dir, fallback=("historical",)):
    """Read the list of base scenarios (as opposed to counterfactuals) from the manifest
    001 writes to `data_dir/base_scenarios.json`. Falls back to `fallback` if the
    manifest doesn't exist."""
    manifest = Path(data_dir) / "base_scenarios.json"
    if manifest.exists():
        return json.loads(manifest.read_text())
    return list(fallback)


SCENARIO_METADATA = {
    "vl": {"model": "REMIND-MAgPIE 3.5-4.11", "scenario": "SSP1 - Very Low Emissions", "version": 5},
    "ln": {"model": "AIM 3.0", "scenario": "SSP2 - Low Overshoot_a", "version": 23},
    "l": {"model": "MESSAGEix-GLOBIOM-GAINS 2.1-M-R12", "scenario": "SSP2 - Low Emissions", "version": 21},
    "ml": {"model": "COFFEE 1.6", "scenario": "SSP2 - Medium-Low Emissions", "version": 14},
    "m": {"model": "IMAGE 3.4", "scenario": "SSP2 - Medium Emissions", "version": 25},
    "hl": {"model": "WITCH 6.0", "scenario": "SSP5 - Medium-Low Emissions_a", "version": 32},
    "h": {"model": "GCAM 8s", "scenario": "SSP3 - High Emissions", "version": 3},
}
"""Marker-scenario metadata (model, long scenario name, version), keyed by short code."""

SCENARIO_LONG_TO_SHORT = {v["scenario"]: k for k, v in SCENARIO_METADATA.items()}


def scenario_short_name(base_scenario):
    """Human-readable short code for a long scenario name, for print/plot text only -
    never for constructing scenario-name keys. Falls back to the long name itself if not
    found in `SCENARIO_METADATA`."""
    return SCENARIO_LONG_TO_SHORT.get(base_scenario, base_scenario)


def load_scenarios(scenario_names, scenarios_db_dir):
    """Load the given scenario names from an OpenSCMDB, renamed to the OpenSCM-Runner
    variable convention (from CMIP7_SCENARIOMIP). Returns the full 1750-2100 range -
    callers that want to supply MAGICC only a later window (e.g. matching the official
    CMIP7 ScenarioMIP workflow's 2015-2100 convention) should slice the result
    themselves; see 002's/004's own `MAGICC_SUPPLY_START_YEAR`."""
    scenarios_db = OpenSCMDB(
        backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=scenarios_db_dir
    )
    complete = scenarios_db.load(out_columns_type=int)
    complete.columns.name = "year"
    complete = complete.loc[complete.index.get_level_values("scenario").isin(scenario_names)]
    return update_index_levels_func(
        complete,
        {
            "variable": lambda v: convert_variable_name(
                v,
                from_convention=SupportedNamingConventions.CMIP7_SCENARIOMIP,
                to_convention=SupportedNamingConventions.OPENSCM_RUNNER,
            )
        },
    )


def load_magicc_cfgs(prob_distribution_path=MAGICC_PROB_DISTRIBUTION_PATH, startyear=MAGICC_START_YEAR, n_members=None, switches=None, overrides=None):
    """Build the MAGICC7 ensemble config list.

    `switches`: list of MAGICC switch keys (e.g. ["CH4_SWITCHFROMCONC2EMIS_YEAR"]) set to
    `startyear` for every member - use to move a species to emissions-driven from the
    start of the run. `overrides`: extra cfg dict applied on top of the switches (e.g.
    the CH4/N2O budget-closure re-anchoring in 004). Both optional; omit for a
    fully-default-config run (the burden-based pipeline, 002).

    `out_ascii_binary`/`out_binary_format` are set explicitly, matching the official
    ScenarioMIP/AR7 `gcages.cmip7_scenariomip.scm_running.load_magicc_cfgs`'s
    `common_cfg`, rather than left at MAGICC's own compiled default."""
    with open(prob_distribution_path) as f:
        distribution = json.load(f)
    cfgs = {
        "MAGICC7": [
            {**member["nml_allcfgs"], "startyear": startyear, "out_ascii_binary": "BINARY", "out_binary_format": 2}
            for member in distribution["configurations"]
        ]
    }
    if n_members is not None:
        cfgs["MAGICC7"] = cfgs["MAGICC7"][:n_members]
    for cfg in cfgs["MAGICC7"]:
        for switch in switches or []:
            cfg[switch] = startyear
        if overrides:
            cfg.update(overrides)
    return cfgs


def run_scms_to_db(
    scenarios_osr,
    scenario_names,
    climate_models_cfgs,
    output_variables,
    out_db_dir,
    max_processes=5,
    batch_size_scenarios=15,
    save_scenarios=True,
):
    """Run MAGICC on each of `scenario_names` (one run_scms call per scenario, to keep
    run_id 0-based per scenario - no cross-scenario batching ambiguity to recover from
    later) and save to `out_db_dir`. Returns the OpenSCMDB."""
    output_db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=out_db_dir)
    grouped_output_db = GroupedSaveDB(output_db)

    for scenario_name in scenario_names:
        scenario_data = scenarios_osr.loc[scenarios_osr.index.get_level_values("scenario") == scenario_name]
        if scenario_data.empty:
            raise ValueError(f"No scenario data found for {scenario_name!r}")
        run_scms(
            scenarios=scenario_data,
            climate_models_cfgs=climate_models_cfgs,
            output_variables=output_variables,
            scenario_group_levels=["model", "scenario"],
            n_processes=max_processes,
            db=grouped_output_db,
            verbose=True,
            progress=True,
            batch_size_scenarios=batch_size_scenarios,
            force_rerun=True,
        )

    if save_scenarios:
        output_db.save(
            assign_level(scenarios_osr, climate_model=CLIMATE_MODEL_NAME),
            groupby=DB_GROUPBY_COLUMNS,
            allow_overwrite=True,
        )
    return output_db


# ---------------------------------------------------------------------------
# ERF delta / QEXTRA forcing-channel machinery
# ---------------------------------------------------------------------------


def slugify(label):
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def write_magicc_extra_rf_file(path, years, values, description, source_name="attribution_common.py"):
    """Write `values` at `years` as a MAGICC FILE_EXTRA_RF-compatible input file."""
    header_lines = [
        "---- HEADER ----",
        "",
        description,
        "",
        "---- METADATA ----",
        "",
        f"source: {source_name}",
        "",
    ]
    spec_lines = [
        "&THISFILE_SPECIFICATIONS",
        "    THISFILE_DATACOLUMNS = 1",
        f"    THISFILE_DATAROWS = {len(years)}",
        f"    THISFILE_FIRSTYEAR = {years[0]}",
        f"    THISFILE_LASTYEAR = {years[-1]}",
        "    THISFILE_ANNUALSTEPS = 1",
        "    THISFILE_UNITS = 'Wpermsuper2'",
        "    THISFILE_DATTYPE = 'NOTUSED'",
        "    THISFILE_REGIONMODE = 'FOURBOX'",
        "    THISFILE_FIRSTDATAROW = PLACEHOLDER",
        "/",
        "",
    ]
    table_header = [
        "   VARIABLE            EXTRA_RF",
        "       TODO                 SET",
        "      UNITS         Wpermsuper2",
        "      YEARS               WORLD",
    ]
    first_data_row = len(header_lines) + len(spec_lines) + len(table_header) + 1
    spec_lines = [line.replace("PLACEHOLDER", str(first_data_row)) for line in spec_lines]

    lines = [*header_lines, *spec_lines, *table_header]
    lines += [f"      {year:6d}    {value:12.6e}" for year, value in zip(years, values)]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_erf(df, scenario, variable, region=REGION):
    """Load one (scenario, variable) timeseries from an already-loaded scm output
    DataFrame `df`, indexed by ensemble member (run_id). Assumes one run_scms call per
    scenario (see `run_scms_to_db`), so run_id is already 0-based - no physical-member
    recovery needed."""
    mask = (
        (df.index.get_level_values("scenario") == scenario)
        & (df.index.get_level_values("variable") == variable)
        & (df.index.get_level_values("region") == region)
    )
    sub = df.loc[mask]
    if sub.empty:
        raise ValueError(f"No data found for scenario={scenario!r}, variable={variable!r}, region={region!r}")
    sub = sub.reset_index().set_index("run_id")
    year_columns = [c for c in sub.columns if isinstance(c, int)]
    return sub[year_columns].sort_index()


def compute_delta(df, base_scenario, counterfactual_scenario, variable, region=REGION):
    """Return ERF(base_scenario) - ERF(counterfactual_scenario) for `variable`, aligned
    on ensemble member."""
    base = load_erf(df, base_scenario, variable, region=region)
    counterfactual = load_erf(df, counterfactual_scenario, variable, region=region)
    common_members = base.index.intersection(counterfactual.index)
    if len(common_members) < max(len(base.index), len(counterfactual.index)):
        raise ValueError(
            f"{base_scenario}, {variable}: baseline has {len(base.index)} members, counterfactual "
            f"has {len(counterfactual.index)} - expected identical ensembles for both."
        )
    return base.loc[common_members].sort_index() - counterfactual.loc[common_members].sort_index()


def run_qextra_channels(
    scenarios_osr,
    driving_scenario_name,
    channel_series,
    combined_label,
    climate_models_cfgs,
    forcing_files_dir,
    out_db_dir,
    max_processes=5,
    emissions_source_scenario=None,
):
    """Write per-member FILE_EXTRA_RF input files for each channel in `channel_series`
    (dict: label -> DataFrame indexed by member, columns=years) plus their sum
    (`combined_label`), then run MAGICC's climate module on each via QEXTRA. The
    resulting scenarios are named f"{driving_scenario_name}_forcing_only_{category}".
    `emissions_source_scenario` (defaults to `driving_scenario_name`) only supplies an
    Emissions| skeleton to drive MAGICC with - QEXTRA overrides the forcing regardless,
    so it need not match `driving_scenario_name`. Returns the output OpenSCMDB."""
    emissions_source_scenario = emissions_source_scenario or driving_scenario_name
    series = dict(channel_series)
    series[combined_label] = sum(series.values())

    manifest_rows = []
    for label, member_df in series.items():
        category_dir = forcing_files_dir / slugify(label)
        for member, row in member_df.iterrows():
            path = category_dir / f"member_{member}.IN"
            write_magicc_extra_rf_file(
                path,
                years=list(row.index),
                values=list(row.values),
                description=f"{label}: {driving_scenario_name}, ensemble member {member}",
            )
            manifest_rows.append({"category": label, "member": member, "path": str(path.resolve())})
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(forcing_files_dir / "manifest.csv", index=False)

    output_db = OpenSCMDB(backend_data=FeatherDataBackend(), backend_index=FeatherIndexBackend(), db_dir=out_db_dir)
    grouped_output_db = GroupedSaveDB(output_db)

    driving_scenario = scenarios_osr.loc[
        (scenarios_osr.index.get_level_values("scenario") == emissions_source_scenario)
        & (scenarios_osr.index.get_level_values("variable").str.startswith("Emissions|"))
    ]

    categories = sorted(series.keys(), key=lambda c: (c == combined_label, c))
    for category in categories:
        category_paths = manifest.loc[manifest["category"] == category].set_index("member")["path"].sort_index()

        cfgs_for_category = copy.deepcopy(climate_models_cfgs)
        cfgs_for_category["MAGICC7"] = cfgs_for_category["MAGICC7"][: len(category_paths)]
        for member, cfg in enumerate(cfgs_for_category["MAGICC7"]):
            cfg["file_extra_rf"] = category_paths.loc[member]
            cfg["rf_extra_read"] = 1
            cfg["rf_total_runmodus"] = "QEXTRA"

        driving = assign_level(driving_scenario, scenario=f"{driving_scenario_name}_forcing_only_{category}")
        run_scms(
            scenarios=driving,
            climate_models_cfgs=cfgs_for_category,
            output_variables=("Surface Air Temperature Change", "Effective Radiative Forcing"),
            scenario_group_levels=["model", "scenario"],
            n_processes=max_processes,
            db=grouped_output_db,
            verbose=True,
            progress=True,
            force_rerun=True,
        )

    return output_db


# ---------------------------------------------------------------------------
# Summary statistics and plotting
# ---------------------------------------------------------------------------


def member_series(df, year):
    """Return `df[year]` indexed by run_id (one value per ensemble member)."""
    return df.reset_index().set_index("run_id")[year].sort_index()


GSAT_REBASE_PERIOD = (1850, 1900)
"""IPCC AR6-standard GSAT pre-industrial reference period."""

GSAT_ASSESSMENT_TIME_PERIOD = (1995, 2014)
GSAT_ASSESSMENT_MEDIAN = 0.85
"""AR6-assessed median warming for 1995-2014 relative to 1850-1900 (Cross-Chapter Box
2.3) - the calibration target `rebase_gsat_matrix` matches every ensemble to."""

_GSAT_REBASE_GROUP_LEVEL = "_gsat_rebase_group"


def rebase_gsat_matrix(matrix, period=GSAT_REBASE_PERIOD):
    """Put this matrix (run_id-indexed, year columns) "in line with the historical
    assessment", via `gcages`' `get_temperatures_in_line_with_assessment`. Two steps:
    (1) subtract this matrix's own per-member mean over `period` (the standard GSAT
    anomaly convention); (2) shift the entire ensemble by one constant so its own median
    warming over `GSAT_ASSESSMENT_TIME_PERIOD` matches `GSAT_ASSESSMENT_MEDIAN` exactly.

    Use this only on standalone/absolute GSAT quantities (a Total GSAT matrix, or a
    channel's own QEXTRA-derived GSAT contribution matrix) - never on leave-one-out
    deltas (e.g. from `compute_delta`): those are already differential and
    baseline-invariant, and rebasing one side of a delta independently would introduce a
    spurious, non-cancelling shift."""
    tagged = matrix.copy()
    tagged.index = pd.MultiIndex.from_arrays(
        [tagged.index, [0] * len(tagged)], names=["run_id", _GSAT_REBASE_GROUP_LEVEL]
    )
    calibrated = get_temperatures_in_line_with_assessment(
        tagged,
        assessment_median=GSAT_ASSESSMENT_MEDIAN,
        assessment_time_period=range(GSAT_ASSESSMENT_TIME_PERIOD[0], GSAT_ASSESSMENT_TIME_PERIOD[1] + 1),
        assessment_pre_industrial_period=range(period[0], period[1] + 1),
        group_cols=[_GSAT_REBASE_GROUP_LEVEL],
    )
    return calibrated.droplevel(_GSAT_REBASE_GROUP_LEVEL)


def gsat_assessment_shift(total_matrix, n_channels=1, period=GSAT_REBASE_PERIOD):
    """Compute the single constant `rebase_gsat_matrix` adds to `total_matrix` (a
    scenario's real, all-forcings-together GSAT) to bring its own ensemble median
    1995-2014 warming to `GSAT_ASSESSMENT_MEDIAN` - exposed separately so the exact same
    shift can be reused for that scenario's individual QEXTRA forcing-channel GSAT
    matrices via `rebase_gsat_channel_matrix`, instead of each channel computing its own,
    independently-wrong, shift.

    `n_channels`: if this shift is going to be split across `n_channels` forcing
    channels (each via `rebase_gsat_channel_matrix`, summed afterward), the returned
    value is pre-divided by `n_channels` - so summing `n_channels` channels each given
    this same returned value reproduces the total's own single, undivided shift, not
    `n_channels` times too much. Leave at the default 1 when the shift is for the total
    itself."""
    baseline_only = total_matrix.sub(total_matrix[list(range(period[0], period[1] + 1))].mean(axis=1), axis=0)
    calibrated = rebase_gsat_matrix(total_matrix, period=period)
    raw_shift = (calibrated - baseline_only).stack().median()
    return raw_shift / n_channels


def rebase_gsat_channel_matrix(channel_matrix, shift, period=GSAT_REBASE_PERIOD):
    """Rebase one QEXTRA forcing channel's own GSAT matrix to the IPCC-standard
    1850-1900 reference period, using a `shift` shared across every channel in the same
    scenario (from `gsat_assessment_shift(total_matrix, n_channels=<how many channels
    this shift will be split across>)`) rather than independently recalibrating this one
    channel to the full `GSAT_ASSESSMENT_MEDIAN` on its own - independent recalibration
    breaks additivity, since most individual channels' own 1995-2014 median warming is
    far below the full AR6-assessed total. The `shift` passed in here must already be
    divided by however many channels it will be summed across
    (`gsat_assessment_shift`'s own `n_channels` argument handles that division): summing
    `n_channels` channels' own `rebase_gsat_channel_matrix` results then reproduces
    `rebase_gsat_matrix(total_matrix)` up to a small QEXTRA non-additivity residual."""
    baseline_only = channel_matrix.sub(channel_matrix[list(range(period[0], period[1] + 1))].mean(axis=1), axis=0)
    return baseline_only + shift


def distribution_summary(series):
    return pd.Series(
        {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "p5": series.quantile(0.05),
            "p17": series.quantile(0.17),
            "p83": series.quantile(0.83),
            "p95": series.quantile(0.95),
        }
    )


def plot_category_bars(values, lower_err, upper_err, title, xlabel, out_path, bucket_colors=None, order=None, show_legend=True):
    """Horizontal bar chart, one bar per category, sorted with the largest positive
    value at the top by default (AR6 WGI Fig 7.6/7.7-style summary bar chart). `values`
    is a pd.Series indexed by category; `lower_err`/`upper_err` are None or
    same-shaped Series for asymmetric error bars.

    `bucket_colors`: optional dict[category -> color] - when given, bars use these
    categorical colors instead of the default sign-based warming/cooling coloring.
    Categories not present in the dict fall back to a neutral gray.
    `order`: optional explicit bottom-to-top category order, overriding the default
    value-sorted order.
    `show_legend`: set False to skip the color legend when `bucket_colors` is given."""
    if order is None:
        order = values.sort_values(ascending=True).index
    else:
        order = pd.Index(order)
    values = values.loc[order]
    if bucket_colors is not None:
        colors = [bucket_colors.get(c, COLOR_MUTED) for c in values.index]
    else:
        colors = [COLOR_WARMING if v >= 0 else COLOR_COOLING for v in values]

    fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    xerr = None
    if lower_err is not None:
        xerr = [lower_err.loc[order].to_numpy(), upper_err.loc[order].to_numpy()]

    bar_kwargs = {"edgecolor": "black", "linewidth": 0.6} if bucket_colors is not None else {}
    ax.barh(
        values.index,
        values.to_numpy(),
        xerr=xerr,
        color=colors,
        height=0.62,
        ecolor=COLOR_SECONDARY_TEXT,
        capsize=3,
        error_kw={"elinewidth": 1.2, "capthick": 1.2},
        **bar_kwargs,
    )

    if bucket_colors is not None and show_legend:
        # `order` is ascending (smallest at the bottom of the chart) - reverse so the
        # legend reads top-to-bottom in the same order the bars appear, largest first.
        legend_order = list(reversed(order))
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=bucket_colors.get(c, COLOR_MUTED), edgecolor="black", linewidth=0.6) for c in legend_order]
        ax.legend(handles, legend_order, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True)

    ax.axvline(0, color=COLOR_BASELINE, linewidth=1.0)
    ax.xaxis.grid(True, color=COLOR_GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    if bucket_colors is not None:
        # Full black box outline, matching plot_scenario_year_stacked_bars' aesthetic
        # (that function leaves matplotlib's default black spines untouched) - the
        # sign-colored default mode below keeps its own lighter, axis-only styling.
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
    else:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(axis="y", colors=COLOR_PRIMARY_TEXT, length=0)
    ax.tick_params(axis="x", colors=COLOR_MUTED)
    ax.set_xlabel(xlabel, color=COLOR_SECONDARY_TEXT)
    ax.set_title(title, color=COLOR_PRIMARY_TEXT, fontsize=12, pad=12)

    lo_ext = (values - lower_err.loc[order]) if lower_err is not None else values
    hi_ext = (values + upper_err.loc[order]) if upper_err is not None else values
    data_min = min(0.0, float(lo_ext.min()))
    data_max = max(0.0, float(hi_ext.max()))
    span = data_max - data_min
    ax.set_xlim(data_min - 0.22 * span, data_max + 0.22 * span)

    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    pad = 0.015 * x_range
    for y, v in enumerate(values.to_numpy()):
        tip = v + (upper_err.loc[order[y]] if upper_err is not None and v >= 0 else 0)
        tip = v - (lower_err.loc[order[y]] if lower_err is not None and v < 0 else 0) if v < 0 else tip
        label_x = tip + pad if v >= 0 else tip - pad
        ha = "left" if v >= 0 else "right"
        ax.text(label_x, y, f"{v:+.3f}", va="center", ha=ha, fontsize=8.5, color=COLOR_SECONDARY_TEXT)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE, bbox_inches="tight")
    return fig, ax


def _draw_grouped_year_bars(ax, values_by_year, lower_err_by_year, upper_err_by_year, years, order, xlabel, bucket_colors=None, hatch="....", bar_height_frac=0.72):
    """Draw one panel's worth of "thinner sub-bar per year, stacked within each
    category's own row" horizontal bars onto `ax` - the shared drawing routine behind
    `plot_category_bars_two_metrics` (two such panels side by side, e.g. ERF | GSAT,
    each with its own x-scale/xlabel but sharing one category order/y-axis). See that
    function for the full parameter semantics."""
    n_years = len(years)
    category_positions = list(range(len(order)))
    sub_height = 0.7 / n_years

    def err_for(year):
        lower_err = lower_err_by_year.get(year) if lower_err_by_year is not None else None
        upper_err = upper_err_by_year.get(year) if upper_err_by_year is not None else None
        return lower_err, upper_err

    data_min, data_max = 0.0, 0.0
    for year in years:
        values = values_by_year[year].loc[order]
        lower_err, upper_err = err_for(year)
        lo_ext = (values - lower_err.loc[order]) if lower_err is not None else values
        hi_ext = (values + upper_err.loc[order]) if upper_err is not None else values
        data_min = min(data_min, float(lo_ext.min()))
        data_max = max(data_max, float(hi_ext.max()))
    span = data_max - data_min
    xlim = (data_min - 0.22 * span, data_max + 0.22 * span)
    x_range = xlim[1] - xlim[0]
    pad = 0.015 * x_range

    for i, year in enumerate(years):
        values = values_by_year[year].loc[order]
        lower_err, upper_err = err_for(year)
        offset = (i - (n_years - 1) / 2) * sub_height
        y_positions = [p + offset for p in category_positions]

        if bucket_colors is not None:
            bar_colors = [bucket_colors.get(c, COLOR_MUTED) for c in values.index]
        else:
            bar_colors = [COLOR_WARMING if v >= 0 else COLOR_COOLING for v in values]

        xerr = None
        if lower_err is not None:
            xerr = [lower_err.loc[order].to_numpy(), upper_err.loc[order].to_numpy()]

        bar_kwargs = {"edgecolor": "black", "linewidth": 0.6} if bucket_colors is not None else {}
        if i != n_years - 1:
            bar_kwargs["hatch"] = hatch

        ax.barh(
            y_positions,
            values.to_numpy(),
            xerr=xerr,
            color=bar_colors,
            height=sub_height * bar_height_frac,
            ecolor=COLOR_SECONDARY_TEXT,
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
            **bar_kwargs,
        )

        for pos, cat, v in zip(y_positions, order, values.to_numpy()):
            lo = lower_err.loc[cat] if lower_err is not None else 0
            hi = upper_err.loc[cat] if upper_err is not None else 0
            tip = v + hi if v >= 0 else v - lo
            label_x = tip + pad if v >= 0 else tip - pad
            ha = "left" if v >= 0 else "right"
            ax.text(label_x, pos, f"{v:+.3f}", va="center", ha=ha, fontsize=7.5, color=COLOR_SECONDARY_TEXT)

    ax.axvline(0, color=COLOR_BASELINE, linewidth=1.0)
    ax.xaxis.grid(True, color=COLOR_GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    if bucket_colors is not None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
    else:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.tick_params(axis="y", colors=COLOR_PRIMARY_TEXT, length=0)
    ax.tick_params(axis="x", colors=COLOR_MUTED)
    ax.set_xlabel(xlabel, color=COLOR_SECONDARY_TEXT)
    ax.set_xlim(xlim)


def _year_legend_handles(n_years, hatch):
    return [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_MUTED, edgecolor="black", linewidth=0.6, hatch=hatch if i != n_years - 1 else None)
        for i in range(n_years)
    ]


def plot_category_bars_two_metrics(
    panels, years, title, out_path,
    bucket_colors=None, order=None, show_legend=True, hatch="....", row_height=0.42, bar_height_frac=0.72, panel_width=5.78,
):
    """Two `_draw_grouped_year_bars`-style panels side by side (a thinner sub-bar per
    year, stacked within each category's own row), sharing one category order/y-axis -
    e.g. ERF on the left, GSAT on the right, each with its own independently-scaled
    x-axis.

    `panels`: list of dicts, one per panel, left to right - each
    `{"values_by_year", "lower_err_by_year", "upper_err_by_year", "xlabel"}`, dict[year
    -> pd.Series] for the first three keys.
    `years`: bottom-to-top sub-bar order within each category's row, shared across both
    panels.
    `order`: shared category order (bottom-to-top) - defaults to sorting by the last
    year of the last panel.
    `show_legend`: the solid-vs-hatched year legend, drawn once inside the leftmost
    panel's upper-left corner."""
    if order is None:
        last_panel = panels[-1]
        order = last_panel["values_by_year"][years[-1]].sort_values(ascending=True).index
    else:
        order = pd.Index(order)

    fig, axes = plt.subplots(
        1, len(panels), figsize=(panel_width * len(panels), 1.0 + row_height * len(order)), facecolor=COLOR_SURFACE, sharey=True
    )
    if len(panels) == 1:
        axes = [axes]

    for ax, panel in zip(axes, panels):
        ax.set_facecolor(COLOR_SURFACE)
        _draw_grouped_year_bars(
            ax, panel["values_by_year"], panel["lower_err_by_year"], panel["upper_err_by_year"], years, order,
            panel["xlabel"], bucket_colors, hatch, bar_height_frac,
        )

    axes[0].set_yticks(list(range(len(order))))
    axes[0].set_yticklabels(order)
    axes[0].set_ylim(-0.5, len(order) - 0.5)

    if show_legend:
        n_years = len(years)
        handles = _year_legend_handles(n_years, hatch)
        axes[0].legend(
            list(reversed(handles)), [str(y) for y in reversed(years)],
            loc="upper left", fontsize=9, frameon=True, title="Year",
        )

    fig.suptitle(title, color=COLOR_PRIMARY_TEXT, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE, bbox_inches="tight")
    return fig, axes


def plot_scenario_year_stacked_bars(
    bucket_values,
    bucket_colors,
    reference_markers,
    scenario_order,
    years,
    title,
    ylabel,
    out_path,
    bar_width=0.6,
    within_gap=0.15,
    group_spacing=0.6,
    ylim=None,
):
    """Condensed cross-scenario summary plot: one bar group per scenario, one stacked
    bar per year within each group, plus overlay reference markers (e.g. a directly-run
    total vs. the sum of the stacked pieces, as an additivity check).

    `bucket_values`: dict[bucket_label -> dict[(scenario, year) -> float]] - already
    member-level-aggregated by the caller (sum same-member series across the bucket's
    underlying categories, *then* take median/mean).
    `bucket_colors`: dict[bucket_label -> hex color], same keys as `bucket_values`, in
    the fixed categorical order to render (see CATEGORICAL_PALETTE).
    `reference_markers`: list of dicts, each `{"label", "values": dict[(scenario, year)
    -> float], "marker", "facecolor"/"color", "edgecolor"}` - drawn as overlay scatter
    points, distinguished by marker shape.
    `scenario_order`/`years`: left-to-right ordering for groups/sub-bars. Pass
    already-display-ready labels in `scenario_order` for the x-axis.
    `ylim`: optional (ymin, ymax) to use verbatim instead of the auto-padded autoscale."""
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(scenario_order)), 5), constrained_layout=True)

    n_years = len(years)
    step = bar_width + within_gap
    group_span = n_years * bar_width + (n_years - 1) * within_gap

    x_positions = {}
    for i, scenario in enumerate(scenario_order):
        for j, year in enumerate(years):
            x_positions[(scenario, year)] = i * (group_span + group_spacing) + j * step

    bottom_pos = {(s, y): 0.0 for s in scenario_order for y in years}
    bottom_neg = {(s, y): 0.0 for s in scenario_order for y in years}

    for bucket_label, colour in bucket_colors.items():
        values = bucket_values[bucket_label]
        xs, pos_vals, neg_vals, bottoms_pos, bottoms_neg = [], [], [], [], []
        for s in scenario_order:
            for y in years:
                v = values.get((s, y), 0.0)
                xs.append(x_positions[(s, y)])
                pos_vals.append(max(v, 0.0))
                neg_vals.append(min(v, 0.0))
                bottoms_pos.append(bottom_pos[(s, y)])
                bottoms_neg.append(bottom_neg[(s, y)])

        # Thin dark edge between stacked segments approximates the "2px surface gap"
        # spec in a static-image context - same approach plot_category_bars already uses.
        ax.bar(xs, pos_vals, bottom=bottoms_pos, width=bar_width, color=colour,
               edgecolor="black", linewidth=0.6, label=bucket_label)
        ax.bar(xs, neg_vals, bottom=bottoms_neg, width=bar_width, color=colour,
               edgecolor="black", linewidth=0.6)

        idx = 0
        for s in scenario_order:
            for y in years:
                bottom_pos[(s, y)] += pos_vals[idx]
                bottom_neg[(s, y)] += neg_vals[idx]
                idx += 1

    for marker_spec in reference_markers:
        values = marker_spec["values"]
        first = True
        for s in scenario_order:
            for y in years:
                if (s, y) not in values:
                    continue
                x = x_positions[(s, y)]
                scatter_kwargs = {
                    "marker": marker_spec.get("marker", "D"),
                    "s": marker_spec.get("size", 40),
                    "zorder": 5,
                    "linewidths": marker_spec.get("linewidths", 1.2),
                    "label": marker_spec["label"] if first else None,
                }
                if "color" in marker_spec:
                    # Single-color marker (e.g. unfilled "x"/"+") - matplotlib ignores
                    # facecolor/edgecolor for these anyway, and passing both alongside
                    # `color` makes `color` silently lose to the facecolor/edgecolor
                    # defaults below, which is exactly the bug this branch avoids.
                    scatter_kwargs["color"] = marker_spec["color"]
                else:
                    scatter_kwargs["facecolor"] = marker_spec.get("facecolor", COLOR_SURFACE)
                    scatter_kwargs["edgecolor"] = marker_spec.get("edgecolor", COLOR_PRIMARY_TEXT)
                ax.scatter(x, values[(s, y)], **scatter_kwargs)
                first = False

    ax.axhline(0, color=COLOR_BASELINE, linewidth=0.8)
    ax.yaxis.grid(True, color=COLOR_GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
 #   for spine in ("top", "right"):
 #       ax.spines[spine].set_visible(False)
 #   ax.spines["left"].set_color(COLOR_BASELINE)
 #   ax.spines["bottom"].set_color(COLOR_BASELINE)
    ax.set_ylabel(ylabel, color="black")
    ax.tick_params(axis="y", colors="black")

    year_ticks = [x_positions[(s, y)] for s in scenario_order for y in years]
    year_labels = [str(y) for _ in scenario_order for y in years]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels, fontsize=8, rotation=45, color=COLOR_SECONDARY_TEXT)
    ax.tick_params(axis="x", colors=COLOR_MUTED)

    group_centers = [
        i * (group_span + group_spacing) + (group_span - bar_width) / 2 for i in range(len(scenario_order))
    ]
    for center, scenario in zip(group_centers, scenario_order):
        ax.annotate(
            str(scenario), xy=(center, 0), xycoords=("data", "axes fraction"),
            xytext=(0, -32), textcoords="offset points", ha="center", va="top",
            fontsize=11, color=COLOR_PRIMARY_TEXT,
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin * 1.1 if ymin < 0 else ymin, ymax * 1.05)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ordered_labels = list(seen.keys())
    ax.legend(
        [seen[l] for l in reversed(ordered_labels)], list(reversed(ordered_labels)),
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9, frameon=True,
    )

    ax.set_title(title, color=COLOR_PRIMARY_TEXT, fontsize=12, pad=14)
    fig.savefig(out_path, dpi=200, facecolor=COLOR_SURFACE, bbox_inches="tight")
    return fig, ax
