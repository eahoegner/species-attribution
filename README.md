# Species-attribution for ERF and GSAT change for emissions scenarios

This repository decomposes the Effective Radiative Forcing (ERF) and Global Surface Air Temperature Change (GSAT) for the CMIP7 ScenarioMIP scenarios assessed with the reduced complexity climate model MAGICCv7.6.0a3 into its contributions by different forcing agents. It further provides a decomposition of ERF and GSAT change from Tropospheric Ozone and Stratospheric Water Vapor into contributions from respective precursor-species (CH4, CO, NOx, VOC), and vice versa, the decomposition of effective ERF and GSAT change from CH4 and NOx into their respective channels (for CH4: CH4, F-Gases, Montreal Protocol Halogens, stratospheric H2O, tropospheric Ozone; for NOx: Aerosols direct and indirect, CH4, N2O, stratospheric H2O and tropospheric Ozone). 

Creates the climate assessment plots for the paper "The role of non-CO2 greenhouse gas emissions in future climate scenarios" (Harmsen et al., in prep.).

## Data inputs
This repo requires the historical emissions for CMIP7 ScenarioMIP emissions harmonisation, available at:
<https://zenodo.org/records/17845154>

Use the `global-workflow-xxx.csv`.


It further requires scenario-data from ScenarioMIP CMIP7, which is currently still under embargo, will be available at:
<https://zenodo.org/records/19825038>

Until then can be requested from the contacts given in the zenodo.

## Installation

We do all our environment management using [uv](https://docs.astral.sh/uv/).
To get started, you will need to make sure that uv is installed
([instructions here](https://docs.astral.sh/uv/getting-started/installation/),
we found that using uv's standalone installer was best on a Mac).

To create the virtual environment, run

```sh
uv sync
uv run pre-commit install
```

These steps are also captured in the `Makefile` so if you want a single
command, you can instead simply run `make virtual-enviroment`.

Having installed your virtual environment, you can now run commands in your
virtual environment using

```sh
uv run <command>
```

For example, to run Python within the virtual environment, run

```sh
uv run python
```

As another example, to run a notebook server, run

```sh
uv run jupyter lab
```

## Original template

This project was generated from this template:
[basic python repository](https://gitlab.com/openscm/copier-basic-python-repository).
[copier](https://copier.readthedocs.io/en/stable/) is used to manage and
distribute this template.
