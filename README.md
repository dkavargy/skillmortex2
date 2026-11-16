# SkillMORTEX 2.0

Code for the paper *Skill MORtality and EXtinction (SkillMORTEX): A framework for modelling the life cycle of
software skills using online job ads* (Kavargyris, Mittas and Angelis).

The repository holds the code only. The online job advertisement data come from the Eurostat Web Intelligence Hub
and are not redistributed here.

## What the code does

| Folder | Script | Purpose |
|---|---|---|
| `analysis/` | `01_aggregate_and_forecast_evaluation.py` | Builds monthly series from the raw files, audits them, and evaluates seven forecasting models against the naive and seasonal naive forecasts (expanding window, seven origins, horizons up to 36 months) |
| | `02_stage_rule.py` | The stage rule (Birth, Growth, Maturity, Decline, Revival, Death) and 2,000 joint draws of its six thresholds |
| | `03_relative_share.py` | The relative share and the stages computed on it |
| | `04_placebo_and_spike_audit.py` | Placebo-date test around November 2022 and audit of the earlier rapid-obsolescence rule |
| | `05_country_and_source_robustness.py` | Stage rule re-run by country, with fixed country weights, and by source stability |
| `figures/` | `fig1_...` to `fig11_...`, `smxfig.py` | One script per figure; `smxfig.py` holds the style, the data loading and the stage rule |
| `tables/` | `make_tables.py` | Writes all tables of the paper as LaTeX |
| `geo/` | `europe_50m.json` | Country outlines for the map (Natural Earth, public domain) |

## Running it

The analysis scripts are written as notebook cells (`# %%`) and are meant to run in order in one Python session
(for example Google Colab), because later scripts use objects defined by earlier ones.

    pip install -r requirements.txt
    export SMX_MODE=relative
    # run analysis/01 ... 05 in order; results are written to OUT_DIR (default /content/smx_r1)

    export SMX_DATA=/path/to/OUT_DIR
    python figures/fig5_stage_map.py        # one script per figure
    cd tables && python make_tables.py      # writes tables_body.tex

## Fixed choices

- Window: October 2018 to March 2025 (78 months); four ISCO-08 unit groups; 13 programming skills (ESCO).
- Relative share: advertisements that mention the skill divided by the sum of that count over the 13 skills, per occupation and month.
- Stage rule, baseline thresholds: W = 12, theta = 0.80, gamma = 0.10, delta = 0.10, rho = 0.20, k = 3.
- Joint threshold draws: 2,000, `numpy.random.default_rng(2026)`.

## Earlier version

The first SkillMORTEX framework, on Stack Overflow data: Kavargyris, D.C., Georgiou, K., Makris, N., Mittas, N.,
Angelis, L. (2027). SkillMORTEX: An Innovation Ecosystem for Monitoring Digital Skill Dynamics Using Stack Overflow
Data. In: Quality of Information and Communications Technology. QUATIC 2025. Lecture Notes in Computer Science,
vol 16316. Springer, Cham. https://doi.org/10.1007/978-3-032-21373-0_2
