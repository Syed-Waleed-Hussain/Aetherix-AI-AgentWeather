# AETHERIX AI — AgentWeather

A free, local-first Karachi weather and air-quality forecasting project. Day 1
provides live Open-Meteo forecast clients. Day 2 adds historical data ingestion,
timestamp alignment, data-quality reporting, causal feature engineering, and
24-hour prediction targets. No API key, paid service, database, or generated
training data is required.

Open-Meteo historical weather values are reanalysis/model estimates.
Air-quality and AQI values are modeled CAMS estimates, not direct sensor
observations and not predictions from a trained AETHERIX model.

## Requirements and installation

- Python 3.12
- Git
- Internet access for the live API smoke test and historical backfill

From the project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The implementation uses the Python standard library plus the existing
`requests` dependency. Tests use the existing `pytest` dependency. All generated
`data/` files are ignored by Git.

## Day 2 architecture

```text
Open-Meteo Historical Weather API ─┐
                                   ├─> data/raw/*.csv
Open-Meteo Air Quality API ─────────┘          │
                                               ├─> exact UTC outer join
                                               ├─> missing-data/QC report
                                               └─> causal features + targets
                                                    data/processed/features.csv
                                                    data/reports/quality_report.*
```

- `aetherix_weather/historical.py`: historical clients, date validation,
  timestamp normalization, source availability, and raw provenance.
- `aetherix_weather/storage.py`: atomic CSV/JSON persistence and timestamp-keyed
  upserts.
- `aetherix_weather/dataset.py`: exact-hour alignment, missing summaries,
  optional limited forward fill, quality checks, features, and targets.
- `scripts/backfill.py`: fetches and persists a historical range, then creates
  the processed dataset and reports.
- `scripts/build_features.py`: rebuilds processed data from existing raw CSVs
  without making API requests.

## Historical sources and supported variables

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
  at `https://archive-api.open-meteo.com/v1/archive`.
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
  at `https://air-quality-api.open-meteo.com/v1/air-quality`.
- Default location: Karachi, Pakistan (`24.8607`, `67.0011`), timezone
  `Asia/Karachi`.
- Weather variables requested: temperature (°C), relative humidity (%),
  precipitation (mm), precipitation probability (%), wind speed (km/h), wind
  direction (degrees), and UV index.
- Air-quality variables requested: US AQI and PM2.5 (µg/m³).

The historical archive accepts hourly precipitation-probability and UV
variables, but the executed 90-day range returned all-null arrays for both
(precipitation probability also had no unit). The pipeline records
`no_historical_values`, keeps those values missing, and creates no targets for
them. AQI and PM2.5 were available from the air-quality endpoint as modeled
historical data; they are not observations from a local sensor. The exact
returned variables and coverage are recorded in `data/raw/manifest.json` and
the quality reports.

The default 90-day request ends six days behind the current date as a
conservative buffer for incomplete/recent archive data. The APIs may still
return a shorter range, null values, or hourly gaps; those are reported rather
than filled with forecast data or fabricated values. Historical weather is a
reanalysis/model record; the air-quality endpoint returns modeled CAMS data.
See the linked API documentation for dataset coverage, update cadence, and
usage terms.

## Run a historical backfill

From the repository root:

```powershell
python -m scripts.backfill --days 90
```

Optional arguments:

```powershell
python -m scripts.backfill --days 30 --end-date 2026-10-01
python -m scripts.backfill --days 90 --max-forward-fill-hours 2
python -m scripts.backfill --days 90 --data-dir .\data
```

Dates are inclusive. An explicit `--end-date` later than the conservative
latest-full-day cutoff is rejected. Re-running a range upserts records by exact
UTC timestamp; it does not append duplicate rows. Successful runs write:

- `data/raw/weather.csv` and `data/raw/air_quality.csv`
- `data/raw/manifest.json` with sources, units, availability, coverage, and
  schemas
- `data/processed/features.csv`
- `data/reports/quality_report.json` and `quality_report.md`

CSV raw rows retain the original API time string (`source_timestamp`), a
timezone-aware UTC key (`timestamp_utc`), Karachi-local time
(`timestamp_local`), and the UTC ingestion timestamp. Units and provider
attribution are maintained in the manifest.

## Rebuild the feature dataset

After a backfill (or when raw CSVs already exist), run:

```powershell
python -m scripts.build_features
```

This command reads only local files and does not make an API request. It also
accepts `--max-forward-fill-hours` and `--data-dir`.

## Alignment, quality, and missing-data policy

Both sources are normalized to aware UTC timestamps while retaining their
Karachi-local representations. The join is an **exact-hour outer join**: no
rounding or shifting occurs, and a row remains if just one source has a value.
Duplicate or non-monotonic timestamps and inconsistent UTC/local conversion
are rejected. Source-only timestamps, missing hourly periods, metric null
percentages, invalid value ranges, and source/feature schema changes are
recorded in the quality report.

Missing values are left unchanged by default. Per-metric missing indicators
are included in processed rows. Optional forward fill is causal and bounded;
it applies only to temperature, humidity, wind, and UV fields. Rainfall,
precipitation probability, AQI, and PM2.5 are never forward-filled. There is no
backfill, backward fill, or centered interpolation. Suspicious measurements
are reported, not automatically deleted.

## Dataset schema and features

Raw CSVs use one row per source timestamp. The processed file contains:

- UTC/local timestamp, source timestamp and source attribution/ingestion fields.
- Weather and air-quality metrics. Internal humidity, temperature, wind,
  precipitation, UV, AQI, and PM2.5 column names preserve their units in their
  names or the manifest.
- `<metric>_was_missing` and `<metric>_imputed` indicators.
- Calendar features: local hour, weekday, month, weekend, cyclical hour, and
  cyclical day-of-year.
- Causal lag features at 1, 3, 6, 12, 24, and 48 hours; trailing rolling means
  at 3, 6, 12, and 24 hours; 24-hour rolling standard deviation; and changes at
  1, 3, and 24 hours.
- Wind direction sine/cosine components, avoiding the 0°/360° discontinuity.

Lag/rolling values require the exact expected hourly timestamps; rolling
windows also require complete observations. A gap or null therefore produces
a null feature rather than borrowing a non-adjacent or future value. Rolling
windows include the current observation and past observations only.

## 24-hour targets

For each timestamp `T`, targets use the observed row at the exact
`T + horizon` UTC timestamp for horizons 1–24. Examples include
`target_temperature_h01`, `target_temperature_h24`,
`target_us_aqi_h01`, and `target_precipitation_mm_h01`. Target columns are
created only where the metric has historical values; a missing future reading
remains null. The newest rows receive no target unless the exact future data
exists.

Historical precipitation probability may be null and has no target unless
actual target values are returned in a later range.
`target_rain_occurred_hNN` is an optional *different task*, derived from
observed/model precipitation (`precipitation_mm > 0`); it is not a probability
forecast. Wind direction targets are output as paired
`target_wind_direction_sin_hNN` and `target_wind_direction_cos_hNN` values.
Recover an angle with `atan2(sin, cos)`; do not treat degrees as an ordinary
linear target.

The report includes target availability counts per horizon and counts rows
with at least one future target. These are preparation statistics, not a claim
that a row is ready for every model or every horizon.

## Data-quality report and attribution

The report uses actual fetched/processed values and includes city/coordinates,
requested and returned dates, source row counts, available/unavailable
variables, missing percentages before and after cleaning, duplicates, gaps,
source alignment differences, imputation counts, schema changes, and target
coverage. It explicitly distinguishes Open-Meteo weather reanalysis from
modeled CAMS air quality.

Attribution and provider details:

- [Open-Meteo API documentation](https://open-meteo.com/en/docs)
- [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
- [Open-Meteo terms and attribution](https://open-meteo.com/en/terms)

## Tests and existing live smoke test

Run all deterministic tests offline:

```powershell
python -m pytest -q
```

The mocked suite covers the Day 1 clients and historical request/date/shape
validation, timezone normalization, exact alignment, missing values, features,
lags, rolling windows, circular wind handling, targets, incomplete horizons,
idempotent persistence, and future-leakage prevention.

The Day 1 live forecast smoke test remains separate:

```powershell
python -m scripts.smoke_test
```

## Known limitations and next steps

- Open-Meteo weather and AQI/PM2.5 values are modeled products, not local
  station/sensor observations.
- The tested historical weather range returned no precipitation-probability
  values. Rain occurrence is a distinct optional target.
- The conservative six-day archive cutoff may exclude some recently available
  records; actual API coverage and gaps appear in reports.
- No training, evaluation, drift detection, dashboard, agent/LLM, database, or
  Hopsworks integration is included in Day 2.
