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

Day 1 and Day 2 use the Python standard library plus `requests`; Day 3 adds
Pandas, Matplotlib, and scikit-learn for local analysis and CPU-only models.
Tests use `pytest`. Generated `data/` and `artifacts/` files are ignored by Git.

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
- `scripts/refresh_historical_uv.py`: updates an existing air-quality dataset
  with cloud-aware hourly UV without re-fetching weather.
- `scripts/backfill_forecast_information.py`: stores forecast probabilities
  separately from observation-aligned features and targets.

## Historical sources and supported variables

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
  at `https://archive-api.open-meteo.com/v1/archive`.
- [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)
  at `https://air-quality-api.open-meteo.com/v1/air-quality`.
- [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
  at `https://historical-forecast-api.open-meteo.com/v1/forecast`.
- Default location: Karachi, Pakistan (`24.8607`, `67.0011`), timezone
  `Asia/Karachi`.
- Weather variables requested: temperature (°C), relative humidity (%),
  precipitation (mm), precipitation probability (%), wind speed (km/h), wind
  direction (degrees), and UV index.
- Air-quality variables requested: US AQI, PM2.5 (µg/m³), and cloud-aware UV
  Index.

The historical weather archive returned all-null hourly UV and precipitation
probability for the 90-day run. A separate 90-day request to the Air Quality
API returned 2,160 non-null hourly `uv_index` values for Karachi (including
nonzero daytime readings). The pipeline uses that cloud-aware UV Index, not
`uv_index_clear_sky`; its unit is the dimensionless UV index (the response's
`hourly_units` value is blank). US AQI and PM2.5 are modeled air-quality values,
not local sensor observations. The joined dataset records whether each UV
value came from the weather archive or the air-quality API.

The Historical Forecast API returned hourly precipitation-probability values
for the same 90-day period (unit `%`). This is **forecast data, not observed
rainfall**. The API response omits per-value model identity, run, issue time,
and lead time, and its documentation describes a continuous series stitched
from successive runs. It is persisted separately and excluded from features
and rain-occurrence targets because its as-of time cannot be verified. UV
availability, forecast metadata, and coverage are recorded in the manifests
and quality report.

The Historical Forecast API also supports daily `uv_index_max`, not hourly UV;
the separate cloud-aware hourly UV source above is used instead.

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
- `data/raw/historical_forecast_precipitation_probability.csv` is created
  separately by the optional forecast-information command below.
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

To refresh UV for an existing Day 2 raw dataset without re-fetching weather:

```powershell
python -m scripts.refresh_historical_uv
python -m scripts.build_features
```

To fetch the separately archived forecast probabilities for the date range in
the raw manifest:

```powershell
python -m scripts.backfill_forecast_information
```

This writes `data/raw/historical_forecast_precipitation_probability.csv` and
`data/raw/historical_forecast_manifest.json`. These forecast probabilities
remain separate from the model feature dataset and observed-rain labels.

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

The historical-weather archive's precipitation-probability values remain
missing; the separate Historical Forecast API probabilities are not targets.
`target_rain_occurred_hNN` is a rain-occurrence *classification label* with
the explicit rule `precipitation_mm > 0.0` (strictly greater than zero).
Missing future precipitation produces a missing label. Because the historical
weather source is reanalysis, this label is a modeled precipitation proxy,
not station-observed ground truth, and is not a probability forecast. Wind
direction targets are output as paired
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
  values. Forecast probabilities are available separately but lack per-record
  issue-time/lead-time metadata, so they are excluded from model inputs.
- Rain-occurrence labels use the archive's reanalysis precipitation as a proxy;
  verified rain-gauge observations are not included.
- The conservative six-day archive cutoff may exclude some recently available
  records; actual API coverage and gaps appear in reports.
- Drift detection, dashboard, agent/LLM, database, and Hopsworks integration
  remain out of scope.

## Day 3: exploratory analysis and first forecasting models

Run the local EDA and baseline experiments from the repository root:

```powershell
python -m scripts.train_baselines
```

The command reads only `data/processed/features.csv` and local manifests. It
does not download data. Optional paths:

```powershell
python -m scripts.train_baselines --data-dir .\data --output-dir .\artifacts\day3
```

It generates:

- `artifacts/day3/eda/eda_report.md` and `dataset_inspection.json`.
- PNG charts for hourly temperature, humidity distribution, rainfall and rain
  occurrence, wind speed and circular direction, AQI/PM2.5, UV, missingness and
  correlations, and local time-of-day behavior.
- `artifacts/day3/model_evaluation.csv` with validation comparisons and final
  test scores by metric, horizon, and model, plus `model_evaluation.md`.
- `artifacts/day3/selected_models.json` with dataset hashes, feature lists,
  date ranges, split boundaries, validation selection scores, and estimator
  settings.
- `artifacts/day3/test_predictions.csv` with explicit UTC issue time, target
  time, horizon, selected-model prediction, and seasonal-naive prediction.
- `artifacts/day3/selected_models.joblib` containing only validation-selected
  fitted estimators (the seasonal-naive baseline needs no saved estimator).

### EDA and input data

EDA uses the actual processed hourly observations. The machine-readable
inspection records column names/types, null percentages, source attribution,
UTC coverage, and the 240 `target_*` columns in the input CSV. All target
columns are excluded from model inputs. The separate Historical Forecast API
probability dataset is not merged or used: it lacks per-value issuance and lead
times. The historical weather archive's all-null precipitation-probability
feature is dropped when its training segment contains no observed values.

### Chronological evaluation and leakage controls

The 70%/15%/15% chronological boundaries are based on issue-time ordering. For
each forecast horizon `h`, the final `h` issue rows before the train/validation
boundary and before the validation/test boundary are purged, keeping each
partition's future labels inside that partition. Rows with missing target
values are excluded per metric/horizon, not imputed. The test period is scored
only after all model choices have been made from validation results.

Ridge preprocessing uses a median imputer and standard scaler inside a
scikit-learn pipeline fitted on training rows only. Random Forest imputers are
also fit only on training rows. The lightweight CPU configuration is
deterministic. Seasonal naive uses the exact previous-day value at the target's
valid time; when that value is unavailable it is left unavailable, and rows
without a baseline value are omitted from like-for-like model comparisons.

Ridge alpha values are compared on validation data. Random Forest regression
and a probability-producing Random Forest classifier are compared with the
seasonal-naive baseline over horizons 1–24. Continuous targets use MAE, RMSE,
and R². Rain occurrence uses Brier score, precision, recall, and F1 at a 0.5
classification threshold. Wind direction is modeled as sine/cosine components
and evaluated with circular angular MAE/RMSE; ordinary linear R² is not used
for this circular target. The precipitation regressor's negative predictions
are floored at 0 mm. Model selection and test performance are reported per
metric and horizon; scores from different units are not combined.

This is an initial short-history experiment: 90 days of modeled/reanalysis data
cannot establish seasonal or year-over-year generalization. Rain labels are a
reanalysis precipitation proxy rather than rain-gauge ground truth; any
validation-selected model that loses to the baseline should be retained as a
negative result, not described as an improvement. LSTM, model registries,
forecast-probability training, dashboard, and Hopsworks are not included.

### Actual local run and observations

The Day 3 run used the existing **2,160** hourly rows (local dates
2026-08-14–2026-11-11), 414 processed columns, and **165 numeric candidate
inputs** before per-training-split removal of all-null features. The source CSV
also has 240 future-target columns; all were excluded from inputs. There were
no duplicate or missing UTC hours. Joined UV is available for all rows, while
archived weather UV and precipitation probability remain 100% missing.

| Metric | Mean | Observed range |
| --- | ---: | ---: |
| Temperature | 28.37 °C | 20.1–37.2 °C |
| Relative humidity | 70.17% | 10–99% |
| Precipitation | 0.0138 mm/hour | 0–1.8 mm |
| Wind speed | 11.06 km/h | 0.3–23.2 km/h |
| US AQI | 82.54 | 59–158 |
| PM2.5 | 25.58 µg/m³ | 8.6–75 µg/m³ |
| UV Index | 1.54 | 0–9.75 |

Rain occurrence (`precipitation_mm > 0`) was positive for 212/2,160 hours
(9.81%). All positive rain hours fell in the training period; there were **no
positive labels in validation or test**. Consequently, the rain test Brier
score of 0 and zero precision/recall/F1 do **not** demonstrate useful rain
forecast skill. Precipitation test MAE/RMSE are likewise 0 because the holdout
periods had no rain; R² is left undefined for a constant target.

The base chronological boundaries contain 1,512 train, 324 validation, and 324
test issue-time rows. Purging and target availability give, at h01, 1,511/323/323
usable train/validation/test rows; at h24, 1,488/300/300. Models are selected
per metric/horizon by validation MAE, circular MAE, or Brier score, then the
test partition is used only for the selected candidate and seasonal-naive
comparison.

Test MAE (or circular MAE / Brier score for the corresponding targets) at h01
and h24 was:

| Metric | h01 selected (score; seasonal naive) | h24 selected (score; seasonal naive) |
| --- | --- | --- |
| Temperature | Ridge α=10 (0.607; 0.889 °C) | Seasonal naive (0.878; 0.878 °C) |
| Humidity | Ridge α=10 (3.392; 11.904 percentage points) | Seasonal naive (12.240; 12.240 percentage points) |
| Precipitation | Seasonal naive (0; 0 mm) | Seasonal naive (0; 0 mm) |
| Wind speed | Ridge α=10 (1.057; 1.837 km/h) | Seasonal naive (1.849; 1.849 km/h) |
| US AQI | Ridge α=0.1 (2.379; 11.189) | Seasonal naive (10.017; 10.017) |
| UV Index | Seasonal naive (0.044; 0.044) | Seasonal naive (0.038; 0.038) |
| Wind direction | Ridge α=10 (19.13°; 22.49° circular MAE) | Seasonal naive (22.65°; 22.65° circular MAE) |
| Rain occurrence | Seasonal naive (Brier 0; 0) | Seasonal naive (Brier 0; 0) |

At wind-direction h01, Ridge improves circular MAE but has higher circular RMSE
(32.72°) than seasonal naive (29.59°). Across all 24 test horizons, a
validation-selected learned model beat the seasonal-naive selection metric on
8 humidity horizons and 15 AQI horizons, plus one temperature, two wind-speed,
and one wind-direction horizon. Random Forest was selected on validation for
wind speed at h02–h13, but its test MAE was worse than the seasonal baseline at
11 of those 12 horizons. The seasonal naive was selected for every
precipitation, UV, and rain-occurrence horizon. These results are short-period
comparisons, not evidence of year-round generalization.

The complete 24-horizon MAE/RMSE/R² and classification metrics, test
predictions, selected-model metadata, and EDA charts are generated locally as
described above; no model scores are hard-coded into the pipeline.
