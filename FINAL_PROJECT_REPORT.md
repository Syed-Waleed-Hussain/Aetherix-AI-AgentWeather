# Aetherix AI AgentWeather

Final internship report · 26 November 2026

## Problem and architecture

AgentWeather forecasts eight hourly weather and air-quality metrics for Karachi at horizons 1–24. The dashboard presents forecasts with their issue and valid times, makes threshold-based alerts accessible to different audiences, and shows the evidence behind model selection and drift decisions.

```text
Open-Meteo archive and air quality → aligned hourly data → versioned features
                                  → validation-selected models and seasonal reference
                                  → Forecast → Alert → Weather Report → Drift
                                  → local SQLite replay and Streamlit dashboard
                                  → reviewed read-only historical snapshot for hosting
```

## Data and features

The validated feature dataset contains **17,448** consecutive UTC hours from 14 November 2024 at 19:00 UTC through 11 November 2026 at 18:00 UTC. The weather archive provides reanalysis estimates. The Air Quality API provides modeled US AQI, PM2.5, and cloud-aware UV where needed. Historical precipitation probability is kept apart because the archived series lacks reliable per-value issue times. The rain target is derived from precipitation. [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) · [Air Quality API](https://open-meteo.com/en/docs/air-quality-api) · [Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)

The feature pipeline aligns timestamps in UTC and uses issue-time calendar features, lags, rolling summaries, and circular wind encoding. Training uses chronological partitions, with boundaries checked against target valid times. Imputers and scalers fit on training data. Validation selects the champion for each target and horizon. The held-out test period and expanding-window backtests serve different evaluation purposes.

## Models and evaluation

The local system uses Ridge, Random Forest regression, Random Forest classification for rain probability, and a previous-day seasonal reference. The statistical experiment evaluates SARIMAX. A CPU PyTorch LSTM evaluates 24 hourly temperature horizons. Wind direction uses circular error. Rain probability uses Brier score, alongside thresholded precision, recall, and F1. Precipitation amount retains the seasonal reference in the saved demo.

The following scores are from the matched temperature holdouts shipped with the demonstration. MAE is in °C. Compare models within a row.

| Experiment and horizon | Common hours | Seasonal naive | Ridge or Random Forest | SARIMAX | LSTM |
| --- | ---: | ---: | ---: | ---: | ---: |
| SARIMAX h01 | 2,617 | — | 0.284 Ridge | 0.238 | — |
| SARIMAX h24 | 2,594 | — | 0.594 Random Forest | 0.612 | — |
| LSTM h01 | 2,594 | 0.573 | 0.284 Ridge | 0.237 | 0.464 |
| LSTM h24 | 2,594 | 0.578 | 0.594 Random Forest | 0.612 | 0.731 |

SARIMAX led the displayed h01 comparison. The seasonal reference had the lowest h24 MAE on the LSTM cohort. The LSTM improved on the seasonal reference at h01 and trailed it at h24. Holdout performance varies by horizon and target. The dashboard exposes sample counts, MAE, RMSE, and R² across 24 horizons. The SARIMAX view verifies exact issue/target matches and equal observed values. The LSTM view verifies common timestamp fingerprints.

## Agents and dashboard

1. **Forecast** uses saved estimators and validation selections to produce horizon-level values and model provenance.
2. **Alert** groups consecutive threshold exceedances into explicitly timed windows for heat, heavy rain, wind, UV, and AQI. The dashboard changes precautions by audience.
3. **Weather Report** summarizes forecast values and alerts. A deterministic provider works offline. Optional Groq formatting uses the same grounded facts.
4. **Drift Detection** compares feature distributions and matured forecast errors against reference windows. It can request manual review. SQLite records decisions and supports replay without inference.

The six sections are Overview, 24-Hour Forecast, Weather Alerts, AI Weather Report, Model Performance, and MLOps & Agent Monitoring. The shipped issue was **11 November 2026, 23:00 PKT**. Its 192 predictions cover temperature, humidity, wind speed and direction, precipitation amount, rain probability, UV, and US AQI. It includes one elevated UV window and a historical drift result with eight feature flags and 56 target/horizon error flags.

## Feature Store, Model Registry, and automation

Hopsworks targets the offline Feature Group `aetherix_hourly_weather_features` version 1, keyed by city and event timestamp. The prepared LSTM registry package carries checkpoint, dataset, and feature-schema metadata. See [verification](FINAL_VERIFICATION.md) for the current remote status.

GitHub Actions checks the Open-Meteo feed each hour at minute 17 UTC and evaluates training eligibility daily at 01:47 UTC. Both allow manual dispatch. Ingestion checks the last 24 completed hours and writes ephemeral output. The daily gate checks persistent dataset, validation, and drift compatibility. Training and champion promotion require a reviewed workflow with durable data and versioned artifacts.

## Limitations and future work

Forecasts address one city. Reanalysis weather and modeled air quality introduce source limitations. The free hosted demonstration reads a fixed snapshot, and current observations and fresh model forecasts are unavailable there. A durable ingestion path with cross-run deduplication, reproducible inference features, and a verified registry would support a future live release.
