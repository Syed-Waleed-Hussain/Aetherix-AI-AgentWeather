# AETHERIX AI — AgentWeather

A zero-cost, beginner-friendly foundation for a Karachi weather and air-quality
forecasting project. Day 1 retrieves hourly forecast data from the public
[Open-Meteo weather API](https://open-meteo.com/en/docs) and
[Open-Meteo air-quality API](https://open-meteo.com/en/docs/air-quality-api).
Neither API requires an API key.

> Open-Meteo data is a forecast from its weather/air-quality models. It is not
> a prediction produced by a future AETHERIX AI machine-learning model.

## Requirements

- Python 3.12
- Git
- Internet access for the live smoke test (unit tests are fully mocked)

## Install

From the project directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The `.env.example` file notes that the public Open-Meteo APIs do not need
credentials. Do not put secrets or local virtual environments in Git.

## Test

Run the mocked unit tests without making network requests:

```powershell
python -m pytest -q
```

GitHub Actions runs this same test command with Python 3.12 on pushes and pull
requests.

## Live API smoke test

With internet access, run:

```powershell
python -m scripts.smoke_test
```

The script fetches the next 24 hourly entries from each API and prints a
concise sample. It exits with a clear error if either request fails.

## Project layout

```text
.
├── .github/workflows/ci.yml
├── aetherix_weather/
│   ├── air_quality.py
│   ├── api.py
│   ├── config.py
│   ├── models.py
│   └── weather.py
├── scripts/
│   └── smoke_test.py
├── tests/
│   └── test_clients.py
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Python modules

- `config.py` holds the default Karachi coordinates, timezone, API URLs, and
  timeout.
- `api.py` shares HTTP request handling and validates timestamps, hourly arrays,
  numeric values, and missing values.
- `models.py` defines typed records for weather and air-quality hours.
- `weather.py` fetches temperature, humidity, precipitation, precipitation
  probability, wind, and UV index.
- `air_quality.py` fetches US AQI, PM2.5, and UV index.
- `scripts/smoke_test.py` checks both live APIs and prints the first hourly
  record from each.
- `tests/test_clients.py` uses mocked HTTP responses to verify client behavior
  offline.

## Scope

This is the Day 1 project foundation and API ingestion only. ML training,
LangGraph agents, Hopsworks, drift detection, and the dashboard are not part of
this phase.
