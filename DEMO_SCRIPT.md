# Jury demonstration

## 0:00–0:35 · Overview

Open **Overview**. Point to **HISTORICAL** and **Live observation · UNAVAILABLE**. Say that the issue was 11 November 2026 at 23:00 Karachi time and that the overview cards show the saved inputs at that hour. The 24-hour forecast uses those archived inputs. Weather is reanalysis and AQI is modeled.

## 0:35–1:15 · Forecast and alerts

Open **24-Hour Forecast**. Show Temperature, then Rain probability or Wind direction. Point out the unit, valid time, and per-hour model in the chart or table. Explain that the Forecast Agent chooses the saved validation champion for each target and horizon. Open **Weather Alerts** and change **Audience** from General public to Outdoor workers. The UV window and threshold stay fixed while the precaution changes.

## 1:15–1:50 · Report and decisions

Open **AI Weather Report**. Connect its UV window and temperature range to the previous two sections. Explain that the Alert Agent turns threshold exceedances into windows and the Weather Report Agent summarizes facts from predictions and alerts. Open **MLOps & Agent Monitoring**. The Drift Agent tracks input shifts and matured error increases, and can request manual review of the model.

## 1:50–2:40 · Holdout results

In **Model Performance**, choose **SARIMAX** at hour 1. Explain that the matched 2,617-hour cohort gives SARIMAX 0.238 °C MAE and Ridge 0.284 °C. Switch to **LSTM**, show hour 24 on its own matched cohort, and compare 0.578 °C seasonal-naive MAE, 0.594 °C Random Forest, and 0.731 °C LSTM. Say that simpler methods were competitive and validation decisions were made before looking at the holdout.

## 2:40–3:45 · Local inference and cloud status

On the local app, open **Run local forecast** and click **Run 24-hour forecast**. While it runs, explain that loaded estimators execute the four-agent graph without training. Then select an earlier run under **Saved run** to show SQLite replay. If using the hosted snapshot, show the disabled forecast button and the saved agent statuses instead.

Explain that scheduled ingestion validates provider hours on an ephemeral runner. Daily drift evaluation checks eligibility for manual review. Hopsworks storage and registry operations are tracked separately from the hosted demo. See [verification](FINAL_VERIFICATION.md) for the observed remote status.
