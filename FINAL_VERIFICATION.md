# Final verification

26 November 2026 · Python 3.12.10 · Day 8 working tree on `main`

## Release audit

| Priority | Finding | Status |
| --- | --- | --- |
| Blocker | A clean checkout lacked forecast and evaluation data because the original inputs and artifacts are local. | Fixed. The default hosted mode loads the reviewed 93,246-byte historical bundle. |
| Important | A provider check can succeed on an ephemeral runner while storing no durable data. | Explicit in the command output and dashboard documentation. The daily gate blocks training without verified persistence. |
| Important | A weather-feed response does not provide inference-ready current features. | Current dashboard observation remains unavailable. The local inference source uses the validated historical feature file and its recorded lineage. |
| Optional | Hosted replay, detailed drift tables, and model artifacts. | The local workspace supplies these views. The hosted demo shows forecasts, report, alerts, evaluation results, and drift summary without them. |

## Tests and user journey

- The full Pytest suite passed **111 tests**, with one opt-in Hopsworks integration case skipped in the default environment. That case passed separately in the Hopsworks SDK environment. Python compilation, installed dependency checks, and YAML parsing passed.
- A real run at **11 November 2026, 07:00 UTC** produced 192 finite forecasts across eight metrics, two UV alert windows, a deterministic report, and a forecast-error increase finding. All four nodes completed. Its replay reproduced the complete recorded state. SQLite returned `integrity_check=ok`.
- In a real browser, the local forecast button generated a new run from **11 November 2026, 18:00 UTC** using the existing estimators, and the earlier **07:00 UTC** issue was reselected via the saved-run control. The two QA runs wrote eight decisions in the separate QA SQLite file.
- Browser navigation rendered all six snapshot sections. Verified all eight Plotly metrics and their units in the clean-checkout browser session, as well as the UV alert's audience-dependent precaution, the deterministic report, both matched performance views, and the drift summary. The Overview was reloaded after fixing a Streamlit expression that emitted an internal object description.
- Clean-source AppTest passed all six sections and eight charts with Plotly 6.9.0, no overrides, and database access and model/cloud imports blocked. Forecast values, source units, rain-probability percentage conversion, model labels, and exact issue-plus-horizon timing were validated against the saved predictions.
- A separate synthetic workflow produced 192 fixture values, five alert windows, and `input_shift`. Its provenance identifies it as synthetic.
- The Python 3.12 Linux dashboard dependency set resolved on PyPI with 36 packages. The first Community Cloud build and resource use remain unverified.
- The targeted secret scan found zero matches in 65 release candidate files and 1,042 reachable Git blobs for configured secret values and common token/key signatures. The snapshot excludes run identifiers, raw decision logs, and credentials. The repository excludes local datasets, checkpoints, SQLite databases, and secret files. `git diff --check` passed.

## Fingerprints

| File | SHA-256 |
| --- | --- |
| Local `features.csv` | `9c36d4b51e0f5fbd3d961e51d7155ccf6df68e2be985a814976e6e2df05c9da6` |
| Local `selected_models.joblib` | `9c3d5e8d58af51d345242689b331569da644ce0efa912477b83bc23a134ffe42` |
| Local `lstm_checkpoint.pt` | `8c871b82582c19cecaf161aa342cc9e795d032218e95d29e3a7791a7fa055cfc` |
| Local `champion_selection.json` | `eb0ce64f4d2143f0a37c1608fd22da9f9b3654011ee9fd84d642e123ca96192c` |
| Bundled `dashboard/demo_snapshot.json` | `2a07493a0cfbbdc1d91fd83e32f469535dc530ef0af7c20ae76ee72c5b3c5a6c` |

The dataset digest matched the saved classical training metadata, validation manifest, and real run. The LSTM digest matched the local registry metadata, and the checkpoint passed the existing load check. The forecast bundle's 192 predictions and alert content equal the archived SQLite-replayed output.

## External services

- GitHub Actions reported success on published commit `31340239ed407ea51e1b4bc620193312b823df80`. The Day 8 working tree has no remote run yet. [CI](https://github.com/Syed-Waleed-Hussain/Aetherix-AI-AgentWeather/actions/runs/45157588509) · [Hourly ingestion](https://github.com/Syed-Waleed-Hussain/Aetherix-AI-AgentWeather/actions/runs/45759194416) · [Daily gate](https://github.com/Syed-Waleed-Hussain/Aetherix-AI-AgentWeather/actions/runs/45595607840)
- Both scheduled jobs use read-only repository permissions, disable persisted checkout credentials, have separate concurrency groups and five-minute timeouts, and can be dispatched manually. Neither needs repository secrets. Ingestion validates the 24 requested UTC keys and rejects duplicates within a run. Overlapping runs still need deduplication in a durable destination.
- The local Open-Meteo check returned **24 provider-modeled hours**, from **25 November at 10:00 UTC** through **26 November at 09:00 UTC**, with `persistent_write: false`. The local eligibility gate returned `blocked_unverified_persistent_storage` and requested manual review. Training and champion promotion did not run. A runner without the dataset stops at `blocked_missing_persistent_dataset`.
- Hopsworks read-only checks verified the project, Feature Store and Model Registry service access, and exactly **24** existing offline rows with the expected schema and values. Local registry verification passed. The remote registry had no equivalent LSTM version.
- Full Feature Group synchronization and Model Registry registration await authorization. The full-sync estimate is **24,700,648 bytes** of available storage and four materialization batches. The project metadata returned no quota usage fields and the quota portal required sign-in. Account-specific remaining storage and compute are unverified. [Hopsworks Free plan](https://www.hopsworks.ai/pricing)

## Prepared Hopsworks commands

The first block performs cloud reads. The second requires authorization and verified Free-plan headroom. Set `$remainingBytes` to the observed available storage before running the full sync. The sync verifies and preserves existing keys.

```powershell
.\.venv-hopsworks\Scripts\python.exe -m scripts.hopsworks_connection_check
.\.venv-hopsworks\Scripts\python.exe -m scripts.sync_hopsworks_features --full --preflight
.\.venv-hopsworks\Scripts\python.exe -m scripts.prepare_model_registry --check-remote
```

```powershell
.\.venv-hopsworks\Scripts\python.exe -m scripts.sync_hopsworks_features --full --execute --confirm-write --confirm-free-plan --free-storage-remaining-bytes $remainingBytes
.\.venv-hopsworks\Scripts\python.exe -m scripts.prepare_model_registry --execute --confirm-write
```

## Completion checklist

- [x] Validate local historical forecast, alerts, report, drift, persistence, and replay.
- [x] Validate read-only mode, the six dashboard sections, and release data provenance.
- [x] Validate the scheduled workflows and link their latest observed remote runs.
- [x] Deliver the project report, deployment guide, jury script, and verification record.
- [ ] Authorize commit and push of the reviewed release files.
- [ ] Authorize public Community Cloud deployment and verify the first build. No public app URL exists for this release.

Cloud writes and a verified live pipeline are outside the snapshot release path.
